import sys
import os
import gc
import pickle
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from v2w_model_definitions import Model

# Allow importing V2W-BERT's Dataset module
sys.path.append(os.path.abspath("./V2W-BERT"))
try:
    from Dataset import getDataset
except ImportError:
    print("[Error] Could not find V2W-BERT/Dataset.py. Make sure you run this from the project root.")


class CWEPredictor:
    def __init__(
        self,
        model_weight_path,
        v2w_dataset_dir="./V2W-BERT/Dataset/CWE/processed/",
        pretrained_model_name='distilbert-base-uncased',
        device='cuda',
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=True)

        print("[System] Loading model weights...")
        mock_args = {'pretrained': pretrained_model_name, 'freeze': 'False', 'lm_lambda': 0.1}
        self.model = Model(**mock_args)

        # Load checkpoint to CPU first to avoid duplicating it on GPU
        print(f"[System] Reading checkpoint to CPU memory: {model_weight_path}")
        checkpoint = torch.load(model_weight_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Strip the 'module.' prefix if the checkpoint was saved from a DataParallel/DDP model
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)

        del checkpoint
        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.to(self.device)
        self.model.eval()

        with open("./V2W-BERT/Dataset/CWE/label2idx_cwe.json", "r") as f:
            raw = json.load(f)
            self.idx2cweid = {int(k): str(v) for k, v in raw.items()}
            self.cweid2idx = {str(v): int(k) for k, v in raw.items()}

        self._load_graph_and_cache_embeddings(v2w_dataset_dir)

    def _load_graph_and_cache_embeddings(self, dataset_dir):
        """Load V2W-BERT's serialized graph and pre-compute CWE class embeddings."""
        print(f"[System] Reading serialized graph and texts from: {dataset_dir}")

        # The two static files V2W-BERT produces during preprocessing
        mask_file = os.path.join(dataset_dir, "NVD_data")
        cve_file = os.path.join(dataset_dir, "NVD_CVE.csv")

        if not os.path.exists(mask_file) or not os.path.exists(cve_file):
            raise FileNotFoundError(
                f"[Error] Required dataset files missing. Expected NVD_data and NVD_CVE.csv under {dataset_dir}"
            )

        # Load the serialized Data object (contains hierarchy and topology)
        # Note: if this raises ModuleNotFoundError: 'ipynb', run `pip install ipynb` and ensure
        # the script is launched from V2W-BERT's parent directory.
        try:
            with open(mask_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"[Error] Failed to load graph structure (NVD_data): {e}")
            raise

        # Reconstruct the level -> CWE-indices map and pull out the root nodes
        self.level0_indices = []
        levels = {}
        if hasattr(data, 'depth'):
            for cwe_idx, depth_vals in data.depth.items():
                if isinstance(depth_vals, int):
                    depth_vals = [depth_vals]
                for d in depth_vals:
                    if d not in levels:
                        levels[d] = []
                    if cwe_idx not in levels[d]:
                        levels[d].append(cwe_idx)
            if 0 in levels:
                self.level0_indices = levels[0]

        self.parent_child_idx = data.parent_child if hasattr(data, 'parent_child') else {}

        # Read the CSV that contains the descriptive text for every node
        print("[System] Extracting CWE class descriptions...")
        df_merged = pd.read_csv(cve_file, low_memory=False)

        # V2W-BERT appends pure-CWE rows at the end of the CSV; class_mask flags them
        class_mask_idx = (data.class_mask == True).nonzero().flatten().numpy()

        cwe_sentences = []
        for i in class_mask_idx:
            # Match the truncation behaviour used during training (first 512 chars)
            desc = str(df_merged.iloc[i].get('CVE Description', ''))
            cwe_sentences.append(desc[:512])

        # Encode and cache CWE embeddings
        print(f"[System] Encoding {len(cwe_sentences)} CWE class descriptions...")
        encoded = self.tokenizer(
            cwe_sentences,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            batch = {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask']}
            _, self.cwe_pooled = self.model.base_model(batch)
            self.n_cwes = self.cwe_pooled.shape[0]

        print(f"[System] Ready. Cached {self.n_cwes} CWE embeddings. Found {len(self.level0_indices)} root nodes.")

    def _get_hier_indices(self, parent_idx, probs, k):
        """Return the top-k child indices of `parent_idx` ranked by their probability."""
        if parent_idx not in self.parent_child_idx:
            return []

        children = self.parent_child_idx[parent_idx]
        if not children:
            return []

        child_probs = probs[children]
        k_child = min(len(children), k)
        top_relative_indices = np.argsort(-child_probs)[:k_child]

        return [children[i] for i in top_relative_indices]

    def predict(self, cve_description, k_list=[6, 5, 4, 3, 2, 1]):
        with torch.no_grad():
            encoded = self.tokenizer(
                [cve_description],
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors='pt',
            ).to(self.device)

            _, cve_pooled = self.model.base_model(encoded)
            cve_repeated = cve_pooled.repeat(self.n_cwes, 1)
            _, logits = self.model.link_model(cve_repeated, self.cwe_pooled)
            probs = torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy()

            retrieved_indices = set()

            # Level 0: pick the top roots
            l0_probs = probs[self.level0_indices]
            k0 = min(len(self.level0_indices), k_list[0])
            top_l0_relative = np.argsort(-l0_probs)[:k0]
            top_l0_global = [self.level0_indices[i] for i in top_l0_relative]

            # Walk the hierarchy down to depth 5, pulling the top-k children at each level
            for l0 in top_l0_global:
                retrieved_indices.add(l0)
                level1 = self._get_hier_indices(l0, probs, k_list[1] if len(k_list) > 1 else 1)
                for l1 in level1:
                    retrieved_indices.add(l1)
                    level2 = self._get_hier_indices(l1, probs, k_list[2] if len(k_list) > 2 else 1)
                    for l2 in level2:
                        retrieved_indices.add(l2)
                        level3 = self._get_hier_indices(l2, probs, k_list[3] if len(k_list) > 3 else 1)
                        for l3 in level3:
                            retrieved_indices.add(l3)
                            level4 = self._get_hier_indices(l3, probs, k_list[4] if len(k_list) > 4 else 1)
                            for l4 in level4:
                                retrieved_indices.add(l4)
                                level5 = self._get_hier_indices(l4, probs, k_list[4] if len(k_list) > 4 else 1)
                                for l5 in level5:
                                    retrieved_indices.add(l5)

            if not retrieved_indices:
                # Hierarchical walk found nothing useful; fall back to a flat top-N
                final_indices = np.argsort(-probs)[:sum(k_list)].tolist()
            else:
                final_indices = list(retrieved_indices)

            final_scores = probs[final_indices]
            sorted_relative = np.argsort(-final_scores)

            results = []
            for relative_idx in sorted_relative:
                idx = final_indices[relative_idx]
                results.append({
                    "cwe_id": self.idx2cweid.get(idx, f"UNKNOWN_{idx}"),
                    "probability": float(probs[idx]),
                })

            return results