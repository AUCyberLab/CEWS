import copy
from cwe_predictor_hier import CWEPredictor


class VulnerabilityIntelligencePipeline:
    def __init__(self, model_weight_path, build_static_graph_fn):
        """Initialize the full vulnerability intelligence analysis pipeline."""
        print("Initializing V2W-BERT predictor...")
        self.predictor = CWEPredictor(
            model_weight_path=model_weight_path,
            v2w_dataset_dir="./V2W-BERT/Dataset/CWE/processed/",
            pretrained_model_name='distilbert-base-uncased',
        )

        print("Building CWE-CAPEC static base knowledge graph...")
        self.base_scorer = build_static_graph_fn()
        print("Pipeline initialization complete.")

    def _get_effective_cwes(self, cve_id, description, known_cwes, enable_fallback):
        """
        Manager-level routing: probe the Scorer to decide whether the known CWEs
        produce a high-quality structural path. If the path is a dead-end or the
        top score is too low, fall back to V2W-BERT prediction (when enabled).
        """
        if known_cwes is None:
            known_cwes = []

        # === Case A: known CWEs are available ===
        if len(known_cwes) > 0:
            # Probe the graph in a sandbox so we don't pollute self.base_scorer
            sandbox_scorer = copy.deepcopy(self.base_scorer)
            for cwe in known_cwes:
                sandbox_scorer.add_cve_cwe_edge(cve_id, cwe, is_ground_truth=True)

            # Pull the single highest-scoring CAPEC to gauge connection quality
            test_results = sandbox_scorer.get_top_k_capecs_for_cve(cve_id, k=1, max_depth=4)

            is_valid_path = False
            top_score = 0.0

            if len(test_results) > 0:
                top_capec, top_score = test_results[0]
                # A score of 0.9+ indicates a strong structural mapping
                if top_score >= 0.9:
                    is_valid_path = True

            if is_valid_path:
                # Probe succeeded: known CWEs yield a clean, high-quality path
                return (
                    [{"cwe_id": c, "prob": 1.0, "is_truth": True} for c in known_cwes],
                    "Known_CWE_Direct",
                )

            # Probe failed: dead-end or score too low
            if enable_fallback:
                if len(test_results) == 0:
                    reason = "dead-end (no connected CAPEC)"
                else:
                    reason = f"top score too low ({top_score:.4f} < 0.9)"
                print(f"[{cve_id}] Known-CWE mapping flagged as low-quality [{reason}]. Falling back to V2W-BERT...")
                preds = self.predictor.predict(description)[:10]
                return (
                    [{"cwe_id": p['cwe_id'], "prob": p['probability'], "is_truth": False} for p in preds],
                    "Fallback_To_Prediction",
                )

            # Fallback disabled: keep the suboptimal known CWEs
            reason = "dead-end" if len(test_results) == 0 else "low-score connection"
            print(f"[{cve_id}] Known CWEs marked as {reason}, but fallback is disabled.")
            return (
                [{"cwe_id": c, "prob": 1.0, "is_truth": True} for c in known_cwes],
                "Known_CWE_Suboptimal",
            )

        # === Case B: no known CWEs at all ===
        print(f"[{cve_id}] No known CWEs, calling V2W-BERT directly...")
        preds = self.predictor.predict(description)[:10]
        return (
            [{"cwe_id": p['cwe_id'], "prob": p['probability'], "is_truth": False} for p in preds],
            "Predicted_Directly",
        )

    def evaluate_specific_pair(self, cve_id, target_capec, description, known_cwes=None, enable_fallback=True):
        """Evaluate the final structural score of a specific (CVE, CAPEC) pair."""
        scorer = copy.deepcopy(self.base_scorer)

        effective_cwes, status = self._get_effective_cwes(cve_id, description, known_cwes, enable_fallback)

        used_cwes = []
        for item in effective_cwes:
            scorer.add_cve_cwe_edge(
                cve_id,
                item['cwe_id'],
                is_ground_truth=item['is_truth'],
                bert_prob=item['prob'],
            )
            used_cwes.append(
                item['cwe_id'] if item['is_truth'] else f"{item['cwe_id']}({item['prob']:.2f})"
            )

        final_score, paths = scorer.evaluate_connection(cve_id, target_capec)

        return {
            "CVE ID": cve_id,
            "Target CAPEC": target_capec,
            "Link Status": status,
            "Used CWEs": ", ".join(used_cwes),
            "Structure Score": final_score,
            "Top Evidence Path": " -> ".join(paths[0]['path']) if paths else "No Path Found",
        }

    def generate_candidate_pool(self, cve_id, description, known_cwes=None, top_k=30, enable_fallback=True):
        """Retrieval task: generate a Top-K CAPEC candidate pool for a CVE."""
        scorer = copy.deepcopy(self.base_scorer)

        effective_cwes, status = self._get_effective_cwes(cve_id, description, known_cwes, enable_fallback)

        for item in effective_cwes:
            scorer.add_cve_cwe_edge(
                cve_id,
                item['cwe_id'],
                is_ground_truth=item['is_truth'],
                bert_prob=item['prob'],
            )

        top_capecs = scorer.get_top_k_capecs_for_cve(cve_id, k=top_k, max_depth=4)

        pool = []
        for rank, (capec_id, final_score) in enumerate(top_capecs, 1):
            pool.append({
                "Rank": rank,
                "CAPEC ID": capec_id,
                "Score": final_score,
            })

        return {
            "CVE ID": cve_id,
            "Strategy Triggered": status,
            "Candidate Pool": pool,
        }