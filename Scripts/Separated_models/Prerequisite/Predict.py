import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import re
import argparse

# ================= Configuration =================
parser = argparse.ArgumentParser(description="Run inference for CVE-CAPEC Prerequisite Prediction.")

parser.add_argument("--input_excel", type=str, required=True, help="Path to your Input Excel test set.")
parser.add_argument("--output_excel", type=str, required=True, help="Path to save the predictions Output Excel.")
parser.add_argument("--base_model", type=str, default="./base_model", help="Path to the base model directory.")
parser.add_argument("--lora_weights", type=str, required=True, help="Path to the trained LoRA adapter weights.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

args = parser.parse_args()

INPUT_EXCEL = args.input_excel
OUTPUT_EXCEL = args.output_excel
BASE_MODEL = args.base_model
LORA_WEIGHTS = args.lora_weights
BATCH_SIZE = args.batch_size
SEED = args.seed

print("================ Configuration ================")
print(f"Input Excel  : {INPUT_EXCEL}")
print(f"Output Excel : {OUTPUT_EXCEL}")
print(f"Base Model   : {BASE_MODEL}")
print(f"LoRA Weights : {LORA_WEIGHTS}")
print(f"Batch Size   : {BATCH_SIZE}")
print(f"Seed         : {SEED}")
print("===============================================")


# ================= Reproducibility =================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ================= Load Model =================
print(f"[System] Loading Base Model from: {BASE_MODEL}")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.padding_side = "left"
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa"
)

print(f"[System] Loading LoRA Adapter from: {LORA_WEIGHTS}")
if not os.path.exists(LORA_WEIGHTS):
    print(f"Error: LoRA path not found: {LORA_WEIGHTS}")
    exit(1)

adapter_config_path = os.path.join(LORA_WEIGHTS, "adapter_config.json")
if os.path.exists(adapter_config_path):
    with open(adapter_config_path, 'r') as f:
        config = json.load(f)
    if "alora_invocation_tokens" in config:
        print("[System] Detected non-standard key 'alora_invocation_tokens' in adapter config. Removing it...")
        del config["alora_invocation_tokens"]
        with open(adapter_config_path, 'w') as f:
            json.dump(config, f, indent=2)

model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)
model.eval()

# ================= Prompt Template =================
SYSTEM_INSTRUCTION = (
    "You are an expert cybersecurity analyst. Your task is to evaluate the logical match between a provided CVE description and a target CAPEC's prerequisites. If the CAPEC prerequisites are missing or empty, evaluate the match based on the CAPEC Name.\n\n"
    "Determine a match score: 1.0 (Strongly/Explicitly Matched), 0.5 (Implicitly/Partially Matched), or 0.0 (Mismatched/Contradicted).\n\n"
    "You must first provide a step-by-step reasoning process enclosed strictly within <think></think> tags, covering: [Step 1: Condition Extraction], [Step 2: Evidence Mapping], [Step 3: Gap Analysis], and [Step 4: Scoring Decision]. After the reasoning block, output exactly one JSON object containing the \"match_score\"."
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

# ================= Batch Inference =================

def run_prediction():
    set_seed(SEED)

    print(f"[System] Reading data from {INPUT_EXCEL}...")
    try:
        df = pd.read_excel(INPUT_EXCEL)
        df.fillna('', inplace=True)
        sort_cols = [c for c in ['CVE-ID', 'CAPEC-ID'] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    results = []
    parse_fail_count = 0
    data_records = df.to_dict('records')
    total_records = len(data_records)

    print(f"[System] Starting BATCH inference (Batch Size: {BATCH_SIZE})...")

    for i in tqdm(range(0, total_records, BATCH_SIZE)):
        batch_records = data_records[i : i + BATCH_SIZE]

        prompts = []
        for row in batch_records:
            cve_id = row.get('CVE-ID', '')
            cve_desc = row.get('CVE_Description', '')
            capec_id = row.get('CAPEC-ID', '')
            capec_name = row.get('CAPEC_Name', '')
            capec_prereq = row.get('CAPEC_Prerequisites', '')

            input_text = (
                f"Target CAPEC:\n"
                f"CAPEC-ID: {capec_id}\n"
                f"CAPEC Name: {capec_name}\n"
                f"CAPEC Prerequisites: {capec_prereq}\n\n"
                f"Evidence to Evaluate:\n"
                f"CVE-ID: {cve_id}\n"
                f"CVE Description: {cve_desc}"
            )
            prompts.append(alpaca_prompt.format(instruction=SYSTEM_INSTRUCTION, input=input_text))

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,          
                num_beams=1,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_len:]

        responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for j, response in enumerate(responses):
            row = batch_records[j]
            think_part = ""
            json_part = response
            match_score = None

            if "<think>" in response and "</think>" in response:
                parts = response.split("</think>")
                think_part = parts[0].replace("<think>", "").strip()
                json_part = parts[1].strip() if len(parts) > 1 else ""
            elif "<think>" in response:
                think_part = response.replace("<think>", "").strip()
                json_part = ""

            try:
                json_matches = re.findall(r'\{[^{}]*"match_score"[^{}]*\}', json_part, re.DOTALL)
                if json_matches:
                    parsed_json = json.loads(json_matches[-1])
                    match_score = parsed_json.get("match_score", None)
                else:
                    score_match = re.search(r'"match_score"\s*:\s*([0-9.]+)', json_part)
                    if score_match:
                        match_score = float(score_match.group(1))
            except Exception:
                pass

            if match_score is None:
                parse_fail_count += 1

            results.append({
                "CVE-ID": row.get('CVE-ID', ''),
                "CVE_Description": row.get('CVE_Description', ''),
                "CAPEC-ID": row.get('CAPEC-ID', ''),
                "CAPEC_Name": row.get('CAPEC_Name', ''),
                "Raw_Response": response,
                "Model_Reasoning": think_part,
                "Extracted_JSON": json_part,
                "Match_Score": match_score
            })

    output_dir = os.path.dirname(OUTPUT_EXCEL)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result_df = pd.DataFrame(results)
    result_df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n[System] Done! Results saved to {OUTPUT_EXCEL}")
    print(f"[System] Total samples: {len(results)} | Score parse failures: {parse_fail_count}")


if __name__ == "__main__":
    run_prediction()