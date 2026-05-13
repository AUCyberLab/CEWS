# CEWS: Auditable CVE-to-CAPEC Linking with CWE-Grounded Multi-Dimensional Evidence

CEWS (Constrained Evidence-Weighted Scoring) is an auditable framework that links CVEs to CAPEC attack patterns by combining **CWE-grounded structural traceability** with **multi-dimensional semantic evidence** across mechanism, prerequisites, description, and impact. Unlike prior approaches that conflate graph reachability or textual similarity with operational correctness, CEWS produces interpretable, dimension-level scores together with a final applicability score.


## Repository Structure

```
.
├── data/                     # Datasets (see Dataset section)
│   ├── Description/          # Dataset to train the description model
│   ├── Impact/               # Dataset to train the impact model
│   ├── Mechanism/            # Dataset to train the mechanism model
│   ├── Prerequisite/         # Dataset to train the Prerequisite model
│   ├── RankingEx/            # Ranking experiments dataset 
│   ├── Early_late_dataset/   # Dataset for temporal evaluation
│   ├── Fan-out-reduction/    # Dataset for candidate reduciton evaluation
│   ├── Test_dataset/         # Test and evaluation datasets
│   ├── capec_database.xlsx   # CAPEC info
│   └── cwe_parsed_data.xlsx  # CWE info
├── Scripts/                  # Training, inference, and evaluation scripts
│   ├── Evaluations/          # Evaluation scripts
│   ├── Separated_models/     # Fine-tune LLMs for features
│   ├── Ranking_Exp/          # Ranking experiments
│   ├── Test                  # Testing script
│   └── Train                 # Training script
├── prompts/                  # Prompt templates for the four semantic scorers
├── Mappings/                 
│   ├── cve_capec_mappings_656.xlsx/     # Our 656 curated CVE–CAPEC pairs
│   ├── cve_capec_mappings_human_labelled_177.xlsx/        # High-confidence expert-labeled subset
│   └── capec_example_GT_63.xlsx/        # CVE-CAPEC pairs extracted from CAPEC  'Example Instance' section
├── requirements.txt
└── README.md
```

### Requirements

- Python `>= 3.10`
- CUDA-capable GPU (recommended for fine-tuning and large-scale inference)
- Key dependencies: `transformers`, `peft`, `bitsandbytes`, `accelerate`, `torch`, `scikit-learn`, `pandas`

### Setup

```bash
git clone https://github.com/AUCyberLab/CEWS.git
cd CEWS

# (optional) create a clean environment
conda create -n cews python=3.10 -y
conda activate cews

pip install -r requirements.txt
```

## Usage

<!-- TODO: replace with your actual entry points -->

Training a semantic scorer:
Each of the four semantic dimensions (mechanism, prerequisite, description, impact) is fine-tuned independently as a LoRA adapter on top of a 4-bit NF4-quantized DeepSeek-distilled LLaMA-3.1-8B base model.
Example training command (description scorer):

```bash
python Scripts/Train.py \
    --model_name_or_path ./base_model \
    --data_path data/Description/Processed_Data/Description_train_data_all.json \
    --output_dir output/cve_des_finetune \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-4 \
    --max_seq_length 4096 \
    --train_on_output_only
```

Running CEWS inference:

Batch inference loads the fine-tuned LoRA adapter on top of the base model and produces per-pair scores.
Example inference command (description scorer):

```bash
python scripts/Description/Pre_notemp.py \
    --input_excel  data/New_test_metrics/Processed_Data/<test_set>.xlsx \
    --output_excel data/New_test_metrics/output/Description_predictions.xlsx \
    --base_model   ./base_model \
    --lora_weights output/cve_des_finetune/final_adapter \
    --batch_size 16
```

---

## Contact

Yingxin Xu — `yingxin.xu@adelaide.edu.au`