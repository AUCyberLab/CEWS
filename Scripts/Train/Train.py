import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
import os
import argparse

# ================= 0. Argument Parsing =================
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="./base_model", help="Path to the base model")
parser.add_argument("--data_path", type=str, required=True, help="Path to training data (.jsonl or .json)")
parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--max_seq_length", type=int, default=4096)

parser.add_argument("--disable_lora", action="store_true", help="Disable LoRA if set")
parser.add_argument("--no_bf16", action="store_true", help="Disable bf16 if set")
parser.add_argument("--train_on_output_only", action="store_true", help="Compute loss only on the Response portion")

args = parser.parse_args()

USE_LORA = not args.disable_lora
USE_BF16 = not args.no_bf16

print(f"Model Path: {args.model_name_or_path}")
print(f"Data Path:  {args.data_path}")
print(f"Output Dir: {args.output_dir}")
print(f"BF16 Mode:  {USE_BF16}")
print(f"Train on Output Only: {args.train_on_output_only}")

# ================= 1. Load Model and Tokenizer =================
print("Loading tokenizer and model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="sdpa"
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# ================= 2. Configure LoRA =================
if USE_LORA:
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

# ================= 3. Prepare Dataset =================
alpaca_prompt_base = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
"""

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]

    if args.train_on_output_only:
        prompts = []
        completions = []
        for instruction, input_text, output_text in zip(instructions, inputs, outputs):
            prompt = alpaca_prompt_base.format(instruction, input_text)
            completion = output_text + tokenizer.eos_token
            prompts.append(prompt)
            completions.append(completion)
        return {"prompt": prompts, "completion": completions}
    else:
        texts = []
        for instruction, input_text, output_text in zip(instructions, inputs, outputs):
            text = alpaca_prompt_base.format(instruction, input_text) + output_text + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

if not os.path.exists(args.data_path):
    raise FileNotFoundError(f"Data file not found at: {args.data_path}")

dataset = load_dataset("json", data_files=args.data_path, split="train")

original_columns = dataset.column_names
dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=original_columns)

# ================= 4. Training Configuration =================
print("Starting training setup...")

sft_config_kwargs = dict(
    output_dir=args.output_dir,
    max_length=args.max_seq_length,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=3,
    learning_rate=args.learning_rate,
    optim="paged_adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=not USE_BF16,
    bf16=USE_BF16,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False}
)

if args.train_on_output_only:
    sft_config_kwargs["dataset_text_field"] = None
else:
    sft_config_kwargs["dataset_text_field"] = "text"

sft_config = SFTConfig(**sft_config_kwargs)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    processing_class=tokenizer,
    args=sft_config,
    peft_config=peft_config,
)

# ================= 5. Train and Save =================
print("Training started...")
trainer.train()

print("Saving final model...")
final_save_path = os.path.join(args.output_dir, "final_adapter")

trainer.model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)

print(f"Training complete! Model saved to {final_save_path}")