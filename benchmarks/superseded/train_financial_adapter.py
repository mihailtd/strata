"""DEPRECATED -- use scripts/train_stock_lora_bf16.py --domain financial_planning

Superseded because per-domain trainers drifted apart in methodology: this script
was the ONLY one applying Liger fused kernels, so the financial expert was
trained differently from astral and postgres while all three were compared as if
matched. train_stock_lora_bf16.py now carries the same bf16 + Liger path for
every domain and writes regime.json so provenance is recorded rather than
inferred.

Kept only so the exact provenance of the existing ctl_lora_fin_a128 stays
readable. Do not use for new work.

Fine-tune Stock LoRA (r=8, alpha=128, scaling 16) on cleaned financial planning dataset.

Dataset: data/financial_planning/training_data.jsonl (304 clean records)
Output: results/adapters/ctl_lora_fin_a128
"""

import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.novel_peft import set_hard_vram_cap  # noqa: E402

DATASET_PATH = REPO_ROOT / "data/financial_planning/training_data.jsonl"
OUT_DIR = REPO_ROOT / "results/adapters/ctl_lora_fin_a128"


def load_dataset_records():
    records = []
    with open(DATASET_PATH) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                records.append({"text": data["text"]})
    return records


def main():
    set_hard_vram_cap(22.0)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🚀 TRAINING FINANCIAL STOCK LORA (r=8, alpha=128) ON {device_name}")
    print("═" * 90)

    model_id = "Qwen/Qwen3.5-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Applying Liger Kernel specific patch for Qwen 3.5...")
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
    apply_liger_kernel_to_qwen3_5()

    print("Loading base model Qwen/Qwen3.5-4B in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset_records = load_dataset_records()
    print(f"Loaded {len(dataset_records)} dataset records from {DATASET_PATH}")

    # Use datasets Library for SFTTrainer
    from datasets import Dataset

    train_dataset = Dataset.from_list(dataset_records)

    sft_config = SFTConfig(
        output_dir=str(OUT_DIR / "checkpoints"),
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_steps=150,
        logging_steps=10,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        save_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("Starting SFT Training (150 steps)...")
    trainer.train()

    print(f"Saving final Stock LoRA adapter to {OUT_DIR}...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print("✅ Training complete and adapter saved successfully!")


if __name__ == "__main__":
    main()
