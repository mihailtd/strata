"""Unsloth AMD Fast Training Micro-Probe Benchmark (500 steps, Qwen3.5-2B).

Runs 500-step QLoRA/LoRA micro-probe training using Unsloth on AMD ROCm 7.2.
Logs throughput, peak VRAM, step timing, and loss curves via the Python `mlflow` SDK.
"""

import time
import sys
from pathlib import Path
import torch
import mlflow
from datasets import Dataset

# Add parent directory to path to reuse dataset loader
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / "src"))
from gnn_experiment.micro_probe.dataset import load_astral_micro_dataset

from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq


def run_unsloth_micro_probe(
    model_name: str = "Qwen/Qwen3.5-2B-Instruct",
    max_steps: int = 500,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    experiment_name: str = "micro_probe_benchmark",
):
    # Configure MLflow via python SDK
    mlflow_db = str(Path(__file__).resolve().parent.parent.parent.parent / "mlruns.db")
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment(experiment_name)

    print(f"--- Loading Micro-Probe Model: {model_name} via Unsloth ---")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto detection
        load_in_4bit=True,
    )

    # Apply fast LoRA adapter
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load 1,000+ astral docs dataset
    raw_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "astral_docs" / "raw")
    sft_file = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "astral_docs" / "sft" / "astral_expert_sft.jsonl")
    dataset = load_astral_micro_dataset(raw_docs_dir=raw_dir, sft_file=sft_file)

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        output_dir="outputs_micro_probe",
        save_strategy="no",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        packing=False,
        args=training_args,
    )

    print(f"--- Starting Unsloth 500-Step Training Micro-Probe ---")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    with mlflow.start_run(run_name=f"unsloth_qwen3.5_2b_{max_steps}steps"):
        mlflow.log_params({
            "framework": "unsloth_amd",
            "model_name": model_name,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "dataset_size": len(dataset),
            "max_seq_length": max_seq_length,
        })

        train_result = trainer.train()
        elapsed_s = time.perf_counter() - start_time

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        total_tokens = max_steps * batch_size * max_seq_length
        tok_per_sec = total_tokens / max(1e-5, elapsed_s)
        steps_per_sec = max_steps / max(1e-5, elapsed_s)

        metrics = {
            "wall_time_s": elapsed_s,
            "peak_vram_gb": peak_vram_gb,
            "throughput_tok_sec": tok_per_sec,
            "steps_per_sec": steps_per_sec,
            "final_loss": train_result.training_loss,
        }
        mlflow.log_metrics(metrics)

        print("\n=== Unsloth Micro-Probe Results ===")
        print(f"Wall time: {elapsed_s:.2f} s ({steps_per_sec:.2f} steps/s)")
        print(f"Throughput: {tok_per_sec:.2f} tokens/s")
        print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
        print(f"Final loss: {train_result.training_loss:.4f}")
        print("Logged to MLflow successfully.")

    return metrics


if __name__ == "__main__":
    run_unsloth_micro_probe()
