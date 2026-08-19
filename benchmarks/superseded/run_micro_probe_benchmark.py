"""Unified Micro-Probe (Qwen3.5-2B) Benchmark & PyTorch Forward Hook Harness.

Executes:
1. Dataset loading (1,000+ items from data/astral).
2. Qwen3.5-2B QLoRA model loading & PEFT adapter setup.
3. PyTorch forward hook registration & layer velocity (\\Delta h_l) diagnostic run.
4. 500-step training loop with gradient stability measurement.
5. Zero-overhead metric logging to results/micro_probe_runs.jsonl.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from gnn_experiment.micro_probe.dataset import load_astral_micro_dataset
from gnn_experiment.micro_probe.forward_hooks import (
    MicroProbeForwardHooks,
    compute_gradient_stability,
)
from gnn_experiment.utils.logger import log_benchmark_metric

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "micro_probe.yaml"))
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--model-name", type=str, default=None)
    return p.parse_args()


def run_benchmark():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg.get("model_name", "Qwen/Qwen3.5-2B")
    max_steps = args.max_steps or cfg.get("training", {}).get("max_steps", 500)
    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 2)
    learning_rate = args.lr or cfg.get("training", {}).get("learning_rate", 2e-4)
    max_length = cfg.get("training", {}).get("max_length", 512)
    epsilon = cfg.get("hooks", {}).get("velocity_epsilon", 0.05)

    exp_name = cfg.get("logging", {}).get("experiment_name", "micro_probe_benchmark")

    print("==================================================")
    print(f" Running Goal 1 Micro-Probe Benchmark ({model_name})")
    print("==================================================")

    # 2. Load Dataset (1,000+ Astral Docs)
    dataset = load_astral_micro_dataset(
        raw_docs_dir=str(REPO_ROOT / cfg.get("dataset_raw_dir", "data/astral/raw")),
        sft_file=str(REPO_ROOT / cfg.get("dataset_sft_file", "data/astral/training_data.jsonl")),
    )

    # 3. Load Tokenizer & Model
    print(f"Loading tokenizer & 4-bit model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 4. Attach PEFT QLoRA Adapter
    peft_cfg = cfg.get("peft", {})
    lora_config = LoraConfig(
        r=peft_cfg.get("r", 8),
        lora_alpha=peft_cfg.get("alpha", 16),
        target_modules=peft_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable_params, total_params = model.get_nb_trainable_parameters()

    # 5. Register PyTorch Forward Hooks for Hidden-State Velocity
    print("Registering PyTorch forward hooks for hidden-state velocity diagnostics...")
    hook_mgr = MicroProbeForwardHooks(model, epsilon=epsilon)

    # 6. Tokenize Dataset
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")

    tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=str(REPO_ROOT / "results" / "micro_probe" / "checkpoints"),
        per_device_train_batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
    )

    trainer = Trainer(model=model, args=train_args, train_dataset=tokenized_ds, data_collator=collator)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"Starting {max_steps}-step micro-probe training pass...")
    start_time = time.perf_counter()

    train_result = trainer.train()
    elapsed_s = time.perf_counter() - start_time

    # Compute gradient stability & forward hook metrics
    grad_stats = compute_gradient_stability(model)
    hook_summary = hook_mgr.get_summary()
    hook_mgr.remove_hooks()

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    total_tokens = max_steps * batch_size * max_length
    tok_per_sec = total_tokens / max(1e-5, elapsed_s)
    steps_per_sec = max_steps / max(1e-5, elapsed_s)

    metrics = {
        "experiment": exp_name,
        "run_name": f"micro_probe_{model_name.replace('/', '_')}_{max_steps}s",
        "model_name": model_name,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": 100 * trainable_params / total_params,
        "velocity_epsilon": epsilon,
        "dataset_size": len(dataset),
        "wall_time_s": elapsed_s,
        "steps_per_sec": steps_per_sec,
        "throughput_tok_sec": tok_per_sec,
        "peak_vram_gb": peak_vram_gb,
        "final_loss": train_result.training_loss,
        "overall_avg_velocity": hook_summary.get("overall_avg_velocity", 0.0),
        "quiet_layer_ratio": hook_summary.get("quiet_layer_ratio", 0.0),
        "total_grad_norm": grad_stats.get("total_grad_norm", 0.0),
        "max_grad_norm": grad_stats.get("max_grad_norm", 0.0),
    }

    log_benchmark_metric(metrics, filepath="results/micro_probe_runs.jsonl")

    # Save summary artifact
    out_dir = REPO_ROOT / "results" / "micro_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "micro_probe_results.json"
    with open(summary_path, "w") as f:
        json.dump(
            {**metrics, "grad_stats": grad_stats, "hook_summary": hook_summary},
            f,
            indent=2,
        )

    print("\n==================================================")
    print(" Micro-Probe Results Summary")
    print("==================================================")
    print(f" Model: {model_name}")
    print(f" Wall Time: {elapsed_s:.2f} s ({steps_per_sec:.2f} steps/s)")
    print(f" Throughput: {tok_per_sec:.2f} tokens/s")
    print(f" Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f" Training Loss: {train_result.training_loss:.4f}")
    print(f" Avg Layer Velocity (\\Delta h_l): {hook_summary.get('overall_avg_velocity', 0.0):.4f}")
    print(f" Quiet Layer Ratio (<{epsilon}): {hook_summary.get('quiet_layer_ratio', 0.0) * 100:.1f}%")
    print(f" Max Grad Norm: {grad_stats.get('max_grad_norm', 0.0):.4f}")
    print(" Results Logged Successfully to 'results/micro_probe_runs.jsonl'!")

    return metrics


if __name__ == "__main__":
    run_benchmark()
