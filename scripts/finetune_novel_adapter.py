"""Fine-tune the Astral micro-probe adapter using one of the novel training
techniques from GOAL_1.md, benchmarked head-to-head against the existing
QLoRA baseline (`scripts/export_adapter.py`, MLflow run
`finetune_Qwen_Qwen3.5-4B_150steps`, experiment `astral_fine_tuning`):

  wall_time_s=207.76  peak_vram_gb=8.81  steps_per_sec=0.722  final_loss=1.244

Variants (all share the same target_modules, steps, batch size, LR, dataset
as the baseline so the comparison is apples-to-apples):

  custom_standard  Our own hand-rolled per-layer independent LoRA (A, B).
                   Not one of the two novel ideas -- a control variant that
                   isolates "cost of reimplementing LoRA ourselves" from
                   "cost/benefit of the novel architecture", since both
                   novel variants are built on the same custom module code
                   path rather than peft's.
  tucker           Cross-Layer Tucker Factorization: one shared (U_in, U_out)
                   factor pair per (module, shape) group across all layers,
                   plus a tiny private core matrix per layer.
  velocity         Velocity-Masked SFT: per-decoder-layer hidden-state
                   velocity EMA gates whether that layer's adapters run at
                   all this step (standard independent A/B otherwise).
  combined         Tucker factorization + velocity masking together.

Logs to MLflow experiment `astral_fine_tuning` (same DB as the baseline) and
saves the adapter to `results/adapters/astral_qwen3.5_micro_<variant>/`
(never touches the original baseline adapter directory).
"""

import argparse
import time
from pathlib import Path
from typing import NamedTuple

import mlflow
import torch
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from gnn_experiment.micro_probe.dataset import load_astral_micro_dataset
from gnn_experiment.novel_peft import (
    TARGET_MODULES,
    VelocityGate,
    apply_novel_lora,
    get_decoder_layers,
    save_novel_adapter,
    set_hard_vram_cap,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class VariantConfig(NamedTuple):
    mode: str
    gated: bool


VARIANT_MODES: dict[str, VariantConfig] = {
    "custom_standard": VariantConfig(mode="standard", gated=False),
    "tucker": VariantConfig(mode="tucker", gated=False),
    "velocity": VariantConfig(mode="standard", gated=True),
    "combined": VariantConfig(mode="tucker", gated=True),
}


class VelocityGateCallback(TrainerCallback):
    def __init__(self, gate: VelocityGate):
        self.gate = gate

    def on_step_end(self, args, state, control, **kwargs):
        self.gate.end_step()


def optimizer_state_bytes(optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


def finetune_novel(
    variant: str,
    model_name: str = "Qwen/Qwen3.5-4B",
    out_dir: str | None = None,
    train_steps: int = 150,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    rank: int = 8,
    rank_in: int = 8,
    rank_out: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    velocity_threshold: float = 0.05,
    velocity_ema_decay: float = 0.9,
    velocity_warmup_steps: int = 20,
    velocity_max_quiet_fraction: float = 0.9,
    experiment_name: str = "astral_fine_tuning",
    vram_cap_gb: float = 20.0,
):
    if variant not in VARIANT_MODES:
        raise ValueError(f"Unknown variant '{variant}'. Options: {sorted(VARIANT_MODES)}")
    cfg = VARIANT_MODES[variant]

    # Safety first (see set_hard_vram_cap docstring): user is fine with up to 20 GB
    # of the 24 GB card, but the cap itself matters regardless of where it sits --
    # it forces any overrun to fail as a clean Python OutOfMemoryError instead of
    # ROCm-over-WSL silently spilling into shared host RAM.
    set_hard_vram_cap(vram_cap_gb)

    out_dir = out_dir or str(REPO_ROOT / "results" / "adapters" / f"astral_qwen3.5_micro_{variant}")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mlflow_db = REPO_ROOT / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment(experiment_name)

    print(f"--- [{variant}] Loading {model_name} in 4-bit for novel-adapter fine-tuning ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    assert tokenizer is not None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
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
    # Matches export_adapter.py's baseline exactly: prepare_model_for_kbit_training()
    # defaults to use_gradient_checkpointing=True, which is why the 8.81 GB baseline
    # peak is achievable at all on a 24 GB card at this batch size/seq length --
    # leaving it off OOMs (activation memory for all 32 layers held simultaneously).
    model = prepare_model_for_kbit_training(model)

    gate = None
    if cfg.gated:
        num_layers = len(get_decoder_layers(model))
        gate = VelocityGate(
            num_layers=num_layers,
            threshold=velocity_threshold,
            ema_decay=velocity_ema_decay,
            warmup_steps=velocity_warmup_steps,
            max_quiet_fraction=velocity_max_quiet_fraction,
        )

    summary = apply_novel_lora(
        model,
        mode=cfg.mode,
        target_modules=TARGET_MODULES,
        rank=rank,
        rank_in=rank_in,
        rank_out=rank_out,
        alpha=alpha,
        dropout=dropout,
        velocity_gate=gate,
    )
    print(
        f"Wrapped {summary['wrapped_count']} Linear layers | "
        f"trainable={summary['trainable_params']:,} / total={summary['total_params']:,} "
        f"({100 * summary['trainable_params'] / summary['total_params']:.3f}%)"
    )
    # transformers' Trainer.__init__ -> validate_quantization_for_training() blocks
    # fine-tuning a quantized model unless `_hf_peft_config_loaded` is set (normally
    # done by peft.get_peft_model()). We legitimately do have trainable adapters
    # attached -- just not peft's classes -- so this is the correct signal to set,
    # not a bypass of the underlying check.
    model._hf_peft_config_loaded = True

    print("Loading Astral docs dataset...")
    dataset = load_astral_micro_dataset()

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=str(REPO_ROOT / "results" / "tmp_export" / variant),
        per_device_train_batch_size=batch_size,
        max_steps=train_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
    )

    callbacks: list[TrainerCallback] = [VelocityGateCallback(gate)] if gate is not None else []
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    print(f"--- [{variant}] Executing {train_steps}-step novel-adapter training pass ---")
    train_result = trainer.train()

    wall_time_s = time.perf_counter() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    opt_state_mb = optimizer_state_bytes(trainer.optimizer) / (1024**2)

    print(f"Saving novel adapter ({variant}) to {out_path}...")
    meta = {
        "variant": variant,
        "mode": cfg.mode,
        "gated": cfg.gated,
        "target_modules": TARGET_MODULES,
        "rank": rank,
        "rank_in": rank_in,
        "rank_out": rank_out,
        "alpha": alpha,
        "model_name": model_name,
        "velocity_threshold": velocity_threshold if gate is not None else None,
        "velocity_ema_decay": velocity_ema_decay if gate is not None else None,
        "velocity_warmup_steps": velocity_warmup_steps if gate is not None else None,
        "velocity_max_quiet_fraction": velocity_max_quiet_fraction if gate is not None else None,
    }
    save_novel_adapter(model, out_path, meta)
    tokenizer.save_pretrained(out_path)

    quiet_fraction_avg = None
    quiet_fraction_final = None
    if gate is not None and gate.quiet_fraction_history:
        post_warmup = gate.quiet_fraction_history[velocity_warmup_steps:]
        if post_warmup:
            quiet_fraction_avg = sum(post_warmup) / len(post_warmup)
            quiet_fraction_final = post_warmup[-1]

    with mlflow.start_run(run_name=f"novel_{variant}_{model_name.replace('/', '_')}_{train_steps}steps"):
        mlflow.set_tags(
            {
                "variant": variant,
                "novel_architecture": "true",
                "mode": cfg.mode,
                "gated": str(cfg.gated),
            }
        )
        mlflow.log_params(
            {
                "model_name": model_name,
                "train_steps": train_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_seq_length": max_seq_length,
                "dataset_size": len(dataset),
                "variant": variant,
                "mode": cfg.mode,
                "rank": rank,
                "rank_in": rank_in,
                "rank_out": rank_out,
                "alpha": alpha,
                "dropout": dropout,
                "wrapped_linear_count": summary["wrapped_count"],
                "velocity_threshold": velocity_threshold if gate is not None else -1,
                "velocity_ema_decay": velocity_ema_decay if gate is not None else -1,
                "velocity_warmup_steps": velocity_warmup_steps if gate is not None else -1,
                "velocity_max_quiet_fraction": velocity_max_quiet_fraction if gate is not None else -1,
            }
        )
        metrics = {
            "wall_time_s": wall_time_s,
            "peak_vram_gb": peak_vram_gb,
            "final_loss": train_result.training_loss,
            "steps_per_sec": train_steps / max(0.001, wall_time_s),
            "trainable_params": summary["trainable_params"],
            "total_params": summary["total_params"],
            "trainable_pct": 100 * summary["trainable_params"] / summary["total_params"],
            "optimizer_state_mb": opt_state_mb,
        }
        if quiet_fraction_avg is not None:
            assert quiet_fraction_final is not None  # always set together, see above
            metrics["quiet_layer_fraction_avg"] = quiet_fraction_avg
            metrics["quiet_layer_fraction_final"] = quiet_fraction_final
        mlflow.log_metrics(metrics)

    print(f"\n=== [{variant}] Novel Adapter Fine-Tuning Results ===")
    print(f"Wall time: {wall_time_s:.2f} s ({metrics['steps_per_sec']:.3f} steps/s)")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Final loss: {train_result.training_loss:.4f}")
    print(f"Trainable params: {summary['trainable_params']:,} ({metrics['trainable_pct']:.3f}%)")
    print(f"Optimizer state: {opt_state_mb:.3f} MB")
    if quiet_fraction_avg is not None:
        print(f"Avg quiet-layer fraction (post-warmup): {quiet_fraction_avg:.1%}")
    print(f"Adapter saved to: {out_path}")
    print("Logged to MLflow successfully.")

    return {**metrics, "out_dir": str(out_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_MODES))
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--rank-in", type=int, default=8)
    parser.add_argument("--rank-out", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--velocity-threshold", type=float, default=0.05)
    parser.add_argument("--velocity-ema-decay", type=float, default=0.9)
    parser.add_argument("--velocity-warmup-steps", type=int, default=20)
    parser.add_argument("--velocity-max-quiet-fraction", type=float, default=0.9)
    parser.add_argument("--experiment-name", default="astral_fine_tuning")
    parser.add_argument(
        "--vram-cap-gb",
        type=float,
        default=20.0,
        help="Hard allocator ceiling; see set_hard_vram_cap.",
    )
    args = parser.parse_args()

    finetune_novel(
        variant=args.variant,
        model_name=args.model_name,
        out_dir=args.out,
        train_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        rank=args.rank,
        rank_in=args.rank_in,
        rank_out=args.rank_out,
        alpha=args.alpha,
        dropout=args.dropout,
        velocity_threshold=args.velocity_threshold,
        velocity_ema_decay=args.velocity_ema_decay,
        velocity_warmup_steps=args.velocity_warmup_steps,
        velocity_max_quiet_fraction=args.velocity_max_quiet_fraction,
        experiment_name=args.experiment_name,
        vram_cap_gb=args.vram_cap_gb,
    )
