"""Fine-tune the Astral micro-probe adapter using one of the novel training
techniques, benchmarked head-to-head against the existing
QLoRA baseline (`scripts/train/export_adapter.py`, MLflow run
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

Velocity gating uses a dynamic bottom-percentile cutoff (quiet_percentile,
default 25%), not a fixed absolute threshold. A fixed threshold (this gate's
original design used 0.05; this repo later recalibrated to 0.45)
has to be hand-retuned any time the training data changes, and silently goes
inert -- zero quiet layers, zero effect -- if the real distribution drifts
away from it without warning; that's exactly what was measured happening on
this session's astral/postgres runs after the dataset composition changed
(0.45 landed in a noisy middle zone nothing consistently cleared). A forward-
only diagnostic across real batches from both current datasets (see
scratchpad/diagnose_velocity_real.py) also found this model's per-layer
velocity is NOT the monotonic "top-heavy" pattern often assumed for
fine-tuning depth dynamics -- depth-vs-velocity correlation is slightly
*negative* (-0.30), dominated by layer 0 (an outlier ~4.0, ~6-8x every other
layer -- it's transforming raw embeddings, not "doing more domain logic")
and a spike on the final layer, with a noisy, fairly flat 0.32-0.76 band in
between. Percentile gating self-calibrates to whatever the real shape is
rather than assuming one, which is why it replaces the threshold entirely
here instead of sitting alongside it.
"""

import argparse
import time
from pathlib import Path
from typing import NamedTuple

import torch
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from transformers.utils import is_flash_attn_2_available
from trl import SFTConfig, SFTTrainer

from runtime.micro_probe.dataset import load_astral_micro_dataset, load_micro_dataset
from runtime.novel_peft import (
    TARGET_MODULES,
    VelocityGate,
    apply_novel_lora,
    get_decoder_layers,
    save_novel_adapter,
    set_hard_vram_cap,
)
from runtime.utils.logger import log_benchmark_metric

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
# Optimisation 3: use PyTorch SDPA (dispatches ROCm fused efficient-attention hip
# kernel) without needing the flash_attn pip package (not installed on this rig).
_SDPA_AVAILABLE = is_flash_attn_2_available() or hasattr(torch.nn.functional, "scaled_dot_product_attention")
_ATTN_IMPL = "sdpa" if _SDPA_AVAILABLE else "eager"


class VariantConfig(NamedTuple):
    mode: str
    gated: bool
    selection: str = "velocity"
    correct_fan_in_init: bool = False


VARIANT_MODES: dict[str, VariantConfig] = {
    "custom_standard": VariantConfig(mode="standard", gated=False),
    "tucker": VariantConfig(mode="tucker", gated=False),
    "velocity": VariantConfig(mode="standard", gated=True),
    "combined": VariantConfig(mode="tucker", gated=True),
    "random_mask": VariantConfig(mode="standard", gated=True, selection="random"),
    "krotucker": VariantConfig(mode="krotucker", gated=False),
    "id_kron": VariantConfig(mode="id_kron", gated=False),
    "master_basis": VariantConfig(mode="master_basis", gated=False),
    # Control for custom_standard: identical in every way EXCEPT the down-
    # projection init uses the true fan_in. custom_standard's `lora_a` is
    # stored (in, rank), so kaiming reads fan_in=rank and over-initialises it
    # by sqrt(in/rank) ~= 17.9x; the trained delta comes out 18.29x larger
    # than peft LoRA's for an otherwise identical config. This variant tests
    # whether custom_standard's 17.7pp lead over peft LoRA (58.17% vs 40.47%)
    # is caused by that inflated update rather than anything architectural.
    "custom_standard_fixedinit": VariantConfig(mode="standard", gated=False, correct_fan_in_init=True),
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
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    rank: int = 8,
    rank_in: int = 8,
    rank_out: int = 8,
    alpha: int = 16,
    # Optimisation 1: default dropout=0.0 — removes stochastic noise, frees one
    # masked-dropout kernel per layer per step; zero quality loss at fine-tune scale.
    dropout: float = 0.0,
    velocity_quiet_percentile: float = 25.0,
    velocity_ema_decay: float = 0.9,
    velocity_warmup_steps: int = 20,
    velocity_max_quiet_fraction: float = 0.9,
    experiment_name: str = "astral_fine_tuning",
    vram_cap_gb: float = 20.0,
    raw_docs_dir: str | None = None,
    sft_file: str | None = None,
    dataset_fingerprint: str | None = None,
    basis_bank: str | None = None,
    init_coefficients_from: str | None = None,
):
    if variant not in VARIANT_MODES:
        raise ValueError(f"Unknown variant '{variant}'. Options: {sorted(VARIANT_MODES)}")
    cfg = VARIANT_MODES[variant]

    # Safety first (see set_hard_vram_cap docstring): user is fine with up to 20 GB
    # of the 24 GB card, but the cap itself matters regardless of where it sits --
    # it forces any overrun to fail as a clean Python OutOfMemoryError.
    set_hard_vram_cap(vram_cap_gb)

    out_dir = out_dir or str(REPO_ROOT / "results" / "adapters" / f"astral_qwen3.5_micro_{variant}")
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"--- [{variant}] Loading {model_name} in 4-bit for novel-adapter fine-tuning ---")
    print(f"  Attention implementation : {_ATTN_IMPL}")
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
        # Optimisation 3: fused SDPA / efficient-attention hip kernel on ROCm.
        attn_implementation=_ATTN_IMPL,
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
            quiet_percentile=velocity_quiet_percentile,
            ema_decay=velocity_ema_decay,
            warmup_steps=velocity_warmup_steps,
            max_quiet_fraction=velocity_max_quiet_fraction,
            selection=cfg.selection,
        )
        # Must live on the same device as the model: VelocityGate.record() writes
        # GPU-tensor velocities straight into its buffers with no host sync (that's
        # the whole fix -- see its docstring), which only holds if the buffers are
        # already on-device. Left on CPU by default (nn.Module buffers default
        # there), that same write would silently force a device transfer per call.
        gate = gate.to(next(model.parameters()).device)

    # A random basis bank spans nothing task-relevant (measured: 12.4-13.4%
    # adherence, flat across a 16x alpha sweep), so master_basis is only
    # meaningful with a bank extracted from real adapters.
    preloaded_basis = None
    if basis_bank:
        bank_path = Path(basis_bank)
        if not bank_path.is_absolute():
            bank_path = REPO_ROOT / bank_path
        preloaded_basis = torch.load(bank_path, map_location="cpu", weights_only=False)["bank"]
        print(f"Loaded basis bank: {bank_path.name} ({len(preloaded_basis)} shape-groups)")

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
        # False, not True: skip_recompute's torch.is_grad_enabled() heuristic
        # assumes *reentrant* checkpointing (no_grad during the original
        # forward, grad enabled only on recompute). This script's checkpointing
        # is non-reentrant (the modern transformers default), where grad stays
        # enabled throughout -- so that heuristic misfires on every single
        # hook call and silently records nothing, ever, with no error. Measured
        # directly: with skip_recompute=True, gate.ema_velocity stayed 100% NaN
        # and quiet_layers was permanently empty across full training runs on
        # both domains (0.0% quiet, no crash, no warning -- see
        # scratchpad/check_checkpointing_state.py). False costs the (real, but
        # small at this model's scale) redundant recompute-pass measurement;
        # True costs the entire technique silently doing nothing.
        velocity_skip_recompute=False,
        correct_fan_in_init=cfg.correct_fan_in_init,
        preloaded_basis=preloaded_basis,
    )
    print(
        f"Wrapped {summary['wrapped_count']} Linear layers | "
        f"trainable={summary['trainable_params']:,} / total={summary['total_params']:,} "
        f"({100 * summary['trainable_params'] / summary['total_params']:.3f}%)"
    )

    if init_coefficients_from:
        # Warm start: begin from the least-squares projection instead of zeros.
        # Zero-init + Adam is hopeless here -- Adam moves each scalar by ~lr per
        # step, so from 0 the coefficients cannot reach the O(1-8) magnitudes the
        # projection shows are needed (measured: dominant |c| median 7.25).
        init_path = Path(init_coefficients_from)
        if not init_path.is_absolute():
            init_path = REPO_ROOT / init_path
        init_state = torch.load(init_path / "novel_adapter.pt", map_location="cpu")
        msd = model.state_dict()
        loaded, missing = 0, []
        for k, v in init_state.items():
            if k in msd:
                msd[k].data.copy_(v.to(msd[k].dtype).to(msd[k].device))
                loaded += 1
            else:
                missing.append(k)
        if missing:
            raise RuntimeError(f"warm-start keys not present in wrapped model: {missing[:5]} ({len(missing)} total)")
        print(f"Warm-started {loaded} coefficient tensors from {init_path.name}")
    # transformers' Trainer.__init__ -> validate_quantization_for_training() blocks
    # fine-tuning a quantized model unless `_hf_peft_config_loaded` is set (normally
    # done by peft.get_peft_model()). We legitimately do have trainable adapters
    # attached -- just not peft's classes -- so this is the correct signal to set,
    # not a bypass of the underlying check.
    model._hf_peft_config_loaded = True

    if raw_docs_dir or sft_file:
        print(f"Loading dataset override (raw_docs_dir={raw_docs_dir}, sft_file={sft_file})...")
        dataset = load_micro_dataset(raw_docs_dir, sft_file, fingerprint=dataset_fingerprint, conversational=True)
    else:
        print("Loading Astral docs dataset...")
        dataset = load_astral_micro_dataset(conversational=True)

    # Response-only loss masking: SFTConfig(assistant_only_loss=True) masks all
    # user-prompt tokens with -100 so the cross-entropy loss is computed exclusively
    # on the assistant's response tokens. This focuses 100% of LoRA gradient updates
    # on output correctness rather than splitting capacity between predicting user
    # questions and assistant answers. Requires a genuinely conversational dataset
    # (raw `messages`, not pre-rendered text) -- SFTTrainer applies the chat template
    # itself internally to track assistant token spans (confirmed against trl==1.9.2's
    # actual source; an earlier version of this script used
    # `trl.DataCollatorForCompletionOnlyLM`, which does not exist in this trl version).
    # pad_to_multiple_of=8 keeps sequence lengths aligned to tensor-core boundaries.
    sft_config = SFTConfig(
        output_dir="/tmp/sft_trainer_scratch",
        # Optimisation 2: larger default batch + grad accumulation scaling.
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        max_steps=train_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        max_length=max_seq_length,
        packing=False,
        pad_to_multiple_of=8,
        assistant_only_loss=True,
    )

    callbacks: list[TrainerCallback] = [VelocityGateCallback(gate)] if gate is not None else []
    # peft_config intentionally omitted (defaults to None): this model is already
    # wrapped by apply_novel_lora() above via in-place module surgery, not
    # peft.get_peft_model() -- it's a plain PreTrainedModel, not a PeftModel.
    # Confirmed against trl's source that every peft-specific code path in
    # SFTTrainer.__init__ is gated behind `peft_config is not None` / `is_peft_model
    # (model)`, so it passes through unchanged and trains like a normal full/custom
    # fine-tune, exactly as the previous plain-Trainer setup did.
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
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
        # Recorded so load_novel_adapter restores the EXACT bank the coefficients
        # were fitted against -- evaluating them over a different (or random)
        # basis silently produces a meaningless adapter rather than an error.
        "basis_bank": basis_bank,
        "init_coefficients_from": init_coefficients_from,
        "target_modules": TARGET_MODULES,
        "rank": rank,
        "rank_in": rank_in,
        "rank_out": rank_out,
        "alpha": alpha,
        "model_name": model_name,
        "velocity_selection": cfg.selection if gate is not None else None,
        "velocity_quiet_percentile": velocity_quiet_percentile if gate is not None else None,
        "velocity_ema_decay": velocity_ema_decay if gate is not None else None,
        "velocity_warmup_steps": velocity_warmup_steps if gate is not None else None,
        "velocity_max_quiet_fraction": velocity_max_quiet_fraction if gate is not None else None,
    }
    save_novel_adapter(model, out_path, meta)
    tokenizer.save_pretrained(str(out_path))

    quiet_fraction_avg = None
    quiet_fraction_final = None
    quiet_churn_avg = None
    if gate is not None and gate.quiet_fraction_history:
        post_warmup = gate.quiet_fraction_history[velocity_warmup_steps:]
        if post_warmup:
            quiet_fraction_avg = sum(post_warmup) / len(post_warmup)
            quiet_fraction_final = post_warmup[-1]
    if gate is not None and gate.quiet_churn_history:
        # How much the quiet set actually moves step to step: 0.0 = the same
        # layers are masked every step (a static capacity cut, NOT stochastic
        # depth), 1.0 = fully disjoint each step. The distinguishing metric
        # between the velocity arm and the random control.
        quiet_churn_avg = sum(gate.quiet_churn_history) / len(gate.quiet_churn_history)

    metrics = {
        "experiment": experiment_name,
        "run_name": f"novel_{variant}_{model_name.replace('/', '_')}_{train_steps}steps",
        "variant": variant,
        "mode": cfg.mode,
        "gated": cfg.gated,
        "selection": cfg.selection if cfg.gated else "n/a",
        "model_name": model_name,
        "train_steps": train_steps,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch": batch_size * grad_accum_steps,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "dataset_size": len(dataset),
        "rank": rank,
        "rank_in": rank_in,
        "rank_out": rank_out,
        "alpha": alpha,
        "dropout": dropout,
        "attn_implementation": _ATTN_IMPL,
        "wrapped_linear_count": summary["wrapped_count"],
        "velocity_quiet_percentile": velocity_quiet_percentile if gate is not None else -1,
        "velocity_ema_decay": velocity_ema_decay if gate is not None else -1,
        "velocity_warmup_steps": velocity_warmup_steps if gate is not None else -1,
        "velocity_max_quiet_fraction": velocity_max_quiet_fraction if gate is not None else -1,
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
    if quiet_churn_avg is not None:
        metrics["quiet_layer_churn_avg"] = quiet_churn_avg

    log_benchmark_metric(metrics, filepath="results/novel_adapter_runs.jsonl")

    print(f"\n=== [{variant}] Novel Adapter Fine-Tuning Results ===")
    print(f"Wall time: {wall_time_s:.2f} s ({metrics['steps_per_sec']:.3f} steps/s)")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"Final loss: {train_result.training_loss:.4f}")
    print(f"Trainable params: {summary['trainable_params']:,} ({metrics['trainable_pct']:.3f}%)")
    print(f"Optimizer state: {opt_state_mb:.3f} MB")
    if quiet_fraction_avg is not None:
        print(f"Avg quiet-layer fraction (post-warmup): {quiet_fraction_avg:.1%}")
    if quiet_churn_avg is not None:
        print(f"Avg quiet-set churn (0=static set, 1=fully disjoint): {quiet_churn_avg:.1%}")
        if gate is not None:
            print(f"Final quiet layers ({cfg.selection}): {sorted(gate.quiet_layers)}")
    print(f"Adapter saved to: {out_path}")
    print("Logged to MLflow successfully.")

    return {**metrics, "out_dir": str(out_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANT_MODES))
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device train batch size. Default 4 (was 2). Increase to fill VRAM.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps. Lower proportionally when raising batch-size.",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--rank-in", type=int, default=8)
    parser.add_argument("--rank-out", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="LoRA dropout. Default 0.0 (zero-loss opt: removes stochastic noise).",
    )
    parser.add_argument(
        "--velocity-quiet-percentile",
        type=float,
        default=25.0,
        help="Bottom N%% of layers by EMA velocity go quiet each step (dynamic, self-calibrating).",
    )
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
    parser.add_argument(
        "--raw-docs-dir",
        default=None,
        help="Override raw markdown docs dir (default: Astral docs). Pass with --sft-file for a different corpus.",
    )
    parser.add_argument(
        "--sft-file",
        default=None,
        help="Override generated SFT jsonl path (default: Astral's). E.g. an EPUB-derived dataset.",
    )
    parser.add_argument(
        "--dataset-fingerprint",
        default=None,
        help="Explicit datasets-cache fingerprint; default derives one from the resolved paths.",
    )
    parser.add_argument(
        "--basis-bank",
        default=None,
        help="master_basis only: path to a bank from scripts/old/extract_svd_basis.py. "
        "Without this the bank is random, which scores ~base (measured).",
    )
    parser.add_argument(
        "--init-coefficients-from",
        default=None,
        help="master_basis only: warm-start coefficients from a projected adapter "
        "(scripts/old/project_adapter_to_basis.py) instead of zeros.",
    )
    args = parser.parse_args()

    finetune_novel(
        variant=args.variant,
        model_name=args.model_name,
        out_dir=args.out,
        train_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        rank=args.rank,
        rank_in=args.rank_in,
        rank_out=args.rank_out,
        alpha=args.alpha,
        dropout=args.dropout,
        velocity_quiet_percentile=args.velocity_quiet_percentile,
        velocity_ema_decay=args.velocity_ema_decay,
        velocity_warmup_steps=args.velocity_warmup_steps,
        velocity_max_quiet_fraction=args.velocity_max_quiet_fraction,
        experiment_name=args.experiment_name,
        vram_cap_gb=args.vram_cap_gb,
        raw_docs_dir=args.raw_docs_dir,
        sft_file=args.sft_file,
        dataset_fingerprint=args.dataset_fingerprint,
        basis_bank=args.basis_bank,
        init_coefficients_from=args.init_coefficients_from,
    )
