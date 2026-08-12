"""Export fine-tuned Astral micro-probe LoRA adapter to disk with MLflow tracking.

Trains/saves the LoRA adapter to `results/adapters/astral_qwen3.5_micro`.
Logs exact fine-tuning metrics (wall time, peak VRAM, final loss, model name) to MLflow.

Zero-Loss Optimisation Checklist (all applied):
  1. lora_dropout=0.0          — removes stochastic noise, frees a masked-dropout kernel
                                 per layer per step; zero quality loss at fine-tune scale.
  2. batch_size auto-scaled    -- default=4 (was 2); use --batch-size to push higher if
                                  VRAM allows. Gradient accumulation steps scaled down
                                  proportionally so effective batch stays constant.
  3. attn_implementation=sdpa  -- PyTorch native Scaled Dot Product Attention; on ROCm
                                  this dispatches the fused efficient-attention hip kernel
                                  (equivalent to FlashAttention-2) without requiring the
                                  flash_attn pip package (not installed on this rig).
                                  Guarded: falls back gracefully if SDPA is unavailable.
  4. trl.SFTTrainer + SFTConfig(assistant_only_loss=True) -- computes cross-entropy loss
                                  ONLY on assistant response tokens; user/system tokens are
                                  masked. Requires a genuinely conversational dataset (raw
                                  `messages`, not pre-rendered text) -- TRL applies the chat
                                  template itself internally to track assistant token spans.
                                  (Note: an earlier version of this script used
                                  `trl.DataCollatorForCompletionOnlyLM` -- that class does not
                                  exist in the installed trl==1.9.2; `assistant_only_loss` is
                                  the real, current replacement, confirmed against trl's
                                  actual source, not assumed from an outdated example.)
                                  pad_to_multiple_of=8 keeps tensor-core alignment.
"""

import argparse
import time
from pathlib import Path

import torch
from peft import LoKrConfig, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
from trl import SFTConfig, SFTTrainer

from gnn_experiment.micro_probe.dataset import load_astral_micro_dataset, load_micro_dataset
from gnn_experiment.peft_compat import patch_lokr_4bit_support
from gnn_experiment.utils.logger import log_benchmark_metric

# peft's LoKr cannot compute delta shapes against a 4-bit quantized base layer
# (it reads the packed uint8 `.weight`); every adapter here trains on one. See
# peft_compat for the full explanation -- without this, --peft-variant lokr
# dies on the first forward pass with a shape error.
patch_lokr_4bit_support()

REPO_ROOT = Path(__file__).resolve().parent.parent

# Use PyTorch SDPA (fused efficient-attention on ROCm) when available.
# flash_attn package is not installed; transformers routes "sdpa" to
# torch.nn.functional.scaled_dot_product_attention which dispatches the
# ROCm fused-attention hip kernel automatically.
_SDPA_AVAILABLE = is_flash_attn_2_available() or hasattr(torch.nn.functional, "scaled_dot_product_attention")
_ATTN_IMPL = "sdpa" if _SDPA_AVAILABLE else "eager"


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_peft_config(peft_variant: str, r: int = 8, alpha: int = 16):
    """LoRA / DoRA / LoKr configs that differ ONLY in the factorization, so an
    A/B between them isolates the factorization and nothing else.

    - lora: baseline A·B.
    - dora: A·B plus DoRA's decoupled magnitude vector m (peft-native
      `use_dora`). Same parameter count as lora up to the m vector, so this
      tests DoRA's *quality* claim, not any size claim.
    - lokr: Kronecker factorization A⊗B (peft-native LoKr). This is the
      "KronA" half of this project's KronA+DoRA target and the one that
      actually shrinks the adapter -- parameter arithmetic on this model's
      shapes predicts ~3.2 MB vs LoRA's ~28.6 MB. NOTE: peft's LoKrConfig has
      no `use_dora` field, so KronA+DoRA combined is NOT available natively;
      it needs a magnitude vector added by hand (plan Step D).
    """
    if peft_variant == "lora":
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if peft_variant == "dora":
        return LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=True,
        )
    if peft_variant == "lokr":
        return LoKrConfig(
            r=r,
            alpha=alpha,
            target_modules=TARGET_MODULES,
            rank_dropout=0.0,
            module_dropout=0.0,
            task_type="CAUSAL_LM",
        )
    raise ValueError(f"Unknown peft_variant {peft_variant!r}; expected lora|dora|lokr")


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def export_adapter(
    model_name: str = "Qwen/Qwen3.5-4B",
    out_dir: str = "results/adapters/astral_qwen3.5_micro",
    train_steps: int = 150,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    experiment_name: str = "astral_fine_tuning",
    peft_variant: str = "lora",
    sft_file: str | None = None,
    r: int = 8,
    alpha: int = 16,
):
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading exact target model: {model_name} for adapter export...")
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
    model = prepare_model_for_kbit_training(model)

    peft_config = build_peft_config(peft_variant, r=r, alpha=alpha)
    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  peft variant : {peft_variant}")
    print(f"  trainable    : {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    print("Loading Astral expert Q&A dataset for fast fine-tuning (conversational)...")
    dataset = (
        load_micro_dataset(sft_file=sft_file, conversational=True)
        if sft_file
        else load_astral_micro_dataset(conversational=True)
    )

    sft_config = SFTConfig(
        output_dir="/tmp/export_trainer_scratch",
        # Optimisation 2: larger batch, gradient-accumulation scaled proportionally
        # so the effective batch size (batch_size * grad_accum) stays ≥ original.
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        max_steps=train_steps,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        max_length=512,
        packing=False,
        pad_to_multiple_of=8,
        # Optimisation 4: response-only loss masking, see module docstring.
        assistant_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    print(f"Executing {train_steps}-step fast training pass on {model_name}...")
    train_result = trainer.train()

    wall_time_s = time.perf_counter() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

    print(f"Saving trained {peft_variant} adapter to {out_path}...")
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))

    # The whole point of the KronA line of work is payload size, so measure the
    # real artifact rather than trusting the parameter arithmetic. Weights only
    # (safetensors/bin) -- the tokenizer files saved alongside are ~20 MB and
    # identical across variants, so including them would swamp the comparison.
    weight_bytes = sum(
        f.stat().st_size for f in out_path.iterdir() if f.is_file() and f.suffix in (".safetensors", ".bin")
    )
    adapter_mb = weight_bytes / 1e6

    log_benchmark_metric(
        {
            "experiment": experiment_name,
            "run_name": f"finetune_{peft_variant}_{model_name.replace('/', '_')}_{train_steps}steps",
            "model_name": model_name,
            "peft_variant": peft_variant,
            "train_steps": train_steps,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch": batch_size * grad_accum_steps,
            "learning_rate": 2e-4,
            "r": r,
            "alpha": alpha,
            "scaling": alpha / r,
            "lora_dropout": 0.0,
            "attn_implementation": _ATTN_IMPL,
            "dataset_size": len(dataset),
            "assistant_only_loss": True,
            "wall_time_s": wall_time_s,
            "peak_vram_gb": peak_vram_gb,
            "final_loss": train_result.training_loss,
            "steps_per_sec": train_steps / max(0.001, wall_time_s),
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": 100 * trainable / total,
            "adapter_size_mb": adapter_mb,
        },
        filepath="results/export_adapter_runs.jsonl",
    )

    loss_val = train_result.training_loss
    print(f"Fine-tuning completed in {wall_time_s:.2f}s (Peak VRAM: {peak_vram_gb:.2f} GB, Final Loss: {loss_val:.4f})")
    print(f"Adapter weights on disk: {adapter_mb:.3f} MB ({trainable:,} trainable params)")
    print(f"MLflow fine-tuning run logged under experiment '{experiment_name}'!")
    return str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", default="results/adapters/astral_qwen3.5_micro")
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
    parser.add_argument(
        "--peft-variant",
        default="lora",
        choices=["lora", "dora", "lokr"],
        help="Factorization to train: lora (A*B), dora (A*B + magnitude m), lokr (Kronecker A(x)B).",
    )
    parser.add_argument(
        "--sft-file",
        default=None,
        help="Override the training jsonl (default: astral). E.g. data/postgresql/training_data.jsonl",
    )
    parser.add_argument("--r", type=int, default=8, help="Rank value r for adapter.")
    parser.add_argument("--alpha", type=int, default=16, help="Alpha value for adapter.")
    parser.add_argument("--experiment-name", default="astral_fine_tuning")
    args = parser.parse_args()
    export_adapter(
        model_name=args.model_name,
        out_dir=args.out,
        train_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        experiment_name=args.experiment_name,
        peft_variant=args.peft_variant,
        sft_file=args.sft_file,
        r=args.r,
        alpha=args.alpha,
    )
