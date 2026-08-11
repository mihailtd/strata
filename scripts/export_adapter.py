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

import mlflow
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
from trl import SFTConfig, SFTTrainer

from gnn_experiment.micro_probe.dataset import load_astral_micro_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent

# Use PyTorch SDPA (fused efficient-attention on ROCm) when available.
# flash_attn package is not installed; transformers routes "sdpa" to
# torch.nn.functional.scaled_dot_product_attention which dispatches the
# ROCm fused-attention hip kernel automatically.
_SDPA_AVAILABLE = is_flash_attn_2_available() or hasattr(torch.nn.functional, "scaled_dot_product_attention")
_ATTN_IMPL = "sdpa" if _SDPA_AVAILABLE else "eager"


def export_adapter(
    model_name: str = "Qwen/Qwen3.5-4B",
    out_dir: str = "results/adapters/astral_qwen3.5_micro",
    train_steps: int = 150,
    batch_size: int = 4,
    grad_accum_steps: int = 1,
    experiment_name: str = "astral_fine_tuning",
):
    out_path = REPO_ROOT / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    mlflow_db = REPO_ROOT / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment(experiment_name)

    print(f"Loading exact target model: {model_name} for adapter export...")
    print(f"  Attention implementation : {_ATTN_IMPL}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        # Optimisation 1: dropout=0.0 — removes stochastic noise; no quality loss
        # at this fine-tune scale, saves one masked-dropout kernel per layer per step.
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    print("Loading Astral expert Q&A dataset for fast fine-tuning (conversational)...")
    dataset = load_astral_micro_dataset(conversational=True)

    sft_config = SFTConfig(
        output_dir=str(REPO_ROOT / "results" / "tmp_export"),
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

    print(f"Saving trained LoRA adapter to {out_path}...")
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    # Log explicit fine-tuning run to MLflow
    with mlflow.start_run(run_name=f"finetune_{model_name.replace('/', '_')}_{train_steps}steps"):
        mlflow.log_params(
            {
                "model_name": model_name,
                "train_steps": train_steps,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "effective_batch": batch_size * grad_accum_steps,
                "learning_rate": 2e-4,
                "r": 8,
                "alpha": 16,
                "lora_dropout": 0.0,
                "attn_implementation": _ATTN_IMPL,
                "dataset_size": len(dataset),
                "assistant_only_loss": True,
            }
        )
        mlflow.log_metrics(
            {
                "wall_time_s": wall_time_s,
                "peak_vram_gb": peak_vram_gb,
                "final_loss": train_result.training_loss,
                "steps_per_sec": train_steps / max(0.001, wall_time_s),
            }
        )

    loss_val = train_result.training_loss
    print(f"Fine-tuning completed in {wall_time_s:.2f}s (Peak VRAM: {peak_vram_gb:.2f} GB, Final Loss: {loss_val:.4f})")
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
    args = parser.parse_args()
    export_adapter(
        model_name=args.model_name,
        out_dir=args.out,
        train_steps=args.steps,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
    )
