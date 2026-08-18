"""CURRENT training methodology (m2). Use this for all new expert training.

METHODOLOGY VERSIONING
----------------------
The filename carries `CURRENT` and the methodology id. Exactly one trainer at a
time is CURRENT. When the methodology changes:

    1. rename this file to `train_expert_m2.py` (drop CURRENT) and move it to
       scripts/superseded/
    2. create `train_expert_CURRENT_m3.py` with METHODOLOGY = "m3"
    3. adapters trained under it are named `m3_<domain>_r<rank>a<alpha>`

so the methodology is readable from both the script name and every adapter it
produces, and results trained under different methodologies can never be
silently compared.

    m1  4-bit NF4 base, no fused kernels   (export_adapter.py,
                                            finetune_novel_adapter.py -- both
                                            now legacy; adapters learn a
                                            correction to quantized weights and
                                            were then folded into bf16 ones)
    m2  bf16 base + Liger fused kernels    THIS FILE -- fused_linear_cross_entropy
                                            (no logit materialisation at 248320
                                            vocab), rms_norm, swiglu; rope off

Hyperparameters fixed across every domain so experts stay comparable: r=8,
alpha=128 (scaling 16), 7 projections, 150 steps, batch 2, grad-accum 2,
lr 2e-4, cosine, max_length 512.

WHY THIS EXISTS
---------------
`export_adapter.py` trains against a **4-bit NF4** base (`load_in_4bit=True`),
but every folding/speculation/stacking benchmark loads a **bf16** base. Adapters
produced there learn a correction to quantized weights and are then folded into
unquantized ones. That seam is measurable: the financial expert moved from
-5.00pp (4-bit-trained) to +4.17pp (bf16-trained) on its own domain, with a data
change that altered zero rubric terms.

`train_financial_adapter.py` trains in bf16 but is hardcoded to the financial
dataset. This is that script generalised, so astral and postgres can be brought
into the same regime instead of being silently mixed into a bf16 stack.

Hyperparameters are held identical to the controlled experts so results stay
comparable: r=8, alpha=128 (scaling 16), 7 projections, 150 steps, batch 2,
grad-accum 2, lr 2e-4, cosine schedule, max_length 512.

    uv run --env-file .env scripts/train_stock_lora_bf16.py --domain astral
    uv run --env-file .env scripts/train_stock_lora_bf16.py --domain postgresql
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import set_hard_vram_cap  # noqa: E402

METHODOLOGY = "m2"  # bf16 + Liger fused kernels; see docstring

# domain -> (training data, output adapter dir). Adapter names carry the
# methodology id so an adapter's training regime is readable from its path.
DOMAINS = {
    "astral": ("data/astral/training_data.jsonl", "results/adapters/m2_astral_r8a128"),
    "postgresql": ("data/postgresql/training_data.jsonl", "results/adapters/m2_postgresql_r8a128"),
    "financial_planning": (
        "data/financial_planning/training_data.jsonl",
        "results/adapters/m2_financial_r8a128",
    ),
}


def load_dataset_records(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "text" in d:
                records.append({"text": d["text"]})
            elif "messages" in d:
                u = d["messages"][0]["content"]
                a = d["messages"][1]["content"]
                records.append({"text": f"### Question:\n{u}\n\n### Answer:\n{a}"})
    return records


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--domain", choices=sorted(DOMAINS), required=True)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument(
        "--init-lora-weights",
        default="true",
        help=(
            "peft init scheme: 'true' (stock LoRA, B=0 so dW=0 at init), 'pissa', "
            "'pissa_niter_<N>', 'olora', 'eva', 'gaussian'. NOTE: pissa/olora MUTATE "
            "the base weights (W_res = W0 - scaling*B0@A0), so the trained adapter is a "
            "delta on W_res, NOT on pristine W0. This engine folds onto a pristine W0 "
            "buffer, so such adapters are converted back to standard LoRA at save time "
            "(peft's path_initial_model_for_weight_conversion). That conversion emits "
            "rank 2r, not r -- see the printed warning."
        ),
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="override the domain's training file (e.g. a corpus revision). Recorded in "
             "regime.json so an adapter never loses track of what it was trained on.",
    )
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--no-liger", action="store_true", help="disable Liger fused kernels (A/B baseline)")
    ap.add_argument("--out", default=None, help="override the default output dir")
    ap.add_argument(
        "--logging-steps", type=int, default=10,
        help="loss logging interval. Use 1 to capture a per-step convergence curve; "
             "the default 10 gives only 15 points on a 150-step run, which is too "
             "coarse to locate a plateau or to drive any early-stopping rule.",
    )
    ap.add_argument(
        "--loss-curve-out", default=None,
        help="dump the full per-step loss history to this JSON path",
    )
    args = ap.parse_args()

    data_rel, out_rel = DOMAINS[args.domain]
    if args.dataset:
        data_rel = args.dataset
    dataset_path = REPO_ROOT / data_rel
    out_dir = Path(args.out) if args.out else REPO_ROOT / out_rel

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"TRAINING bf16 STOCK LORA [{args.domain}] r={args.rank} alpha={args.alpha} on {dev}")
    print(f"  data: {dataset_path}")
    print(f"  out:  {out_dir}")
    print("=" * 88)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Liger fused Triton kernels. MUST be applied BEFORE from_pretrained: with
    # model=None the patcher rebinds at class level, so only models constructed
    # afterwards pick up the RMSNorm/SwiGLU swaps. Reversing these two lines
    # makes the patch silently do nothing.
    #
    # rope is left at its default False: liger raises NotImplementedError for
    # Qwen3.5 ("not available"), because of the hybrid Gated DeltaNet/attention
    # mix. There is no per-layer detection -- it is a blanket opt-out.
    #
    # Verified equivalent: Liger's RMSNorm uses offset=1.0 + casting_mode
    # "gemma", matching stock Qwen3_5RMSNorm's output * (1.0 + weight.float())
    # then cast. Loss trajectories match a non-Liger run (1.941->0.859 vs
    # 1.943->0.868).
    liger_applied = False
    if not args.no_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

        apply_liger_kernel_to_qwen3_5()
        liger_applied = True
        print("Liger fused kernels applied (fused_linear_cross_entropy, rms_norm, swiglu; rope=off)")

    # bf16, NOT load_in_4bit -- this is the whole point of the script
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # peft wants the bool True for stock LoRA and a string for every other scheme.
    init_scheme: object = True if args.init_lora_weights.lower() == "true" else args.init_lora_weights
    mutates_base = args.init_lora_weights.lower().startswith(("pissa", "olora"))

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=init_scheme,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # PiSSA/OLoRA subtract their init from the base weights, so the adapter that
    # comes out is a delta on W_res, not on W0. Folding it onto this engine's
    # pristine W0 buffer would add the principal component twice. peft can convert
    # back to an equivalent standard LoRA if it is handed the INITIAL adapter, so
    # stash that before a single gradient step touches it.
    init_adapter_dir = None
    if mutates_base:
        init_adapter_dir = out_dir / "_pissa_init"
        init_adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(init_adapter_dir))
        print(
            f"  [{args.init_lora_weights}] base weights MUTATED (W_res = W0 - scaling*B0@A0).\n"
            f"  Initial adapter stashed -> {init_adapter_dir}\n"
            "  It will be converted back to a pristine-W0 LoRA at save time; expect rank 2r."
        )

    from datasets import Dataset

    records = load_dataset_records(dataset_path)
    print(f"Loaded {len(records)} records from {dataset_path}")
    train_dataset = Dataset.from_list(records)

    sft_config = SFTConfig(
        output_dir=str(out_dir / "checkpoints"),
        dataset_text_field="text",
        max_length=512,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
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

    print(f"Starting SFT training ({args.max_steps} steps)...")
    trainer.train()

    if args.loss_curve_out:
        curve_path = Path(args.loss_curve_out)
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        effective_batch = 2 * 2  # per_device_train_batch_size * gradient_accumulation_steps
        curve_path.write_text(
            json.dumps(
                {
                    "domain": args.domain,
                    "methodology": METHODOLOGY,
                    "init_lora_weights": args.init_lora_weights,
                    "rank": args.rank,
                    "alpha": args.alpha,
                    "max_steps": args.max_steps,
                    "logging_steps": args.logging_steps,
                    "lr": args.lr,
                    "lr_scheduler_type": "cosine",
                    "warmup_ratio": 0.03,
                    "n_records": len(records),
                    "effective_batch": effective_batch,
                    "epochs_seen": args.max_steps * effective_batch / max(1, len(records)),
                    "log_history": trainer.state.log_history,
                },
                indent=2,
            )
        )
        print(f"Loss curve -> {curve_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if init_adapter_dir is not None:
        # Emits dW = scaling*(B_trained@A_trained - B0@A0) refactorised as a plain
        # LoRA on pristine W0 -- which is what WeightFoldingEngine requires. The
        # refactorisation of a difference of two rank-r products is rank 2r.
        model.save_pretrained(
            str(out_dir), path_initial_model_for_weight_conversion=str(init_adapter_dir)
        )
    else:
        model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    # Record the regime in the adapter itself. 0 of 69 existing adapters do this,
    # so provenance was previously recoverable only from directory-layout side
    # effects (export_adapter.py leaves no checkpoints/ subdir; this one does).
    (out_dir / "regime.json").write_text(
        json.dumps(
            {
                "methodology": METHODOLOGY,
                "precision": "bfloat16",
                "quantization": None,
                "liger_fused_kernels": liger_applied,
                "trained_by": "scripts/train_expert_CURRENT_m2.py",
                "domain": args.domain,
                "rank": args.rank,
                "alpha": args.alpha,
                "scaling": args.alpha / args.rank,
                "init_lora_weights": args.init_lora_weights,
                "base_weights_mutated_during_training": mutates_base,
                "converted_to_pristine_w0_lora": init_adapter_dir is not None,
                "max_steps": args.max_steps,
                "lr": args.lr,
                "dataset": data_rel,
                "n_records": len(records),
            },
            indent=2,
        )
    )
    print(f"Saved adapter + regime.json to {out_dir}")


if __name__ == "__main__":
    main()
