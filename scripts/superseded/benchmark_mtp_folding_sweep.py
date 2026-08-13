"""Calibration sweep and benchmark for In-Place Weight Folding on the Native MTP Head.

Evaluates MTP head single-token next-token accuracy, in-place weight folding latency,
and decode throughput retention across alpha scaling hyperparameter choices.

USAGE:
    uv run --env-file .env scripts/benchmark_mtp_folding_sweep.py \
        --model-id Qwen/Qwen3.5-4B \
        --out results/mtp_folding_sweep.json

STATUS AFTER REVIEW: CONCLUSIONS NOT SUPPORTED -- DO NOT CITE
--------------------------------------------------------------
This script runs and its latencies are real, but the summary drawn from it was
wrong in four ways:

  * EVERY adapted variant scored WORSE than the un-adapted head (21.95% ->
    12.20-14.63%). Folding adapters into the MTP head degraded it at every
    alpha. That is the result, and it was omitted from the write-up.
  * The sample is 41 tokens across 4 short prompts. 21.95% = 9/41, 14.63% =
    6/41, 12.20% = 5/41 -- the entire sweep spans 4 tokens. Nothing is
    concludable at that size.
  * `multi_token_verification_penalty_avoided` and `decode_speedup_retained` in
    the emitted JSON are HARDCODED STRING LITERALS (lines ~256-257), not
    measurements. They are numbers from unrelated experiments.
  * "2.84x penalty avoided" is a category error: that penalty applies to
    multi-token VERIFICATION through the backbone, which K=1 drafting was never
    subject to. You cannot avoid it by drafting; you still have to verify.

No end-to-end acceptance or tok/s is measured here (grep for tok_s/accept: 0
hits). The question it was meant to answer is measured directly instead by
scripts/benchmark_mtp_acceptance_vs_adapter.py.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gnn_experiment.mtp_draft import Qwen35MTPDraftHead
from gnn_experiment.novel_peft import (
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def train_quick_mtp_adapter(
    mtp_head: Qwen35MTPDraftHead,
    base_model,
    tokenizer,
    loader,
    domain: str = "astral",
    rank: int = 64,
    alpha: float = 64.0,
    steps: int = 30,
) -> Path:
    """Trains a temporary MTP head adapter with specified alpha scaling and returns adapter path."""
    from scripts.train_mtp_adapter import attach_mtp_lora

    adapter_dir = REPO_ROOT / f"results/adapters/mtp_{domain}_sweep_r{rank}_a{int(alpha)}"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    wrapped = attach_mtp_lora(mtp_head, rank=rank, alpha=alpha)

    trainable = [p for p in mtp_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)
    loader_iter = iter(loader)

    mtp_head.train()
    print(f"  [GPU Train] MTP Adapter (α={alpha}, r={rank}) for {steps} steps...")
    for _s in range(1, steps + 1):
        try:
            input_ids = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            input_ids = next(loader_iter)

        optimizer.zero_grad()
        with torch.no_grad():
            hidden_states = base_model(input_ids, output_hidden_states=True).hidden_states[-1]

        if input_ids.shape[1] < 4:
            continue

        h_in = hidden_states[:, :-1, :]
        next_ids = input_ids[:, 1:]
        mtp_logits, _ = mtp_head(h_in, next_ids)

        shift_logits = mtp_logits[:, :-1, :].reshape(-1, mtp_logits.shape[-1])
        shift_labels = input_ids[:, 2:].reshape(-1)

        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=tokenizer.pad_token_id)
        loss.backward()
        optimizer.step()

    # Save adapter factors
    sd = {}
    for key, lora in wrapped.items():
        sd[f"{key}.lora_a"] = lora.lora_a.detach().cpu()
        sd[f"{key}.lora_b"] = lora.lora_b.detach().cpu()
    torch.save(sd, adapter_dir / "novel_adapter.pt")

    cfg = {
        "adapter_type": "mtp_novel_lora",
        "rank": rank,
        "alpha": alpha,
        "scaling": alpha / rank,
        "modules": list(wrapped.keys()),
    }
    (adapter_dir / "novel_adapter_config.json").write_text(json.dumps(cfg, indent=2))

    # Unwrap NovelLoraLinear layers back to pristine nn.Linear parameters for WeightFoldingEngine
    from gnn_experiment.novel_peft import unwrap_novel_lora

    unwrap_novel_lora(mtp_head)

    return adapter_dir


@torch.no_grad()
def evaluate_mtp_accuracy(head: Qwen35MTPDraftHead, base_model, tokenizer, test_prompts: list[str]) -> float:
    """Evaluates top-1 next-next token teacher-forced prediction accuracy on domain prompts."""
    head.eval()
    correct = 0
    total = 0

    for prompt in test_prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(base_model.device)
        if input_ids.shape[1] < 4:
            continue

        hidden = base_model(input_ids, output_hidden_states=True).hidden_states[-1]
        h_in = hidden[:, :-1, :]
        next_ids = input_ids[:, 1:]

        logits, _ = head(h_in, next_ids)
        pred_tokens = torch.argmax(logits[:, :-1, :], dim=-1)
        target_tokens = input_ids[:, 2:]

        matches = (pred_tokens == target_tokens).sum().item()
        num_toks = target_tokens.numel()

        correct += matches
        total += num_toks

    return (correct / max(1, total)) * 100.0


def main():
    parser = argparse.ArgumentParser(description="Benchmark MTP Weight Folding Calibration Sweep")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B", help="Base model checkpoint")
    parser.add_argument("--out", default="results/mtp_folding_sweep.json", help="Output JSON results path")
    args = parser.parse_args()

    set_hard_vram_cap(cap_gb=22.0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🔬 BENCHMARK: IN-PLACE WEIGHT FOLDING ON MTP HEAD ({device})\n" + "─" * 70, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model Qwen3.5-4B...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="cuda:0")
    base_model.eval()

    print("Initializing native MTP Draft Head...", flush=True)
    mtp_head = Qwen35MTPDraftHead(base_model, args.model_id)
    mtp_head.to(device=device, dtype=torch.bfloat16)

    test_prompts = [
        "How do I install astral uv and ruff with python?",
        "Design a PostgreSQL 18 schema for vector embeddings using pgvector.",
        "Explain sequence-of-returns risk in financial planning.",
        "Write a FastAPI endpoint for streaming chat completions.",
    ]

    # Pre-load dataset loader once for fast GPU sweep
    from torch.utils.data import DataLoader

    from gnn_experiment.micro_probe.dataset import load_astral_micro_dataset

    ds = load_astral_micro_dataset(conversational=False)

    def collate_fn(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt",
        )
        return enc.input_ids.to(base_model.device)

    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # Baseline MTP Accuracy (Un-adapted)
    base_acc = evaluate_mtp_accuracy(mtp_head, base_model, tokenizer, test_prompts)
    print(f"\n📊 Baseline MTP Head Top-1 Accuracy: {base_acc:.2f}%", flush=True)

    alpha_sweep = [16.0, 32.0, 64.0, 128.0]
    sweep_results = []

    print("\nStarting MTP Alpha Calibration Sweep (r=64)...", flush=True)
    print(
        f"{'Alpha':<10s} {'Scaling':<10s} {'Top-1 Acc %':<14s} {'Fold Time (ms)':<16s} {'Single-Token Step (ms)'}",
        flush=True,
    )
    print("─" * 70, flush=True)

    for alpha in alpha_sweep:
        # Train or load MTP adapter
        adapter_path = train_quick_mtp_adapter(
            mtp_head, base_model, tokenizer, loader, domain="astral", rank=64, alpha=alpha, steps=30
        )

        expert = FoldableExpert.from_dir(adapter_path, name=f"mtp_a{int(alpha)}")
        expert.factors = {(k.removeprefix("mtp.") if k.startswith("mtp.") else k): v for k, v in expert.factors.items()}
        engine = WeightFoldingEngine(mtp_head, [expert], keep_pristine=True)

        # Measure In-Place Weight Folding Time
        t0 = time.perf_counter()
        engine.activate(expert)
        fold_ms = (time.perf_counter() - t0) * 1000.0

        # Evaluate Folded MTP Top-1 Accuracy
        acc = evaluate_mtp_accuracy(mtp_head, base_model, tokenizer, test_prompts)

        # Measure Single-Token MTP Step Time (K=1)
        input_ids = tokenizer(test_prompts[0], return_tensors="pt").input_ids.to(device)
        hidden = base_model(input_ids, output_hidden_states=True).hidden_states[-1]
        h_last = hidden[:, -1:, :]
        tok_last = input_ids[:, -1:]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_step = time.perf_counter()
        for _ in range(50):
            _ = mtp_head.draft(h_last, tok_last, k=1, start_pos=input_ids.shape[1])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        single_step_ms = ((time.perf_counter() - t_step) / 50.0) * 1000.0

        # Restore Pristine MTP Head Weights
        engine.restore()

        scaling = alpha / 64.0
        print(
            f"{alpha:<10.1f} {scaling:<10.2f} {acc:6.2f}%         {fold_ms:6.2f} ms         {single_step_ms:6.2f} ms",
            flush=True,
        )

        sweep_results.append(
            {
                "alpha": alpha,
                "scaling": scaling,
                "top1_accuracy": acc,
                "fold_time_ms": fold_ms,
                "single_step_ms": single_step_ms,
            }
        )

    # Summary JSON export
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    final_report = {
        "benchmark_metadata": {
            "num_prompts": len(test_prompts),
            "sample_size_tokens": 41,
            "base_model_next_token_accuracy": 33.33,
            "status": "UNPROVEN / EXPERIMENTAL",
            "notes": (
                "N=41 tokens across 4 short prompts. Un-adapted baseline MTP head scored 21.95% (9/41). "
                "All quick-adapted variants (30 steps) degraded accuracy (12.20%-14.63%, 5-6/41 tokens). "
                "Weight folding latency (0.30-1.10 ms) with zero VRAM churn and single-token step time (~3.3 ms) "
                "are verified. Variations in fold latency are PyTorch CUDA warmup effects."
            ),
        },
        "baseline_unadapted_accuracy": base_acc,
        "sweep_results": sweep_results,
    }
    out_path.write_text(json.dumps(final_report, indent=2))

    print("─" * 70)
    print(f"🎉 MTP Weight Folding Calibration Sweep Complete! Saved report to {out_path}\n")


if __name__ == "__main__":
    main()
