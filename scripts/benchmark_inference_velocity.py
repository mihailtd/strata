"""Inference-time velocity skipping benchmark: does dynamically bypassing the
LoRA adapter on "quiet" layers during token-by-token generation (not just
during training) produce a real decode-speed win, and does it hold the
83.70% modern-tooling adherence measured for the `velocity` adapter under
normal (always-on) generation?

Two conditions, same model load, same 15 eval questions, same scoring:
  baseline  Adapter always fires every layer every token (current default
            generation behavior -- what the reported 83.70% adherence and
            eval-time timings already reflect).
  dynamic   A fresh VelocityGate per prompt drives per-token masking via
            VelocityGateTickLogitsProcessor (threshold empirically verified
            for this regime, see scratchpad/diagnose_inference_velocity.py
            -- training-time 0.45 transfers reasonably: inference-time
            median ~0.43 vs training-time ~0.48, same distribution shape).

Single model load throughout (the known stall is from a *second*
from_pretrained() in one process, not from switching gate configs on an
already-loaded model).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from gnn_experiment.eval.eval_suite import LEGACY_TERMS, MODERN_TERMS, count_matches  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    VelocityGate,
    VelocityGateTickLogitsProcessor,
    get_decoder_layers,
    load_novel_adapter,
    set_hard_vram_cap,
)


def generate_and_score(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    logits_processor: LogitsProcessorList | None = None,
) -> dict:
    formatted = f"### Question:\n{prompt}\n\n### Answer:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    start_t = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            logits_processor=logits_processor,
        )
    gen_time_s = time.perf_counter() - start_t

    new_tokens = outputs[0][inputs.input_ids.shape[1] :]
    num_tokens = new_tokens.shape[0]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    modern_hits = count_matches(response_text, MODERN_TERMS)
    legacy_hits = count_matches(response_text, LEGACY_TERMS)
    total_hits = modern_hits + legacy_hits
    adherence_pct = (modern_hits / max(1, total_hits)) * 100.0 if total_hits > 0 else 0.0

    return {
        "response": response_text,
        "gen_time_s": gen_time_s,
        "num_tokens": num_tokens,
        "tok_per_sec": num_tokens / max(1e-6, gen_time_s),
        "modern_hits": modern_hits,
        "legacy_hits": legacy_hits,
        "adherence_pct": adherence_pct,
    }


def aggregate(results: list[dict]) -> dict:
    total_tokens = sum(r["num_tokens"] for r in results)
    total_time = sum(r["gen_time_s"] for r in results)
    modern_total = sum(r["modern_hits"] for r in results)
    legacy_total = sum(r["legacy_hits"] for r in results)
    warm = results[1:] if len(results) > 1 else results  # drop first prompt (cold-start kernel selection)
    warm_tokens = sum(r["num_tokens"] for r in warm)
    warm_time = sum(r["gen_time_s"] for r in warm)
    return {
        "all_tok_per_sec": total_tokens / max(1e-6, total_time),
        "warm_tok_per_sec": warm_tokens / max(1e-6, warm_time),
        "avg_adherence_pct": sum(r["adherence_pct"] for r in results) / len(results),
        "modern_hits_total": modern_total,
        "legacy_hits_total": legacy_total,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
    }


def run_benchmark(
    model_name: str = "Qwen/Qwen3.5-4B",
    adapter_dir: str | None = None,
    questions_file: str = "data/astral/evaluation_data.jsonl",
    max_new_tokens: int = 256,
    velocity_threshold: float = 0.45,
    velocity_ema_decay: float = 0.9,
    velocity_warmup_tokens: int = 3,
    velocity_max_quiet_fraction: float = 0.9,
    experiment_name: str = "astral_inference_benchmark",
    vram_cap_gb: float = 20.0,
    max_questions: int | None = None,
) -> dict:
    set_hard_vram_cap(vram_cap_gb)
    adapter_dir = adapter_dir or str(REPO_ROOT / "results" / "adapters" / "astral_qwen3.5_micro_velocity")

    questions_path = Path(questions_file)
    if not questions_path.is_absolute():
        questions_path = REPO_ROOT / questions_path
    with open(questions_path) as f:
        questions = [json.loads(line) for line in f if line.strip()]
    if max_questions:
        questions = questions[:max_questions]

    mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlruns.db'}")
    mlflow.set_experiment(experiment_name)

    print(f"Loading {model_name} + velocity adapter from {adapter_dir} (single load)")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    assert tokenizer is not None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    load_novel_adapter(model, adapter_dir, velocity_gate=None)
    model.eval()
    num_layers = len(get_decoder_layers(model))

    # Warm-up: one throwaway generation so kernel/algorithm selection overhead
    # (rocBLAS/MIOpen autotuning on first-seen shapes) doesn't pollute either
    # condition's timing.
    print("Warm-up generation...")
    generate_and_score(model, tokenizer, questions[0]["prompt"], max_new_tokens=32)

    print(f"\n=== Baseline: adapter always active ({len(questions)} questions) ===")
    baseline_results = []
    for idx, q in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] baseline: '{q['prompt'][:60]}...'")
        res = generate_and_score(model, tokenizer, q["prompt"], max_new_tokens=max_new_tokens)
        res["id"] = q["id"]
        baseline_results.append(res)
    baseline_agg = aggregate(baseline_results)
    print(
        f"Baseline: {baseline_agg['all_tok_per_sec']:.2f} tok/s (all) / "
        f"{baseline_agg['warm_tok_per_sec']:.2f} tok/s (warm) | "
        f"adherence {baseline_agg['avg_adherence_pct']:.2f}%"
    )

    print(f"\n=== Dynamic: velocity-gated per-token skipping ({len(questions)} questions) ===")
    dynamic_results = []
    quiet_fractions = []
    for idx, q in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] dynamic: '{q['prompt'][:60]}...'")
        gate = VelocityGate(
            num_layers=num_layers,
            threshold=velocity_threshold,
            ema_decay=velocity_ema_decay,
            warmup_steps=velocity_warmup_tokens,
            max_quiet_fraction=velocity_max_quiet_fraction,
        )
        gate = gate.to(next(model.parameters()).device)
        handles = _install_gate(model, gate)
        processor = LogitsProcessorList([VelocityGateTickLogitsProcessor(gate)])
        res = generate_and_score(
            model, tokenizer, q["prompt"], max_new_tokens=max_new_tokens, logits_processor=processor
        )
        for h in handles:
            h.remove()
        res["id"] = q["id"]
        post_warmup = gate.quiet_fraction_history[velocity_warmup_tokens:]
        res["quiet_fraction_avg"] = sum(post_warmup) / len(post_warmup) if post_warmup else 0.0
        quiet_fractions.append(res["quiet_fraction_avg"])
        dynamic_results.append(res)
    dynamic_agg = aggregate(dynamic_results)
    avg_quiet = sum(quiet_fractions) / len(quiet_fractions)
    print(
        f"Dynamic: {dynamic_agg['all_tok_per_sec']:.2f} tok/s (all) / "
        f"{dynamic_agg['warm_tok_per_sec']:.2f} tok/s (warm) | "
        f"adherence {dynamic_agg['avg_adherence_pct']:.2f}% | avg quiet fraction {avg_quiet:.1%}"
    )

    speedup = dynamic_agg["warm_tok_per_sec"] / max(1e-6, baseline_agg["warm_tok_per_sec"])

    with mlflow.start_run(run_name=f"inference_velocity_{model_name.replace('/', '_')}"):
        mlflow.set_tags({"variant": "velocity", "phase": "inference"})
        mlflow.log_params(
            {
                "model_name": model_name,
                "adapter_path": str(adapter_dir),
                "num_questions": len(questions),
                "max_new_tokens": max_new_tokens,
                "velocity_threshold": velocity_threshold,
                "velocity_warmup_tokens": velocity_warmup_tokens,
            }
        )
        mlflow.log_metrics(
            {
                "baseline_tok_per_sec_all": baseline_agg["all_tok_per_sec"],
                "baseline_tok_per_sec_warm": baseline_agg["warm_tok_per_sec"],
                "baseline_adherence_pct": baseline_agg["avg_adherence_pct"],
                "dynamic_tok_per_sec_all": dynamic_agg["all_tok_per_sec"],
                "dynamic_tok_per_sec_warm": dynamic_agg["warm_tok_per_sec"],
                "dynamic_adherence_pct": dynamic_agg["avg_adherence_pct"],
                "dynamic_avg_quiet_fraction": avg_quiet,
                "speedup_warm": speedup,
            }
        )
        mlflow.log_dict(
            {"baseline_results": baseline_results, "dynamic_results": dynamic_results},
            "inference_benchmark_detail.json",
        )

    out = {
        "baseline": baseline_agg,
        "dynamic": dynamic_agg,
        "dynamic_avg_quiet_fraction": avg_quiet,
        "speedup_warm": speedup,
    }
    out_path = REPO_ROOT / "results" / "inference_velocity_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Summary ===")
    print(
        f"Baseline (always-on):  {baseline_agg['warm_tok_per_sec']:.2f} tok/s, "
        f"{baseline_agg['avg_adherence_pct']:.2f}% adherence"
    )
    print(
        f"Dynamic (velocity-skip): {dynamic_agg['warm_tok_per_sec']:.2f} tok/s, "
        f"{dynamic_agg['avg_adherence_pct']:.2f}% adherence, {avg_quiet:.1%} avg quiet fraction"
    )
    print(f"Speedup: {speedup:.3f}x")
    print(f"Saved to {out_path}")
    return out


def _install_gate(model, gate: VelocityGate) -> list:
    from gnn_experiment.novel_peft import NovelLoraLinear, install_velocity_hooks

    layers = get_decoder_layers(model)
    for module in model.modules():
        if isinstance(module, NovelLoraLinear):
            module._is_quiet_fn = gate.is_quiet
    return install_velocity_hooks(layers, gate, skip_recompute=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--questions", default="data/astral/evaluation_data.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--velocity-threshold", type=float, default=0.45)
    parser.add_argument("--velocity-warmup-tokens", type=int, default=3)
    parser.add_argument("--vram-cap-gb", type=float, default=20.0)
    parser.add_argument("--max-questions", type=int, default=None, help="Limit question count (for smoke tests).")
    parser.add_argument("--experiment-name", default="astral_inference_benchmark")
    args = parser.parse_args()

    run_benchmark(
        model_name=args.model_name,
        adapter_dir=args.adapter_dir,
        questions_file=args.questions,
        max_new_tokens=args.max_new_tokens,
        velocity_threshold=args.velocity_threshold,
        velocity_warmup_tokens=args.velocity_warmup_tokens,
        vram_cap_gb=args.vram_cap_gb,
        max_questions=args.max_questions,
        experiment_name=args.experiment_name,
    )
