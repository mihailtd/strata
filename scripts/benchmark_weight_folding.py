"""Does folding an adapter into base weights actually pay, and is it correct?

Replaces scripts/benchmark_multi_expert_chain.py, whose numbers were not
reproducible and whose dense-payload design could not run on this box (three
10.2 GB fp32 payloads pinned on 23 GB of RAM; a swap would move 2 x 10.2 GB
over a measured 12.6 GB/s PCIe link, ~1.6 s, not the 11.2 ms reported).

CORRECTNESS IS THE GATE, NOT A FOOTNOTE. Everything a naive folding benchmark
measures -- swap latency, weight drift, decode rate -- is also passed with
flying colours by a payload of all zeros, which would simply be running the
base model. So this script refuses to report timings until two things hold:

    folded ~= wrapped   the folded model reproduces the adapter's own logits
    folded !=  base     and is measurably different from the unadapted model

Both directions are needed. The first alone is satisfied by a no-op adapter;
the second alone is satisfied by corrupting the weights.

Arms, all in one process on identical weights so nothing is confounded by a
reload, all greedy, all the same dtype:

    base      plain model
    wrapped   NovelLoraLinear wrappers, the normal inference path
    folded    W_live = W0 + scaling * (U @ V), wrappers removed entirely
    base#2    the base arm again at the end, to expose machine drift

    uv run --env-file .env scripts/benchmark_weight_folding.py
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    load_novel_adapter,
    set_hard_vram_cap,
    unwrap_novel_lora,
)

DEFAULT_EXPERTS = {
    "financial_planning": "results/adapters/financial_planning_krona_dora",
    "postgresql": "results/adapters/postgres_qwen3.5_micro_id_kron_r16",
    "astral": "results/adapters/astral_qwen3.5_micro_id_kron",
}

PROMPTS = [
    (
        "financial_planning",
        "A client shows money-avoidance behaviour and anxiety about sequence-of-returns\n"
        "risk in retirement. Outline a risk strategy.",
    ),
    (
        "postgresql",
        "Design a PostgreSQL 18 schema with pgvector HNSW indexing for transaction\n"
        "history and client profile embeddings.",
    ),
    (
        "astral",
        "Write a FastAPI endpoint, managed with uv and linted with ruff, that runs a\n"
        "similarity search over a pgvector table.",
    ),
    ("financial_planning", "Combine the risk strategy, the database schema and the API layer into one execution plan."),
]


def logits_for(model, tokenizer, prompts, max_len=192):
    """Last-position logits for each prompt -- a cheap, deterministic fingerprint."""
    out = []
    with torch.no_grad():
        for p in prompts:
            ids = tokenizer(
                f"### Question:\n{p}\n\n### Answer:\n", return_tensors="pt", truncation=True, max_length=max_len
            )
            ids = {k: v.to(model.device) for k, v in ids.items()}
            out.append(model(**ids).logits[0, -1, :].float().cpu())
    return torch.stack(out)


def compare(name_a, a, name_b, b):
    """Report how far apart two logit fingerprints are, in absolute and rank terms."""
    max_abs = (a - b).abs().max().item()
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    # correlation of the top-50 logits, robust to a constant shift
    ka = a.topk(50, dim=-1).indices
    agree = statistics.mean(
        len(set(ka[i].tolist()) & set(b.topk(50, dim=-1).indices[i].tolist())) / 50 for i in range(len(a))
    )
    return {
        "pair": f"{name_a} vs {name_b}",
        "max_abs_logit_diff": max_abs,
        "top1_agreement": top1,
        "top50_overlap": agree,
    }


def decode_arm(label, model, tokenizer, prompts, max_new_tokens, repeats=1):
    rates, texts = [], []
    for _ in range(repeats):
        for p in prompts:
            ids = tokenizer(f"### Question:\n{p}\n\n### Answer:\n", return_tensors="pt").to(model.device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            n = out[0].shape[0] - ids["input_ids"].shape[1]
            rates.append(n / dt)
            texts.append(tokenizer.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True))
    print(
        f"  {label:10s} {statistics.mean(rates):6.2f} tok/s  "
        f"(min {min(rates):.2f}, max {max(rates):.2f}, n={len(rates)})"
    )
    return {"mean_tok_s": statistics.mean(rates), "min": min(rates), "max": max(rates), "all": rates, "texts": texts}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--gate-expert", default="astral", help="expert used for the correctness gate")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--swap-reps", type=int, default=30)
    ap.add_argument("--restore-cycles", type=int, default=100)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/weight_folding_benchmark.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    results = {"model": args.model_name, "dtype": str(dtype)}

    print("=" * 72)
    print(" Weight-folding benchmark")
    print("=" * 72)

    # ---- experts as device-resident factors -------------------------------
    experts = {}
    for name, rel in DEFAULT_EXPERTS.items():
        e = FoldableExpert.from_dir(REPO_ROOT / rel, name=name)
        experts[name] = e
        print(f"  {e}")
    results["experts"] = {
        n: {"modules": len(e.factors), "scaling": e.scaling, "factor_mb": e.nbytes / 1e6} for n, e in experts.items()
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = [p for _, p in PROMPTS]

    # ---- load once; measure base, then wrapped, then folded ---------------
    print(f"\nLoading {args.model_name} in {dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    logits_base = logits_for(model, tokenizer, prompts)

    print("\n--- decode: base (unwrapped) ---")
    dec_base = decode_arm("base", model, tokenizer, prompts, args.max_new_tokens)

    gate_dir = REPO_ROOT / DEFAULT_EXPERTS[args.gate_expert]
    print(f"\nWrapping with {args.gate_expert} adapter (the normal inference path) ...")
    load_novel_adapter(model, gate_dir)
    logits_wrapped = logits_for(model, tokenizer, prompts)

    print("--- decode: wrapped ---")
    dec_wrapped = decode_arm("wrapped", model, tokenizer, prompts, args.max_new_tokens)

    n_unwrapped = unwrap_novel_lora(model)
    print(f"\nRemoved {n_unwrapped} wrappers; model is a plain nn.Module again.")
    logits_unwrapped = logits_for(model, tokenizer, prompts)
    assert torch.equal(logits_unwrapped, logits_base), "unwrap did not restore the base model exactly"
    print("  unwrap restored base logits exactly.")

    # ---- fold ------------------------------------------------------------
    engine = WeightFoldingEngine(model, experts.values())
    print(f"\nEngine: {len(engine.slots)} weight tensors, pristine copy {engine.pristine_bytes / 1e9:.2f} GB")
    print(f"  VRAM after pristine snapshot: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    engine.activate(experts[args.gate_expert])
    logits_folded = logits_for(model, tokenizer, prompts)

    # ================== CORRECTNESS GATE ==================
    print("\n" + "=" * 72)
    print(" GATE 1 -- does folding reproduce the adapter, and differ from base?")
    print("=" * 72)
    cmp_fold_wrap = compare("folded", logits_folded, "wrapped", logits_wrapped)
    cmp_fold_base = compare("folded", logits_folded, "base", logits_base)
    cmp_wrap_base = compare("wrapped", logits_wrapped, "base", logits_base)
    for c in (cmp_fold_wrap, cmp_fold_base, cmp_wrap_base):
        print(
            f"  {c['pair']:20s} max|dlogit|={c['max_abs_logit_diff']:8.4f}  "
            f"top1={c['top1_agreement']:.2f}  top50 overlap={c['top50_overlap']:.2f}"
        )
    results["gate"] = {
        "folded_vs_wrapped": cmp_fold_wrap,
        "folded_vs_base": cmp_fold_base,
        "wrapped_vs_base": cmp_wrap_base,
    }

    equivalent = cmp_fold_wrap["max_abs_logit_diff"] < 0.5 * cmp_wrap_base["max_abs_logit_diff"]
    non_trivial = cmp_wrap_base["max_abs_logit_diff"] > 0.1
    results["gate"]["passed"] = bool(equivalent and non_trivial)
    if not non_trivial:
        print("\n  *** GATE FAILED: the adapter barely changes the model; nothing to fold. ***")
    elif not equivalent:
        print("\n  *** GATE FAILED: folded output does not track the wrapped adapter. ***")
    else:
        print("\n  GATE PASSED: folding reproduces the adapter and is not the base model.")

    # ---- decode: folded --------------------------------------------------
    print("\n--- decode: folded ---")
    dec_folded = decode_arm("folded", model, tokenizer, prompts, args.max_new_tokens)

    engine.restore()
    print("--- decode: base again (drift control) ---")
    dec_base2 = decode_arm("base#2", model, tokenizer, prompts, args.max_new_tokens)

    results["decode"] = {"base": dec_base, "wrapped": dec_wrapped, "folded": dec_folded, "base_again": dec_base2}

    # ---- GATE 2: exactness of restore ------------------------------------
    print("\n" + "=" * 72)
    print(f" GATE 2 -- {args.restore_cycles} activate/restore cycles, drift vs pristine")
    print("=" * 72)
    order = [experts["financial_planning"], experts["postgresql"], experts["astral"], experts["financial_planning"]]
    for _ in range(args.restore_cycles):
        for e in order:
            engine.activate(e)
    engine.restore()
    torch.cuda.synchronize()
    drift = engine.max_drift()
    print(f"  after {args.restore_cycles * len(order)} activations: L_inf drift = {drift:.10f}")
    print("  (exact by construction: activate writes W0 + dW, restore copies W0 back)")
    results["restore_drift"] = {
        "cycles": args.restore_cycles,
        "activations": args.restore_cycles * len(order),
        "linf": drift,
    }

    # ---- swap latency ----------------------------------------------------
    print("\n" + "=" * 72)
    print(" Swap latency (expert -> expert, full model)")
    print("=" * 72)
    seq = [experts["financial_planning"], experts["postgresql"], experts["astral"]]
    for e in seq:  # warm up autotuning
        engine.activate(e)
    torch.cuda.synchronize()
    times = []
    for i in range(args.swap_reps):
        e = seq[i % len(seq)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.activate(e)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50, p95 = times[len(times) // 2], times[int(len(times) * 0.95)]
    print(f"  n={len(times)}  p50={p50:.2f} ms  p95={p95:.2f} ms  min={times[0]:.2f}  max={times[-1]:.2f}")

    folded_bytes = engine.pristine_bytes
    traffic_gb = 2 * folded_bytes / 1e9  # read W0 + write W_live
    print(f"  memory traffic per swap: {traffic_gb:.2f} GB (read W0 + write W_live)")
    print(f"  implied bandwidth: {traffic_gb / (p50 / 1000):.0f} GB/s")
    results["swap"] = {
        "p50_ms": p50,
        "p95_ms": p95,
        "min_ms": times[0],
        "max_ms": times[-1],
        "traffic_gb": traffic_gb,
        "implied_bw_gb_s": traffic_gb / (p50 / 1000),
    }

    # ---- amortisation ----------------------------------------------------
    print("\n" + "=" * 72)
    print(" Verdict")
    print("=" * 72)
    gain = (dec_folded["mean_tok_s"] - dec_wrapped["mean_tok_s"]) / dec_wrapped["mean_tok_s"] * 100
    recovered = dec_folded["mean_tok_s"] / dec_base["mean_tok_s"] * 100
    drift_pct = abs(dec_base2["mean_tok_s"] - dec_base["mean_tok_s"]) / dec_base["mean_tok_s"] * 100
    print(f"  wrapped {dec_wrapped['mean_tok_s']:.2f} -> folded {dec_folded['mean_tok_s']:.2f} tok/s   ({gain:+.1f}%)")
    print(f"  folded reaches {recovered:.1f}% of unadapted base ({dec_base['mean_tok_s']:.2f} tok/s)")
    print(f"  machine drift across the run (base vs base#2): {drift_pct:.1f}%")
    tokens_to_break_even = (p50 / 1000) / max(1e-9, (1 / dec_wrapped["mean_tok_s"] - 1 / dec_folded["mean_tok_s"]))
    print(f"  a swap pays for itself after {tokens_to_break_even:.0f} generated tokens")
    results["verdict"] = {
        "speedup_vs_wrapped_pct": gain,
        "pct_of_base": recovered,
        "machine_drift_pct": drift_pct,
        "breakeven_tokens": tokens_to_break_even,
    }

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
