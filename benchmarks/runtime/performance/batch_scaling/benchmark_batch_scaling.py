"""Which batch-1 conclusions survive at batch >= 2?

Every performance finding in this repo was measured at batch 1: the 15.6%
adapter-wrapper tax, the chunked-kernel penalty that sets speculation's
break-even, and the launch-overhead analysis. All three are batch-size
dependent in principle, and nobody has measured batch 2.

WHAT DECIDES IT
---------------
At batch 1 a decode step reads ~8 GB of weights to produce ONE token. If that
read dominates, the same read produces B tokens at batch B for nearly the same
wall-clock:

    bandwidth-bound  -> step latency ~FLAT in B, aggregate tok/s scales ~linearly
    compute-bound    -> step latency ~PROPORTIONAL to B, aggregate tok/s flat

(The common intuition is backwards here: latency *doubling* at B=2 means
batching bought nothing, not that the system is bandwidth-bound.)

Sanity anchor: batch-1 decode at ~31.5 tok/s is 31.7 ms/token for 8 GB of bf16
weights = ~250 GB/s against a ~960 GB/s card. That is only ~26% of peak, so
this model is NOT cleanly bandwidth-bound at B=1 -- there is overhead headroom,
which is exactly why batching might pay.

THE THREE MEASUREMENTS
----------------------
1. decode step latency and aggregate throughput vs B  -> is batching ~free?
2. K=1 vs K=4 forward cost vs B                       -> speculation break-even.
   Break-even is cost(K)/K-ish; if the chunked penalty shrinks at higher B,
   speculation gets cheaper. If it grows, speculation is a batch-1-only trick.
3. wrapped vs folded decode vs B                      -> does the +21% folding
   win amortize away once the GPU has more work per launch?

    uv run --env-file .env scripts/benchmark_batch_scaling.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if torch.cuda.is_available():  # fla's device probe is @cache'd at import
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    load_novel_adapter,
    set_hard_vram_cap,
    unwrap_novel_lora,
)

# DEFAULTS: the m2 expert set (bf16 + Liger, methodology-matched). Verify with
# `uv run python scripts/audit_adapters.py`. Do NOT default to m1 (4-bit NF4)
# adapters -- every benchmark here loads a bf16 base, so an m1 adapter folds a
# correction-to-quantized-weights into unquantized ones.
ADAPTER = "results/adapters/m2_astral_r8a128"


@torch.no_grad()
def decode_step_ms(model, B, prompt_len, k, reps=20):
    """Median ms for one forward of k token(s) per sequence, with a warm cache."""
    ids = torch.randint(1000, 50000, (B, prompt_len), device=model.device)
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    step = torch.randint(1000, 50000, (B, k), device=model.device)
    for _ in range(5):
        model(step, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(step, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--adapter", default=ADAPTER, help="adapter for the wrapped-vs-folded arm")
    ap.add_argument("--out", default="results/batch_scaling.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    results = {}

    # ---- 1. decode scaling (k=1) -------------------------------------------
    print("1. DECODE STEP (k=1): is batching ~free?")
    print(f"   {'B':>3s} {'ms/step':>9s} {'aggregate tok/s':>16s} {'per-req tok/s':>14s} {'vs B=1 latency':>15s}")
    base_ms = None
    dec = {}
    for B in args.batches:
        ms = decode_step_ms(model, B, args.prompt_len, k=1)
        if base_ms is None:
            base_ms = ms
        agg = B * 1000.0 / ms
        dec[B] = {"ms": ms, "aggregate_tok_s": agg, "per_req_tok_s": agg / B, "latency_vs_b1": ms / base_ms}
        print(f"   {B:3d} {ms:8.2f} {agg:15.2f} {agg / B:13.2f} {ms / base_ms:14.2f}x")
    results["decode_k1"] = dec
    b_max = max(args.batches)
    scale = dec[b_max]["aggregate_tok_s"] / dec[1]["aggregate_tok_s"]
    print(f"   -> aggregate throughput B=1..{b_max}: {scale:.2f}x  (perfect scaling would be {b_max}x)")

    # ---- 2. chunked penalty vs batch (speculation break-even) ---------------
    print("\n2. K=4 VERIFICATION vs K=1: does speculation's break-even move with B?")
    print(f"   {'B':>3s} {'k=1 ms':>8s} {'k=4 ms':>8s} {'ratio (break-even tokens)':>26s}")
    chunk = {}
    for B in args.batches:
        m1 = dec[B]["ms"]
        m4 = decode_step_ms(model, B, args.prompt_len, k=4)
        chunk[B] = {"k1_ms": m1, "k4_ms": m4, "ratio": m4 / m1}
        print(f"   {B:3d} {m1:7.2f} {m4:7.2f} {m4 / m1:25.2f}")
    results["chunk_penalty"] = chunk

    # ---- 3. wrapper tax vs batch (does folding still pay?) -----------------
    print("\n3. WRAPPED vs FOLDED decode: does the +21% folding win survive batching?")
    expert = FoldableExpert.from_dir(REPO_ROOT / args.adapter, "astral")
    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
    engine.activate(expert)
    folded = {B: decode_step_ms(model, B, args.prompt_len, k=1) for B in args.batches}
    engine.restore()

    # Stock LoRA ships adapter_model.safetensors and no novel_adapter.pt, so the
    # novel wrapper cannot load it. Use peft's own wrapper -- which is also the
    # baseline that matters for LoRA, since that is how LoRA is actually served.
    # NOTE: the two wrapped arms are therefore DIFFERENT wrappers (NovelLoraLinear
    # vs peft.LoraLayer); the folding win is only comparable within an architecture.
    if (REPO_ROOT / args.adapter / "adapter_model.safetensors").exists():
        from peft import PeftModel

        pm = PeftModel.from_pretrained(model, str(REPO_ROOT / args.adapter))
        pm.eval()
        wrapped = {B: decode_step_ms(pm, B, args.prompt_len, k=1) for B in args.batches}
        pm.unload()
    else:
        load_novel_adapter(model, REPO_ROOT / args.adapter)
        wrapped = {B: decode_step_ms(model, B, args.prompt_len, k=1) for B in args.batches}
        unwrap_novel_lora(model)

    print(f"   {'B':>3s} {'wrapped ms':>11s} {'folded ms':>10s} {'folding win':>12s}")
    tax = {}
    for B in args.batches:
        win = wrapped[B] / folded[B]
        tax[B] = {"wrapped_ms": wrapped[B], "folded_ms": folded[B], "folding_speedup": win}
        print(f"   {B:3d} {wrapped[B]:10.2f} {folded[B]:9.2f} {win:11.2f}x")
    results["wrapper_tax"] = tax

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
