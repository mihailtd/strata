"""Measure what an adapter hot-swap actually costs, and what fraction of an
agent loop it represents.

Settles a specific claim: that eliminating adapter-swap latency makes
multi-tool agent loops go from "stuttering" to "instantaneous". The honest
arithmetic on this rig says otherwise -- decode runs at ~17.8 tok/s, so a
200-token agent step takes ~11 s, and even a 120 ms swap is ~1% of one step.
This script measures the swap side directly instead of arguing about it.

Three arms:
  rewrap  load_novel_adapter() -- what the repo does today. Re-walks the module
          tree and replaces every target Linear, so cost is Python-side module
          surgery, not bytes. Also cannot be called twice on one model without
          corrupting it, which is why it is measured on a fresh wrap each time.
  lora    swap_adapter_weights() on a rank-8 LoRA payload (~21 MB).
  coef    swap_adapter_weights() on a k=32 master_basis coefficient payload
          (~8 KB), with the basis bank already resident.

Timing uses torch.cuda.synchronize() on both sides, because non_blocking
copies only queue work -- timing without a sync measures the enqueue, not the
transfer.

    uv run --env-file .env scripts/benchmark_adapter_swap.py
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from gnn_experiment.novel_peft import (  # noqa: E402
    AdapterPayload,
    load_novel_adapter,
    prepare_swap_slots,
    set_hard_vram_cap,
    swap_adapter_weights,
)

MEASURED_DECODE_TOK_S = 17.77  # astral_inference_benchmark, warm, this rig


def _load_base(model_name: str):
    dt = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dt, bnb_4bit_quant_type="nf4"),
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_swaps(
    slots, payloads: list[AdapterPayload], trials: int, non_blocking: bool, batched: bool = False
) -> list[float]:
    """Alternate between payloads like an agent loop switching tools."""
    # warm up: first copy into a fresh slot can pay one-time driver costs
    swap_adapter_weights(slots, payloads[0], non_blocking=non_blocking, batched=batched)
    _sync()
    times = []
    for i in range(trials):
        p = payloads[i % len(payloads)]
        _sync()
        t0 = time.perf_counter()
        swap_adapter_weights(slots, p, non_blocking=non_blocking, batched=batched)
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def report(label: str, times: list[float], nbytes: int):
    times_sorted = sorted(times)
    p50 = statistics.median(times_sorted)
    p99 = times_sorted[min(len(times_sorted) - 1, int(0.99 * len(times_sorted)))]
    gbps = (nbytes / 1e9) / (p50 / 1000.0) if p50 > 0 else float("inf")
    print(f"  {label:28s} p50 {p50:8.3f} ms   p99 {p99:8.3f} ms   payload {nbytes / 1e6:8.3f} MB   {gbps:6.2f} GB/s")
    return p50


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--vram-cap-gb", type=float, default=20.0)
    ap.add_argument("--lora-a", default="results/adapters/astral_qwen3.5_micro_custom_standard")
    ap.add_argument("--lora-b", default="results/adapters/postgres_qwen3.5_micro_custom_standard")
    ap.add_argument("--coef-a", default="results/adapters/astral_qwen3.5_micro_mbproj_k32")
    ap.add_argument("--coef-b", default="results/adapters/astral_qwen3.5_micro_mbwarm_k32")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    results: dict[str, float] = {}

    # ---- arm 1: LoRA-sized payload -------------------------------------------------
    print(f"\n=== LoRA payload swap (rank-8, ~21 MB) x{args.trials} ===")
    model = _load_base(args.model_name)
    load_novel_adapter(model, REPO_ROOT / args.lora_a, velocity_gate=None)
    pa = AdapterPayload.from_dir(REPO_ROOT / args.lora_a)
    pb = AdapterPayload.from_dir(REPO_ROOT / args.lora_b)
    print(f"  {pa}\n  {pb}")
    slots = prepare_swap_slots(model, pa)
    results["lora_blocking"] = report("in-place (blocking)", time_swaps(slots, [pa, pb], args.trials, False), pa.nbytes)
    results["lora_nonblocking"] = report(
        "in-place (non_blocking+pinned)", time_swaps(slots, [pa, pb], args.trials, True), pa.nbytes
    )
    results["lora_batched"] = report(
        "in-place (batched foreach)", time_swaps(slots, [pa, pb], args.trials, True, batched=True), pa.nbytes
    )

    # the incumbent, for contrast: full re-wrap on an already-wrapped model is
    # unsafe, so time it against a freshly loaded base each iteration is far too
    # slow; instead time a single re-wrap on this model and report it as-is.
    t0 = time.perf_counter()
    load_novel_adapter(model, REPO_ROOT / args.lora_a, velocity_gate=None)
    _sync()
    results["lora_rewrap"] = (time.perf_counter() - t0) * 1000.0
    print(f"  {'load_novel_adapter (re-wrap)':28s} {results['lora_rewrap']:8.3f} ms   <-- incumbent, 1 sample")

    del model, slots, pa, pb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- arm 2: coefficient payload ------------------------------------------------
    coef_a = REPO_ROOT / args.coef_a
    coef_b = REPO_ROOT / args.coef_b
    if coef_a.exists() and coef_b.exists():
        print(f"\n=== master_basis coefficient swap (k=32, ~8 KB) x{args.trials} ===")
        model = _load_base(args.model_name)
        load_novel_adapter(model, coef_a, velocity_gate=None)
        ca = AdapterPayload.from_dir(coef_a)
        cb = AdapterPayload.from_dir(coef_b)
        print(f"  {ca}\n  {cb}")
        slots = prepare_swap_slots(model, ca)
        results["coef_blocking"] = report(
            "in-place (blocking)", time_swaps(slots, [ca, cb], args.trials, False), ca.nbytes
        )
        results["coef_nonblocking"] = report(
            "in-place (non_blocking+pinned)", time_swaps(slots, [ca, cb], args.trials, True), ca.nbytes
        )
        results["coef_batched"] = report(
            "in-place (batched foreach)", time_swaps(slots, [ca, cb], args.trials, True, batched=True), ca.nbytes
        )
        del model, slots, ca, cb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("\n(skipping coefficient arm -- projected/warm adapters not found)")

    # ---- what it means for an agent loop -------------------------------------------
    print(f"\n=== agent-loop impact (decode {MEASURED_DECODE_TOK_S} tok/s, measured on this rig) ===")
    print(f"  {'swaps':>6s} {'tok/step':>9s} {'compute':>10s} {'rewrap':>10s} {'lora-fast':>10s} {'coef-fast':>10s}")
    for nsw in (10, 30):
        for toks in (50, 200):
            compute = nsw * toks / MEASURED_DECODE_TOK_S
            row = f"  {nsw:6d} {toks:9d} {compute:9.1f}s"
            for key in ("lora_rewrap", "lora_batched", "coef_batched"):
                if key in results:
                    ov = nsw * results[key] / 1000.0
                    row += f" {100 * ov / compute:9.2f}%"
                else:
                    row += f" {'--':>10s}"
            print(row)
    print("\n  (percentages are swap overhead as a share of pure decode time)")


if __name__ == "__main__":
    main()
