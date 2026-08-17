"""Prefill vs decode at AGENT TURN shapes — is the decode loop worth optimising?

WHY THIS EXISTS ALONGSIDE benchmark_prefill_share.py
----------------------------------------------------
That script measured prefill at **1.5% @2k, 3.0% @4k, 5.9% @8k** and the repo
concluded prefill is "off the critical path". The measurement is sound. The
workload shape is not the one the engine actually serves: it decodes
`max_new_tokens=1024`, so the prefill cost is amortised over 1024 tokens.

An agent turn is the opposite shape. It feeds back a tool result — file contents,
a query result, a stack trace — and emits a short action: **~3,000 tokens in,
~150 out**. Rearranging that script's own numbers, at 8k the prefill is 5.9% of a
1024-token turn, so decode-per-token is 0.941/1024, and at 150 output tokens the
prefill share becomes 0.059/(0.059+0.138) ≈ **30%** — five times what is on
record.

That matters because every remaining optimisation on the board (fused speculative
verification, AITER RMSNorm/SwiGLU, FlashNorm) targets **decode**. If decode is
70% of an agent turn rather than 94%, their ceilings drop accordingly — and the
repo has been here before: the swap cost was optimised for months before anyone
measured it at 0.86% of wall clock.

WHAT THIS MEASURES
    For each prompt length, ONE prefill and then 300 decode steps with the
    per-token time recorded individually. Shares for any output length <= 300 are
    then summed from real per-token measurements, not extrapolated from a mean —
    decode slows as the KV cache grows, so a mean would misreport short turns.

    Reported per cell: prefill share, and the Amdahl ceiling of a decode-only
    optimisation (what a 2x faster decode loop actually buys end-to-end).

    uv run --env-file .env python \
        benchmarks/runtime/performance/prefill_vs_decode/benchmark_agent_turn_split.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# fla's device probe is @cache'd at import; touch CUDA before transformers.
torch.zeros(1, device="cuda")

from transformers import AutoModelForCausalLM  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import set_hard_vram_cap  # noqa: E402

# Turn shapes. 3000/150 is the stated agent case; the rest bracket it so the
# conclusion can be read off a curve rather than a single point.
PROMPT_LENS = [500, 1000, 2000, 3000, 5000, 8000]
OUT_LENS = [50, 150, 300]


@torch.no_grad()
def measure(model, input_ids, n_decode: int) -> tuple[float, list[float]]:
    """One prefill, then n_decode steps timed individually.

    Per-token timing matters: decode slows as the cache grows, so a 150-token
    turn is not 150 x (mean of 300). Each step is synchronised, which adds a
    small fixed overhead to every step equally and does not distort the ratio.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1).unsqueeze(0)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    per_token: list[float] = []
    for _ in range(n_decode):
        t = time.perf_counter()
        out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = torch.argmax(out.logits[:, -1, :], -1).unsqueeze(0)
        torch.cuda.synchronize()
        per_token.append(time.perf_counter() - t)
    return prefill_s, per_token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-decode", type=int, default=max(OUT_LENS))
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/agent_turn_split.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  PREFILL vs DECODE AT AGENT TURN SHAPES")
    print("=" * 100)
    print(f"  device={dev}  prompt lens={PROMPT_LENS}  output lens={OUT_LENS}\n")

    # No tokenizer: prompts are random token ids. Decode cost does not depend on
    # token VALUES, only on sequence length, and random ids guarantee exactly
    # `max_decode` steps with no EOS terminating a run early and skewing a cell.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()

    # Warmup: the first forward at a new shape pays lazy kernel compilation, and
    # ~33 s of Triton JIT landing inside a timed region is exactly what produced
    # this repo's retracted "prefill = 54%" claim.
    print("  warmup (kernel compilation OUTSIDE the timed region)...")
    _ = measure(model, torch.randint(1000, 40000, (1, 512), device="cuda"), 8)
    _ = measure(model, torch.randint(1000, 40000, (1, 4096), device="cuda"), 8)
    print("  done\n")

    rows = {}
    for plen in PROMPT_LENS:
        ids = torch.randint(1000, 40000, (1, plen), device="cuda")
        prefill_s, per_token = measure(model, ids, args.max_decode)
        rows[plen] = {"prefill_s": prefill_s, "per_token_s": per_token}
        cum = sum(per_token)
        print(f"  prompt={plen:>5}  prefill={prefill_s * 1000:8.1f} ms   "
              f"decode({args.max_decode})={cum:6.2f} s   "
              f"mean/tok={1000 * cum / len(per_token):5.2f} ms   "
              f"first/last tok={1000 * per_token[0]:.2f}/{1000 * per_token[-1]:.2f} ms")

    print("\n" + "=" * 100)
    print("  PREFILL SHARE OF WALL CLOCK, BY TURN SHAPE")
    print("=" * 100)
    header = f"  {'prompt':>8}" + "".join(f"{f'out={o}':>13}" for o in OUT_LENS)
    print(header + f"{'out=1024*':>13}")
    print("  " + "-" * 96)

    report: dict = {"device": dev, "prompt_lens": PROMPT_LENS, "out_lens": OUT_LENS, "cells": {}}
    for plen in PROMPT_LENS:
        pf = rows[plen]["prefill_s"]
        pts = rows[plen]["per_token_s"]
        cells = []
        for o in OUT_LENS:
            dec = sum(pts[:o])
            share = 100.0 * pf / (pf + dec)
            cells.append(f"{share:11.1f}%")
            report["cells"][f"{plen}x{o}"] = {
                "prefill_s": pf, "decode_s": dec,
                "prefill_share_pct": share,
                "total_s": pf + dec,
            }
        # 1024 is what the existing benchmark used; projected from the measured
        # tail rate and marked with * because it is the one number NOT measured.
        tail = sum(pts[-50:]) / 50
        dec1024 = sum(pts) + tail * (1024 - len(pts))
        cells.append(f"{100.0 * pf / (pf + dec1024):11.1f}%")
        report["cells"][f"{plen}x1024_projected"] = {
            "prefill_share_pct": 100.0 * pf / (pf + dec1024), "projected": True
        }
        print(f"  {plen:>8}" + "".join(cells))
    print("\n  * out=1024 is PROJECTED from the measured tail rate — it is the shape the")
    print("    existing benchmark_prefill_share.py used, shown here for reconciliation only.")

    print("\n" + "=" * 100)
    print("  WHAT A DECODE-ONLY OPTIMISATION BUYS  (Amdahl, end-to-end turn speedup)")
    print("=" * 100)
    print(f"  {'turn shape':>14}{'prefill%':>10}{'decode 1.5x':>14}{'decode 2x':>12}"
          f"{'decode inf':>12}")
    print("  " + "-" * 96)
    for plen in PROMPT_LENS:
        for o in OUT_LENS:
            c = report["cells"][f"{plen}x{o}"]
            pf, dec = c["prefill_s"], c["decode_s"]
            tot = pf + dec
            row = {}
            for f in (1.5, 2.0):
                row[f] = tot / (pf + dec / f)
            row["inf"] = tot / pf
            star = "  <- agent" if (plen, o) == (3000, 150) else ""
            print(f"  {f'{plen}x{o}':>14}{c['prefill_share_pct']:>9.1f}%"
                  f"{row[1.5]:>13.2f}x{row[2.0]:>11.2f}x{row['inf']:>11.2f}x{star}")
            c["amdahl_decode_1.5x"] = row[1.5]
            c["amdahl_decode_2x"] = row[2.0]
            c["amdahl_decode_infinite"] = row["inf"]

    agent = report["cells"]["3000x150"]
    print("\n" + "=" * 100)
    print("  VERDICT FOR THE STATED AGENT TURN (3000 in, 150 out)")
    print("=" * 100)
    print(f"    prefill {agent['prefill_s'] * 1000:.0f} ms  ({agent['prefill_share_pct']:.1f}% of the turn)")
    print(f"    decode  {agent['decode_s'] * 1000:.0f} ms  ({100 - agent['prefill_share_pct']:.1f}%)")
    print(f"    total   {agent['total_s'] * 1000:.0f} ms")
    print(f"\n    An INFINITELY fast decode loop caps the turn speedup at "
          f"{agent['amdahl_decode_infinite']:.2f}x.")
    print(f"    A realistic 2x decode win returns {agent['amdahl_decode_2x']:.2f}x end-to-end.")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
