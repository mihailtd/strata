r"""Is "surgical stacking" surgery, or is it a rounding error? -- CPU, seconds.

WHY
---
`WeightFoldingEngine.activate_many(scale_mode="surgical")` is the runtime DEFAULT
(DECISIONS.md §52) and the claimed mechanism behind §50/§51. Measured on the real v6
adapters, what it actually does to a 2-expert stack:

    30 neurons zeroed, across 2 of 128 weight matrices  (0.0039% of output neurons)
    -> removes 0.005% - 0.009% of the stacked perturbation

`top_k=15` and `max_conflict_modules=2` are hardcoded defaults that were never swept.
This probe sweeps them and asks the question that decides whether the knob is worth
having:

    Is notching SELECTIVE? Does it remove cross-talk faster than it removes the
    in-domain signal it is supposed to protect?

If crosstalk and signal fall at the same rate, notching is not surgery -- it is
deleting neurons, and the LV-GLasso result behind it is a diagnosis with no
corresponding treatment.

METHOD -- everything is analytic in the r x r factors, no dense dW is ever built
--------------------------------------------------------------------------------
    ||dW_i||_F^2      = s_i^2 tr( (U_i^T U_i)(V_i V_i^T) )
    <dW_i, dW_j>      = s_i s_j tr( (U_i^T U_j)(V_j V_i^T) )
    row-wise crosstalk_r = s_i s_j * <row_r(U_i) V_i, row_r(U_j) V_j>

Masking multiplies rows of U, so every quantity above stays exact under a mask.

    CUDA_VISIBLE_DEVICES="" uv run python \
        benchmarks/factory/geometry/surgical_notch_sweep/probe_notch_sweep.py
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from runtime.canon import CANON, REPO_ROOT, adapter_path
from runtime.novel_peft import FoldableExpert, compute_surgical_notch_masks

PAIRS = [("astral", "python_modern"), ("postgresql", "duckdb"), ("astral", "postgresql")]
MLP = ("gate_proj", "down_proj", "up_proj")
TOP_K = [15, 50, 200, 1000]
MAX_MODULES = [2, 8, 32, 128]


def pair_stats(ea: FoldableExpert, eb: FoldableExpert, keys: list[str]) -> dict:
    """Per-key row-wise crosstalk and signal, in the factor basis."""
    out = {}
    for key in keys:
        ua, va = (t.double() for t in ea.factors[key])
        ub, vb = (t.double() for t in eb.factors[key])
        sa, sb = ea.scaling, eb.scaling
        VaVa = va @ va.T
        VbVb = vb @ vb.T
        VaVb = va @ vb.T
        # row_r(dW_i) . row_r(dW_j) for every output neuron r, without expanding dW
        cross = sa * sb * torch.sum((ua @ VaVb) * ub, dim=1)      # (d_out,)
        sig_a = (sa ** 2) * torch.sum((ua @ VaVa) * ua, dim=1)    # ||row_r(dW_a)||^2
        sig_b = (sb ** 2) * torch.sum((ub @ VbVb) * ub, dim=1)
        out[key] = {"cross": cross, "sig_a": sig_a, "sig_b": sig_b,
                    "sharpness": float(cross.abs().max() / (cross.abs().mean() + 1e-12))}
    return out


def apply(stats: dict, masks: dict[str, torch.Tensor]) -> dict:
    """Totals under a mask set. Absent key = no mask = untouched."""
    tot = {"cross_abs": 0.0, "cross_signed": 0.0, "sig": 0.0, "neurons": 0, "modules": 0}
    for key, s in stats.items():
        m = masks.get(key)
        if m is None:
            keep = torch.ones_like(s["cross"])
        else:
            keep = m.double()
            tot["modules"] += 1
            tot["neurons"] += int((m == 0).sum())
        tot["cross_abs"] += float((s["cross"].abs() * keep).sum())
        tot["cross_signed"] += float((s["cross"] * keep).sum())
        tot["sig"] += float(((s["sig_a"] + s["sig_b"]) * keep).sum())
    return tot


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/benchmarks/surgical_notch_sweep.json")
    args = ap.parse_args()
    if torch.cuda.is_available():
        raise SystemExit('  CPU-only probe. Re-run with CUDA_VISIBLE_DEVICES="".')

    t0 = time.time()
    print("=" * 96)
    print(f" SURGICAL NOTCH SWEEP   adapters={CANON.ADAPTER_VERSION}   CPU")
    print(" shipped defaults: top_k=15, max_conflict_modules=2, sharpness gate > 3.0")
    print("=" * 96)

    experts = {d: FoldableExpert.from_dir(adapter_path(d), d)
               for d in {x for p in PAIRS for x in p}}
    keys = sorted(k for k in experts[PAIRS[0][0]].factors if any(m in k for m in MLP))
    print(f"  {len(keys)} MLP weight matrices per adapter "
          f"({len(experts[PAIRS[0][0]].factors)} total)\n")

    results = {}
    for a, b in PAIRS:
        st = pair_stats(experts[a], experts[b], keys)
        base = apply(st, {})
        sharp = np.array([st[k]["sharpness"] for k in keys])
        n_gate = int((sharp > 3.0).sum())

        print("=" * 96)
        print(f" {a} + {b}")
        print(f"   sharpness (max/mean of row crosstalk) over {len(keys)} MLP matrices:")
        print(f"     min {sharp.min():.2f}   median {np.median(sharp):.2f}   "
              f"p90 {np.percentile(sharp, 90):.2f}   max {sharp.max():.2f}")
        print(f"     pass the >3.0 gate: {n_gate}/{len(keys)}  "
              f"-- the shipped config then keeps only the top 2")

        rows = []
        for mm in MAX_MODULES:
            for tk in TOP_K:
                masks = compute_surgical_notch_masks(
                    [experts[a], experts[b]], top_k=tk, max_conflict_modules=mm)
                cur = apply(st, masks)
                d_cross = 1 - cur["cross_abs"] / base["cross_abs"]
                d_sig = 1 - cur["sig"] / base["sig"]
                rows.append({"top_k": tk, "max_modules": mm,
                             "modules": cur["modules"], "neurons": cur["neurons"],
                             "crosstalk_removed": d_cross, "signal_removed": d_sig,
                             "selectivity": (d_cross / d_sig) if d_sig > 1e-12 else float("nan")})

        print(f"\n   {'top_k':>6} {'maxmod':>7} {'mods':>5} {'neurons':>8} "
              f"{'crosstalk-':>11} {'signal-':>9} {'selectivity':>12}")
        print("   " + "-" * 66)
        for r in rows:
            star = "  <- SHIPPED" if (r["top_k"] == 15 and r["max_modules"] == 2) else ""
            print(f"   {r['top_k']:6d} {r['max_modules']:7d} {r['modules']:5d} "
                  f"{r['neurons']:8d} {r['crosstalk_removed']:10.4%} "
                  f"{r['signal_removed']:8.4%} {r['selectivity']:12.2f}{star}")
        results[f"{a}+{b}"] = {"sharpness": sharp.tolist(), "gate_pass": n_gate,
                               "n_keys": len(keys), "sweep": rows}
        print()

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": CANON.stamp() | {"pairs": [list(p) for p in PAIRS]},
                               "results": results,
                               "elapsed_seconds": time.time() - t0}, indent=2))
    print("=" * 96)
    print(" selectivity = (fraction of crosstalk removed) / (fraction of signal removed).")
    print(" 1.0 means notching destroys signal exactly as fast as crosstalk -- i.e. it is")
    print(" deleting neurons, not performing surgery. Above 1.0 it is buying something.")
    print(f"\n  {time.time()-t0:.1f}s   wrote {args.out}")


if __name__ == "__main__":
    main()
