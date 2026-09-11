"""How much overlap is there for an orthogonalisation penalty to actually remove?

THE CLAIM UNDER TEST
--------------------
"Explicit Subspace Orthogonalization Penalty" (and its fleet-scale twin) proposes
adding a training-time loss term that pushes each expert's adapter subspace away
from the others, so that experts can coexist without interfering.

The repo has retired both by *inference*: cross-task overlap was measured at
1.10–1.28x chance, so the subspaces are "already orthogonal" and a penalty would
enforce something that happens for free. That is an argument from an adjacent
measurement, and this repo has just been burned by exactly that move — the same
subspace reasoning "predicted" PiSSA's failure, then a matched-alpha control
showed the geometry had nothing to do with the result (`docs/DECISIONS.md` §2).

So this measures the penalty's headroom directly, on the three LIVE experts, and
states what a penalty could win even in the best case.

WHAT IS MEASURED, PER MODULE, PER EXPERT PAIR
    Both sides of the update matter and they are not the same question:

      OUTPUT side   column space of U (out, r) — which output directions the
                    expert writes into.
      INPUT side    row space of V (r, in) — which input directions it reads.
                    This is the one that governs interference: stacking hurts
                    when expert B READS the directions task A's inputs live in.

    Overlap uses the mean squared cosine of principal angles between orthonormal
    bases, ||Qa^T Qb||_F^2 / r, normalised by the chance floor for two random
    r-dimensional subspaces in ambient dimension d, which is r/d. A ratio of 1.0
    is indistinguishable from random; r/1 would be complete coincidence.

    Also reported: the direct delta-interaction ||dWa^T dWb||_F / (||dWa|| ||dWb||),
    which needs no subspace machinery and answers "do these two updates see each
    other at all" without assuming the low-rank factorisation is the right lens.

    uv run --env-file .env python \
        benchmarks/factory/geometry/orthogonality_headroom/probe_orthogonality_headroom.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
import sys
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.novel_peft import FoldableExpert, set_hard_vram_cap  # noqa: E402

EXPERTS = {
    "astral": "results/adapters/m2_astral_r8a128",
    "postgresql": "results/adapters/m2_postgresql_r8a128",
    "financial": "results/adapters/m2_financial_r8a128",
}


def basis(m: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis for the column space of m, via QR."""
    q, _ = torch.linalg.qr(m.float())
    return q


def overlap_ratio(a: torch.Tensor, b: torch.Tensor, ambient: int) -> float:
    """Mean squared cosine of principal angles, over the random-subspace floor.

    E[||Qa^T Qb||_F^2] = r^2/d for two independent random r-subspaces of R^d, so
    the per-dimension chance level is r/d and the ratio below is 1.0 for random.
    """
    qa, qb = basis(a), basis(b)
    r = qa.shape[1]
    m = (qa.T @ qb).pow(2).sum().item() / r
    return m / (r / ambient)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument(
        "--experts", nargs="+", default=None,
        help="override as name=path pairs, e.g. astral=results/adapters/m2_astral_r8a128 "
             "pg_orth=results/adapters/probe_orth_lam1.0_postgresql",
    )
    ap.add_argument("--out", default="results/orthogonality_headroom.json")
    args = ap.parse_args()

    global EXPERTS
    if args.experts:
        EXPERTS = dict(x.split("=", 1) for x in args.experts)

    set_hard_vram_cap(args.vram_cap_gb)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 100)
    print("  ORTHOGONALITY HEADROOM — what could a subspace penalty actually remove?")
    print("=" * 100)

    experts = {}
    for name, rel in EXPERTS.items():
        e = FoldableExpert.from_dir(REPO_ROOT / rel, name)
        experts[name] = e
        print(f"  loaded {name:<12} modules={len(e.factors)}  scaling={e.scaling}")

    common = set.intersection(*(set(e.factors) for e in experts.values()))
    print(f"\n  modules common to all three experts: {len(common)}")

    report: dict = {"experts": list(EXPERTS), "n_modules": len(common), "pairs": {}}

    print(f"\n  {'pair':<26}{'INPUT side':>26}{'OUTPUT side':>26}")
    print(f"  {'':<26}{'mean x chance':>14}{'max':>12}{'mean x chance':>14}{'max':>12}")
    print("  " + "-" * 96)

    for a, b in itertools.combinations(EXPERTS, 2):
        ins, outs, inter = [], [], []
        for mod in sorted(common):
            ua, va = (t.to(dev).float() for t in experts[a].factors[mod])
            ub, vb = (t.to(dev).float() for t in experts[b].factors[mod])
            # V is (r, in): its ROW space is the read subspace -> transpose to columns.
            ins.append(overlap_ratio(va.T, vb.T, ambient=va.shape[1]))
            outs.append(overlap_ratio(ua, ub, ambient=ua.shape[0]))
            # Direct interaction, no factorisation assumptions.
            dwa = (ua @ va) * experts[a].scaling
            dwb = (ub @ vb) * experts[b].scaling
            inter.append(
                (dwa.T @ dwb).norm().item()
                / max(1e-12, dwa.norm().item() * dwb.norm().item())
            )
        report["pairs"][f"{a}|{b}"] = {
            "input_overlap_x_chance": {"mean": st.mean(ins), "max": max(ins), "min": min(ins)},
            "output_overlap_x_chance": {"mean": st.mean(outs), "max": max(outs), "min": min(outs)},
            "delta_interaction": {"mean": st.mean(inter), "max": max(inter)},
        }
        print(f"  {a + ' | ' + b:<26}{st.mean(ins):>13.3f}x{max(ins):>11.3f}x"
              f"{st.mean(outs):>13.3f}x{max(outs):>11.3f}x")

    print("\n" + "=" * 100)
    print("  DIRECT DELTA INTERACTION  ||dWa^T dWb||_F / (||dWa|| ||dWb||)")
    print("=" * 100)
    print("  0 = the two updates are mutually invisible; 1 = fully aligned.")
    for pair, v in report["pairs"].items():
        print(f"    {pair:<26} mean {v['delta_interaction']['mean']:.4f}   "
              f"max {v['delta_interaction']['max']:.4f}")

    # --- the decision ------------------------------------------------------
    all_in = [v["input_overlap_x_chance"]["mean"] for v in report["pairs"].values()]
    worst = max(all_in)
    print("\n" + "=" * 100)
    print("  HEADROOM FOR A PENALTY")
    print("=" * 100)
    print(f"    worst pairwise INPUT-side overlap: {worst:.3f}x chance")
    excess = 100.0 * (worst - 1.0) / worst
    print(f"    a PERFECT penalty drives this to 1.0x, i.e. removes {excess:.1f}% of the")
    print("    overlap that is present — and 1.0x is random, not zero, so that is the")
    print("    entire budget. There is no configuration in which it removes more.")
    report["worst_input_overlap_x_chance"] = worst
    report["max_removable_pct"] = excess

    if worst < 1.5:
        print("\n    => The subspaces are already at chance. A penalty has essentially")
        print("       nothing to remove, and its only consumer (adapter stacking) is")
        print("       retired for being worth 10.4 ms in a single-tenant engine.")
    else:
        print("\n    => Non-trivial overlap exists. A penalty has something to act on;")
        print("       whether removing it helps anything is a separate training test.")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
