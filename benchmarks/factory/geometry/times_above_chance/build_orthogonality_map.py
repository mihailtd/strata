"""Cross-task subspace orthogonality map over every trained domain adapter.

Answers one question before any shared-basis work is attempted: do adapters for
different tasks occupy overlapping low-rank subspaces, or independent ones?

READ THE RATIO, NOT THE PERCENTAGE. Projecting onto a k-dimensional subspace of
an (in*out)-dimensional matrix space captures k/(in*out) of the energy by pure
chance -- about 0.0005% at k=32 for these shapes. So a raw "retention < 1%"
test can never fail and carries no information. Calibrated on this repo's own
adapters:

    same adapter vs itself          26569x chance   (ceiling)
    astral a256 vs astral a128       7.15x chance   (same task, real structure)
    astral vs financial_planning     1.28x chance   (different tasks, orthogonal)

A raw-percentage test called ALL THREE "orthogonal", including the same-task
pair. This script reports x-above-chance, and includes the self and same-task
pairs as positive controls so the null is visible in the same table.

    uv run scripts/build_orthogonality_map.py
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# the probes live under benchmarks/, not scripts/ -- the old path silently
# broke this module (ModuleNotFoundError) when it moved
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'preflight_svd_probe'))
from probe_subspace_overlap import evaluate_subspace_overlap  # noqa: E402

# One representative peft-format LoRA per domain (the probe reads peft or novel).
ADAPTERS = {
    "astral": "results/adapters/astral_qwen3.5_micro_lora_a256",
    "astral_alt": "results/adapters/astral_qwen3.5_micro_lora_a128",
    "postgres": "results/adapters/postgres_qwen3.5_micro_custom_standard",
    "financial": "results/adapters/financial_planning_standard_lora",
}


def summarise(a: Path, b: Path, k: int) -> tuple[float, float, float]:
    res = evaluate_subspace_overlap(a, b, k=k)
    if not res:
        return float("nan"), float("nan"), float("nan")
    ret = sum(m["retained_energy_pct"] for m in res.values()) / len(res)
    floor = sum(m["random_floor_pct"] for m in res.values()) / len(res)
    return ret, floor, ret / max(1e-30, floor)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default="results/orthogonality_map.json")
    args = ap.parse_args()

    present = {n: REPO_ROOT / p for n, p in ADAPTERS.items() if (REPO_ROOT / p).exists()}
    for n, p in ADAPTERS.items():
        if n not in present:
            print(f"[warn] missing {n}: {p}")

    pairs = []
    names = sorted(present)
    # self pairs (ceiling) + every unordered cross pair
    for n in names:
        pairs.append((f"{n} vs itself", n, n))
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs.append((f"{a} vs {b}", a, b))

    print(f"cross-task orthogonality map (k={args.k})\n")
    print(f"{'pair':38s} {'retention':>11s} {'chance':>10s} {'x chance':>10s}  {'reading':s}")
    rows = []
    for label, a, b in pairs:
        ret, floor, x = summarise(present[a], present[b], args.k)
        if x != x:  # nan -> no shared shape groups
            print(f"{label:38s}  (no shared shape groups)")
            continue
        reading = "ceiling" if a == b else ("ORTHOGONAL" if x < 2.0 else "shared structure")
        print(f"{label:38s} {ret:10.5f}% {floor:9.5f}% {x:9.2f}x  {reading}")
        rows.append({"pair": label, "a": a, "b": b, "retention_pct": ret, "chance_pct": floor, "x_chance": x})

    cross = [r for r in rows if r["a"] != r["b"]]
    if cross:
        print("\nCross-pairs only:")
        worst = max(cross, key=lambda r: r["x_chance"])
        print(f"  highest overlap: {worst['pair']} at {worst['x_chance']:.2f}x chance")
        if worst["x_chance"] < 2.0:
            print("  -> every task pair is at chance. Shared-basis compression cannot work here.")
        else:
            print("  -> some pair is above chance; re-test basis sharing directly before trusting it.")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"k": args.k, "rows": rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
