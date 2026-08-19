"""Reads per-step training loss curves and answers, from data, three questions.

    1. Do the domains converge at different rates? (the hypothesis behind
       per-domain step budgets)
    2. Would early stopping have fired, and where?
    3. Is any domain still descending at max_steps — i.e. UNDER-trained, which
       early stopping cannot fix and would make worse?

Everything here is descriptive. No thresholds are baked in as recommendations;
the EMA/patience grid is swept so the sensitivity is visible rather than assumed.

    uv run python scripts/audit/analyze_loss_curves.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
def ema(xs: list[float], alpha: float) -> list[float]:
    out, cur = [], None
    for x in xs:
        cur = x if cur is None else alpha * x + (1 - alpha) * cur
        out.append(cur)
    return out


def load(path: Path) -> tuple[dict, list[int], list[float]]:
    d = json.loads(path.read_text())
    steps, losses = [], []
    for e in d["log_history"]:
        if "loss" in e and "step" in e:
            steps.append(int(e["step"]))
            losses.append(float(e["loss"]))
    return d, steps, losses


def tail_slope(steps: list[int], losses: list[float], frac: float = 0.2) -> float:
    """Least-squares slope over the last `frac` of training, in loss per step.

    Negative and large => still descending => stopping early would truncate real
    learning. Near zero => converged.
    """
    n = max(2, int(len(steps) * frac))
    xs, ys = steps[-n:], losses[-n:]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom if denom else 0.0


def first_within(losses: list[float], final: float, tol: float) -> int | None:
    """First step index whose EMA is already within `tol` of the final EMA."""
    for i, v in enumerate(losses):
        if abs(v - final) <= tol:
            return i + 1
    return None


def early_stop_step(sm: list[float], min_delta: float, patience: int) -> int | None:
    """Where EMA+patience early stopping would have fired."""
    best, wait = float("inf"), 0
    for i, v in enumerate(sm):
        if v < best - min_delta:
            best, wait = v, 0
        else:
            wait += 1
            if wait >= patience:
                return i + 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/loss_curves")
    ap.add_argument("--alpha", type=float, default=0.1, help="EMA smoothing factor")
    ap.add_argument("--out", default="results/loss_curve_analysis.json")
    args = ap.parse_args()

    curve_dir = REPO_ROOT / args.dir
    files = sorted(p for p in curve_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No curves in {curve_dir}")

    report: dict = {"ema_alpha": args.alpha, "domains": {}}

    print("=" * 92)
    print("  PER-STEP LOSS CURVE ANALYSIS")
    print("=" * 92)
    print(f"  EMA alpha = {args.alpha}\n")

    print(f"  {'domain':<20}{'steps':>7}{'epochs':>8}{'loss[0]':>9}{'final':>8}"
          f"{'min':>8}{'tail slope/step':>17}{'verdict':>16}")
    print("  " + "-" * 88)

    for f in files:
        d, steps, losses = load(f)
        if not losses:
            print(f"  {f.stem:<20} no loss entries")
            continue
        sm = ema(losses, args.alpha)
        slope = tail_slope(steps, sm)
        # "still descending" = the last 20% is still falling by more than the
        # step-to-step noise floor of the smoothed curve
        noise = sum(abs(sm[i] - sm[i - 1]) for i in range(1, len(sm))) / max(1, len(sm) - 1)
        still_descending = slope < -noise
        verdict = "STILL DESCENDING" if still_descending else "converged"

        print(f"  {f.stem:<20}{len(steps):>7}{d.get('epochs_seen', 0):>8.2f}"
              f"{losses[0]:>9.3f}{sm[-1]:>8.3f}{min(sm):>8.3f}"
              f"{slope:>17.5f}{verdict:>16}")

        report["domains"][f.stem] = {
            "n_steps": len(steps),
            "epochs_seen": d.get("epochs_seen"),
            "n_records": d.get("n_records"),
            "first_loss": losses[0],
            "final_ema": sm[-1],
            "min_ema": min(sm),
            "tail_slope_per_step": slope,
            "smoothed_noise_floor": noise,
            "still_descending_at_max_steps": still_descending,
        }

    # --- where would you have to stop to keep the final loss? ---
    print("\n" + "=" * 92)
    print("  STEPS NEEDED TO REACH WITHIN X OF FINAL LOSS  (what a budget cut would cost)")
    print("=" * 92)
    print(f"  {'domain':<20}{'within 0.05':>13}{'within 0.02':>13}{'within 0.01':>13}{'of max':>10}")
    print("  " + "-" * 88)
    for name, _r in report["domains"].items():
        d, steps, losses = load(curve_dir / f"{name}.json")
        sm = ema(losses, args.alpha)
        row = {}
        cells = []
        for tol in (0.05, 0.02, 0.01):
            k = first_within(sm, sm[-1], tol)
            row[f"within_{tol}"] = k
            cells.append(f"{k}" if k else "never")
        pct = f"{100*row['within_0.02']/len(steps):.0f}%" if row.get("within_0.02") else "—"
        print(f"  {name:<20}{cells[0]:>13}{cells[1]:>13}{cells[2]:>13}{pct:>10}")
        report["domains"][name]["steps_to_reach_final"] = row

    # --- would early stopping fire, and where? sweep the grid ---
    print("\n" + "=" * 92)
    print("  WHERE EMA EARLY STOPPING WOULD FIRE  (grid sweep — sensitivity is the point)")
    print("=" * 92)
    grid = [(0.005, 15), (0.005, 25), (0.01, 15), (0.02, 15), (0.01, 30)]
    header = "  " + f"{'domain':<20}" + "".join(f"{f'd={d},p={p}':>13}" for d, p in grid)
    print(header)
    print("  " + "-" * 88)
    for name in report["domains"]:
        _, steps, losses = load(curve_dir / f"{name}.json")
        sm = ema(losses, args.alpha)
        cells, fired = [], {}
        for md, pat in grid:
            k = early_stop_step(sm, md, pat)
            fired[f"min_delta={md},patience={pat}"] = k
            cells.append(f"{k}" if k else f"never({len(steps)})")
        print("  " + f"{name:<20}" + "".join(f"{c:>13}" for c in cells))
        report["domains"][name]["early_stop_would_fire_at"] = fired

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
