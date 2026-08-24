"""Does a smarter LoRA initialisation reach the target loss in fewer steps?

PiSSA/OLoRA/CorDA were retired in `docs/DECISIONS.md` §2 on a geometric argument:
the trained `dW` puts only 1.37x chance-level energy inside `W0`'s top-8 subspace,
so initialising there was judged pointless. That argument is about where training
ENDS. PiSSA's claim is about where it STARTS. Those are different claims, and no
adapter under any of these schemes had ever been trained here.

This compares matched runs -- same data, same r=8/alpha=128, same lr, same 150
steps, same seed path -- differing only in `init_lora_weights`.

TWO DIFFERENT WINS, REPORTED SEPARATELY (per PISSA_ASSESSMENT.md §3):

  (a) ITERATION WIN  reaches the baseline's FINAL loss in fewer steps
                     -> cut max_steps, bank wall-clock. This is the stated claim.
  (b) QUALITY WIN    reaches a LOWER loss at the same 150 steps
                     -> keep the budget, bank the quality. NOT the paper's claim,
                        so it must not be justified by citing the paper.

SANITY CHECK THAT GATES EVERYTHING. PiSSA at init is mathematically identical to
the base model (`W_res + scaling*B0@A0 == W0`), and stock LoRA starts at `dW = 0`,
which is also the base model. So step-1 loss MUST agree across schemes. If it does
not, the comparison is not matched and no other number here means anything.

n=1 run per (domain, scheme): loss curves are noisy, so a difference smaller than
the smoothed noise floor is not a result. Three domains give three semi-independent
replications of the DIRECTION, which is the evidence that actually counts.

    uv run python scripts/old/compare_init_schemes.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
DOMAINS = ["astral", "postgresql", "financial_planning"]


def ema(xs: list[float], alpha: float) -> list[float]:
    out, cur = [], None
    for x in xs:
        cur = x if cur is None else alpha * x + (1 - alpha) * cur
        out.append(cur)
    return out


def load(path: Path) -> tuple[dict, list[float]]:
    d = json.loads(path.read_text())
    return d, [float(e["loss"]) for e in d["log_history"] if "loss" in e]


def noise_floor(sm: list[float]) -> float:
    return sum(abs(sm[i] - sm[i - 1]) for i in range(1, len(sm))) / max(1, len(sm) - 1)


def first_at_or_below(sm: list[float], target: float) -> int | None:
    for i, v in enumerate(sm):
        if v <= target:
            return i + 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/loss_curves")
    ap.add_argument("--alpha", type=float, default=0.1, help="EMA smoothing factor")
    ap.add_argument("--schemes", nargs="+", default=["pissa", "olora"])
    ap.add_argument("--out", default="results/init_scheme_comparison.json")
    args = ap.parse_args()

    curve_dir = REPO_ROOT / args.dir
    report: dict = {"ema_alpha": args.alpha, "schemes": args.schemes, "domains": {}}

    print("=" * 100)
    print("  LoRA INITIALISATION SCHEMES — does a smarter init train faster?")
    print("=" * 100)

    # --- gate: the arms must start from the same model ------------------------
    print("\n  MATCHING CHECK — step-1 loss must agree (all schemes start at dW==base)")
    print("  " + "-" * 96)
    print(f"  {'domain':<22}{'stock':>10}{'pissa':>10}{'olora':>10}{'max spread':>14}  verdict")
    matched = True
    for dom in DOMAINS:
        base_p = curve_dir / f"{dom}.json"
        if not base_p.exists():
            continue
        firsts = {}
        _, lb = load(base_p)
        firsts["stock"] = lb[0]
        for s in args.schemes:
            p = curve_dir / f"{dom}_{s}.json"
            if p.exists():
                _, lv = load(p)
                firsts[s] = lv[0]
        spread = max(firsts.values()) - min(firsts.values())
        ok = spread < 0.15
        matched &= ok
        cells = "".join(f"{firsts.get(k, float('nan')):>10.3f}" for k in ["stock", *args.schemes])
        print(f"  {dom:<22}{cells}{spread:>14.3f}  {'ok' if ok else '⚠️ NOT MATCHED'}")
        report["domains"].setdefault(dom, {})["step1_loss"] = firsts
        report["domains"][dom]["step1_spread"] = spread
    if not matched:
        print("\n  ⚠️  Step-1 losses disagree. The arms are not matched; treat everything below as void.")

    # --- (a) the iteration win ------------------------------------------------
    print("\n" + "=" * 100)
    print("  (a) ITERATION WIN — steps to reach the STOCK run's final loss  [the stated PiSSA claim]")
    print("=" * 100)
    print(f"  {'domain':<22}{'target':>9}{'stock':>9}" +
          "".join(f"{s:>11}" for s in args.schemes) + f"{'best cut':>11}")
    print("  " + "-" * 96)

    for dom in DOMAINS:
        base_p = curve_dir / f"{dom}.json"
        if not base_p.exists():
            continue
        _, lb = load(base_p)
        sb = ema(lb, args.alpha)
        target = sb[-1]
        stock_steps = first_at_or_below(sb, target) or len(sb)
        row = {"target_loss": target, "stock_steps": stock_steps, "variants": {}}
        cells, cuts = [], []
        for s in args.schemes:
            p = curve_dir / f"{dom}_{s}.json"
            if not p.exists():
                cells.append(f"{'—':>11}")
                continue
            _, lv = load(p)
            sv = ema(lv, args.alpha)
            k = first_at_or_below(sv, target)
            row["variants"][s] = {"steps_to_target": k, "final_ema": sv[-1]}
            if k:
                cut = 100.0 * (stock_steps - k) / stock_steps
                cuts.append((cut, s))
                cells.append(f"{k:>11}")
            else:
                cells.append(f"{'never':>11}")
        best = f"{max(cuts)[0]:+.0f}% ({max(cuts)[1]})" if cuts else "none"
        print(f"  {dom:<22}{target:>9.3f}{stock_steps:>9}" + "".join(cells) + f"{best:>11}")
        report["domains"][dom]["iteration_win"] = row

    # --- (b) the quality win --------------------------------------------------
    print("\n" + "=" * 100)
    print("  (b) QUALITY WIN — final loss at the same 150 steps  [NOT the paper's claim]")
    print("=" * 100)
    print(f"  {'domain':<22}{'stock':>10}" + "".join(f"{s:>11}" for s in args.schemes) +
          f"{'noise floor':>13}  resolvable?")
    print("  " + "-" * 96)

    for dom in DOMAINS:
        base_p = curve_dir / f"{dom}.json"
        if not base_p.exists():
            continue
        _, lb = load(base_p)
        sb = ema(lb, args.alpha)
        nf = noise_floor(sb)
        cells, deltas = [], {}
        for s in args.schemes:
            p = curve_dir / f"{dom}_{s}.json"
            if not p.exists():
                cells.append(f"{'—':>11}")
                continue
            _, lv = load(p)
            sv = ema(lv, args.alpha)
            deltas[s] = sv[-1] - sb[-1]
            cells.append(f"{sv[-1]:>11.3f}")
        big = [s for s, d in deltas.items() if abs(d) > nf]
        verdict = ", ".join(f"{s} {deltas[s]:+.3f}" for s in big) if big else "no — all within noise"
        print(f"  {dom:<22}{sb[-1]:>10.3f}" + "".join(cells) + f"{nf:>13.4f}  {verdict}")
        report["domains"][dom]["quality"] = {
            "stock_final_ema": sb[-1],
            "noise_floor": nf,
            "deltas_vs_stock": deltas,
        }

    # --- wall clock: the win only banks if steps*seconds/step actually falls ----
    # A scheme that reaches the target in 30% fewer steps but costs 40% more per
    # step is a loss. Cross-session wall clock can vary based on load,
    # so this is reported as a check on the iteration win, not as a result itself.
    print("\n" + "=" * 100)
    print("  WALL CLOCK — a step cut only banks if seconds/step does not rise to eat it")
    print("=" * 100)
    print(f"  {'domain':<22}{'scheme':<10}{'train_s':>10}{'s/step':>10}{'vs stock':>11}"
          f"{'steps→target':>14}{'projected_s':>13}{'net':>9}")
    print("  " + "-" * 96)

    def runtime_of(d: dict) -> float | None:
        for e in reversed(d["log_history"]):
            if "train_runtime" in e:
                return float(e["train_runtime"])
        return None

    for dom in DOMAINS:
        base_p = curve_dir / f"{dom}.json"
        if not base_p.exists():
            continue
        db, lb = load(base_p)
        rt_b = runtime_of(db)
        n_b = len(lb)
        sps_b = rt_b / n_b if rt_b else None
        it = report["domains"][dom].get("iteration_win", {})
        stock_steps = it.get("stock_steps", n_b)
        proj_b = sps_b * stock_steps if sps_b else None
        print(f"  {dom:<22}{'stock':<10}{rt_b or 0:>10.1f}{sps_b or 0:>10.3f}{'—':>11}"
              f"{stock_steps:>14}{proj_b or 0:>13.1f}{'—':>9}")
        report["domains"][dom].setdefault("wall_clock", {})["stock"] = {
            "train_s": rt_b, "s_per_step": sps_b, "projected_s_to_target": proj_b
        }
        for s in args.schemes:
            p = curve_dir / f"{dom}_{s}.json"
            if not p.exists():
                continue
            dv, lv = load(p)
            rt_v = runtime_of(dv)
            sps_v = rt_v / len(lv) if rt_v else None
            k = it.get("variants", {}).get(s, {}).get("steps_to_target")
            proj_v = sps_v * k if (sps_v and k) else None
            ratio = f"{sps_v / sps_b:.2f}x" if (sps_v and sps_b) else "—"
            net = f"{100 * (proj_b - proj_v) / proj_b:+.0f}%" if (proj_b and proj_v) else "—"
            print(f"  {dom:<22}{s:<10}{rt_v or 0:>10.1f}{sps_v or 0:>10.3f}{ratio:>11}"
                  f"{(k if k else 'never'):>14}{proj_v or 0:>13.1f}{net:>9}")
            report["domains"][dom]["wall_clock"][s] = {
                "train_s": rt_v, "s_per_step": sps_v,
                "s_per_step_vs_stock": (sps_v / sps_b) if (sps_v and sps_b) else None,
                "projected_s_to_target": proj_v,
                "net_wall_clock_saving_pct": (100 * (proj_b - proj_v) / proj_b)
                if (proj_b and proj_v) else None,
            }

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
