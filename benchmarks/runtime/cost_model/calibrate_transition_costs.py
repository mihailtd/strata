"""Calibrates the VRAM state-transition cost model against real hardware.

This is the script that retired the shortest-path router. It originally assumed
a decomposed cost model:
    C(u, v) = t_restore + t_fold * |v|      (a "restore then fold" two-step)

But `WeightFoldingEngine.activate()` always writes W0 + dW in ONE fused addmm
per slot -- it never restores first. So the real cost may be destination-only:
    C(u, v) = f(v)                          (independent of the source state u)

That distinction decides whether shortest-path routing can ever help: under a
destination-only cost, every detour through an intermediate state adds a
strictly positive f(intermediate), so the direct edge is always optimal and
Floyd-Warshall is provably a no-op.

This script measures the transition matrix directly and reports which model holds.

USAGE:
    uv run --env-file .env python \
        benchmarks/runtime/router/vram_state_routing/calibrate_transition_costs.py \
        --repeats 30 --out results/vram_transition_costs.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

DOMAINS = {
    "astral": "results/adapters/m2_astral_r8a128",
    "postgresql": "results/adapters/m2_postgresql_r8a128",
    "financial": "results/adapters/m2_financial_r8a128",
}


def timed(fn, repeats: int, warmup: int = 5) -> dict:
    """Times a callable with CUDA synchronisation, returning median/mean/stdev in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "n": len(samples),
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate VRAM transition costs")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--vram-cap-gb", type=float, default=22.0)
    parser.add_argument("--out", default="results/vram_transition_costs.json")
    args = parser.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)

    print("=" * 78)
    print("  VRAM State Transition Cost Calibration")
    print("=" * 78)
    print(f"  Model   : {args.model_id}")
    print(f"  Repeats : {args.repeats} (plus 5 warmup)")

    print("\nLoading base model (bfloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    base_model.eval()

    expert_map: dict[str, FoldableExpert] = {}
    for domain, rel_path in DOMAINS.items():
        adapter_path = REPO_ROOT / rel_path
        if not adapter_path.exists():
            raise SystemExit(f"Missing adapter: {adapter_path}")
        expert_map[domain] = FoldableExpert.from_dir(adapter_path, name=domain)
        print(f"  Loaded expert [{domain}]")

    engine = WeightFoldingEngine(base_model, list(expert_map.values()), keep_pristine=True)
    names = list(expert_map.keys())

    report: dict = {"model_id": args.model_id, "repeats": args.repeats}

    # --- 1. restore() cost (pure copy, no arithmetic) ---
    print("\n[1] restore()  -- pure pristine copy")
    engine.activate(expert_map[names[0]])
    r = timed(lambda: engine.restore(), args.repeats)
    report["restore"] = r
    print(f"    median {r['median_ms']:.3f} ms  (stdev {r['stdev_ms']:.3f})")

    # --- 2. activate() from every possible source state ---
    # If C(u,v) is destination-only, every row of this matrix is identical.
    print("\n[2] activate(v) measured from each source state u")
    print("    (destination-only cost <=> all rows identical)")
    sources = ["pristine"] + names
    matrix: dict[str, dict[str, dict]] = {}

    for src in sources:
        matrix[src] = {}
        for dst in names:
            def setup_and_go(src=src, dst=dst):
                # Re-enter the source state, then time only the transition
                pass

            # Establish source state, time the destination activate
            samples = []
            for _ in range(args.repeats + 5):
                if src == "pristine":
                    engine.restore()
                else:
                    engine.activate(expert_map[src])
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                engine.activate(expert_map[dst])
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - t0) * 1000.0)
            samples = samples[5:]  # drop warmup
            matrix[src][dst] = {
                "median_ms": statistics.median(samples),
                "mean_ms": statistics.mean(samples),
                "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                "n": len(samples),
            }

    report["activate_matrix"] = matrix

    header = "    " + f"{'from \\ to':<14}" + "".join(f"{d:>16}" for d in names)
    print(header)
    for src in sources:
        row = "    " + f"{src:<14}"
        for dst in names:
            row += f"{matrix[src][dst]['median_ms']:>15.3f} "
        print(row)

    # --- 3. activate_many() for stacked states ---
    print("\n[3] activate_many([a, b])  -- stacked (2-expert) states")
    stacked = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pair = (names[i], names[j])
            experts = [expert_map[pair[0]], expert_map[pair[1]]]
            s = timed(lambda e=experts: engine.activate_many(e), args.repeats)
            stacked["+".join(pair)] = s
            print(f"    {'+'.join(pair):<28} median {s['median_ms']:.3f} ms")
    report["activate_many"] = stacked

    # --- 4. Verdict: is the cost destination-only? ---
    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)

    # For each destination, compare the spread across source states against
    # the within-cell measurement noise.
    dest_only = True
    for dst in names:
        col = [matrix[src][dst]["median_ms"] for src in sources]
        noise = max(matrix[src][dst]["stdev_ms"] for src in sources)
        spread = max(col) - min(col)
        verdict = "source-independent" if spread <= 3 * noise else "SOURCE-DEPENDENT"
        if spread > 3 * noise:
            dest_only = False
        print(
            f"  activate({dst:<12}) spread across sources = {spread:.3f} ms "
            f"| noise (max stdev) = {noise:.3f} ms  -> {verdict}"
        )

    report["destination_only_cost"] = dest_only

    restore_ms = report["restore"]["median_ms"]
    fold_ms = statistics.median([matrix["pristine"][d]["median_ms"] for d in names])
    report["t_restore_ms"] = restore_ms
    report["t_fold_single_ms"] = fold_ms
    stack_ms = statistics.median([v["median_ms"] for v in stacked.values()])
    report["t_fold_stacked2_ms"] = stack_ms

    print(f"\n  Measured t_restore        = {restore_ms:.3f} ms")
    print(f"  Measured t_fold (1 expert)= {fold_ms:.3f} ms")
    print(f"  Measured t_fold (2 stack) = {stack_ms:.3f} ms")

    if dest_only:
        print(
            "\n  => Transition cost is DESTINATION-ONLY: C(u,v) = f(v).\n"
            "     Under a destination-only cost, any detour u -> k -> v costs\n"
            "     f(k) + f(v) > f(v), so the direct edge is always optimal.\n"
            "     Floyd-Warshall is therefore provably a no-op on this graph,\n"
            "     for ANY expert set -- not just the current 3."
        )
    else:
        print("\n  => Transition cost DEPENDS ON SOURCE. Shortest-path routing may help.")

    # Cross-check the router's hardcoded assumption
    assumed_swap = 8.0 + 10.0
    measured_swap = statistics.median(
        [matrix[s][d]["median_ms"] for s in names for d in names if s != d]
    )
    print(
        f"\n  Router assumes a cross-expert swap costs {assumed_swap:.1f} ms "
        f"(t_restore 8.0 + t_fold 10.0)."
    )
    print(f"  Measured cross-expert swap = {measured_swap:.3f} ms (a single fused addmm).")
    report["assumed_swap_ms"] = assumed_swap
    report["measured_swap_ms"] = measured_swap

    drift = engine.max_drift()
    engine.restore()
    drift_after = engine.max_drift()
    report["max_drift_live"] = drift
    report["max_drift_after_restore"] = drift_after
    print(f"\n  max_drift while folded         = {drift:.8f}")
    print(f"  max_drift after restore()      = {drift_after:.8f} (0.0 == bit-exact)")

    out_file = REPO_ROOT / args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved calibration to: {out_file}")


if __name__ == "__main__":
    main()
