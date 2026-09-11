"""Hardware-in-the-loop queueing benchmark for the VRAM State Router.

WHAT IS REAL AND WHAT IS MODELLED
---------------------------------
Real: every state transition is an actual `WeightFoldingEngine` call on the GPU,
timed with `torch.cuda.synchronize()` around it. Transition counts and transition
milliseconds are measurements.

Modelled: token generation is charged at a fixed per-token rate rather than
actually decoded, so that a 100+ request sweep finishes in minutes instead of
hours. Every regime is charged the identical rate, so the comparison between
regimes is fair, but the absolute latency figures are simulation output, not
measured serving latency. For measured end-to-end latency against a live server,
use `benchmark_router_e2e.py`.

WHY THE ARRIVAL-RATE SWEEP
--------------------------
The previous version of this benchmark ran a single arrival rate of 10 req/s
against a service rate of 1.74 req/s -- an offered load of rho = 5.76. That queue
is unstable by construction: latency is dominated by unbounded queue growth, all
regimes report ~96% SLA violations, and scheduling policy is invisible in the
result. This version sweeps rho on both sides of saturation so the regime where
routing actually matters is visible.

USAGE:
    uv run --env-file .env python benchmarks/runtime/router/vram_state_routing/benchmark_vram_router.py \
        --num-requests 200 --out results/vram_router_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)
from runtime.router.vram_state_router import (  # noqa: E402
    PendingRequest,
    TransitionCosts,
    VRAMState,
    VRAMStateGraph,
    VRAMStateScheduler,
)

DOMAINS = {
    "astral": "results/adapters/m2_astral_r8a128",
    "postgresql": "results/adapters/m2_postgresql_r8a128",
    "financial": "results/adapters/m2_financial_r8a128",
}

# Measured decode throughput of the folded CUDA-graph path (see README).
DEFAULT_TOK_S = 32.89


@dataclass
class WorkloadRequest:
    req_id: str
    domain: str
    target_state: VRAMState
    arrival_time_s: float
    sla_deadline_s: float
    gen_tokens: int = 32


def generate_synthetic_workload(
    num_requests: int,
    arrival_rate: float,
    sla_deadline_s: float,
    gen_tokens: int,
    seed: int = 42,
) -> list[WorkloadRequest]:
    """Poisson arrivals with a Zipfian domain mix."""
    rng = random.Random(seed)
    domains = ["astral", "postgresql", "financial"]
    weights = [0.50, 0.35, 0.15]

    workload = []
    curr_time = 0.0
    for i in range(num_requests):
        curr_time += rng.expovariate(arrival_rate)
        domain = rng.choices(domains, weights=weights)[0]
        workload.append(
            WorkloadRequest(
                req_id=f"req_{i:04d}",
                domain=domain,
                target_state=VRAMState.single(domain),
                arrival_time_s=curr_time,
                sla_deadline_s=sla_deadline_s,
                gen_tokens=gen_tokens,
            )
        )
    return workload


def paired_bootstrap_ci(
    a: list[float], b: list[float], n_boot: int = 5000, seed: int = 42
) -> tuple[float, float, float]:
    """95% CI on the PAIRED mean difference (a - b), plus the point estimate.

    The regimes serve the identical workload, so the requests pair up one-to-one.
    Comparing two independent marginal CIs (as the previous version did) throws
    that pairing away and is the wrong test -- marginal CIs can overlap heavily
    while the paired difference is unambiguous.
    """
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    point = sum(diffs) / n
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return point, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(int(q * len(sorted_vals)), len(sorted_vals) - 1)
    return sorted_vals[idx]


def run_workload_simulation(
    regime_name: str,
    workload: list[WorkloadRequest],
    engine: WeightFoldingEngine,
    expert_map: dict[str, FoldableExpert],
    scheduler: VRAMStateScheduler | None,
    policy: str,
    sec_per_token: float,
    reload_penalty_s: float = 0.0,
) -> dict:
    """Single-server queueing simulation with REAL hardware state transitions.

    policy: "fifo" | "greedy" | "router"
    reload_penalty_s: measured cost of a cold adapter reload, charged per swap for
        the legacy baseline. Measured, not invented.
    """
    engine.restore()
    torch.cuda.synchronize()

    current_state = VRAMState.pristine()
    total_transition_ms = 0.0
    transition_count = 0
    sla_violations = 0
    latencies_ms: list[float] = []
    per_request_latency: dict[str, float] = {}

    pending = list(workload)
    sim_time = 0.0

    while pending:
        available = [r for r in pending if r.arrival_time_s <= sim_time]
        if not available:
            sim_time = pending[0].arrival_time_s
            available = [pending[0]]

        if policy == "router" and scheduler is not None:
            p_reqs = [
                PendingRequest(
                    req_id=r.req_id,
                    target_state=r.target_state,
                    arrival_time=r.arrival_time_s,
                    sla_deadline_s=r.sla_deadline_s,
                    metadata={"raw": r},
                    service_time_s=r.gen_tokens * sec_per_token,
                )
                for r in available
            ]
            scheduled = scheduler.schedule_batch(
                p_reqs, current_state=current_state, current_time_s=sim_time
            )
            next_req = scheduled[0].metadata["raw"]
        elif policy == "greedy":
            matching = [r for r in available if r.target_state == current_state]
            next_req = matching[0] if matching else available[0]
        else:  # fifo
            next_req = available[0]

        pending.remove(next_req)

        if next_req.target_state != current_state:
            if reload_penalty_s > 0:
                sim_time += reload_penalty_s
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.activate(expert_map[next_req.domain])  # REAL hardware transition
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0

            total_transition_ms += dt_ms
            transition_count += 1
            current_state = next_req.target_state
            sim_time += dt_ms / 1000.0

        sim_time += next_req.gen_tokens * sec_per_token

        latency_s = sim_time - next_req.arrival_time_s
        latencies_ms.append(latency_s * 1000.0)
        per_request_latency[next_req.req_id] = latency_s * 1000.0
        if latency_s > next_req.sla_deadline_s:
            sla_violations += 1

    # Drift must be measured against a RESTORED engine. The previous version
    # called max_drift() while weights were still folded and labelled the
    # resulting 0.0094 "bit-exact"; that number was the live delta, not drift.
    engine.restore()
    torch.cuda.synchronize()
    drift_after_restore = engine.max_drift()

    ordered = sorted(latencies_ms)
    return {
        "regime": regime_name,
        "policy": policy,
        "total_requests": len(workload),
        "total_transitions": transition_count,
        "total_transition_ms": total_transition_ms,
        "total_makespan_s": sim_time,
        "transition_share_of_makespan_pct": (total_transition_ms / 1000.0) / sim_time * 100.0,
        "mean_latency_ms": statistics.mean(latencies_ms),
        "p50_latency_ms": percentile(ordered, 0.50),
        "p95_latency_ms": percentile(ordered, 0.95),
        "p99_latency_ms": percentile(ordered, 0.99),
        "sla_violations": sla_violations,
        "sla_violation_pct": sla_violations / len(workload) * 100.0,
        "max_drift_after_restore": drift_after_restore,
        "_latencies": [per_request_latency[r.req_id] for r in workload],
    }


def measure_reload_penalty(adapter_dir: Path, reps: int = 3) -> float:
    """Measures a real cold adapter load from disk, in seconds.

    The legacy baseline needs a number for 'reload the adapter instead of folding
    a resident one'. The previous version used a hardcoded `time.sleep(0.080)`
    with a comment claiming ~180ms -- so its entire legacy penalty was invented,
    and inconsistent with its own comment. This measures it instead.
    """
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        FoldableExpert.from_dir(adapter_dir, name="reload_probe")
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description="VRAM State Router queueing benchmark")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--num-requests", type=int, default=200)
    parser.add_argument(
        "--rho",
        type=float,
        nargs="+",
        default=[0.5, 0.8, 0.95, 1.5],
        help="Offered loads to sweep. rho<1 is a stable queue; rho>1 saturates.",
    )
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--tok-s", type=float, default=DEFAULT_TOK_S)
    parser.add_argument("--sla-deadline-s", type=float, default=2.0)
    parser.add_argument("--out", default="results/vram_router_benchmark.json")
    parser.add_argument("--vram-cap-gb", type=float, default=22.0)
    args = parser.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    sec_per_token = 1.0 / args.tok_s
    service_time_s = args.gen_tokens * sec_per_token

    print("=" * 90)
    print("  VRAM State Router — Hardware-in-the-Loop Queueing Benchmark")
    print("=" * 90)
    print(f"  Requests/arm  : {args.num_requests}")
    print(f"  Service time  : {service_time_s*1000:.0f} ms/request "
          f"({args.gen_tokens} tokens @ {args.tok_s} tok/s)")
    print(f"  SLA deadline  : {args.sla_deadline_s}s")
    print(f"  Offered loads : {args.rho}")
    print("  Transitions are REAL GPU folds; generation is modelled at a fixed rate.")

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
        path = REPO_ROOT / rel_path
        if not path.exists():
            raise SystemExit(f"Missing adapter: {path}")
        expert_map[domain] = FoldableExpert.from_dir(path, name=domain)
        print(f"  Loaded expert [{domain}]: {expert_map[domain].nbytes / 1e6:.2f} MB")

    engine = WeightFoldingEngine(base_model, list(expert_map.values()), keep_pristine=True)
    print(f"  Pristine buffer: {engine.pristine_bytes / 1e6:.2f} MB")

    reload_penalty_s = measure_reload_penalty(REPO_ROOT / DOMAINS["astral"])
    print(f"  Measured cold adapter reload: {reload_penalty_s*1000:.1f} ms (legacy baseline)")

    calib_path = REPO_ROOT / "results" / "vram_transition_costs.json"
    costs = TransitionCosts.from_calibration(calib_path) if calib_path.exists() else TransitionCosts()
    graph = VRAMStateGraph(list(expert_map.keys()), allow_stacking=False, costs=costs)
    scheduler = VRAMStateScheduler(
        graph,
        default_sla_deadline_s=args.sla_deadline_s,
        default_service_time_s=service_time_s,
    )
    print(f"  Router: {graph.num_nodes} states, SLA-bounded cluster scheduling "
          f"(no shortest-path solve — provably degenerate here)")

    regimes = [
        ("Legacy Reload FIFO", "fifo", reload_penalty_s),
        ("In-Place FIFO", "fifo", 0.0),
        ("Greedy Affinity", "greedy", 0.0),
        ("SLA-Bounded Cluster Router", "router", 0.0),
    ]

    all_results: dict = {
        "config": {
            "model_id": args.model_id,
            "num_requests": args.num_requests,
            "gen_tokens": args.gen_tokens,
            "tok_s": args.tok_s,
            "service_time_s": service_time_s,
            "sla_deadline_s": args.sla_deadline_s,
            "measured_reload_penalty_ms": reload_penalty_s * 1000.0,
            "note": "Transitions are real GPU folds; generation is modelled at a fixed rate.",
        },
        "sweep": {},
    }

    for rho in args.rho:
        arrival_rate = rho / service_time_s
        workload = generate_synthetic_workload(
            args.num_requests, arrival_rate, args.sla_deadline_s, args.gen_tokens
        )
        mix = {d: sum(1 for r in workload if r.domain == d) for d in DOMAINS}

        print("\n" + "═" * 90)
        print(f"  OFFERED LOAD rho = {rho}  (arrival {arrival_rate:.2f} req/s, "
              f"service {1/service_time_s:.2f} req/s)  {'STABLE' if rho < 1 else 'SATURATED'}")
        print(f"  Domain mix: {mix}")
        print("═" * 90)
        print(f"  {'Regime':<28} {'Swaps':>6} {'Swap ms':>9} {'Swap%':>7} "
              f"{'Mean ms':>10} {'P95 ms':>10} {'SLA viol':>10}")
        print("  " + "-" * 86)

        rho_results = {}
        for name, policy, penalty in regimes:
            res = run_workload_simulation(
                name, workload, engine, expert_map, scheduler, policy,
                sec_per_token, reload_penalty_s=penalty,
            )
            rho_results[policy if penalty == 0 else "legacy_reload_fifo"] = res
            print(f"  {name:<28} {res['total_transitions']:>6} "
                  f"{res['total_transition_ms']:>9.1f} "
                  f"{res['transition_share_of_makespan_pct']:>6.2f}% "
                  f"{res['mean_latency_ms']:>10.1f} {res['p95_latency_ms']:>10.1f} "
                  f"{res['sla_violation_pct']:>9.1f}%")

        # Paired comparison: router vs each baseline, on the identical workload.
        print("\n  Paired mean-latency difference vs SLA-Bounded Cluster Router")
        print("  (negative => router is faster; CI excluding 0 => significant)")
        router_lat = rho_results["router"]["_latencies"]
        for key, label in (("fifo", "In-Place FIFO"), ("greedy", "Greedy Affinity")):
            point, lo, hi = paired_bootstrap_ci(router_lat, rho_results[key]["_latencies"])
            sig = "significant" if (lo > 0 or hi < 0) else "not significant"
            print(f"    vs {label:<20} {point:>9.1f} ms  95% CI [{lo:>8.1f}, {hi:>8.1f}]  {sig}")
            rho_results["router"].setdefault("paired_vs", {})[key] = {
                "mean_diff_ms": point, "ci95_lo": lo, "ci95_hi": hi,
                "significant": bool(lo > 0 or hi < 0),
            }

        drift = rho_results["router"]["max_drift_after_restore"]
        print(f"\n  Max weight drift after restore(): {drift:.8f} "
              f"({'bit-exact' if drift == 0.0 else 'NOT bit-exact'})")

        for v in rho_results.values():
            v.pop("_latencies", None)
        all_results["sweep"][f"rho_{rho}"] = {"rho": rho, "arrival_rate": arrival_rate,
                                              "regimes": rho_results}

    out_file = REPO_ROOT / args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\n✅ Saved report to: {out_file}")


if __name__ == "__main__":
    main()
