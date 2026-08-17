"""End-to-End A/B benchmark for the VRAM State Router against a live server.

Fires real HTTP requests at /v1/chat/completions and measures wall-clock latency,
so unlike the offline benchmark nothing here is modelled.

METHODOLOGY
-----------
Two arms, same server process, same loaded model, same warm CUDA graph:

  A. FIFO passthrough        — router disabled via POST /v1/router/config
  B. SLA-bounded clustering  — router enabled

Arms alternate (ABAB...) across repeats so that any monotonic drift in the box
(thermal, cache, fragmentation) is shared between them rather than accruing to
whichever ran second. Telemetry is reset at the start of each arm, so counters
report that arm's traffic only.

The previous version ran a single arm and printed a one-column table, which
cannot show whether routing helps. It also read the cumulative
`transitions_avoided` counter after the run without subtracting its pre-run
value, so any earlier traffic inflated the headline.

Usage:
    uv run --env-file .env python benchmarks/runtime/router/vram_state_routing/benchmark_router_e2e.py \\
        --base-url http://127.0.0.1:8000 --num-requests 30 --repeats 2 \\
        --out results/vram_router_e2e_benchmark.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import aiohttp
except ImportError:
    raise SystemExit(
        "aiohttp is required for the e2e benchmark.\nInstall with: uv pip install aiohttp"
    )


DOMAINS = [
    {
        "model": "qwen3.5-4b-astral",
        "domain": "astral",
        "prompt": "How do I configure uv to use a custom index for package resolution?",
    },
    {
        "model": "qwen3.5-4b-postgresql",
        "domain": "postgresql",
        "prompt": "Write a PostgreSQL query to find the top 10 customers by total order value using window functions.",
    },
    {
        "model": "qwen3.5-4b-financial",
        "domain": "financial_planning",
        "prompt": "What is the optimal asset allocation for a 30-year-old with moderate risk tolerance and a 35-year horizon?",
    },
]


def generate_interleaved_workload(num_requests: int, max_tokens: int, seed: int = 42) -> list[dict]:
    """Round-robin across domains so FIFO pays a transition on nearly every request."""
    rng = random.Random(seed)
    workload = []
    for i in range(num_requests):
        domain = DOMAINS[i % len(DOMAINS)]
        workload.append(
            {
                "req_id": f"e2e_{i:04d}",
                "model": domain["model"],
                "domain": domain["domain"],
                "prompt": domain["prompt"],
                "max_tokens": max_tokens,
            }
        )
    for i in range(0, len(workload) - 1, 2):
        if rng.random() < 0.3:
            workload[i], workload[i + 1] = workload[i + 1], workload[i]
    return workload


def count_fifo_transitions(workload: list[dict]) -> int:
    transitions, prev = 0, None
    for w in workload:
        if prev is not None and w["domain"] != prev:
            transitions += 1
        prev = w["domain"]
    return transitions


@dataclass
class RequestResult:
    req_id: str
    model: str
    domain: str
    wall_clock_ms: float
    swap_time_ms: float
    generation_time_ms: float
    tokens_per_second: float
    completion_tokens: int
    router_queue_position: int | None = None
    router_wait_ms: float | None = None
    router_gpu_state: str | None = None
    error: str | None = None


@dataclass
class ArmResult:
    arm: str
    router_enabled: bool
    total_requests: int
    successful_requests: int
    total_wall_clock_s: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_swap_ms: float
    aggregate_tok_s: float
    server_transitions: int
    transitions_avoided: int
    total_batches: int
    latencies_ms: list[float] = field(default_factory=list)
    request_results: list[dict] = field(default_factory=list)


async def fire_single_request(
    session: aiohttp.ClientSession,
    base_url: str,
    item: dict,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": item["model"],
        "messages": [{"role": "user", "content": item["prompt"]}],
        "max_tokens": max_tokens,
        "stream": False,
    }

    async with semaphore:
        t0 = time.perf_counter()
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                wall_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status != 200:
                    body = await resp.text()
                    return RequestResult(
                        item["req_id"], item["model"], item["domain"], wall_ms,
                        0.0, 0.0, 0.0, 0, error=f"HTTP {resp.status}: {body[:200]}",
                    )
                data = await resp.json()
                usage = data.get("usage", {})
                qp = resp.headers.get("X-Router-Queue-Position")
                wait = resp.headers.get("X-Router-Wait-Ms")
                return RequestResult(
                    req_id=item["req_id"],
                    model=item["model"],
                    domain=item["domain"],
                    wall_clock_ms=wall_ms,
                    swap_time_ms=usage.get("swap_time_ms") or 0.0,
                    generation_time_ms=usage.get("generation_time_ms") or 0.0,
                    tokens_per_second=usage.get("tokens_per_second") or 0.0,
                    completion_tokens=usage.get("completion_tokens") or 0,
                    router_queue_position=int(qp) if qp is not None else None,
                    router_wait_ms=float(wait) if wait is not None else None,
                    router_gpu_state=resp.headers.get("X-Router-GPU-State"),
                )
        except Exception as exc:
            return RequestResult(
                item["req_id"], item["model"], item["domain"],
                (time.perf_counter() - t0) * 1000.0, 0.0, 0.0, 0.0, 0, error=repr(exc),
            )


async def configure_router(
    session,
    base_url: str,
    enabled: bool,
    batch_window_ms: float,
    sla_deadline_s: float,
    service_time_s: float,
) -> dict:
    """Switches arms and zeroes telemetry so each arm measures only itself."""
    async with session.post(
        f"{base_url}/v1/router/config",
        json={
            "enabled": enabled,
            "batch_window_ms": batch_window_ms,
            "sla_deadline_s": sla_deadline_s,
            "service_time_s": service_time_s,
            "reset_telemetry": True,
        },
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


async def run_arm(
    base_url: str,
    workload: list[dict],
    max_tokens: int,
    concurrency: int,
    router_enabled: bool,
    batch_window_ms: float,
    sla_deadline_s: float,
    service_time_s: float,
) -> ArmResult:
    arm_name = "SLA-Bounded Clustering" if router_enabled else "FIFO Passthrough"
    print(f"\n  --- Arm: {arm_name} (router_enabled={router_enabled}) ---")

    async with aiohttp.ClientSession() as session:
        status = await configure_router(
            session, base_url, router_enabled, batch_window_ms, sla_deadline_s, service_time_s
        )
        print(f"      mode={status['router']['mode']}, telemetry reset")

        t0 = time.perf_counter()
        sem = asyncio.Semaphore(concurrency)
        results: list[RequestResult] = await asyncio.gather(
            *[fire_single_request(session, base_url, it, max_tokens, sem) for it in workload]
        )
        wall_s = time.perf_counter() - t0

        async with session.get(f"{base_url}/v1/router/status") as resp:
            post = (await resp.json())["router"]

    successful = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    if failed:
        print(f"      ⚠️  {len(failed)} failed; first: {failed[0].error}")

    if not successful:
        return ArmResult(arm_name, router_enabled, len(workload), 0, wall_s,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         post.get("total_transitions", 0),
                         post.get("transitions_avoided", 0),
                         post.get("total_batches", 0))

    lat = sorted(r.wall_clock_ms for r in successful)
    total_tokens = sum(r.completion_tokens for r in successful)
    gen_s = sum(r.generation_time_ms for r in successful) / 1000.0

    def pct(q):
        return lat[min(int(len(lat) * q), len(lat) - 1)]

    res = ArmResult(
        arm=arm_name,
        router_enabled=router_enabled,
        total_requests=len(workload),
        successful_requests=len(successful),
        total_wall_clock_s=wall_s,
        mean_latency_ms=statistics.mean(lat),
        p50_latency_ms=pct(0.50),
        p95_latency_ms=pct(0.95),
        p99_latency_ms=pct(0.99),
        total_swap_ms=sum(r.swap_time_ms for r in successful),
        aggregate_tok_s=round(total_tokens / gen_s, 2) if gen_s > 0 else 0.0,
        # Counters are post-arm values against a telemetry reset at arm start,
        # so they are this arm's traffic only — no pre/post subtraction needed.
        server_transitions=post.get("total_transitions", 0),
        transitions_avoided=post.get("transitions_avoided", 0),
        total_batches=post.get("total_batches", 0),
        latencies_ms=[r.wall_clock_ms for r in results],
        request_results=[asdict(r) for r in results],
    )

    print(f"      wall {wall_s:.2f}s | mean {res.mean_latency_ms:.0f}ms | "
          f"p95 {res.p95_latency_ms:.0f}ms | GPU swaps {res.server_transitions} | "
          f"swap {res.total_swap_ms:.1f}ms | {res.aggregate_tok_s} tok/s")
    return res


def paired_bootstrap_ci(a: list[float], b: list[float], n_boot: int = 5000, seed: int = 42):
    """95% CI on the paired mean difference (a - b). Both arms serve the same workload."""
    rng = random.Random(seed)
    pairs = [(x, y) for x, y in zip(a, b) if x > 0 and y > 0]
    if not pairs:
        return 0.0, 0.0, 0.0
    diffs = [x - y for x, y in pairs]
    n = len(diffs)
    point = sum(diffs) / n
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return point, means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


async def main():
    parser = argparse.ArgumentParser(description="VRAM State Router E2E A/B benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--num-requests", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2, help="ABAB... alternations per arm")
    parser.add_argument("--batch-window-ms", type=float, default=50.0)
    parser.add_argument(
        "--sla-deadline-s",
        type=float,
        default=20.0,
        help="Per-request deadline the router protects. Must be set relative to "
             "realistic service time under load, or every request is already "
             "unsaveable and the SLA bound has nothing to do.",
    )
    parser.add_argument(
        "--service-time-s",
        type=float,
        default=1.9,
        help="Estimated per-request generation time the scheduler plans against.",
    )
    parser.add_argument("--out", default="results/vram_router_e2e_benchmark.json")
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("  VRAM State Router — End-to-End A/B Benchmark")
    print("=" * 78)
    print(f"  Server: {args.base_url} | requests/arm: {args.num_requests} | "
          f"concurrency: {args.concurrency} | repeats: {args.repeats}")

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(
                f"{args.base_url}/health", timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    raise SystemExit(f"Server health check failed: HTTP {resp.status}")
                print("  Health: ✅ server is up")
        except aiohttp.ClientError as e:
            raise SystemExit(f"Cannot reach server at {args.base_url}: {e}")

    workload = generate_interleaved_workload(args.num_requests, args.max_tokens)
    mix: dict[str, int] = {}
    for w in workload:
        mix[w["domain"]] = mix.get(w["domain"], 0) + 1
    print(f"  Workload: {mix} | FIFO transitions if run in arrival order: "
          f"{count_fifo_transitions(workload)}")

    # Warm the server so neither arm eats first-request cost.
    print("\n  Warming up...")
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(2)
        await asyncio.gather(
            *[fire_single_request(session, args.base_url, w, args.max_tokens, sem)
              for w in workload[:3]]
        )

    fifo_arms: list[ArmResult] = []
    router_arms: list[ArmResult] = []

    for rep in range(args.repeats):
        print(f"\n{'='*78}\n  Repeat {rep + 1}/{args.repeats}\n{'='*78}")
        # Alternate order each repeat so neither arm is systematically favoured.
        order = [False, True] if rep % 2 == 0 else [True, False]
        for enabled in order:
            arm = await run_arm(args.base_url, workload, args.max_tokens,
                                args.concurrency, enabled, args.batch_window_ms,
                                args.sla_deadline_s, args.service_time_s)
            (router_arms if enabled else fifo_arms).append(arm)

    def agg(arms: list[ArmResult], attr: str) -> float:
        return statistics.mean(getattr(a, attr) for a in arms)

    print("\n" + "=" * 78)
    print("  📊 A/B Summary (mean across repeats)")
    print("=" * 78)
    print(f"  {'Metric':<30} {'FIFO':>14} {'Router':>14} {'Delta':>14}")
    print("  " + "-" * 74)
    for label, attr, fmt in [
        ("Wall clock (s)", "total_wall_clock_s", "{:.2f}"),
        ("Mean latency (ms)", "mean_latency_ms", "{:.1f}"),
        ("P50 latency (ms)", "p50_latency_ms", "{:.1f}"),
        ("P95 latency (ms)", "p95_latency_ms", "{:.1f}"),
        ("P99 latency (ms)", "p99_latency_ms", "{:.1f}"),
        ("GPU state transitions", "server_transitions", "{:.1f}"),
        ("Total swap overhead (ms)", "total_swap_ms", "{:.1f}"),
        ("Aggregate tok/s", "aggregate_tok_s", "{:.2f}"),
    ]:
        f, r = agg(fifo_arms, attr), agg(router_arms, attr)
        print(f"  {label:<30} {fmt.format(f):>14} {fmt.format(r):>14} {fmt.format(r - f):>14}")

    # Paired significance across all repeats, request by request.
    fifo_lat = [x for a in fifo_arms for x in a.latencies_ms]
    router_lat = [x for a in router_arms for x in a.latencies_ms]
    point, lo, hi = paired_bootstrap_ci(router_lat, fifo_lat)
    significant = lo > 0 or hi < 0
    print(f"\n  Paired mean latency difference (Router - FIFO): {point:.1f} ms")
    print(f"  95% CI [{lo:.1f}, {hi:.1f}] -> "
          f"{'SIGNIFICANT' if significant else 'NOT significant (CI contains 0)'}")
    if significant and point < 0:
        print("  => Router is measurably faster on this workload.")
    elif significant:
        print("  => Router is measurably SLOWER on this workload.")
    else:
        print("  => No detectable latency difference on this workload.")

    swap_f, swap_r = agg(fifo_arms, "total_swap_ms"), agg(router_arms, "total_swap_ms")
    wall_f = agg(fifo_arms, "total_wall_clock_s")
    print(f"\n  Swap overhead as share of wall clock (FIFO): "
          f"{swap_f / 1000.0 / wall_f * 100:.2f}%  <- the entire budget routing can win")
    print(f"  Swap overhead removed by routing: {swap_f - swap_r:.1f} ms")

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "benchmark": "VRAM State Router E2E A/B",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "base_url": args.base_url,
            "num_requests": args.num_requests,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "repeats": args.repeats,
            "batch_window_ms": args.batch_window_ms,
            "workload_mix": mix,
            "fifo_transitions_if_arrival_order": count_fifo_transitions(workload),
        },
        "arms": {
            "fifo_passthrough": [asdict(a) for a in fifo_arms],
            "sla_bounded_clustering": [asdict(a) for a in router_arms],
        },
        "paired_latency_diff_router_minus_fifo": {
            "mean_ms": point, "ci95_lo": lo, "ci95_hi": hi, "significant": significant,
        },
    }, indent=2, default=str))
    print(f"\n  ✅ Saved report to: {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
