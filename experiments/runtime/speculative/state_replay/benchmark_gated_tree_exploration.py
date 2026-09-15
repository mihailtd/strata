r"""Does the real, validated early-exit gate reduce cost when combined with the
real, validated KV-fork tree-branching primitive?

WIDENED (2026-09-13, second pass): the first run of this script measured ONE
real prompt at ONE gate threshold, and measured savings by forward-pass COUNT
only. That's a first observation, not a trustworthy number. This version
sweeps multiple real prompts (drawn from this project's own domain eval data,
same convention as every other real-prompt benchmark here) x multiple gate
thresholds, and measures real WALL-CLOCK time (torch.cuda.synchronize() +
perf_counter()) for both the ungated and gated tree builds, not just pass
counts -- so it can also answer "does skipping N forward passes actually save
N forward passes' worth of TIME, or does per-node bookkeeping (deepcopy'ing
the gate, restoring the KV chain) eat into the savings?"

WHY THIS EXISTS
---------------
The cross-runtime audit (Category 5) flagged this pairing directly: "state_replay's
branching/tree unlock + weibull_hazard_gating's early-exit gate. A tree with a live
per-branch early-abort gate is exactly what both half-finished ideas are reaching
for separately." Both halves are independently validated already, just never
combined:
  - `benchmark_kv_fork_tree_unlock.py` (docs/DECISIONS.md §75): the fork/restore
    primitive is bit-exact and 326x cheaper than full recompute for reviving a
    branch. Imported here directly, not reimplemented.
  - `apps/runtime/range_statistic_gate.py`'s `RangeStatisticGate` (weibull-hazard +
    Bollinger-band composite): already measured a real +4.7% speedup live on plain
    LINEAR speculative decoding. Never tested against branching/tree exploration.

Crucially, this sidesteps §76's blocker entirely. §76 found batching two divergent
candidates into ONE forward pass numerically unsafe on this hardware (a real
conv1d batch-size-dependence bug, 5.6% argmax-flip rate). This experiment never
batches anything -- the gate decides, one branch at a time, sequentially, whether
to keep extending it. Each forward pass is still batch-size-1. The §76 blocker
cannot appear here by construction.

METHOD
------
Build a real, fully-branching width-ary tree to `max_depth` (every live node gets
`width` children, not just one path per level -- unlike §75's own measurement_3,
which only ever advances one path and uses the other siblings as inert leaves).
This is deliberate: an early-exit gate only saves real work if pruning a node
actually removes a subtree that would otherwise have been expanded further.

At each node, right after the one forward pass that creates it, run
`RangeStatisticGate.should_early_exit` on that step's real logits (each branch
carries its OWN gate instance -- a Python-level `copy.deepcopy` of its parent's
gate, since the gate is a few rolling scalar floats, not GPU tensors -- so
divergent branches accumulate independent volatility/hazard state, exactly as
they would in a real multi-branch speculative session). If the gate fires,
that node becomes a permanent leaf: no children are created from it, and the
forward passes that would have built its subtree are never spent.

Compares, on the same real prompt, same real model, same tree shape:
1. UNGATED: every node at every depth gets its full `width` children (the naive
   full-tree baseline -- what §75's plumbing enables but does not itself decide
   when to stop using).
2. GATED: identical tree, but subtrees the gate flags are never expanded.
Reports real forward-pass count saved, fraction of subtrees pruned, and -- because
gating must not be allowed to quietly break the fork/restore correctness guarantee
it sits on top of -- re-confirms bit-exact resume from an arbitrary SURVIVING leaf
in the gated tree, using the identical from-scratch reference method §75 already
validated (`extend()`, not a manual token replay -- see that script's own bug note).

RUN
    uv run --env-file .env python \
        experiments/runtime/speculative/state_replay/benchmark_gated_tree_exploration.py
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.canon import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime.novel_peft import set_hard_vram_cap  # noqa: E402
from runtime.range_statistic_gate import RangeStatisticGate  # noqa: E402

from benchmark_kv_fork_tree_unlock import (  # noqa: E402
    extend,
    fork_bytes,
    fork_kv_tail,
    fresh_cache,
    kv_lens,
    restore_kv_chain,
    restore_ssm,
    snapshot_ssm,
)

PROMPT = "### Question:\nWrite a PostgreSQL query that returns the top customers by revenue.\n\n### Answer:\n"

# Same real per-domain eval files every other benchmark in this repo draws
# from (data/<dir>/evaluation_data*.jsonl) -- no synthetic prompts.
REAL_PROMPT_FILES = [
    "data/astral/evaluation_data.jsonl",
    "data/postgresql/evaluation_data.jsonl",
    "data/duckdb/evaluation_data.jsonl",
    "data/financial_planning/evaluation_data.jsonl",
    "data/python_modern/evaluation_data_disposition.jsonl",
    "data/python_web/evaluation_data_disposition.jsonl",
]


def load_real_prompts(n: int, seed: int = 0) -> list[str]:
    """Draws n real prompts, spread across this project's real domain eval
    files (not synthetic), formatted the same way every other real-prompt
    benchmark here formats them."""
    import json
    import random

    pool: list[str] = []
    for rel in REAL_PROMPT_FILES:
        f = REPO_ROOT / rel
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            p = row.get("prompt")
            if p:
                pool.append(f"### Question:\n{p}\n\n### Answer:\n")
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n] if len(pool) >= n else pool


@torch.no_grad()
def extend_one_with_logits(model, cache, tok_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single real forward pass: returns (next_token, next_token_logits)."""
    out = model(tok_in, past_key_values=cache, use_cache=True)
    logits = out.logits[:, -1, :]
    nxt = torch.argmax(logits, -1, keepdim=True)
    return nxt, logits


def build_tree(
    model, tok, prompt: str, width: int, max_depth: int, gated: bool, gate_kwargs: dict,
) -> tuple[list[dict], int]:
    """Real width-ary tree. Returns (all nodes, forward_passes_spent)."""
    cache, nxt0, ids = fresh_cache(model, tok, prompt)
    root_lens = kv_lens(cache)
    root_ssm = snapshot_ssm(cache)
    root_gate = RangeStatisticGate(**gate_kwargs)
    root_gate.reset_state()
    root = {
        "depth": -1, "branch": -1, "ssm": root_ssm, "lens": root_lens, "last": nxt0,
        "seq": [], "edge_fork": None, "parent": None, "gate": root_gate, "pruned": False,
    }

    nodes: list[dict] = []
    frontier = [root]
    forward_passes = 0

    for d in range(max_depth):
        next_frontier = []
        for parent in frontier:
            if gated and parent["pruned"]:
                continue  # this subtree was already stopped -- spend nothing
            for w in range(width):
                restore_kv_chain(cache, root_lens, parent)
                restore_ssm(cache, parent["ssm"])
                nxt, logits = extend_one_with_logits(model, cache, parent["last"])
                forward_passes += 1
                ssm = snapshot_ssm(cache)
                edge_fork = fork_kv_tail(cache, parent["lens"])
                lens = kv_lens(cache)

                node_gate = copy.deepcopy(parent["gate"])
                should_exit, range_val, tau_eff = node_gate.should_early_exit(logits, step_idx=d)

                node = {
                    "depth": d, "branch": w, "ssm": ssm, "edge_fork": edge_fork, "lens": lens,
                    "last": nxt, "seq": parent["seq"] + [nxt.item()], "parent": parent,
                    "gate": node_gate, "pruned": bool(gated and should_exit),
                    "range_val": range_val, "tau_eff": tau_eff,
                }
                nodes.append(node)
                next_frontier.append(node)
        frontier = next_frontier

    return nodes, forward_passes


def measurement_correctness_on_gated_tree(model, tok, prompt: str, nodes: list[dict], root_lens: list[int]) -> dict:
    """Bit-exact resume check on an arbitrary SURVIVING (non-pruned) leaf of the
    gated tree -- gating must not corrupt the fork/restore guarantee §75 already
    validated. Same from-scratch reference method as §75 (extend(), not a manual
    token replay -- see that script's own off-by-one bug note)."""
    surviving = [n for n in nodes if not n["pruned"]]
    if not surviving:
        return {"skipped": True, "reason": "gate pruned every node -- nothing to resume from"}

    victim = max(surviving, key=lambda n: n["depth"])

    cache, nxt0, ids = fresh_cache(model, tok, prompt)
    restore_kv_chain(cache, root_lens, victim)
    restore_ssm(cache, victim["ssm"])
    resumed, _ = extend(model, cache, victim["last"], 2)

    ref_cache, ref_nxt, _ = fresh_cache(model, tok, prompt)
    ref_seq, ref_last = extend(model, ref_cache, ref_nxt, len(victim["seq"]))
    assert ref_seq == victim["seq"], f"reference replay diverged from the tree path: {ref_seq} != {victim['seq']}"
    ref_resumed, _ = extend(model, ref_cache, ref_last, 2)

    ok = resumed == ref_resumed
    return {
        "skipped": False, "victim_depth": victim["depth"], "victim_branch": victim["branch"],
        "victim_seq": victim["seq"], "resumed": resumed, "reference": ref_resumed,
        "correct": ok,
    }


def timed_build_tree(model, tok, prompt: str, width: int, max_depth: int, gated: bool, gate_kwargs: dict):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    nodes, passes = build_tree(model, tok, prompt, width, max_depth, gated, gate_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return nodes, passes, wall_ms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--thresholds", default="2.0,2.5,3.0,3.5,4.0,5.0",
                     help="comma-separated RangeStatisticGate thresholds to sweep")
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/benchmarks/gated_tree_exploration_sweep.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    thresholds = [float(t) for t in args.thresholds.split(",")]
    prompts = load_real_prompts(args.num_prompts, seed=args.seed)

    print("=" * 100)
    print("  GATED TREE EXPLORATION -- WIDENED SWEEP (real prompts x real thresholds, wall-clock timed)")
    print("=" * 100)
    print(f"  device={dev}  width={args.width}  max_depth={args.max_depth}")
    print(f"  thresholds={thresholds}")
    print(f"  {len(prompts)} real prompts drawn from: {', '.join(REAL_PROMPT_FILES)}\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()

    naive_full_tree_passes = sum(args.width ** d for d in range(1, args.max_depth + 1))
    t_start = time.time()
    runs: list[dict] = []

    for ti, thr in enumerate(thresholds):
        gate_kwargs = {"threshold": thr}
        for pi, prompt in enumerate(prompts):
            nodes_ungated, passes_ungated, ms_ungated = timed_build_tree(
                model, tok, prompt, args.width, args.max_depth, gated=False, gate_kwargs=gate_kwargs)
            nodes_gated, passes_gated, ms_gated = timed_build_tree(
                model, tok, prompt, args.width, args.max_depth, gated=True, gate_kwargs=gate_kwargs)
            n_pruned = sum(1 for n in nodes_gated if n["pruned"])

            root_lens = kv_lens(fresh_cache(model, tok, prompt)[0])
            corr = measurement_correctness_on_gated_tree(model, tok, prompt, nodes_gated, root_lens)

            pass_savings_pct = 100.0 * (1.0 - passes_gated / max(1, passes_ungated))
            time_savings_pct = 100.0 * (1.0 - ms_gated / max(1e-6, ms_ungated))
            run = {
                "threshold": thr, "prompt_idx": pi,
                "passes_ungated": passes_ungated, "passes_gated": passes_gated, "nodes_pruned": n_pruned,
                "ms_ungated": ms_ungated, "ms_gated": ms_gated,
                "pass_savings_pct": pass_savings_pct, "time_savings_pct": time_savings_pct,
                "correct": corr.get("correct", None), "correctness_skipped": corr.get("skipped", False),
            }
            runs.append(run)
            print(f"  thr={thr:4.1f} prompt={pi:2d}  passes {passes_ungated:3d}->{passes_gated:3d} "
                  f"({pass_savings_pct:+5.1f}%)  time {ms_ungated:6.1f}ms->{ms_gated:6.1f}ms "
                  f"({time_savings_pct:+5.1f}%)  correct={corr.get('correct', 'SKIP')}", flush=True)

    # ---- aggregate per threshold -------------------------------------------
    print("\n" + "=" * 100)
    print("  AGGREGATE PER THRESHOLD (mean +/- std across real prompts)")
    print("=" * 100)
    print(f"  {'threshold':>9s}  {'pass_savings%':>16s}  {'time_savings%':>16s}  {'correct':>10s}")
    agg = {}
    for thr in thresholds:
        rows = [r for r in runs if r["threshold"] == thr]
        pass_s = [r["pass_savings_pct"] for r in rows]
        time_s = [r["time_savings_pct"] for r in rows]
        n_correct = sum(1 for r in rows if r["correct"] is True)
        n_checked = sum(1 for r in rows if not r["correctness_skipped"])

        def mean(xs): return sum(xs) / len(xs) if xs else 0.0
        def std(xs):
            m = mean(xs)
            return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if xs else 0.0

        agg[thr] = {
            "pass_savings_mean": mean(pass_s), "pass_savings_std": std(pass_s),
            "time_savings_mean": mean(time_s), "time_savings_std": std(time_s),
            "correct": f"{n_correct}/{n_checked}",
        }
        print(f"  {thr:9.1f}  {mean(pass_s):+7.1f} +/- {std(pass_s):5.1f}  "
              f"{mean(time_s):+7.1f} +/- {std(time_s):5.1f}  {n_correct:>4d}/{n_checked:<4d}")

    all_correct = all(r["correct"] is not False for r in runs)
    elapsed = time.time() - t_start
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"    {len(prompts)} real prompts x {len(thresholds)} thresholds = {len(runs)} runs")
    print(f"    all correctness checks passed: {all_correct}")
    print(f"    {elapsed:.0f}s total")

    report = {
        "device": dev, "width": args.width, "max_depth": args.max_depth, "thresholds": thresholds,
        "num_prompts": len(prompts), "runs": runs, "aggregate_by_threshold": {str(k): v for k, v in agg.items()},
        "all_correctness_checks_passed": all_correct, "elapsed_seconds": elapsed,
    }
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {out_p}")


if __name__ == "__main__":
    main()
