r"""Does the early-exit gate ever prune a genuinely BETTER candidate branch?

WHY THIS EXISTS
---------------
`benchmark_gated_tree_exploration.py` (docs/DECISIONS.md §78) measured real,
substantial forward-pass and wall-clock savings from gating a branching tree
-- but its branches were built with pure greedy argmax at every node. Under
deterministic greedy decoding, every sibling branch from the same parent
computes the IDENTICAL token, so "pruning a branch" there only ever discarded
a redundant recomputation of an already-decided path, never a genuinely
different candidate. That result is real (the fork+gate plumbing works, and
skipping redundant recompute is a real saving), but it does not yet answer
the actually interesting question for real tree/multi-candidate speculative
decoding: does the gate correctly distinguish a WORSE candidate from a
BETTER one, or does it sometimes throw away the better one?

This script fixes that by building branches with genuine content diversity
(rank-based top-k token selection: branch 0 takes the model's top-1 token,
branch 1 takes its 2nd choice, etc. -- the same mechanism real tree/Medusa-
style speculative drafting uses to generate distinct candidates), then
measures, using the model's OWN token log-probabilities as the quality
signal (a standard, real, non-fabricated proxy for how much the model
itself "likes" a continuation -- no extra model calls needed, it's already
in the logits used to pick the token), whether gate-pruned subtrees ever
contained a better continuation than what was actually kept.

METHOD
------
For each real prompt, build the FULL tree (never actually prune -- "shadow"
mode) using rank-based top-k branching, exactly as this project's own tree-
fork primitive (§75) already validates as bit-exact. At every node, in
addition to its rank-based token, deep-copy the parent's `RangeStatisticGate`
(as §78 already does) and record what it WOULD have decided
(`should_early_exit`) without acting on it -- so the whole tree is always
built, and pruning decisions are recorded but never enforced.

For every node, track:
  - `logprob`: log-softmax probability of the token this node chose (a real,
    already-computed-for-free quantity, not fabricated).
  - `cum_logprob_mean`: path log-probability from root to here, divided by
    depth+1 (length-normalized, so nodes at different depths are comparable).

Post-hoc, for every node whose gate WOULD fire (`shadow_pruned=True`) whose
parent was NOT itself already shadow-pruned (i.e. a real, reachable pruning
decision -- not a moot descendant of an earlier one), compare the BEST
`cum_logprob_mean` achieved anywhere in its own (fully built, un-truncated)
subtree against the best achieved in its KEPT sibling's subtree at the same
parent. If the pruned subtree's best ever exceeds the kept sibling's best,
that is a real, quantified case of the gate discarding a better candidate.

RUN
    uv run --env-file .env python \
        experiments/runtime/speculative/state_replay/benchmark_gated_tree_quality.py
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
    fork_kv_tail,
    fresh_cache,
    kv_lens,
    restore_kv_chain,
    restore_ssm,
    snapshot_ssm,
)
from benchmark_gated_tree_exploration import load_real_prompts  # noqa: E402


@torch.no_grad()
def extend_one_topk(model, cache, tok_in: torch.Tensor, rank: int):
    """Single real forward pass. Picks the (rank+1)-th most likely token
    (rank=0 -> top-1/greedy, rank=1 -> 2nd choice, ...) -- genuine content
    diversity between branches, the same mechanism real tree-speculative
    drafters (Medusa-style) use. Returns (token, logprob_of_chosen, logits)."""
    out = model(tok_in, past_key_values=cache, use_cache=True)
    logits = out.logits[:, -1, :]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    topk = torch.topk(logits, k=rank + 1, dim=-1, largest=True, sorted=True)
    tok = topk.indices[..., rank:rank + 1]
    lp = float(logprobs.gather(-1, tok).item())
    return tok, lp, logits


def build_shadow_tree(model, tok, prompt: str, width: int, max_depth: int, gate_kwargs: dict) -> list[dict]:
    """Builds the FULL tree unconditionally (never actually prunes), using
    rank-based top-k branching for genuine diversity, recording what the
    gate WOULD have decided at every node without enforcing it."""
    cache, nxt0, ids = fresh_cache(model, tok, prompt)
    root_lens = kv_lens(cache)
    root_ssm = snapshot_ssm(cache)
    root_gate = RangeStatisticGate(**gate_kwargs)
    root_gate.reset_state()
    root = {
        "id": 0, "depth": -1, "rank": -1, "ssm": root_ssm, "lens": root_lens, "last": nxt0,
        "parent": None, "parent_id": None, "gate": root_gate, "shadow_pruned": False,
        "logprob": 0.0, "cum_logprob": 0.0, "cum_logprob_mean": 0.0, "edge_fork": None,
    }

    nodes: list[dict] = [root]
    frontier = [root]
    next_id = 1

    for d in range(max_depth):
        next_frontier = []
        for parent in frontier:
            for r in range(width):
                restore_kv_chain(cache, root_lens, parent)
                restore_ssm(cache, parent["ssm"])
                nxt, lp, logits = extend_one_topk(model, cache, parent["last"], rank=r)

                ssm = snapshot_ssm(cache)
                edge_fork = fork_kv_tail(cache, parent["lens"])
                lens = kv_lens(cache)

                node_gate = copy.deepcopy(parent["gate"])
                should_exit, range_val, tau_eff = node_gate.should_early_exit(logits, step_idx=d)

                cum_logprob = parent["cum_logprob"] + lp
                node = {
                    "id": next_id, "depth": d, "rank": r, "ssm": ssm, "edge_fork": edge_fork, "lens": lens,
                    "last": nxt, "parent": parent, "parent_id": parent["id"], "gate": node_gate,
                    "shadow_pruned": bool(should_exit),
                    "logprob": lp, "cum_logprob": cum_logprob, "cum_logprob_mean": cum_logprob / (d + 2),
                }
                next_id += 1
                nodes.append(node)
                next_frontier.append(node)
        frontier = next_frontier

    return nodes


def analyze_quality(nodes: list[dict]) -> dict:
    """Post-hoc, tree-wide: does gating ever prevent the system from reaching
    the best candidate that exists anywhere in the full (ungated) tree?

    NOTE on why this compares tree-wide rather than direct siblings: two
    rank-based siblings (rank=0 and rank=1 from the SAME parent) are computed
    from the IDENTICAL forward pass -- the gate's own input (that step's
    logits) is the same tensor for both, so RangeStatisticGate necessarily
    makes the SAME keep/prune decision for direct siblings, every time
    (confirmed empirically: reachable_pruned == both_siblings_pruned in every
    pilot run). Divergence between lineages only appears at the NEXT step,
    once each branch has fed a different token back in. So the only place a
    real disagreement between "what gating kept" and "what existed" can show
    up is at the population level, not the sibling-pair level."""
    by_id = {n["id"]: n for n in nodes}
    children_of: dict[int, list[dict]] = {n["id"]: [] for n in nodes}
    for n in nodes:
        if n["parent_id"] is not None:
            children_of[n["parent_id"]].append(n)

    # reachable = parent chain has no earlier shadow-pruned node (a real gated run would create this node)
    reachable: dict[int, bool] = {}
    for n in sorted(nodes, key=lambda x: x["depth"]):
        if n["parent_id"] is None:
            reachable[n["id"]] = True
        else:
            reachable[n["id"]] = reachable[n["parent_id"]] and not by_id[n["parent_id"]]["shadow_pruned"]

    # Exclude the root sentinel (depth=-1, zero tokens generated, cum_logprob_mean
    # trivially 0.0 -- since every real node's log-prob is negative, the root would
    # otherwise always "win" the max() and make this comparison vacuous.
    real_nodes = [n for n in nodes if n["depth"] >= 0]
    global_best_node = max(real_nodes, key=lambda n: n["cum_logprob_mean"])
    global_best = global_best_node["cum_logprob_mean"]

    reachable_real_nodes = [n for n in real_nodes if reachable[n["id"]]]
    gated_best_node = max(reachable_real_nodes, key=lambda n: n["cum_logprob_mean"])
    gated_best = gated_best_node["cum_logprob_mean"]

    n_reachable = sum(1 for n in nodes if reachable[n["id"]])
    n_reachable_pruned = sum(1 for n in nodes if reachable[n["id"]] and n["shadow_pruned"] and n["parent_id"] is not None)
    both_siblings_pruned = sum(
        1 for pid, kids in children_of.items()
        if kids and all(k["shadow_pruned"] for k in kids) and reachable.get(pid, False)
    )

    cost = global_best - gated_best  # >= 0 always; how much log-prob quality gating gave up, if any
    return {
        "global_best_cum_logprob_mean": global_best, "global_best_depth": global_best_node["depth"],
        "gated_best_cum_logprob_mean": gated_best, "gated_best_depth": gated_best_node["depth"],
        "gating_cost": cost, "gating_lost_the_global_best": bool(cost > 1e-9),
        "diag_num_reachable": n_reachable,
        "diag_num_reachable_pruned": n_reachable_pruned,
        "diag_num_both_siblings_pruned": both_siblings_pruned,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--thresholds", default="2.5,3.5,4.5")
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/benchmarks/gated_tree_quality.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    thresholds = [float(t) for t in args.thresholds.split(",")]
    prompts = load_real_prompts(args.num_prompts, seed=args.seed)

    print("=" * 100)
    print("  GATED TREE QUALITY CHECK -- does the gate ever prune a genuinely BETTER candidate?")
    print("=" * 100)
    print(f"  device={dev}  width={args.width}  max_depth={args.max_depth}  thresholds={thresholds}")
    print(f"  {len(prompts)} real prompts, rank-based top-k branching (genuine content diversity)\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()

    t0 = time.time()
    all_results = []
    for thr in thresholds:
        gate_kwargs = {"threshold": thr}
        for pi, prompt in enumerate(prompts):
            nodes = build_shadow_tree(model, tok, prompt, args.width, args.max_depth, gate_kwargs)
            n_shadow_pruned = sum(1 for n in nodes if n["shadow_pruned"])
            q = analyze_quality(nodes)
            row = {"threshold": thr, "prompt_idx": pi, "num_nodes": len(nodes),
                   "num_shadow_pruned": n_shadow_pruned, **q}
            all_results.append(row)
            lost = "LOST BEST" if q["gating_lost_the_global_best"] else "kept best"
            print(f"  thr={thr:4.1f} prompt={pi:2d}  nodes={len(nodes):3d} shadow_pruned={n_shadow_pruned:3d} "
                  f"reachable={q['diag_num_reachable']:3d}  "
                  f"global_best={q['global_best_cum_logprob_mean']:+.4f}(d{q['global_best_depth']}) "
                  f"gated_best={q['gated_best_cum_logprob_mean']:+.4f}(d{q['gated_best_depth']})  "
                  f"cost={q['gating_cost']:.5f}  {lost}", flush=True)

    print("\n" + "=" * 100)
    print("  AGGREGATE PER THRESHOLD")
    print("=" * 100)
    agg = {}
    for thr in thresholds:
        rows = [r for r in all_results if r["threshold"] == thr]
        n_lost = sum(1 for r in rows if r["gating_lost_the_global_best"])
        costs = [r["gating_cost"] for r in rows]
        costs_when_lost = [r["gating_cost"] for r in rows if r["gating_lost_the_global_best"]]
        agg[thr] = {
            "num_runs": len(rows), "num_lost_the_global_best": n_lost,
            "lost_rate": n_lost / len(rows) if rows else None,
            "mean_cost_all_runs": (sum(costs) / len(costs)) if costs else None,
            "mean_cost_when_lost": (sum(costs_when_lost) / len(costs_when_lost)) if costs_when_lost else None,
        }
        rate_s = f"{n_lost}/{len(rows)}" if rows else "n/a"
        mc = agg[thr]["mean_cost_when_lost"]
        print(f"  threshold={thr:4.1f}  lost the global best in {rate_s} runs"
              + (f"  (mean log-prob cost when lost: {mc:.5f})" if mc is not None else ""))

    elapsed = time.time() - t0
    print(f"\n{elapsed:.0f}s total.")

    report = {
        "device": dev, "width": args.width, "max_depth": args.max_depth, "thresholds": thresholds,
        "num_prompts": len(prompts), "per_run": all_results,
        "aggregate_by_threshold": {str(k): v for k, v in agg.items()},
        "elapsed_seconds": elapsed,
    }
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {out_p}")


if __name__ == "__main__":
    main()
