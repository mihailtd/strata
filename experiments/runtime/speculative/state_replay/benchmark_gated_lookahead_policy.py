r"""The tree-search policy: closes the loop §75 opened and §78 validated in pieces.

WHY THIS EXISTS
---------------
§75 built a real, bit-exact, cheap KV/SSM fork-and-restore primitive for
switching between branches -- but on its own it had no consumer: cheap
switching only pays off if something actually decides WHEN to branch and
WHICH branch to keep. §78 validated a gate that decides WHICH branches to
prune, and separately measured that pruning has a real, tunable speed/
quality trade-off. Neither result is a decoding policy by itself -- both
were measured on an isolated, always-built tree at one fixed position.

This is the actual policy: **greedy decoding, augmented with a triggered,
bounded lookahead search exactly when the model's own confidence signal
says the greedy choice looks shaky.** The same real gate
(`RangeStatisticGate`) does two jobs it was never asked to do together
before:
  1. TRIGGER: evaluated on the plain greedy candidate's logits at every
     step. If confident, commit the greedy token immediately -- one
     forward pass, exactly as expensive as plain greedy decoding, zero
     overhead.
  2. PRUNE: only when triggered, spend a small, gated, width-ary lookahead
     (the exact mechanism §78 validated) to find a better immediate next
     token than pure greedy, then commit ONLY that one token and continue
     normal decoding -- re-triggering fresh at the next step.

Switching to the winning branch's state after a triggered search is the
fork/restore primitive from §75, finally with a real caller. This is the
"what do we do with the tree" design that was the missing piece after
every prior measurement in this thread.

WHAT THIS MEASURES
------------------
Real, full multi-token generation (not a single isolated branch point) on
real prompts, comparing:
  1. PLAIN GREEDY: the baseline every one of this project's own reference
     scripts already uses.
  2. GATED LOOKAHEAD: the policy above.
Reports, for the same number of committed output tokens on both arms:
  - total forward passes spent (real cost, trigger rate included)
  - the generated sequence's own mean log-probability (real quality signal,
    same metric §78 used, not fabricated)
  - whether the two arms' output text actually differs at all
This is the first measurement in this thread of the COMPLETE system, not
an isolated primitive.

RUN
    uv run --env-file .env python \
        experiments/runtime/speculative/state_replay/benchmark_gated_lookahead_policy.py
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
from benchmark_gated_tree_quality import extend_one_topk  # noqa: E402


def run_plain_greedy(model, tok, prompt: str, n_tokens: int) -> dict:
    cache, nxt0, ids = fresh_cache(model, tok, prompt)
    last = nxt0
    tokens, logprobs = [], []
    for _ in range(n_tokens):
        nxt, lp, logits = extend_one_topk(model, cache, last, rank=0)
        tokens.append(nxt.item())
        logprobs.append(lp)
        last = nxt
    return {"tokens": tokens, "logprobs": logprobs, "forward_passes": n_tokens,
            "mean_logprob": sum(logprobs) / len(logprobs)}


def build_lookahead_tree(model, cache, root_lens, root_ssm, root_last, width, depth,
                          gate_state, seed_rank0=None):
    """Real width-ary lookahead from the CURRENT live generation state (not a
    fresh prompt -- this is what makes it a mid-generation policy rather than
    an isolated experiment). `seed_rank0`, if given, is the already-computed
    (token, logprob, logits) for rank=0 at depth 0, so the trigger decision's
    own forward pass is reused instead of being spent twice."""
    root = {"id": 0, "depth": -1, "ssm": root_ssm, "lens": root_lens, "last": root_last,
            "parent": None, "edge_fork": None, "cum_logprob": 0.0, "cum_logprob_mean": 0.0,
            "root_child": None}
    nodes = [root]
    frontier = [root]
    next_id = 1
    passes = 0

    for d in range(depth):
        next_frontier = []
        for parent in frontier:
            for r in range(width):
                if d == 0 and r == 0 and seed_rank0 is not None:
                    nxt, lp, logits = seed_rank0
                else:
                    restore_kv_chain(cache, root_lens, parent)
                    restore_ssm(cache, parent["ssm"])
                    nxt, lp, logits = extend_one_topk(model, cache, parent["last"], rank=r)
                    passes += 1

                ssm = snapshot_ssm(cache)
                edge_fork = fork_kv_tail(cache, parent["lens"])
                lens = kv_lens(cache)
                g = copy.deepcopy(gate_state)
                should_exit, _, _ = g.should_early_exit(logits, step_idx=d)

                cum = parent["cum_logprob"] + lp
                node = {
                    "id": next_id, "depth": d, "rank": r, "ssm": ssm, "edge_fork": edge_fork,
                    "lens": lens, "last": nxt, "parent": parent, "own_logprob": lp,
                    "cum_logprob": cum, "cum_logprob_mean": cum / (d + 2), "pruned": bool(should_exit),
                    "root_child": parent if d == 0 else parent["root_child"],
                }
                next_id += 1
                nodes.append(node)
                if not should_exit:
                    next_frontier.append(node)
        frontier = next_frontier
        if not frontier:
            break

    return nodes, passes


def pick_winning_root_child(nodes: list[dict]) -> dict:
    """Best root-child by the best cum_logprob_mean reachable in ITS OWN
    (gated, real) subtree -- the actual outcome this policy would deliver
    for each candidate first move, not a hypothetical unpruned best."""
    root_children = [n for n in nodes if n["depth"] == 0]
    best_by_child: dict[int, float] = {}
    for n in nodes:
        if n["depth"] < 0 or n["root_child"] is None:
            continue
        rc_id = n["root_child"]["id"]
        best_by_child[rc_id] = max(best_by_child.get(rc_id, n["cum_logprob_mean"]), n["cum_logprob_mean"])
    return max(root_children, key=lambda n: best_by_child.get(n["id"], n["cum_logprob_mean"]))


def run_gated_lookahead(model, tok, prompt: str, n_tokens: int, width: int, search_depth: int,
                         gate_kwargs: dict) -> dict:
    cache, nxt0, ids = fresh_cache(model, tok, prompt)
    primary_gate = RangeStatisticGate(**gate_kwargs)
    primary_gate.reset_state()

    last = nxt0
    tokens, logprobs = [], []
    total_passes = 0
    n_triggers = 0
    range_vals = []

    for step in range(n_tokens):
        cur_lens = kv_lens(cache)
        cur_ssm = snapshot_ssm(cache)

        nxt0_tok, lp0, logits0 = extend_one_topk(model, cache, last, rank=0)
        total_passes += 1
        # step_idx=0 (fixed, not the outer generation step): the gate's Weibull
        # hazard term escalates tau_eff with step_idx by design, for a BOUNDED
        # speculative-chain depth (this is what build_lookahead_tree correctly
        # passes d for, below). The primary trigger decision has no such
        # concept -- it's evaluating one greedy token's own confidence at the
        # current position, not judging depth into an already-triggered
        # search. Feeding the ever-growing outer step count here made tau_eff
        # grow unboundedly over a 20-40 token generation, guaranteeing
        # near-total triggering by the back half of any response (found via a
        # real pilot run: 15-18 of 20 steps triggered before this fix). The
        # gate's Bollinger/volatility EMA still evolves naturally across
        # steps via its own rolling state -- only the hazard escalation is
        # pinned here.
        should_search, range_val, tau_eff = primary_gate.should_early_exit(logits0, step_idx=0)
        range_vals.append(range_val)

        if not should_search:
            tokens.append(nxt0_tok.item())
            logprobs.append(lp0)
            last = nxt0_tok
            continue

        # TRIGGERED: restore to the pre-step state (the rank=0 forward pass above
        # already advanced `cache` -- reset it before building the real tree, whose
        # own restore_kv_chain calls assume `root_lens`/`root_ssm` are the state
        # BEFORE any of this step's candidates were computed).
        n_triggers += 1
        restore_kv_chain(cache, cur_lens, {"edge_fork": None, "parent": None})
        restore_ssm(cache, cur_ssm)

        nodes, extra_passes = build_lookahead_tree(
            model, cache, cur_lens, cur_ssm, last, width, search_depth,
            gate_state=primary_gate, seed_rank0=(nxt0_tok, lp0, logits0),
        )
        total_passes += extra_passes

        winner = pick_winning_root_child(nodes)
        restore_kv_chain(cache, cur_lens, winner)
        restore_ssm(cache, winner["ssm"])

        tokens.append(winner["last"].item() if winner["rank"] != 0 else nxt0_tok.item())
        # NOTE: winner["last"] IS the committed token for this step regardless of
        # rank -- restated via nxt0_tok only when rank==0 to avoid a redundant
        # tensor->python round trip; both are the same value in that case.
        logprobs.append(winner["own_logprob"])
        last = winner["last"]

    return {"tokens": tokens, "logprobs": logprobs, "forward_passes": total_passes,
            "mean_logprob": sum(logprobs) / len(logprobs), "n_triggers": n_triggers,
            "trigger_rate": n_triggers / n_tokens, "range_vals": range_vals}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--search-depth", type=int, default=3)
    ap.add_argument("--n-tokens", type=int, default=40)
    ap.add_argument("--gate-threshold", type=float, default=3.5)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/benchmarks/gated_lookahead_policy.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    prompts = load_real_prompts(args.num_prompts, seed=args.seed)
    gate_kwargs = {"threshold": args.gate_threshold}

    print("=" * 100)
    print("  GATED LOOKAHEAD POLICY -- greedy decoding + triggered, gated, bounded lookahead search")
    print("=" * 100)
    print(f"  device={dev}  width={args.width}  search_depth={args.search_depth}  "
          f"gate_threshold={args.gate_threshold}  n_tokens={args.n_tokens}")
    print(f"  {len(prompts)} real prompts\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()

    t0 = time.time()
    rows = []
    for pi, prompt in enumerate(prompts):
        greedy = run_plain_greedy(model, tok, prompt, args.n_tokens)
        lookahead = run_gated_lookahead(model, tok, prompt, args.n_tokens, args.width,
                                         args.search_depth, gate_kwargs)
        same_text = greedy["tokens"] == lookahead["tokens"]
        pass_overhead_pct = 100.0 * (lookahead["forward_passes"] / greedy["forward_passes"] - 1.0)
        quality_delta = lookahead["mean_logprob"] - greedy["mean_logprob"]
        row = {
            "prompt_idx": pi, "greedy": greedy, "lookahead": lookahead,
            "same_output": same_text, "pass_overhead_pct": pass_overhead_pct,
            "quality_delta_mean_logprob": quality_delta,
        }
        rows.append(row)
        print(f"  prompt={pi:2d}  greedy_passes={greedy['forward_passes']:3d} "
              f"lookahead_passes={lookahead['forward_passes']:3d} (+{pass_overhead_pct:5.1f}%)  "
              f"triggers={lookahead['n_triggers']:2d}/{args.n_tokens} "
              f"greedy_mlp={greedy['mean_logprob']:+.4f} lookahead_mlp={lookahead['mean_logprob']:+.4f} "
              f"(delta={quality_delta:+.4f})  same_output={same_text}", flush=True)

    n_diff = sum(1 for r in rows if not r["same_output"])
    n_improved = sum(1 for r in rows if r["quality_delta_mean_logprob"] > 1e-9)
    n_worse = sum(1 for r in rows if r["quality_delta_mean_logprob"] < -1e-9)
    mean_overhead = sum(r["pass_overhead_pct"] for r in rows) / len(rows)
    mean_quality_delta = sum(r["quality_delta_mean_logprob"] for r in rows) / len(rows)

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"    {len(rows)} real prompts, {args.n_tokens} tokens each")
    print(f"    output changed vs. plain greedy : {n_diff}/{len(rows)} prompts")
    print(f"    mean forward-pass overhead      : {mean_overhead:+.1f}%")
    print(f"    quality improved / worse / same : {n_improved} / {n_worse} / {len(rows) - n_improved - n_worse}")
    print(f"    mean quality delta (log-prob)   : {mean_quality_delta:+.5f}")

    elapsed = time.time() - t0
    report = {
        "device": dev, "width": args.width, "search_depth": args.search_depth,
        "gate_threshold": args.gate_threshold, "n_tokens": args.n_tokens, "num_prompts": len(prompts),
        "rows": rows,
        "summary": {
            "n_output_changed": n_diff, "mean_pass_overhead_pct": mean_overhead,
            "n_quality_improved": n_improved, "n_quality_worse": n_worse,
            "mean_quality_delta_mean_logprob": mean_quality_delta,
        },
        "elapsed_seconds": elapsed,
    }
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{elapsed:.0f}s total. Wrote {out_p}")


if __name__ == "__main__":
    main()
