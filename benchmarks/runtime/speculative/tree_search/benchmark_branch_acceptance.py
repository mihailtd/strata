"""Does offering W draft branches instead of 1 lift acceptance past break-even?

THE ONLY QUESTION THAT MATTERS FOR TREE SPECULATION HERE
--------------------------------------------------------
Linear speculation on this rig sits at **tau 1.67-2.33 against a ~2.8 break-even**
(`mtp_draft.py`), i.e. it does not currently pay. Every proposed elaboration --
trees, ring-buffer rollback, pointer arithmetic -- is worthless unless acceptance
clears that bar. So measure acceptance first and argue about mechanism second.

WHAT IS AND IS NOT MEASURED HERE
--------------------------------
MEASURED: effective tau when the draft head offers W independent K-token
continuations and the verifier keeps the best one. This is pure acceptance
statistics and does not depend on how the branches are executed.

NOT MEASURED: batched execution. Branches are verified SEQUENTIALLY here (snapshot
-> verify -> restore, per branch), which is W times slower to run but yields the
IDENTICAL acceptance numbers. Real execution would batch them, and that cost is
already on record in the batch-scaling table:

    batch 1   28.68 ms/step        batch 2   28.93 ms (1.009x)
    batch 4   32.31 ms (1.127x)    batch 8   37.92 ms (1.322x)

So W=4 branches cost ~1.13x the latency of one. Combining measured tau with that
table gives the net speedup without building the batched engine -- and if tau does
not move, the engine never needs building.

WHY THE STATED TREE DESIGN IS NOT WHAT GETS BUILT (docs/DECISIONS.md §20-21)
    * "SSMs avoid the KV explosion, so branching is cheap" is backwards at this
      scale: SSM state is 51.90 MB per checkpoint vs 32.77 KB per token of KV.
      Crossover ~1,584 tokens; trees span 8-32.
    * Rollback-based sequential branching wins nothing -- W branches = W forwards.
      Real tree speculation verifies the whole tree in ONE forward via a tree
      ATTENTION MASK.
    * That mask cannot work on the 24 recurrent layers: a sequential scan mixes
      sibling branches into the state and no mask fixes it. Branches must live in
      the BATCH dimension, which is why the batch table above is the relevant cost
      model.

    uv run --env-file .env python \
        benchmarks/runtime/speculative/tree_search/benchmark_branch_acceptance.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.mtp_draft import (  # noqa: E402
    Qwen35MTPDraftHead,
    restore_state,
    snapshot_state,
)
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

PROMPTS_FILE = "data/astral/evaluation_data.jsonl"
ADAPTER = "results/adapters/m2_astral_r8a128"
# measured, benchmarks/runtime/performance/batch_scaling/
BATCH_STEP_MS = {1: 28.68, 2: 28.93, 4: 32.31, 8: 37.92}
BREAK_EVEN_TAU = 2.8  # mtp_draft.py, this rig


@torch.no_grad()
def draft_branches(head, H, seq, nxt, k: int, pos: int, w: int) -> list[torch.Tensor]:
    """W independent K-token continuations, diverging at the FIRST draft position.

    The shipped `head.draft()` is greedy-only, so this reimplements it against the
    head's internals to take the top-W first tokens and greedily extend each. The
    base model is never called, so no recurrent state advances -- same property the
    single-path drafter relies on.
    """
    from transformers.cache_utils import DynamicCache

    branches = []
    cache0 = head.prefill(H, seq)
    fused = head._fuse(H[:, -1:, :], nxt)
    positions = torch.arange(pos, pos + 1, device=fused.device)
    h0 = head._run_layer(fused, positions, cache0)
    logits0 = head._lm_head(h0)[:, -1, :]
    top = torch.topk(logits0, w, dim=-1).indices[0]  # (w,)

    for b in range(w):
        tok = top[b].view(1, 1)
        toks = [tok]
        h = h0
        cache = DynamicCache()
        # re-seed this branch's own draft cache from the shared prefix
        _ = head.prefill(H, seq)
        for i in range(1, k):
            fused_i = head._fuse(h, tok)
            pos_i = torch.arange(pos + i, pos + i + 1, device=fused_i.device)
            h = head._run_layer(fused_i, pos_i, cache)
            tok = torch.argmax(head._lm_head(h)[:, -1, :], dim=-1, keepdim=True)
            toks.append(tok)
        branches.append(torch.cat(toks, dim=-1))
    return branches


@torch.no_grad()
def run(model, tok, head, prompt: str, n_new: int, k: int, w: int) -> dict:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    eos = tok.eos_token_id
    out = model(ids, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    hids = [out.hidden_states[-1]]
    seq = ids
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    pos = ids.shape[1]
    steps = accepted = 0
    branch_wins = [0] * w
    done = toks[0] == eos

    while len(toks) < n_new and not done:
        H = torch.cat(hids, dim=1)
        branches = draft_branches(head, H, seq, nxt, k, pos - 1, w)

        # Verify each branch from the SAME state. Sequential here on purpose:
        # acceptance is independent of execution layout, and this needs no
        # batched-cache machinery to measure.
        snap = snapshot_state(cache)
        best_n, best_i, best_out = -1, 0, None
        for i, draft in enumerate(branches):
            restore_state(cache, snap)
            chunk = torch.cat([nxt, draft], dim=-1)
            o = model(chunk, past_key_values=cache, use_cache=True, output_hidden_states=True)
            target = torch.argmax(o.logits[0], -1)
            n_acc = 0
            for j in range(k):
                if draft[0, j].item() == target[j].item():
                    n_acc += 1
                else:
                    break
            if n_acc > best_n:
                best_n, best_i, best_out = n_acc, i, (draft, target, o)
        branch_wins[best_i] += 1

        draft, target, o = best_out
        # replay the winner so the cache ends in the winner's state
        restore_state(cache, snap)
        chunk = torch.cat([nxt, draft], dim=-1)
        o = model(chunk, past_key_values=cache, use_cache=True, output_hidden_states=True)
        target = torch.argmax(o.logits[0], -1)

        steps += 1
        accepted += best_n
        committed = torch.cat([nxt, draft[:, :best_n]], dim=-1)
        if best_n < k:
            restore_state(cache, snap)
            o = model(committed, past_key_values=cache, use_cache=True, output_hidden_states=True)
            new_h = o.hidden_states[-1]
        else:
            new_h = o.hidden_states[-1][:, : best_n + 1, :]

        bonus = target[best_n].item()
        for t in draft[0, :best_n].tolist() + [bonus]:
            if len(toks) >= n_new:
                break
            toks.append(t)
            if t == eos:
                done = True
                break

        hids.append(new_h)
        seq = torch.cat([seq, committed], dim=-1)
        pos += best_n + 1
        nxt = torch.tensor([[bonus]], device=ids.device)

    return {"steps": steps, "accepted": accepted, "tokens": len(toks),
            "branch_wins": branch_wins}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--widths", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/branch_acceptance.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  DOES BRANCHING LIFT ACCEPTANCE PAST BREAK-EVEN?")
    print("=" * 100)
    print(f"  device={dev}  K={args.k}  widths={args.widths}  "
          f"break-even tau={BREAK_EVEN_TAU}\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()
    head = Qwen35MTPDraftHead(model, args.model_name)
    expert = FoldableExpert.from_dir(REPO_ROOT / ADAPTER, "astral")
    WeightFoldingEngine(model, [expert], keep_pristine=True).activate(expert)

    rows = [json.loads(x) for x in (REPO_ROOT / PROMPTS_FILE).read_text().splitlines() if x.strip()]
    prompts = [f"### Question:\n{r['prompt']}\n\n### Answer:\n" for r in rows[: args.n_prompts]]

    run(model, tok, head, prompts[0], 8, args.k, 1)  # warmup

    report: dict = {"device": dev, "k": args.k, "break_even_tau": BREAK_EVEN_TAU,
                    "batch_step_ms": BATCH_STEP_MS, "widths": {}}
    print(f"  {'W':>3}{'steps':>8}{'accepted':>10}{'tau_eff':>10}{'vs W=1':>10}"
          f"{'clears 2.8?':>13}{'branch wins':>26}")
    print("  " + "-" * 96)

    base_tau = None
    for w in args.widths:
        tot = {"steps": 0, "accepted": 0}
        wins = [0] * w
        t0 = time.perf_counter()
        for p in prompts:
            r = run(model, tok, head, p, args.tokens, args.k, w)
            tot["steps"] += r["steps"]
            tot["accepted"] += r["accepted"]
            wins = [a + b for a, b in zip(wins, r["branch_wins"], strict=True)]
        secs = time.perf_counter() - t0
        tau = tot["accepted"] / max(1, tot["steps"])
        if base_tau is None:
            base_tau = tau
        wpct = ", ".join(f"{100 * x / max(1, sum(wins)):.0f}%" for x in wins)
        print(f"  {w:>3}{tot['steps']:>8}{tot['accepted']:>10}{tau:>10.3f}"
              f"{tau - base_tau:>+10.3f}{'YES' if tau >= BREAK_EVEN_TAU else 'no':>13}"
              f"{wpct:>26}")
        report["widths"][w] = {"steps": tot["steps"], "accepted": tot["accepted"],
                               "tau_eff": tau, "branch_wins": wins,
                               "measure_seconds": secs}

    # --- net speedup, combining measured tau with the measured batch cost ------
    print("\n" + "=" * 100)
    print("  NET SPEEDUP  (measured tau x measured batch cost; batched engine NOT built)")
    print("=" * 100)
    print("  Per step a speculative round emits tau+1 tokens and costs one verify")
    print("  forward at batch W, plus drafting. Autoregressive emits 1 token per step.")
    print(f"\n  {'W':>3}{'tau_eff':>10}{'tokens/step':>13}{'batch cost':>12}"
          f"{'tokens/ms':>12}{'vs autoregressive':>20}")
    print("  " + "-" * 96)
    auto_ms = BATCH_STEP_MS[1]
    for w in args.widths:
        tau = report["widths"][w]["tau_eff"]
        step_ms = BATCH_STEP_MS.get(w, BATCH_STEP_MS[max(BATCH_STEP_MS)])
        # verify + the commit re-forward that fires on a partial accept
        cost = step_ms + auto_ms * (1.0 if tau < args.k else 0.0)
        tps = (tau + 1) / cost
        print(f"  {w:>3}{tau:>10.3f}{tau + 1:>13.2f}{step_ms:>11.2f}ms"
              f"{tps:>12.4f}{tps / (1 / auto_ms):>19.3f}x")
        report["widths"][w]["net_vs_autoregressive"] = tps / (1 / auto_ms)

    best = max(args.widths, key=lambda w: report["widths"][w]["net_vs_autoregressive"])
    print(f"\n  Best width: W={best} at "
          f"{report['widths'][best]['net_vs_autoregressive']:.3f}x vs autoregressive.")
    if report["widths"][best]["net_vs_autoregressive"] <= 1.0:
        print("  => Tree speculation does NOT pay here. Acceptance never clears the bar,")
        print("     so no amount of branching machinery makes speculation profitable.")
    else:
        print("  => Worth building the batched verifier.")

    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
