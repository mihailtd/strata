"""Does EAGLE-style speculation with Qwen3.5's shipped MTP head beat plain decode?

STATUS: SUPERSEDED for headline numbers -- cite
scripts/benchmark_mtp_indomain_speculation_matrix.py instead (9 cells, 3
interleaved repeats, correctness gate). This file is kept because it is the
only place the snapshot/restore mechanism is exercised end to end.

WHAT MAKES THIS WORK ON A HYBRID MODEL
--------------------------------------
`transformers` refuses assisted generation here ("assisted generation is not
supported with stateful models, such as Qwen3_5ForCausalLM"): 24 of 32 layers
are GatedDeltaNet, whose recurrent state after K tokens cannot be recovered
from the state after M<K. A KV cache truncates; a recurrent state does not.

The state is fixed-size though, so we snapshot it before verification and
restore on partial acceptance. That converts an impossible rollback into a
cheap copy. Verified: the 52.5 MB snapshot round-trips exactly -- replay after
restore reproduces the identical token.

VERIFICATION COST -- CURRENT, NOT THE ORIGINAL ESTIMATE
-------------------------------------------------------
An earlier version of this docstring stated `fla`/`causal_conv1d` were "not
buildable on this AMD rig" and built a cost model on it:

    SUPERSEDED:  K=1 33.56ms (1.00x) | K=2 95.44ms (2.84x) | K=4 94.33ms (2.81x)

That is wrong twice over. `fla` and `causal_conv1d` are two INDEPENDENT
dependencies with separate fallbacks -- conflating them is what kept this
blocked. `fla` (flash-linear-attention 0.5.2, triton-rocm 3.7.1) binds and runs
on gfx1100; `causal_conv1d` is still absent and its torch fallback costs only
3.8% of wall time. Measured with `fla` active (results/fla_verification_profile.json,
`fla_active: true`):

    CURRENT:     K=1 27.87ms (1.00x) | K=2 38.06ms (1.37x) | K=4 33.27ms (1.19x)

So verification costs ~1.19 single-token steps, not ~2.8, putting break-even at
tau ~ 1.39 against a measured tau of ~2.3-2.4.

NOTE: `fla`'s device probe is @cache'd at import. Touch CUDA BEFORE importing
transformers or the process latches to a fallback for its whole lifetime.

ON EXACTNESS -- THE OLD "DO NOT CITE" RATIONALE IS RETRACTED
------------------------------------------------------------
This file previously carried "STATUS: NOT CORRECT YET -- DO NOT CITE", because
the loop diverged from plain greedy on 2-3 of 4 prompts. Two things changed:

  * The suspected bug was real and is FIXED: the draft head's KV cache was
    discarded and rebuilt each round, which is why acceptance fell from 66.7%
    isolated to 24-48% in the loop. Accumulating hidden states restored it to
    44-69%.
  * The remaining divergence is NOT a loop defect. Measured at n=60 (20 prompts
    x 3 domains): speculative-vs-greedy 80.0%, against a *non-speculative*
    control (plain greedy vs teacher-forced) of 83.3%, with a determinism
    control at 100.0%. Speculation adds ~3pp of divergence over paths that
    involve no speculation at all.

The cause is bf16 kernel path-dependence: the chunked multi-token kernel and
the single-token recurrent kernel genuinely disagree (divergence at tokens 2-4
of 32 on 3/3 prompts with zero speculation involved). Verification is
inherently multi-token, so it cannot use the single-token kernel.

**Token-exactness against plain greedy is therefore UNREACHABLE on this stack**
-- for this loop and for plain chunked decode alike. Do not treat a mismatch
here as proof of a bug, and do not report these speedups as exact-output
speedups. That is a real limitation, not a clean bill of health.

    uv run --env-file .env scripts/benchmark_mtp_speculative.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.mtp_draft import (  # noqa: E402
    Qwen35MTPDraftHead,
    restore_state,
    snapshot_state,
    state_nbytes,
)
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

# Domain experts, folded into the backbone to test whether adaptation erodes the
# speculative win. Acceptance measurements say astral is the one that hurts
# drafting (-0.86 accepted tokens vs un-adapted), so it is the real stress case.
ADAPTERS = {
    "financial": "results/adapters/m2_financial_r8a128",
    "astral": "results/adapters/m2_astral_r8a128",
    "postgres": "results/adapters/m2_postgresql_r8a128",
}

PROMPTS = [
    "### Question:\nExplain loss aversion in one sentence.\n\n### Answer:\n",
    "### Question:\nHow do I add a dependency with uv?\n\n### Answer:\n",
    "### Question:\nWhat is pgvector used for?\n\n### Answer:\n",
    "### Question:\nWhat is a money script?\n\n### Answer:\n",
]


@torch.no_grad()
def plain_greedy(model, tok, ids, n_new):
    # Let the model build its own cache. A bare DynamicCache() fails here: the
    # hybrid model needs LinearAttentionLayer entries for its 24 GatedDeltaNet
    # layers, and constructing the cache manually yields an empty layer list
    # (IndexError inside update_conv_state on the first decode step).
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    for _ in range(n_new - 1):
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks.append(nxt.item())
    return toks


@torch.no_grad()
def speculative(model, tok, head, ids, n_new, k):
    """Speculative decode with the MTP head, keeping the head's context intact.

    The earlier version rebuilt the head's KV cache as an EMPTY DynamicCache
    every round while still drafting at absolute position `pos-1`. The head
    therefore attended to one token of context instead of the whole sequence,
    and acceptance fell from 66.7% (isolated) to 24-48%. Here the committed
    hidden states are accumulated and the head is re-primed over them each
    round, which is what it was trained to see.
    """
    out = model(ids, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    hids = [out.hidden_states[-1]]
    seq = ids
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    pos = ids.shape[1]
    stats = {"steps": 0, "drafted": 0, "accepted": 0, "full_accepts": 0}

    while len(toks) < n_new:
        H = torch.cat(hids, dim=1)
        dcache = head.prefill(H, seq)
        draft = head.draft(H[:, -1:, :], nxt, k=k, start_pos=pos - 1, cache=dcache)

        snap = snapshot_state(cache)
        chunk = torch.cat([nxt, draft], dim=-1)
        out = model(chunk, past_key_values=cache, use_cache=True, output_hidden_states=True)
        target = torch.argmax(out.logits[0], -1)

        n_acc = 0
        for i in range(k):
            if draft[0, i].item() == target[i].item():
                n_acc += 1
            else:
                break
        stats["steps"] += 1
        stats["drafted"] += k
        stats["accepted"] += n_acc
        stats["full_accepts"] += int(n_acc == k)

        committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)  # positions pos .. pos+n_acc
        if n_acc < k:
            # state advanced by k+1, only n_acc+1 tokens are real -> restore and redo
            restore_state(cache, snap)
            out = model(committed, past_key_values=cache, use_cache=True, output_hidden_states=True)
            new_h = out.hidden_states[-1]
        else:
            new_h = out.hidden_states[-1][:, : n_acc + 1, :]

        bonus = target[n_acc].item()
        for t in draft[0, :n_acc].tolist() + [bonus]:
            if len(toks) < n_new:
                toks.append(t)

        hids.append(new_h)
        seq = torch.cat([seq, committed], dim=-1)
        pos += n_acc + 1
        nxt = torch.tensor([[bonus]], device=ids.device)

    return toks[:n_new], stats


@torch.no_grad()
def chunked_reference(model, ids, toks):
    """What the CHUNKED kernel says the greedy continuation is.

    Speculative verification runs the multi-token (chunked) kernel, while plain
    decode runs the single-token recurrent kernel. Measured: those two paths
    disagree on 1 of 3 prompts (divergence at token 9/32). So token-exactness
    against plain decode is NOT achievable on this model, and the honest
    reference for a speculative decoder is the path its verifier actually uses.
    Both references are reported.
    """
    full = torch.cat([ids, torch.tensor([toks], device=ids.device)], dim=-1)
    T = ids.shape[1]
    logits = model(full, use_cache=False).logits
    return torch.argmax(logits[0, T - 1 : -1, :], -1).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--adapter", default="none", choices=["none", *ADAPTERS])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    head = Qwen35MTPDraftHead(model, args.model_name)

    engine = None
    if args.adapter != "none":
        expert = FoldableExpert.from_dir(REPO_ROOT / ADAPTERS[args.adapter], args.adapter)
        engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
        engine.activate(expert)
        print(f"folded domain expert into backbone: {args.adapter}\n")
    out_path = args.out or f"results/mtp_speculative_{args.adapter}.json"

    ids0 = tok(PROMPTS[0], return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        c = model(ids0, use_cache=True).past_key_values
    print(f"recurrent+kv state snapshot size: {state_nbytes(c) / 1e6:.1f} MB\n")

    # baseline
    base_rates = []
    refs = {}
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(model.device)
        plain_greedy(model, tok, ids, 4)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        refs[p] = plain_greedy(model, tok, ids, args.tokens)
        torch.cuda.synchronize()
        base_rates.append(args.tokens / (time.perf_counter() - t0))
    base = sum(base_rates) / len(base_rates)
    print(f"baseline plain greedy: {base:.2f} tok/s\n")

    results = {"adapter": args.adapter, "baseline_tok_s": base, "arms": {}}
    print(
        f"{'K':>3s} {'tok/s':>8s} {'speedup':>8s} {'accept%':>8s} {'acc/step':>9s} "
        f"{'vs 1tok':>8s} {'vs chunk':>9s}"
    )
    for k in args.k:
        rates, accs, exact, exact_chunk = [], [], 0, 0
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(model.device)
            speculative(model, tok, head, ids, 4, k)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            got, st = speculative(model, tok, head, ids, args.tokens, k)
            torch.cuda.synchronize()
            rates.append(args.tokens / (time.perf_counter() - t0))
            accs.append(st)
            exact += int(got == refs[p])
            chunk_ref = chunked_reference(model, ids, got)
            exact_chunk += int(got == chunk_ref)
        r = sum(rates) / len(rates)
        tot_d = sum(a["drafted"] for a in accs)
        tot_a = sum(a["accepted"] for a in accs)
        steps = sum(a["steps"] for a in accs)
        flag = "OK" if exact_chunk == len(PROMPTS) else "*** DIVERGED ***"
        print(
            f"{k:3d} {r:7.2f} {r / base:7.2f}x {100 * tot_a / tot_d:7.1f}% "
            f"{tot_a / steps:8.2f} {exact}/{len(PROMPTS)} {exact_chunk}/{len(PROMPTS)} {flag}"
        )
        results["arms"][k] = {
            "tok_s": r,
            "speedup": r / base,
            "accept_pct": 100 * tot_a / tot_d,
            "accepted_per_step": tot_a / steps,
            "exact_vs_single_token": exact,
            "exact_vs_chunked": exact_chunk,
            "n_prompts": len(PROMPTS),
        }

    out = REPO_ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
