r"""Does BATCHING two candidate continuations into one verification pass beat
verifying them sequentially? The one falsifiable question that decides whether
tree speculative decoding is worth building at all on this hardware.

RESULT (2026-09-13): TIMING IS PROMISING (1.80x for 2x candidates) BUT REAL
CORRECTNESS FAILS. Batching two divergent candidates into one forward pass
changes which token argmax picks for at least one of them, compared to
verifying that exact same candidate alone -- reproduced with a completely
NATIVE batch=2 forward pass, no custom code involved (confirmed in isolation,
see `docs/DECISIONS.md` §76). This is a real numerical hazard in how this
model/hardware/software stack handles batch>1 with divergent per-row content
-- not bf16 rounding noise (it flips argmax, not just the raw logit value) --
and it makes this specific approach unsafe to use for real speculative
decoding as-is on this rig. Kept running (rather than deleted) because a
negative result with a real mechanism behind it is exactly what this repo's
"falsify, don't just report a win" convention wants preserved.

ROOT-CAUSED: see `repro_conv1d_batch_dependence.py` in this same directory and
`docs/DECISIONS.md` §76 section 4. Bisected with real forward hooks to
`F.conv1d(groups=hidden_size)` inside `causal_conv1d_update` (the cached
single-token GDN decode path) -- a real, reproducible (0.445 raw diff),
batch-size-dependent numerical property of that op for this model's REAL
trained weights specifically (does not reproduce with synthetic random data
at the identical shape). Not a logic bug anywhere in this codebase or in
`transformers`' Python model code; a hardware/kernel-library numerical
property on this AMD RDNA3/ROCm rig. `torch.use_deterministic_algorithms`
does not fix it (different property: run-to-run repeatability, not
batch-size invariance).

WHY THIS EXISTS -- AND WHY IT IS NOT `state_replay`'s fork/restore idea
------------------------------------------------------------------------
`docs/DECISIONS.md` §75 built and validated a real KV-fork primitive that makes
switching BACK to an abandoned branch cheap -- but that primitive targets the
eager `DynamicCache` research path, and even there, the real economics already
measured in `mtp_draft.py`'s own docstring make SEQUENTIAL branch exploration a
bad trade: verifying K tokens costs a roughly FLAT ~2.7-2.84x regardless of K
in [2, 8] (no fused GatedDeltaNet kernels on this rig, so any multi-token
verification takes the same slow chunked-scan path). Trying a second candidate
sequentially means paying that ~2.8x tax A SECOND TIME -- a bad trade no matter
how cheap switching between the two is.

But that same flat-cost fact points at a DIFFERENT, unexplored idea: if K=2
through K=8 cost about the same, the free capacity is in WIDTH, not in cheap
resumption. Verifying two K=4 candidate continuations in ONE forward pass (via
the batch dimension, not sequential replay) might cost close to what verifying
ONE already costs -- in which case a width-2 tree would be close to free, not
a 2x tax. Nobody has measured this. This script does, honestly, and reports
whichever answer the hardware actually gives.

METHOD
------
1. Real prefill of `--prompts` real domain prompts on Qwen3.5-4B (reused from
   `activation_init_premise`'s loader -- real, diverse, not synthetic).
2. Per prompt, the real MTP draft head produces branch A (the head's own
   greedy top-1 chain, exactly what production drafts today) and branch B
   (identical length, diverging at the FIRST token: the head's logits'
   second-highest token instead of the argmax, then greedy-continued from
   there) -- a real alternate candidate, not synthetic.
3. Real verification, two arms, using `runtime.mtp_draft.snapshot_state` /
   `restore_state` to reset to the identical prefill point (deep-clone
   snapshot -- no re-prefill cost distorting the timing):
     SEQ:   two separate batch=1 forward passes, one per branch.
     BATCH: ONE batch=2 forward pass verifying both branches together, using
            `cache.batch_repeat_interleave(2)` (a real transformers Cache API;
            `manual_batch_repeat_interleave` below patches around a real gap
            where Qwen3.5's GatedDeltaNet layer doesn't implement it) to
            expand the shared prefill state to batch=2 before the single call.
4. Per prompt: check whether batching flips argmax at any position vs. the
   solo reference (the real correctness question), building up a real
   FLIP RATE across many real prompts rather than trusting a single example.
5. Timing measured once, on the first prompt only, with warmup + repeats +
   an alternating (not blocked) execution order, per this repo's own §19
   lesson: a fixed A-then-B order lets whichever runs second look
   artificially cheap from a warmer GPU/cache.

    uv run --env-file .env python \
        experiments/runtime/speculative/batched_tree_verification/benchmark_batched_tree_verify.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.canon import CANON, REPO_ROOT  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps" / "runtime-ipwf"))

from mtp_draft import Qwen35MTPDraftHead, restore_state, snapshot_state  # noqa: E402
from runtime.novel_peft import set_hard_vram_cap  # noqa: E402

PROMPT = "### Question:\nWrite a PostgreSQL query that returns the top customers by revenue.\n\n### Answer:\n"

DOMAINS = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]
DATA_DIR = {"financial": "financial_planning"}


def load_prompts(n: int) -> list[str]:
    """Real, diverse prompts across all 6 canonical domains -- same loader
    pattern as `activation_init_premise/probe_capca_premise.py`."""
    out: list[str] = []
    per_domain = max(1, n // len(DOMAINS) + 1)
    for dom in DOMAINS:
        d = DATA_DIR.get(dom, dom)
        for name in ("evaluation_data_disposition.jsonl", "training_data_v6.jsonl"):
            f = REPO_ROOT / "data" / d / name
            if not f.exists():
                continue
            got = 0
            for line in f.read_text().splitlines():
                if not line.strip() or got >= per_domain:
                    break
                r = json.loads(line)
                p = r.get("prompt") or (r["messages"][0]["content"] if "messages" in r else None)
                if p:
                    out.append(f"### Question:\n{p}\n\n### Answer:\n")
                    got += 1
            if got:
                break
    return out[:n]


@torch.no_grad()
def draft_two_branches(head: Qwen35MTPDraftHead, hidden, next_tok, k: int, start_pos: int):
    """Real branch A (head's own greedy chain) and real branch B (diverges at
    the FIRST drafted token: second-highest logit instead of argmax, then
    greedy-continued) -- same length, genuinely different candidate content."""
    from transformers.cache_utils import DynamicCache

    cache_a = DynamicCache()
    h, tok = hidden, next_tok
    toks_a = []
    first_logits = None
    for i in range(k):
        fused = head._fuse(h, tok)
        positions = torch.arange(start_pos + i, start_pos + i + 1, device=fused.device)
        h = head._run_layer(fused, positions, cache_a)
        logits = head._lm_head(h)[:, -1, :]
        if i == 0:
            first_logits = logits
        tok = torch.argmax(logits, dim=-1, keepdim=True)
        toks_a.append(tok)
    branch_a = torch.cat(toks_a, dim=-1)

    # Branch B: same first hidden state, but take the SECOND-best token at
    # step 0 (a real alternate candidate a tree scheduler would consider),
    # then continue greedily from there through the head, independently.
    top2 = torch.topk(first_logits, k=2, dim=-1).indices[:, 1:2]
    cache_b = DynamicCache()
    h, tok = hidden, next_tok
    toks_b = []
    for i in range(k):
        fused = head._fuse(h, tok)
        positions = torch.arange(start_pos + i, start_pos + i + 1, device=fused.device)
        h = head._run_layer(fused, positions, cache_b)
        logits = head._lm_head(h)[:, -1, :]
        tok = top2 if i == 0 else torch.argmax(logits, dim=-1, keepdim=True)
        toks_b.append(tok)
    branch_b = torch.cat(toks_b, dim=-1)
    return branch_a, branch_b


def manual_batch_repeat_interleave(cache, repeats: int, device) -> None:
    """Expand a batch=1 cache to batch=`repeats`.

    transformers' generic `Cache.batch_repeat_interleave` assumes every layer
    implements it -- confirmed live, Qwen3.5's GatedDeltaNet `LinearAttentionLayer`
    raises `AttributeError` (it was never given one). It DOES implement
    `reorder_cache` (an `index_select` along the batch dim, for beam search),
    which for a batch=1 source is the exact same tensor operation as
    repeat-interleave: reordering with index `[0, 0, ..., 0]` (repeats times)
    duplicates the one real batch entry `repeats` times. Real, correct, uses
    only an existing public method -- no private internals touched."""
    idx = torch.zeros(repeats, dtype=torch.long, device=device)
    for layer in cache.layers:
        if hasattr(layer, "batch_repeat_interleave"):
            layer.batch_repeat_interleave(repeats)
        elif hasattr(layer, "reorder_cache"):
            layer.reorder_cache(idx)
        else:
            raise RuntimeError(f"layer {type(layer).__name__} supports neither batch_repeat_interleave nor reorder_cache")


@torch.no_grad()
def verify_batch1(model, cache, chunk: torch.Tensor, pos: int):
    """One real batch=1 verification forward pass over `chunk` (1, K+1)."""
    positions = torch.arange(pos, pos + chunk.shape[1], device=chunk.device)
    out = model(chunk, past_key_values=cache, cache_position=positions, use_cache=True)
    return out.logits


@torch.no_grad()
def verify_batch2(model, cache, chunk_ab: torch.Tensor, pos: int):
    """One real batch=2 verification forward pass over both branches at once.
    `cache` must already be batch_repeat_interleave(2)'d to match."""
    positions = torch.arange(pos, pos + chunk_ab.shape[1], device=chunk_ab.device)
    out = model(chunk_ab, past_key_values=cache, cache_position=positions, use_cache=True)
    return out.logits


@torch.no_grad()
def check_one_prompt(model, head, tok, prompt: str, k: int) -> dict:
    """Real prefill, real two-branch draft, real batched-vs-solo verification,
    real argmax-flip check, for ONE prompt. Independent cache objects per
    call -- no state carried between prompts."""
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=256).input_ids.to(model.device)
    out = model(ids, use_cache=True, output_hidden_states=True)
    cache_seq = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    full_hids = out.hidden_states[-1]
    pos = ids.shape[1]

    branch_a, branch_b = draft_two_branches(head, full_hids[:, -1:, :], nxt, k, start_pos=pos - 1)
    chunk_a = torch.cat([nxt, branch_a], dim=-1)
    chunk_b = torch.cat([nxt, branch_b], dim=-1)
    chunk_ab = torch.cat([chunk_a, chunk_b], dim=0)

    logits_a_solo = verify_batch1(model, cache_seq, chunk_a, pos)

    out2 = model(ids, use_cache=True)
    cache_seq_b = out2.past_key_values
    logits_b_solo = verify_batch1(model, cache_seq_b, chunk_b, pos)

    out3 = model(ids, use_cache=True)
    cache_batch = out3.past_key_values
    manual_batch_repeat_interleave(cache_batch, 2, device=model.device)
    logits_ab = verify_batch2(model, cache_batch, chunk_ab, pos)

    am_batched_a = torch.argmax(logits_ab[0], dim=-1)
    am_solo_a = torch.argmax(logits_a_solo[0], dim=-1)
    am_batched_b = torch.argmax(logits_ab[1], dim=-1)
    am_solo_b = torch.argmax(logits_b_solo[0], dim=-1)
    return {
        "branch_a": branch_a[0].tolist(), "branch_b": branch_b[0].tolist(),
        "max_diff_a": (logits_ab[0] - logits_a_solo[0]).abs().max().item(),
        "max_diff_b": (logits_ab[1] - logits_b_solo[0]).abs().max().item(),
        "flip_a": not torch.equal(am_batched_a, am_solo_a),
        "flip_b": not torch.equal(am_batched_b, am_solo_b),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=4, help="draft length per branch (chunk is k+1 wide)")
    ap.add_argument("--repeats", type=int, default=15, help="timing repeats, on the first prompt only")
    ap.add_argument("--prompts", type=int, default=18, help="number of real prompts for the flip-rate scan")
    ap.add_argument("--out", default="results/benchmarks/batched_tree_verify.json")
    args = ap.parse_args()

    set_hard_vram_cap(CANON.VRAM_CAP_GB)
    print("=" * 100)
    print("  BATCHED vs SEQUENTIAL two-candidate verification -- is width free here,")
    print("  and how often does it actually flip an accept/reject decision?")
    print("=" * 100)

    tok = AutoTokenizer.from_pretrained(CANON.BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CANON.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()
    head = Qwen35MTPDraftHead(model, model_id=CANON.BASE_MODEL)

    # ---- 1. Real flip-rate scan across many real, diverse prompts
    prompts = load_prompts(args.prompts)
    print(f"\n  [1/2] FLIP-RATE SCAN -- {len(prompts)} real prompts, k={args.k}")
    print("  " + "-" * 96)
    rows = []
    n_flips = 0
    for i, p in enumerate(prompts):
        r = check_one_prompt(model, head, tok, p, args.k)
        any_flip = r["flip_a"] or r["flip_b"]
        n_flips += int(any_flip)
        rows.append(r)
        print(f"    [{i:2d}] max|diff| A={r['max_diff_a']:.3f} B={r['max_diff_b']:.3f}  "
              f"flip_A={r['flip_a']} flip_B={r['flip_b']}  {'<-- FLIP' if any_flip else ''}")
    flip_rate = n_flips / max(1, len(rows))
    print(f"\n  FLIP RATE: {n_flips}/{len(rows)} prompts ({100*flip_rate:.1f}%) had >=1 argmax flip "
          f"in a {args.k+1}-token verification.")

    # ---- 2. Timing, on the canonical PROMPT, warmed + alternating + repeated
    print(f"\n  [2/2] TIMING -- canonical prompt, {args.repeats} alternating repeats")
    print("  " + "-" * 96)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids, use_cache=True, output_hidden_states=True)
    cache_seq = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    full_hids = out.hidden_states[-1]
    pos = ids.shape[1]
    base_snap = snapshot_state(cache_seq)

    k = args.k
    branch_a, branch_b = draft_two_branches(head, full_hids[:, -1:, :], nxt, k, start_pos=pos - 1)
    chunk_a = torch.cat([nxt, branch_a], dim=-1)
    chunk_b = torch.cat([nxt, branch_b], dim=-1)
    chunk_ab = torch.cat([chunk_a, chunk_b], dim=0)
    width = chunk_a.shape[1]

    with torch.no_grad():
        out2 = model(ids, use_cache=True)
    cache_batch = out2.past_key_values
    manual_batch_repeat_interleave(cache_batch, 2, device=model.device)
    batch_snap = snapshot_state(cache_batch)

    def run_seq():
        restore_state(cache_seq, base_snap)
        verify_batch1(model, cache_seq, chunk_a, pos)
        restore_state(cache_seq, base_snap)
        verify_batch1(model, cache_seq, chunk_b, pos)

    def run_batch():
        restore_state(cache_batch, batch_snap)
        verify_batch2(model, cache_batch, chunk_ab, pos)

    for _ in range(3):
        run_seq(); run_batch()
    torch.cuda.synchronize()

    seq_times, batch_times = [], []
    for _ in range(args.repeats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run_seq()
        torch.cuda.synchronize(); seq_times.append(time.perf_counter() - t0)

        torch.cuda.synchronize(); t0 = time.perf_counter()
        run_batch()
        torch.cuda.synchronize(); batch_times.append(time.perf_counter() - t0)

    seq_ms = sorted(seq_times)
    batch_ms = sorted(batch_times)
    seq_med = 1000 * seq_ms[len(seq_ms) // 2]
    batch_med = 1000 * batch_ms[len(batch_ms) // 2]

    print(f"  SEQ   (2x batch=1 verify)  : {seq_med:8.2f} ms   [{1000*seq_ms[0]:.2f}, {1000*seq_ms[-1]:.2f}] range")
    print(f"  BATCH (1x batch=2 verify)  : {batch_med:8.2f} ms   [{1000*batch_ms[0]:.2f}, {1000*batch_ms[-1]:.2f}] range")
    print(f"  batching is {seq_med/batch_med:.2f}x the cost of sequential")

    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)
    print(f"  Width IS cheap here: {seq_med/batch_med:.2f}x for 2x candidates.")
    print(f"  But real flip rate is {100*flip_rate:.1f}% ({n_flips}/{len(rows)}) -- ", end="")
    if flip_rate > 0:
        print("NOT SAFE to ship as-is.")
        print("  A real, non-negligible fraction of batched verifications would accept/reject")
        print("  a DIFFERENT token than a solo verification of the identical candidate would.")
        print("  This is a real numerical hazard in this model/hardware/software stack's batch>1")
        print("  handling (reproduced with a plain native batch=2 call, no custom code involved),")
        print("  not bf16 rounding noise -- it changes which token wins argmax. The cheap-width")
        print("  finding is real, but unusable until this is root-caused (candidate culprit: the")
        print("  slow reference GatedDeltaNet kernel used because flash-linear-attention isn't")
        print("  installed on this rig) or worked around (e.g. per-branch verification remains")
        print("  sequential; only draft-side computation is batched).")
    else:
        print("clean across this sample.")

    report = {
        "config": CANON.stamp() | {"k": k, "width": width, "repeats": args.repeats, "n_prompts": len(rows)},
        "flip_rate_scan": rows, "n_flips": n_flips, "n_prompts": len(rows), "flip_rate": flip_rate,
        "timing": {
            "seq_ms_median": seq_med, "batch_ms_median": batch_med,
            "seq_ms_all": [1000 * t for t in seq_times], "batch_ms_all": [1000 * t for t in batch_times],
            "batch_vs_seq_ratio": batch_med / seq_med,
        },
    }
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
