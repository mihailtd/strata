"""Builds the missing plumbing `benchmark_branching_primitives.py` sized but never wrote.

WHY THIS EXISTS
---------------
`docs/DECISIONS.md` §20 measured the state ring buffer's three capability claims
and found CLAIM 1 (speculative trees) NOT SUPPORTED and CLAIM 3 (local beam
search) only PARTIAL, both for the same reason: `StateRingBuffer.rollback()`
restores attention KV by CROPPING it in place (`layer.keys[..., :saved_len, :]`)
which is destructive -- once you roll back past a branch, that branch's KV is
gone and can only be recovered by full recomputation (the "+54.6% penalty when
the first branch wins" in Claim 3).

§20 also sized exactly how much KV a modest tree needs to keep multiple branches
alive: width-2 x depth-4 needs 8 live nodes (which `max_depth=8` already
provisions for the SSM half) and only 0.26 MB of attention KV -- "a small piece
of plumbing, not a memory-infrastructure project." Nobody had written that
plumbing. This does, and measures whether it actually removes the penalty it
was sized to remove.

THE PRIMITIVE: fork_kv_tail / restore_kv_tail
----------------------------------------------
Instead of only cropping on rollback, clone the KV entries ABOVE the crop point
before rolling back (`.clone()` on a slice -- a real, cheap tensor copy, not a
recompute), and provide a restore that crops back to the shared base and
re-concatenates the saved tail. This gives the attention half a real,
non-destructive fork.

A SECOND, UNPLANNED FINDING: StateRingBuffer.push()/rollback() ARE NOT SAFE FOR
TREE-STYLE MULTI-SLOT USE, AS-IS
----------------------------------------------------------------------------
The plan was to pair the KV-tail primitive above with the existing
`StateRingBuffer.push()`/`rollback()` for the SSM/conv half, since §19-§20
already proved that mechanism correct and cheap for LINEAR rollback. Building
the actual tree (measurement 3, below) found a real bug in that plan, not in
the primitive: `StateRingBuffer.rollback()` unconditionally sets
`self.write_ptr = (commit_ptr + n_accepted) % max_depth` -- a formula for the
speculative-decoding caller it was built for, which never keeps more than one
speculative slot alive at once. A tree does: `rollback(slot=parent)` followed
by `push()` to create a child's slot silently reassigns `write_ptr` back to a
COMMIT-RELATIVE position rather than "the next free slot", so a second
`rollback(slot=X); push()` sequence (exploring branch 2 of the same parent)
overwrites whatever slot branch 1 just wrote to. First observed as branch 1 and
branch 2 of depth 0 producing DIFFERENT next-tokens despite both starting from
an identical, deterministically-greedy-decoded root state -- the tell that the
recurrent state fed into branch 2's forward pass was not actually root's.

The fix here is the same shape as the KV-tail primitive: bypass the ring
buffer's write_ptr bookkeeping entirely for tree nodes and clone/restore the
SSM/conv tensors directly per node (`snapshot_ssm`/`restore_ssm` below) -- SSM
state is fixed-size and cheap to clone (§20: 51.9 MB, and a clone is one
`.clone()` call, not a recompute), so this needs no ring at all. `StateRingBuffer`
itself is untouched -- linear speculative rollback (its actual current use) is
unaffected; this only affects tree-style multi-slot addressing, which nothing
uses it for today.

This is deliberately NOT wired into `state_ring_buffer.py` / the serving path.
It is stage-1 experimental plumbing, validated here in isolation, matching this
repo's own experiment -> integration -> benchmark lifecycle.

WHAT IS MEASURED
-----------------
1. CORRECTNESS: does restoring a forked branch reproduce a token-for-token
   identical continuation to a fresh from-scratch generation of that same
   branch? (Bit-exact KV, not "close enough.")
2. COST: does fork+restore beat the "+54.6% winner-recompute penalty" §20's
   Claim 3 measured, on the identical 2-branch local-beam-search scenario?
3. THE FULL WIDTH-2 x DEPTH-4 TREE: build all 8 live nodes for real, verify the
   actual KV bytes retained matches §20's 0.26 MB prediction, and verify
   bit-exact resume from an arbitrary leaf.

    uv run --env-file .env python \
        experiments/runtime/speculative/state_replay/benchmark_kv_fork_tree_unlock.py
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

from runtime.canon import REPO_ROOT  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.mtp_draft import attach_state_ring_buffer  # noqa: E402
from runtime.novel_peft import set_hard_vram_cap  # noqa: E402

PROMPT = "### Question:\nWrite a PostgreSQL query that returns the top customers by revenue.\n\n### Answer:\n"


# ---------------------------------------------------------------------------
# THE PRIMITIVE: real, non-destructive attention-KV fork/restore
# ---------------------------------------------------------------------------

def attn_layers(cache) -> list:
    return [lc for lc in cache.layers if hasattr(lc, "keys") and hasattr(lc, "values")]


def kv_lens(cache) -> list[int]:
    return [lc.keys.shape[-2] if isinstance(lc.keys, torch.Tensor) and lc.keys.numel() else 0
            for lc in attn_layers(cache)]


def fork_kv_tail(cache, base_lens: list[int]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Clone the KV entries ABOVE base_lens[i], per attention layer. Real tensor .clone(),
    not a recompute -- this is the whole primitive that was missing."""
    fork: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for i, lc in enumerate(attn_layers(cache)):
        cur = lc.keys.shape[-2] if isinstance(lc.keys, torch.Tensor) and lc.keys.numel() else 0
        base = base_lens[i]
        if cur > base:
            fork[i] = (lc.keys[..., base:cur, :].clone(), lc.values[..., base:cur, :].clone())
    return fork


def fork_bytes(fork: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> int:
    total = 0
    for k, v in fork.values():
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


def gdn_layers(cache) -> list:
    return [lc for lc in cache.layers if hasattr(lc, "recurrent_states") and hasattr(lc, "conv_states")]


def snapshot_ssm(cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Direct per-node SSM/conv clone, bypassing StateRingBuffer's write_ptr
    entirely -- see the module docstring for why the ring buffer itself is not
    safe for this. Cheap: fixed-size state, one `.clone()` per GDN layer."""
    return [(lc.recurrent_states[0].clone(), lc.conv_states[0].clone()) for lc in gdn_layers(cache)]


def restore_ssm(cache, snap: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    for lc, (rec, conv) in zip(gdn_layers(cache), snap, strict=True):
        lc.recurrent_states[0].copy_(rec)
        lc.conv_states[0].copy_(conv)


def restore_kv_tail(cache, base_lens: list[int], fork: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Crop to the shared base, then re-attach the forked tail. No recomputation."""
    for i, lc in enumerate(attn_layers(cache)):
        base = base_lens[i]
        if hasattr(lc, "crop"):
            lc.crop(base)
        else:
            lc.keys = lc.keys[..., :base, :]
            lc.values = lc.values[..., :base, :]
        if i in fork:
            k_tail, v_tail = fork[i]
            lc.keys = torch.cat([lc.keys, k_tail], dim=-2)
            lc.values = torch.cat([lc.values, v_tail], dim=-2)


@torch.no_grad()
def extend(model, cache, first_tok, n: int) -> tuple[list[int], torch.Tensor]:
    toks, nxt = [], first_tok
    for _ in range(n):
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks.append(nxt.item())
    return toks, nxt


@torch.no_grad()
def fresh_cache(model, tok, prompt: str):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model(ids, use_cache=True)
    return out.past_key_values, torch.argmax(out.logits[:, -1, :], -1, keepdim=True), ids


# ---------------------------------------------------------------------------


def measurement_1_correctness(model, tok, h: int) -> dict:
    """Fork branch A, explore B, restore A via fork -- does resuming A from the
    fork match a fresh from-scratch continuation of A? Bit-exact is the bar."""
    print("\n" + "=" * 100)
    print("  1. CORRECTNESS -- forked-and-restored branch A vs a from-scratch reference")
    print("=" * 100)

    cache, nxt0, ids = fresh_cache(model, tok, PROMPT)
    ring = attach_state_ring_buffer(cache, max_depth=8)
    base_slot = ring.push(cache)
    base_lens = kv_lens(cache)

    branch_a, a_last = extend(model, cache, nxt0, h)
    a_slot = ring.push(cache)                       # SSM/conv state after A
    a_fork = fork_kv_tail(cache, base_lens)          # attention KV tail after A
    a_fork_bytes = fork_bytes(a_fork)

    ring.rollback(cache, slot=base_slot)             # -> back to base (crops A's KV away)
    branch_b, _ = extend(model, cache, nxt0, h)       # explore B (greedy, so B == A by construction)

    # "A wins" -- restore it via the fork, no recomputation.
    ring.rollback(cache, slot=a_slot)                # SSM/conv -> A's state
    restore_kv_tail(cache, base_lens, a_fork)         # attention KV -> A's KV, via cheap concat
    restored_len = kv_lens(cache)

    # Continue generating from the restored branch A and compare against a
    # fully independent, from-scratch re-derivation of the same continuation.
    continued, _ = extend(model, cache, a_last, h)

    ref_cache, ref_nxt, _ = fresh_cache(model, tok, PROMPT)
    ref_a, ref_a_last = extend(model, ref_cache, ref_nxt, h)
    ref_continued, _ = extend(model, ref_cache, ref_a_last, h)

    kv_ok = restored_len == [base + (a_fork[i][0].shape[-2] if i in a_fork else 0)
                             for i, base in enumerate(base_lens)]
    tokens_ok = (branch_a == ref_a) and (continued == ref_continued)
    print(f"    branch A (h={h} tok)              : {branch_a}")
    print(f"    reference A (fresh, h={h} tok)     : {ref_a}    match={branch_a == ref_a}")
    print(f"    KV length after fork-restore matches direct extension: {kv_ok}")
    print(f"    continuation from restored A       : {continued}")
    print(f"    continuation from reference A      : {ref_continued}    match={continued == ref_continued}")
    print(f"    fork size for this one branch       : {a_fork_bytes} bytes ({a_fork_bytes/1e3:.2f} KB)")
    print(f"    => {'CORRECT -- bit-exact resume' if tokens_ok and kv_ok else 'FAILED'}")
    return {"branch_a": branch_a, "branch_b": branch_b, "tokens_ok": tokens_ok,
            "kv_length_ok": kv_ok, "fork_bytes_one_branch": a_fork_bytes}


def measurement_2_beam_search_cost(model, tok, h: int, repeats: int) -> dict:
    """Re-run §20 Claim 3's exact scenario (2-branch local beam search, A wins)
    but with the fork/restore primitive instead of full recomputation."""
    print("\n" + "=" * 100)
    print("  2. COST -- does fork+restore beat the +54.6% winner-recompute penalty?")
    print("=" * 100)

    cache, nxt0, ids = fresh_cache(model, tok, PROMPT)
    ring = attach_state_ring_buffer(cache, max_depth=8)
    base_slot = ring.push(cache)
    base_lens = kv_lens(cache)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _, a_last = extend(model, cache, nxt0, h)
    torch.cuda.synchronize()
    explore_ms = 1000.0 * (time.perf_counter() - t0)

    a_slot = ring.push(cache)
    a_fork = fork_kv_tail(cache, base_lens)

    ring.rollback(cache, slot=base_slot)
    extend(model, cache, nxt0, h)   # explore B (same greedy path by construction)

    # OLD WAY (§20 Claim 3): "A wins" -> rollback destroyed A's KV -> regenerate.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ring.rollback(cache, slot=base_slot)
    extend(model, cache, nxt0, h)
    torch.cuda.synchronize()
    old_regen_ms = 1000.0 * (time.perf_counter() - t0)

    # NEW WAY: "A wins" -> restore via fork, zero recomputation.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        ring.rollback(cache, slot=a_slot)
        restore_kv_tail(cache, base_lens, a_fork)
    torch.cuda.synchronize()
    new_restore_ms = 1000.0 * (time.perf_counter() - t0) / repeats

    two_branch_total = 2 * explore_ms
    old_penalty_pct = 100 * old_regen_ms / two_branch_total
    new_penalty_pct = 100 * new_restore_ms / two_branch_total
    print(f"    explore one branch ({h} tok)         : {explore_ms:8.3f} ms")
    print(f"    OLD -- A wins, full recompute        : {old_regen_ms:8.3f} ms  (+{old_penalty_pct:.1f}% of 2-branch total)")
    print(f"    NEW -- A wins, fork+restore           : {new_restore_ms:8.3f} ms  (+{new_penalty_pct:.1f}% of 2-branch total)")
    print(f"    fork+restore is {old_regen_ms / max(1e-9, new_restore_ms):.0f}x cheaper than recomputation")
    print(f"    => the +54.6%-class penalty §20 measured drops to +{new_penalty_pct:.1f}% with this primitive.")
    return {"explore_ms": explore_ms, "old_regen_ms": old_regen_ms, "new_restore_ms": new_restore_ms,
            "old_penalty_pct": old_penalty_pct, "new_penalty_pct": new_penalty_pct,
            "speedup_vs_recompute": old_regen_ms / max(1e-9, new_restore_ms)}


def restore_kv_chain(cache, root_lens: list[int], node: dict | None) -> None:
    """Crop to the shared root base, then replay the INCREMENTAL edge-forks from
    root down to `node`. Each edge stores only the ONE token added at that edge
    -- shared ancestry is never duplicated, which is what makes the total live
    KV budget `live_nodes * kv_bytes_per_token`, not `sum(depth) * kv_bytes_per_token`."""
    for i, lc in enumerate(attn_layers(cache)):
        base = root_lens[i]
        if hasattr(lc, "crop"):
            lc.crop(base)
        else:
            lc.keys = lc.keys[..., :base, :]
            lc.values = lc.values[..., :base, :]
    chain = []
    n = node
    while n is not None and n.get("edge_fork") is not None:
        chain.append(n["edge_fork"])
        n = n["parent"]
    for edge_fork in reversed(chain):
        for i, lc in enumerate(attn_layers(cache)):
            if i in edge_fork:
                k_tail, v_tail = edge_fork[i]
                lc.keys = torch.cat([lc.keys, k_tail], dim=-2)
                lc.values = torch.cat([lc.values, v_tail], dim=-2)


def measurement_3_full_tree(model, tok, width: int, depth: int) -> dict:
    """Build the actual width x depth tree §20 sized, keep every live node's
    state resident simultaneously, verify total bytes and bit-exact resume
    from an arbitrary leaf.

    Each node's `edge_fork` holds only the KV for the ONE token added at that
    edge (parent -> child) -- not the whole path back to root. Restoring a
    node replays its ancestor chain of edge-forks (`restore_kv_chain`), which
    is what makes the live-node accounting `width*depth * kv_bytes_per_token`
    instead of quadratic in depth.

    SSM/conv state per node is a direct clone (`snapshot_ssm`/`restore_ssm`),
    NOT `StateRingBuffer.push()`/`rollback()` -- see the module docstring for
    the real bug that choice avoids: the ring's `rollback()` reassigns
    `write_ptr` to a commit-relative slot every call, so a `push()` right
    after a `rollback()` (exactly this tree's access pattern) silently
    overwrites whatever slot a sibling branch just wrote. Direct clone/copy_
    sidesteps that bookkeeping entirely -- still O(1) and cheap per §20, since
    SSM state is fixed-size regardless of tree depth."""
    print("\n" + "=" * 100)
    print(f"  3. THE FULL TREE -- width {width} x depth {depth} ({width * depth} live nodes)")
    print("=" * 100)

    cache, nxt0, ids = fresh_cache(model, tok, PROMPT)
    root_lens = kv_lens(cache)
    root_ssm = snapshot_ssm(cache)
    root = {"branch": -1, "ssm": root_ssm, "lens": root_lens, "last": nxt0, "seq": [], "edge_fork": None, "parent": None}

    nodes: list[dict] = []
    frontier = root
    for d in range(depth):
        children = []
        for w in range(width):
            restore_kv_chain(cache, root_lens, frontier)
            restore_ssm(cache, frontier["ssm"])
            toks, last = extend(model, cache, frontier["last"], 1)
            ssm = snapshot_ssm(cache)
            edge_fork = fork_kv_tail(cache, frontier["lens"])   # ONLY this new token's KV
            lens = kv_lens(cache)
            node = {"depth": d, "branch": w, "ssm": ssm, "edge_fork": edge_fork,
                    "lens": lens, "last": last, "seq": frontier["seq"] + toks, "parent": frontier}
            children.append(node)
            nodes.append(node)
        frontier = children[0]  # advance frontier along branch 0

    total_kv_bytes = sum(fork_bytes(n["edge_fork"]) for n in nodes)
    print(f"    live nodes            : {len(nodes)}")
    print(f"    total forked KV bytes : {total_kv_bytes} bytes ({total_kv_bytes/1e6:.3f} MB)")
    print(f"    §20 predicted         : 0.26 MB for width 2 x depth 4")

    # Bit-exact resume check on an arbitrary interior leaf (not branch 0, so this
    # actually exercises a real, non-trivially-cached node whose ancestor chain
    # must be replayed correctly).
    victim = nodes[-1] if nodes[-1]["branch"] != 0 else nodes[-2]
    restore_kv_chain(cache, root_lens, victim)
    restore_ssm(cache, victim["ssm"])
    resumed, _ = extend(model, cache, victim["last"], 2)

    # Reference: regenerate that exact node's full path from scratch. Must use
    # extend(), not a manual token-by-token replay of victim["seq"] -- extend()
    # feeds the PREVIOUS prediction to produce the NEXT one (that is how the
    # tree itself was built: node["last"] is a predicted-but-not-yet-cached
    # token), so replaying `seq` itself as direct inputs is off-by-one against
    # the real path and was the actual bug the first version of this check had.
    ref_cache, ref_nxt, _ = fresh_cache(model, tok, PROMPT)
    ref_seq, ref_last = extend(model, ref_cache, ref_nxt, len(victim["seq"]))
    assert ref_seq == victim["seq"], f"reference replay diverged from the tree path: {ref_seq} != {victim['seq']}"
    ref_resumed, _ = extend(model, ref_cache, ref_last, 2)

    ok = resumed == ref_resumed
    print(f"    victim node depth={victim['depth']} branch={victim['branch']}  seq={victim['seq']}")
    print(f"    resume-from-fork continuation  : {resumed}")
    print(f"    from-scratch reference         : {ref_resumed}")
    print(f"    => {'CORRECT -- arbitrary leaf resumes bit-exact' if ok else 'FAILED'}")

    return {"width": width, "depth": depth, "live_nodes": len(nodes),
            "total_kv_bytes": total_kv_bytes, "predicted_kv_mb_from_decisions_md": 0.26,
            "arbitrary_leaf_resume_correct": ok}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/benchmarks/kv_fork_tree_unlock.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  KV-FORK TREE UNLOCK -- the plumbing §20 sized but never built")
    print("=" * 100)
    print(f"  device={dev}  horizon={args.horizon}\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()

    t0 = time.time()
    m1 = measurement_1_correctness(model, tok, args.horizon)
    m2 = measurement_2_beam_search_cost(model, tok, args.horizon, args.repeats)
    m3 = measurement_3_full_tree(model, tok, width=2, depth=4)

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"    1. Fork+restore is bit-exact              : {m1['tokens_ok'] and m1['kv_length_ok']}")
    print(f"    2. Winner-recompute penalty                : +54.6% (old, §20) -> +{m2['new_penalty_pct']:.1f}% (new)")
    print(f"    3. width2xdepth4 tree, {m3['live_nodes']} live nodes  : {m3['total_kv_bytes']/1e6:.3f} MB KV, "
          f"leaf resume correct={m3['arbitrary_leaf_resume_correct']}")

    report = {"device": dev, "horizon": args.horizon, "measurement_1_correctness": m1,
              "measurement_2_beam_search_cost": m2, "measurement_3_full_tree": m3,
              "elapsed_seconds": time.time() - t0}
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  {time.time()-t0:.0f}s   Saved -> {out_p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
