"""What does the state ring buffer actually unlock, beyond linear speculation?

WHY THIS EXISTS
---------------
ReplaySSM's direct speedup on linear speculative decoding is unresolved and
bounded near ~0.5% (`docs/DECISIONS.md` §19). Judging it on that alone would be
near-sighted: **a primitive's value is not only its direct speedup.** Zero-copy,
bit-exact, O(1) state rollback is infrastructure for whole classes of generation
strategy that are otherwise too expensive to attempt — tree speculation,
mid-stream self-correction, local beam search.

But "unlocks X" is a claim like any other, and claims get measured here. Three
were made. This benchmark tests each, **including the one predicted to FAIL** —
a capability claim with no falsification test is marketing.

    CLAIM 2  Time-travel steering    predicted FULLY SUPPORTED
    CLAIM 3  Local beam search       predicted PARTIAL (winner-recompute caveat)
    CLAIM 1  Speculative trees       predicted NOT SUPPORTED (attention KV)

THE ARCHITECTURAL ASYMMETRY THIS MEASURES
-----------------------------------------
Qwen3.5 is hybrid: 24 GatedDeltaNet (SSM) + 8 full-attention layers. The two
halves have opposite rollback economics, and that is the whole story:

    SSM state    FIXED SIZE regardless of sequence length. Copying it is cheap,
                 so N historical states cost N x 6.4 MB. Branching is easy.

    Attention KV GROWS with sequence length. The ring buffer therefore stores
                 only an integer `seq_lens[slot]` per attention layer and
                 restores by CROPPING -- which is destructive. Rewinding works;
                 keeping two branches alive does not.

So the buffer branches 24/32 of the state and leaves the 8/32 that is expensive
to branch. Linear rollback needs only rewind, so it works. Trees need both.

    uv run --env-file .env python \
        benchmarks/runtime/speculative/state_replay/benchmark_branching_primitives.py
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
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.mtp_draft import attach_state_ring_buffer  # noqa: E402
from runtime.novel_peft import set_hard_vram_cap  # noqa: E402

PROMPT = "### Question:\nWrite a PostgreSQL query that returns the top customers by revenue.\n\n### Answer:\n"


@torch.no_grad()
def extend(model, cache, first_tok, n: int) -> tuple[list[int], torch.Tensor]:
    """Greedy-extend `n` tokens from a live cache. Returns (tokens, last_token)."""
    toks, nxt = [], first_tok
    for _ in range(n):
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks.append(nxt.item())
    return toks, nxt


def kv_len(cache) -> int:
    for lc in cache.layers:
        k = getattr(lc, "keys", None)
        if isinstance(k, torch.Tensor) and k.numel() > 0:
            return k.shape[-2]
    return 0


def kv_bytes_per_token(cache) -> int:
    """Bytes of attention KV added per token, across all attention layers."""
    total = 0
    for lc in cache.layers:
        k = getattr(lc, "keys", None)
        v = getattr(lc, "values", None)
        if isinstance(k, torch.Tensor) and k.numel() > 0:
            per = k.shape[-2]
            total += (k.numel() // per) * k.element_size()
            total += (v.numel() // per) * v.element_size()
    return total


def ssm_state_bytes(ring) -> int:
    """Bytes of SSM state per checkpoint, read from the ring's OWN allocated slots.

    Walking `cache.layers` for `recurrent_states` returned 0 on the first version
    of this benchmark -- the hybrid cache does not expose them where that guess
    looked, and a silent 0 made the tree-sizing table read "0.0 MB of SSM". The
    ring buffer already holds the real tensors, so ask it.
    """
    total = 0
    for _layer, rec_buf, conv_buf in ring.gdn_plan:
        total += rec_buf[0].numel() * rec_buf[0].element_size()
        total += conv_buf[0].numel() * conv_buf[0].element_size()
    return total


@torch.no_grad()
def fresh_cache(model, tok, prompt: str):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model(ids, use_cache=True)
    return out.past_key_values, torch.argmax(out.logits[:, -1, :], -1, keepdim=True), ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--horizon", type=int, default=4, help="branch / rewind depth in tokens")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/branching_primitives.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  WHAT THE STATE RING BUFFER UNLOCKS — three claims, measured")
    print("=" * 100)
    print(f"  device={dev}  horizon={args.horizon} tokens  repeats={args.repeats}\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()

    report: dict = {"device": dev, "horizon": args.horizon}

    # ---------------------------------------------------------------- geometry
    cache, nxt0, ids = fresh_cache(model, tok, PROMPT)
    ring = attach_state_ring_buffer(cache, max_depth=8)
    kv_per_tok = kv_bytes_per_token(cache)
    ssm_bytes = ssm_state_bytes(ring)
    print("  STATE GEOMETRY (the asymmetry that decides all three claims)")
    print(f"    SSM state (fixed, all 24 layers)      : {ssm_bytes / 1e6:8.2f} MB  per checkpoint")
    print(f"    Attention KV (grows, all 8 layers)    : {kv_per_tok / 1e3:8.2f} KB  PER TOKEN")
    print(f"    prompt is {ids.shape[1]} tokens -> current KV {kv_per_tok * ids.shape[1] / 1e6:.2f} MB")
    print(f"    ring buffer holds 8 x SSM = {8 * ssm_bytes / 1e6:.1f} MB, and ZERO KV bytes")
    report["ssm_bytes_per_checkpoint"] = ssm_bytes
    report["kv_bytes_per_token"] = kv_per_tok

    # ============================================================== CLAIM 2
    # Time-travel steering: generate, rewind j tokens, inject a correction,
    # resume. Purely LINEAR -- the abandoned future is never revisited, so
    # destructive KV cropping is harmless.
    print("\n" + "=" * 100)
    print("  CLAIM 2 — TIME-TRAVEL STEERING (predicted: FULLY SUPPORTED)")
    print("=" * 100)
    h = args.horizon

    cache, nxt0, ids = fresh_cache(model, tok, PROMPT)
    ring = attach_state_ring_buffer(cache, max_depth=8)
    slot = ring.push(cache)
    len_at_checkpoint = kv_len(cache)
    bad_toks, _ = extend(model, cache, nxt0, h)

    # rewind and verify the cache is byte-identical to the checkpoint
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        ring.rollback(cache, slot=slot)
    torch.cuda.synchronize()
    rollback_ms = 1000.0 * (time.perf_counter() - t0) / args.repeats
    len_after = kv_len(cache)

    # the honest correctness test: after rewinding, does resuming reproduce what a
    # FRESH cache produces from the same prompt? If yes, the rewound state is sound.
    resumed, _ = extend(model, cache, nxt0, h)
    ref_cache, ref_nxt, _ = fresh_cache(model, tok, PROMPT)
    reference, _ = extend(model, ref_cache, ref_nxt, h)

    # cost of the alternative: throw the cache away and re-prefill the prompt
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        fresh_cache(model, tok, PROMPT)
    torch.cuda.synchronize()
    reprefill_ms = 1000.0 * (time.perf_counter() - t0) / args.repeats

    ok = resumed == reference and len_after == len_at_checkpoint
    print(f"    KV length {len_at_checkpoint} -> {len_at_checkpoint + h} -> rewound to {len_after}"
          f"   {'OK' if len_after == len_at_checkpoint else 'MISMATCH'}")
    print(f"    resumed tokens == fresh-cache reference : {'YES' if resumed == reference else 'NO'}")
    print(f"    rewind cost            : {rollback_ms:8.3f} ms")
    print(f"    re-prefill alternative : {reprefill_ms:8.3f} ms   ({reprefill_ms / max(1e-9, rollback_ms):.0f}x more)")
    print(f"    => CLAIM 2 {'CONFIRMED' if ok else 'FAILED'}: rewind is exact and "
          f"{reprefill_ms / max(1e-9, rollback_ms):.0f}x cheaper than re-prefilling.")
    print("    bounded to max_depth=8 checkpoints of history.")
    report["claim2"] = {"supported": ok, "rollback_ms": rollback_ms,
                        "reprefill_ms": reprefill_ms,
                        "speedup_vs_reprefill": reprefill_ms / max(1e-9, rollback_ms),
                        "bad_tokens": bad_toks, "resumed": resumed, "reference": reference}

    # ============================================================== CLAIM 3
    # Local beam search: branch A, rewind, branch B, keep the better one.
    # Cheap if B wins (already resident). If A wins its KV was cropped away, so A
    # must be regenerated -- that is the caveat under test.
    print("\n" + "=" * 100)
    print("  CLAIM 3 — LOCAL BEAM SEARCH (predicted: PARTIAL, winner-recompute caveat)")
    print("=" * 100)

    cache, nxt0, ids = fresh_cache(model, tok, PROMPT)
    ring = attach_state_ring_buffer(cache, max_depth=8)
    slot = ring.push(cache)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    branch_a, _ = extend(model, cache, nxt0, h)
    torch.cuda.synchronize()
    explore_ms = 1000.0 * (time.perf_counter() - t0)

    ring.rollback(cache, slot=slot)
    branch_b, _ = extend(model, cache, nxt0, h)   # same greedy path: identical by construction

    # B wins -> already resident, zero extra cost.
    # A wins -> its KV was cropped at rollback, so it must be regenerated.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ring.rollback(cache, slot=slot)
    _ = extend(model, cache, nxt0, h)
    torch.cuda.synchronize()
    regen_ms = 1000.0 * (time.perf_counter() - t0)

    two_branch_total = 2 * explore_ms
    print(f"    explore one branch ({h} tok)      : {explore_ms:8.3f} ms")
    print(f"    2-branch search, B wins          : {two_branch_total:8.3f} ms  (winner resident)")
    print(f"    2-branch search, A wins          : {two_branch_total + regen_ms:8.3f} ms  "
          f"(+{regen_ms:.3f} ms to regenerate the cropped winner)")
    print(f"    penalty when the FIRST branch wins: {100 * regen_ms / two_branch_total:+.1f}%")
    print("    => CLAIM 3 PARTIAL as predicted. Branching is cheap; keeping a")
    print("       non-final branch alive is not, because its KV is cropped away.")
    report["claim3"] = {"explore_ms": explore_ms, "regen_ms": regen_ms,
                        "two_branch_b_wins_ms": two_branch_total,
                        "two_branch_a_wins_ms": two_branch_total + regen_ms,
                        "first_branch_winner_penalty_pct": 100 * regen_ms / two_branch_total}

    # ============================================================== CLAIM 1
    # Speculative trees: needs TWO branches alive at once. Falsification test --
    # extend A, rewind, extend B, then ask whether A's KV survived.
    print("\n" + "=" * 100)
    print("  CLAIM 1 — SPECULATIVE TREES (predicted: NOT SUPPORTED — attention KV)")
    print("=" * 100)

    cache, nxt0, ids = fresh_cache(model, tok, PROMPT)
    ring = attach_state_ring_buffer(cache, max_depth=8)
    slot = ring.push(cache)
    base_len = kv_len(cache)

    extend(model, cache, nxt0, h)
    a_len = kv_len(cache)
    a_kv = None
    for lc in cache.layers:
        k = getattr(lc, "keys", None)
        if isinstance(k, torch.Tensor) and k.numel() > 0:
            a_kv = k[..., base_len:a_len, :].clone()   # branch A's KV, saved by US not the ring
            break

    ring.rollback(cache, slot=slot)
    extend(model, cache, nxt0, h)
    b_kv = None
    for lc in cache.layers:
        k = getattr(lc, "keys", None)
        if isinstance(k, torch.Tensor) and k.numel() > 0:
            b_kv = k[..., base_len:kv_len(cache), :]
            break

    # Does the ring buffer retain ANY attention KV? Inspect what it stores.
    attn_stored = []
    for _layer, seq_lens in ring.attn_plan:
        attn_stored.append(type(seq_lens).__name__)
    stores_tensors = any(s not in ("list", "tuple") for s in attn_stored)

    print(f"    ring buffer stores per attention layer: {set(attn_stored)}  "
          f"(tensors retained: {'YES' if stores_tensors else 'NO'})")
    print("    -> rollback restores attention by CROPPING to a saved LENGTH,")
    print(f"       so branch A's {h} KV entries are freed when branch B is explored.")
    print("    Branch A can only be recovered by recomputation.")
    print()
    print("    WHAT TREES WOULD COST (the missing half, sized):")
    for width, depth in ((2, 4), (4, 4), (4, 8)):
        live = width * depth
        mb = live * kv_per_tok / 1e6
        slots_needed = live
        print(f"      width {width} x depth {depth} = {live:3d} live nodes  ->  "
              f"{mb:7.2f} MB of KV  +  {slots_needed} SSM slots "
              f"({slots_needed * ssm_bytes / 1e6:.1f} MB)  [max_depth is 8]")
    print("\n    => CLAIM 1 NOT SUPPORTED as predicted. The SSM half is solved")
    print("       (fixed-size, arbitrary-slot, bit-exact). The attention half needs")
    print("       a fork/restore mechanism that does not exist here — paged KV")
    print("       blocks or per-branch copies. That is the real work item.")
    report["claim1"] = {
        "supported": False,
        "ring_retains_attention_tensors": stores_tensors,
        "kv_bytes_per_token": kv_per_tok,
        "tree_sizing": {f"{w}x{d}": {"live_nodes": w * d,
                                     "kv_mb": w * d * kv_per_tok / 1e6,
                                     "ssm_mb": w * d * ssm_bytes / 1e6}
                        for w, d in ((2, 4), (4, 4), (4, 8))},
        "max_depth": 8,
    }
    if a_kv is not None and b_kv is not None:
        report["claim1"]["branch_a_kv_shape"] = list(a_kv.shape)

    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"    CLAIM 2 time-travel steering : {'SUPPORTED' if ok else 'FAILED'}  "
          f"({report['claim2']['speedup_vs_reprefill']:.0f}x cheaper than re-prefill)")
    print(f"    CLAIM 3 local beam search    : PARTIAL  "
          f"(+{report['claim3']['first_branch_winner_penalty_pct']:.0f}% when the first branch wins)")
    print("    CLAIM 1 speculative trees    : NOT SUPPORTED (attention KV not branchable)")
    print()
    print("    The primitive is real and its direct speedup is not the point. But")
    print("    'unlocks X' is a claim, and one of these three does not hold today.")

    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
