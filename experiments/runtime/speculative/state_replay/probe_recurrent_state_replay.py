"""Can a recurrent state replay replace the 34.96 ms `commit_fwd` re-forward?

THE PROBLEM
-----------
`commit_fwd` is **29.4% of the speculative loop** (`docs/DECISIONS.md` §10). On a
partial accept the chunked scan has already advanced the recurrent state past the
rejected tokens, so the loop restores a 52.5 MB snapshot and runs a FULL MODEL
FORWARD over the committed prefix purely to re-derive that state — 34.96 ms on 62
of 89 steps at tau=2.02.

WHY NOT JUST ASK fla FOR THE INTERMEDIATE STATE
-----------------------------------------------
It cannot provide one, structurally. `chunk_gated_delta_rule` returns
`(o, final_state)` with `final_state` of shape `[N, HV, K, V]` — exactly one
state — and its `chunk_size` is **64**, so a K+1 = 5 token verification window
contains ZERO chunk boundaries. Chunked scans compute a block's aggregate state
transition and deliberately never materialise per-position states; that is what
chunking is. Materialising them would be re-deriving the sequential recurrence.

THE IDEA UNDER TEST
-------------------
`commit_fwd` recomputes the whole stack to recover ONE tensor per GatedDeltaNet
layer. The other things it produces are cheap by other means: attention KV entries
are truncatable, hidden states are a slice. And `fused_recurrent_gated_delta_rule`
takes the SAME arguments as the chunked call — `(q, k, v, g=, beta=,
initial_state=, output_final_state=, use_qk_l2norm_in_kernel=True)` — so the
committed prefix can be replayed through the recurrent kernel from the snapshot,
using only the scan rather than a full forward.

TWO THINGS DECIDE IT, AND BOTH ARE MEASURED HERE
    1. COST.      replay time vs the full re-forward it replaces.
    2. NUMERICS.  does the replayed state match the re-forwarded one? fla's
                  chunked and recurrent kernels are known to disagree on this
                  stack — that disagreement is the source of the 83.3%
                  non-speculative exact-match control — so this is the risk, not
                  an afterthought. A cheap replay that drifts is worthless.

Reusing the projections captured during the chunked verify is causally valid:
q/k/v/g/beta at position i are per-position linear functions of the layer input at
i, and position i depends only on positions <= i. Any difference is numerical, and
measuring that difference is the point.

    uv run --env-file .env python \
        benchmarks/runtime/speculative/state_replay/probe_recurrent_state_replay.py
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

from runtime.mtp_draft import restore_state, snapshot_state  # noqa: E402
from runtime.novel_peft import set_hard_vram_cap  # noqa: E402

PROMPT = "### Question:\nHow should I install external Python packages for a new project?\n\n### Answer:\n"


def find_gdn_layers(model) -> list:
    return [m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=4, help="draft length; chunk is k+1 wide")
    ap.add_argument("--n-acc", type=int, default=2, help="accepted drafts; tau measures ~2.0")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/recurrent_state_replay.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  RECURRENT STATE REPLAY vs FULL commit_fwd RE-FORWARD")
    print("=" * 100)
    print(f"  device={dev}  K={args.k}  n_acc={args.n_acc}  (commit prefix = {args.n_acc + 1} tokens)\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()

    gdn = find_gdn_layers(model)
    print(f"  GatedDeltaNet layers: {len(gdn)}")
    if not gdn:
        raise RuntimeError("no Qwen3_5GatedDeltaNet layers found")
    if gdn[0].chunk_gated_delta_rule is None:
        raise RuntimeError("chunk_gated_delta_rule is None — fla not bound; aborting")

    # --- capture the chunked call's inputs, per layer ------------------------
    captured: dict[int, dict] = {}

    def wrap(layer, idx):
        orig = layer.chunk_gated_delta_rule

        def timed(q, k, v, g=None, beta=None, initial_state=None,
                  output_final_state=False, **kw):
            captured[idx] = {
                "q": q, "k": k, "v": v, "g": g, "beta": beta,
                "initial_state": initial_state,
            }
            return orig(q, k, v, g=g, beta=beta, initial_state=initial_state,
                        output_final_state=output_final_state, **kw)

        layer.chunk_gated_delta_rule = timed
        return orig

    originals = [wrap(m, i) for i, m in enumerate(gdn)]
    recurrent_fn = gdn[0].recurrent_gated_delta_rule

    ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)

    # a K+1 wide verification chunk, exactly as the speculative loop builds it
    chunk = torch.cat([nxt] + [nxt for _ in range(args.k)], dim=-1)
    snap = snapshot_state(cache)

    captured.clear()
    with torch.no_grad():
        model(chunk, past_key_values=cache, use_cache=True)
    print(f"  captured chunked-scan inputs from {len(captured)} layers")
    n_cap = len(captured)
    if n_cap != len(gdn):
        print(f"  ⚠️  only {n_cap}/{len(gdn)} layers captured — some took the recurrent path")

    m = args.n_acc + 1  # committed prefix length

    # --- ARM A: the current full-model re-forward ---------------------------
    def arm_a():
        restore_state(cache, snap)
        with torch.no_grad():
            model(chunk[:, :m], past_key_values=cache, use_cache=True)

    restore_state(cache, snap)
    with torch.no_grad():
        model(chunk[:, :m], past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    for _ in range(3):
        arm_a()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        arm_a()
    torch.cuda.synchronize()
    a_ms = 1000.0 * (time.perf_counter() - t0) / args.repeats

    # --- ARM B: recurrent replay of the committed prefix only ---------------
    def arm_b() -> list:
        outs = []
        for i in range(n_cap):
            c = captured[i]
            s, fs = recurrent_fn(
                c["q"][:, :m], c["k"][:, :m], c["v"][:, :m],
                g=c["g"][:, :m] if c["g"] is not None else None,
                beta=c["beta"][:, :m] if c["beta"] is not None else None,
                initial_state=c["initial_state"],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            outs.append(fs)
        return outs

    with torch.no_grad():
        for _ in range(3):
            arm_b()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.repeats):
            states_b = arm_b()
        torch.cuda.synchronize()
        b_ms = 1000.0 * (time.perf_counter() - t0) / args.repeats

    # --- ARM C: the reference — chunked scan over the committed prefix -------
    # This is what a full re-forward's scan produces for the same prefix, so it is
    # the state ARM A installs. Comparing B against C isolates recurrent-vs-chunked
    # numerics from everything else in the forward pass.
    with torch.no_grad():
        states_c = []
        for i in range(n_cap):
            c = captured[i]
            _, fs = originals[i](
                c["q"][:, :m], c["k"][:, :m], c["v"][:, :m],
                g=c["g"][:, :m] if c["g"] is not None else None,
                beta=c["beta"][:, :m] if c["beta"] is not None else None,
                initial_state=c["initial_state"],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            states_c.append(fs)

    print("\n" + "=" * 100)
    print("  1. COST")
    print("=" * 100)
    print(f"    ARM A  full model re-forward over {m} tokens : {a_ms:8.2f} ms")
    print(f"    ARM B  recurrent replay, {n_cap} layers only  : {b_ms:8.2f} ms")
    print(f"    -> replay is {a_ms / max(1e-9, b_ms):.1f}x cheaper, saving {a_ms - b_ms:.2f} ms per partial accept")

    print("\n" + "=" * 100)
    print("  2. NUMERICS — replayed state vs the chunked state it must match")
    print("=" * 100)
    worst_rel = 0.0
    rows = []
    for i, (b, c) in enumerate(zip(states_b, states_c, strict=True)):
        if b is None or c is None:
            continue
        d = (b.float() - c.float()).abs()
        denom = c.float().abs().max().clamp_min(1e-12)
        rel = (d.max() / denom).item()
        worst_rel = max(worst_rel, rel)
        rows.append({"layer": i, "max_abs": d.max().item(), "rel": rel})
    best = sorted(rows, key=lambda r: -r["rel"])[:5]
    print(f"    layers compared: {len(rows)}")
    for r in best:
        print(f"      layer {r['layer']:<3} max|Δ| {r['max_abs']:.3e}   relative {r['rel']:.3e}")
    print(f"\n    WORST relative divergence across all layers: {worst_rel:.3e}")
    if worst_rel < 1e-2:
        print("    => within bf16 slack. The replay reproduces the chunked state.")
    else:
        print("    => DIVERGENT. A cheap replay that drifts is worthless; this kills the idea.")

    speculative_loop_share = 0.294
    saving = (a_ms - b_ms) / max(1e-9, a_ms) * speculative_loop_share
    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)
    print(f"    commit_fwd is {100 * speculative_loop_share:.1f}% of the speculative loop.")
    print(f"    Replacing it at this cost ratio removes {100 * saving:.1f}% of the loop")
    print(f"    -> loop speedup ceiling {1 / (1 - saving):.3f}x")
    print("    Serving path only, and speculation is not wired into server.py today.")

    report = {
        "device": dev, "k": args.k, "n_acc": args.n_acc, "commit_prefix": m,
        "layers": n_cap,
        "arm_a_full_reforward_ms": a_ms,
        "arm_b_recurrent_replay_ms": b_ms,
        "speedup": a_ms / max(1e-9, b_ms),
        "worst_relative_divergence": worst_rel,
        "per_layer": rows,
        "loop_speedup_ceiling": 1 / (1 - saving),
    }
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
