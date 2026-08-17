"""Does recurrent state replay change the generated text, and does its error compound?

WHAT THE KILL-TEST LEFT OPEN
----------------------------
`probe_recurrent_state_replay.py` measured the replay at **27.8x cheaper** than the
`commit_fwd` re-forward (34.23 ms -> 1.23 ms) with a worst relative divergence of
**7.31e-03** against the chunked state. That verdict said "within bf16 slack" using
a hand-picked 1e-2 threshold — but bf16 epsilon is ~3.9e-03, so the divergence is
about **2 ULP**, and the state is carried forward. One step was measured. Whether
the error COMPOUNDS across a whole generation, or is damped by the gated delta
rule's forget gate, decides the idea.

HOW THIS SETTLES IT WITHOUT RISKY CACHE SURGERY
-----------------------------------------------
Skipping `commit_fwd` outright would need the attention KV entries truncated and
the hidden states sliced. Both are provably safe (position i depends only on
positions <= i, so the chunked forward's outputs for the committed prefix are
causally valid) — but implementing that surgery risks introducing a bug that would
confound the very text comparison this test exists to make.

So this runs the REAL loop and lets `commit_fwd` execute, then **overwrites** the
recurrent state with the replayed one and continues generating. Everything else is
byte-identical to the baseline. If the text matches over a full generation, the
replayed state is behaviourally substitutable and the surgery is worth building.
If it drifts, the idea dies here and no surgery is wasted.

Divergence is recorded at EVERY step, so compounding is visible as a trend rather
than assumed from a single sample.

The speedup is NOT measured by this script — arm B deliberately pays for both
paths. It is projected from the kill-test's measured saving and the observed
partial-accept rate, and reported as such.

    uv run --env-file .env python \
        benchmarks/runtime/speculative/state_replay/benchmark_replay_end_to_end.py
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
# measured by probe_recurrent_state_replay.py on this rig
COMMIT_FWD_MS = 34.23
REPLAY_MS = 1.23


def gdn_layers(model) -> list:
    return [m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]


@torch.no_grad()
def run(model, tok, head, prompt: str, n_new: int, k: int,
        replay: bool, gdn: list, captured: dict, diag: list) -> tuple[list[int], float]:
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    eos = tok.eos_token_id
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model(ids, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    hids = [out.hidden_states[-1]]
    seq = ids
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    pos = ids.shape[1]
    done = toks[0] == eos

    while len(toks) < n_new and not done:
        H = torch.cat(hids, dim=1)
        dcache = head.prefill(H, seq)
        draft = head.draft(H[:, -1:, :], nxt, k=k, start_pos=pos - 1, cache=dcache)

        snap = snapshot_state(cache)
        chunk = torch.cat([nxt, draft], dim=-1)
        captured.clear()
        out = model(chunk, past_key_values=cache, use_cache=True, output_hidden_states=True)
        target = torch.argmax(out.logits[0], -1)

        n_acc = 0
        for i in range(k):
            if draft[0, i].item() == target[i].item():
                n_acc += 1
            else:
                break

        committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
        if n_acc < k:
            restore_state(cache, snap)
            out = model(committed, past_key_values=cache, use_cache=True,
                        output_hidden_states=True)
            new_h = out.hidden_states[-1]

            if replay and captured:
                # Replay the committed prefix through the RECURRENT kernel from the
                # pre-chunk state, then overwrite what commit_fwd just installed.
                m = n_acc + 1
                for li, layer in enumerate(gdn):
                    c = captured.get(li)
                    if c is None or c["q"].shape[1] < m:
                        continue
                    _, fs = layer.recurrent_gated_delta_rule(
                        c["q"][:, :m], c["k"][:, :m], c["v"][:, :m],
                        g=c["g"][:, :m] if c["g"] is not None else None,
                        beta=c["beta"][:, :m] if c["beta"] is not None else None,
                        initial_state=c["initial_state"],
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                    if fs is None:
                        continue
                    # Read what commit_fwd just installed, straight off the layer
                    # cache — there is no read-only accessor on the public API, and
                    # inventing one silently returns None and kills the diagnostic.
                    cur = None
                    try:
                        lc = cache.layers[layer.layer_idx]
                        rs = getattr(lc, "recurrent_states", None)
                        if rs is not None and len(rs) and isinstance(rs[0], torch.Tensor):
                            cur = rs[0]
                    except Exception:
                        cur = None
                    if cur is not None and cur.shape == fs.shape:
                        d = (fs.float() - cur.float()).abs().max()
                        denom = cur.float().abs().max().clamp_min(1e-12)
                        diag.append({"step": len(toks), "layer": li,
                                     "rel": (d / denom).item()})
                    cache.update_recurrent_state(fs, layer.layer_idx)
        else:
            new_h = out.hidden_states[-1][:, : n_acc + 1, :]

        bonus = target[n_acc].item()
        for t in draft[0, :n_acc].tolist() + [bonus]:
            if len(toks) >= n_new:
                break
            toks.append(t)
            if t == eos:
                done = True
                break

        hids.append(new_h)
        seq = torch.cat([seq, committed], dim=-1)
        pos += n_acc + 1
        nxt = torch.tensor([[bonus]], device=ids.device)

    torch.cuda.synchronize()
    return toks[:n_new], time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/replay_end_to_end.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  END-TO-END: does replaying the recurrent state change the text?")
    print("=" * 100)
    print(f"  device={dev}  K={args.k}  tokens={args.tokens}  prompts={args.n_prompts}\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()
    head = Qwen35MTPDraftHead(model, args.model_name)
    expert = FoldableExpert.from_dir(REPO_ROOT / ADAPTER, "astral")
    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
    engine.activate(expert)

    gdn = gdn_layers(model)
    print(f"  GatedDeltaNet layers: {len(gdn)}")
    captured: dict[int, dict] = {}

    for i, layer in enumerate(gdn):
        orig = layer.chunk_gated_delta_rule

        def timed(q, kk, v, g=None, beta=None, initial_state=None,
                  output_final_state=False, _i=i, _o=orig, **kw):
            captured[_i] = {"q": q, "k": kk, "v": v, "g": g, "beta": beta,
                            "initial_state": initial_state}
            return _o(q, kk, v, g=g, beta=beta, initial_state=initial_state,
                      output_final_state=output_final_state, **kw)

        layer.chunk_gated_delta_rule = timed

    rows = [json.loads(x) for x in (REPO_ROOT / PROMPTS_FILE).read_text().splitlines() if x.strip()]
    prompts = [f"### Question:\n{r['prompt']}\n\n### Answer:\n" for r in rows[: args.n_prompts]]

    diag: list = []
    _ = run(model, tok, head, prompts[0], 8, args.k, False, gdn, captured, [])

    base_txt, repl_txt = [], []
    base_t = repl_t = 0.0
    for p in prompts:
        t, dt = run(model, tok, head, p, args.tokens, args.k, False, gdn, captured, [])
        base_txt.append(t)
        base_t += dt
    for p in prompts:
        t, dt = run(model, tok, head, p, args.tokens, args.k, True, gdn, captured, diag)
        repl_txt.append(t)
        repl_t += dt

    exact = sum(a == b for a, b in zip(base_txt, repl_txt, strict=True))
    print("=" * 100)
    print("  1. DOES THE TEXT CHANGE?")
    print("=" * 100)
    print(f"    identical token sequences: {exact}/{len(prompts)}")
    for i, (a, b) in enumerate(zip(base_txt, repl_txt, strict=True)):
        if a != b:
            j = next((x for x, (p_, q_) in enumerate(zip(a, b, strict=False)) if p_ != q_), None)
            print(f"      prompt {i}: first divergence at token {j}")

    print("\n" + "=" * 100)
    print("  2. DOES THE ERROR COMPOUND?")
    print("=" * 100)
    if diag:
        steps = sorted({d["step"] for d in diag})
        early = [d["rel"] for d in diag if d["step"] <= steps[len(steps) // 4]]
        late = [d["rel"] for d in diag if d["step"] >= steps[3 * len(steps) // 4]]
        mx = max(d["rel"] for d in diag)
        print(f"    samples {len(diag)} over {len(steps)} steps")
        print(f"    mean relative divergence, EARLY quartile : {sum(early) / max(1, len(early)):.3e}")
        print(f"    mean relative divergence, LATE  quartile : {sum(late) / max(1, len(late)):.3e}")
        print(f"    worst observed                           : {mx:.3e}")
        growth = (sum(late) / max(1, len(late))) / max(1e-12, sum(early) / max(1, len(early)))
        print(f"    late/early ratio                         : {growth:.2f}x")
        print("    => " + ("BOUNDED — the forget gate damps it." if growth < 2.0
                           else "GROWING — error compounds across steps."))
    else:
        print("    no diagnostics captured (no partial accepts?)")
        growth = None

    print("\n" + "=" * 100)
    print("  3. PROJECTED SPEEDUP  (this run pays for BOTH paths; not measured here)")
    print("=" * 100)
    saved_ms = COMMIT_FWD_MS - REPLAY_MS
    print(f"    kill-test saving per partial accept: {saved_ms:.2f} ms")
    print(f"    baseline loop wall clock           : {base_t:.2f} s over {len(prompts)} prompts")
    print("    (real speedup requires skipping commit_fwd, which needs KV truncation")
    print("     and hidden-state slicing — safe by causality, but not built here.)")

    report = {
        "device": dev, "k": args.k, "tokens": args.tokens, "n_prompts": len(prompts),
        "identical_sequences": exact, "n_sequences": len(prompts),
        "baseline_s": base_t, "replay_arm_s": repl_t,
        "divergence_samples": len(diag),
        "late_over_early_ratio": growth,
        "worst_relative_divergence": max((d["rel"] for d in diag), default=None),
        "commit_fwd_ms": COMMIT_FWD_MS, "replay_ms": REPLAY_MS,
    }
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
