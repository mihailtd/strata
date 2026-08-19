"""Skip `commit_fwd` entirely: replay the state instead of re-forwarding the stack.

THE SURGERY
-----------
On a partial accept the baseline loop restores a 52.5 MB snapshot and runs a FULL
MODEL FORWARD over the committed prefix (34.23 ms, on 70% of steps) purely to
re-derive state the chunked scan has already advanced past. This arm skips that
forward and repairs the cache directly.

`commit_fwd` advances THREE pieces of state, and each has a cheaper route:

    recurrent SSM state   replay the committed prefix through
                          `fused_recurrent_gated_delta_rule` from the pre-chunk
                          state, reusing the projections the chunked verify
                          already computed.  (~1.23 ms for all 24 layers)

    conv state            the short conv's rolling window. `update_conv_state`
                          stores the last `conv_kernel_size` columns of
                          [cached_conv_state ++ new inputs], so the state after m
                          tokens is recoverable by slicing the SAME concatenated
                          tensor the conv consumed. Captured with a pre-hook on
                          `layer.conv1d`, whose input is exactly that tensor.
                          ⚠️ This one is easy to forget — the first version of
                          this work accounted only for the recurrent state.

    attention KV          `DynamicLayer.crop()`. The chunked forward appended
                          K+1 entries; drop the last (K - n_acc).

Hidden states are sliced from the chunked forward's own output, which is causally
valid: position i depends only on positions <= i.

WHY TOKEN EQUALITY IS NOT THE PASS CRITERION
--------------------------------------------
This stack changes text on 33% of prompts from kernel numerics ALONE, with no
measurable quality cost (+0.19pp, CI [-0.77, +1.17], §7). So "the text changed" is
not by itself a defect here. This reports token equality as a diagnostic and
leaves the verdict to the quality harness if the arms diverge.

    uv run --env-file .env python \
        benchmarks/runtime/speculative/state_replay/benchmark_commit_fwd_surgery.py
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

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
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


class Capture:
    """Per-layer capture of everything the surgery needs from the chunked verify."""

    def __init__(self):
        self.scan: dict[int, dict] = {}
        self.conv: dict[int, torch.Tensor] = {}

    def clear(self):
        self.scan.clear()
        self.conv.clear()


def install(model, cap: Capture) -> list:
    gdn = [m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]
    for i, layer in enumerate(gdn):
        orig = layer.chunk_gated_delta_rule

        def wrapped(q, k, v, g=None, beta=None, initial_state=None,
                    output_final_state=False, _i=i, _o=orig, **kw):
            cap.scan[_i] = {"q": q, "k": k, "v": v, "g": g, "beta": beta,
                            "initial_state": initial_state}
            return _o(q, k, v, g=g, beta=beta, initial_state=initial_state,
                      output_final_state=output_final_state, **kw)

        layer.chunk_gated_delta_rule = wrapped
        # conv1d's INPUT is [cached_conv_state ++ new inputs] -- exactly the tensor
        # update_conv_state slices its window from.
        layer.conv1d.register_forward_pre_hook(
            lambda m, inp, _i=i: cap.conv.__setitem__(_i, inp[0])
        )
    return gdn


def repair(cache, gdn: list, cap: Capture, m: int, k: int) -> None:
    """Install the state for a committed prefix of length m, without a forward."""
    for li, layer in enumerate(gdn):
        ksz = layer.conv_kernel_size
        cat = cap.conv.get(li)
        if cat is not None and cat.shape[-1] >= ksz + m:
            cache.update_conv_state(cat[:, :, : ksz + m][:, :, -ksz:].contiguous(),
                                    layer.layer_idx)
        c = cap.scan.get(li)
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
        if fs is not None:
            cache.update_recurrent_state(fs, layer.layer_idx)

    drop = k - (m - 1)
    if drop > 0:
        for lc in cache.layers:
            keys = getattr(lc, "keys", None)
            if isinstance(keys, torch.Tensor) and keys.numel() > 0 and hasattr(lc, "crop"):
                lc.crop(keys.shape[-2] - drop)


@torch.no_grad()
def run(model, tok, head, prompt: str, n_new: int, k: int, surgery: bool,
        gdn: list, cap: Capture) -> tuple[list[int], float, dict]:
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
    stats = {"steps": 0, "accepted": 0, "partial": 0}
    done = toks[0] == eos

    while len(toks) < n_new and not done:
        H = torch.cat(hids, dim=1)
        dcache = head.prefill(H, seq)
        draft = head.draft(H[:, -1:, :], nxt, k=k, start_pos=pos - 1, cache=dcache)

        snap = None if surgery else snapshot_state(cache)
        cap.clear()
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
        stats["accepted"] += n_acc

        committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
        if n_acc < k:
            stats["partial"] += 1
            if surgery:
                repair(cache, gdn, cap, n_acc + 1, k)
                new_h = out.hidden_states[-1][:, : n_acc + 1, :]
            else:
                restore_state(cache, snap)
                out = model(committed, past_key_values=cache, use_cache=True,
                            output_hidden_states=True)
                new_h = out.hidden_states[-1]
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
    return toks[:n_new], time.perf_counter() - t0, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/commit_fwd_surgery.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  SKIPPING commit_fwd — state repair vs full re-forward")
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
    WeightFoldingEngine(model, [expert], keep_pristine=True).activate(expert)

    cap = Capture()
    gdn = install(model, cap)
    print(f"  GatedDeltaNet layers instrumented: {len(gdn)}  "
          f"(conv_kernel_size={gdn[0].conv_kernel_size})\n")

    rows = [json.loads(x) for x in (REPO_ROOT / PROMPTS_FILE).read_text().splitlines() if x.strip()]
    prompts = [f"### Question:\n{r['prompt']}\n\n### Answer:\n" for r in rows[: args.n_prompts]]

    for s in (False, True):
        run(model, tok, head, prompts[0], 8, args.k, s, gdn, cap)

    res = {}
    for name, surgery in (("baseline", False), ("surgery", True)):
        txts, total, st = [], 0.0, {"steps": 0, "accepted": 0, "partial": 0}
        for p in prompts:
            t, dt, s = run(model, tok, head, p, args.tokens, args.k, surgery, gdn, cap)
            txts.append(t)
            total += dt
            for kk in st:
                st[kk] += s[kk]
        res[name] = {"texts": txts, "s": total, "stats": st}
        tau = st["accepted"] / max(1, st["steps"])
        print(f"  {name:<10} {total:6.2f} s   tau={tau:.3f}   "
              f"partial accepts {st['partial']}/{st['steps']} "
              f"({100 * st['partial'] / max(1, st['steps']):.0f}%)")

    b, s = res["baseline"], res["surgery"]
    speedup = b["s"] / max(1e-9, s["s"])
    exact = sum(x == y for x, y in zip(b["texts"], s["texts"], strict=True))

    print("\n" + "=" * 100)
    print("  RESULT")
    print("=" * 100)
    tb = b["stats"]["accepted"] + b["stats"]["steps"]
    ts = s["stats"]["accepted"] + s["stats"]["steps"]
    tps_b, tps_s = tb / max(1e-9, b["s"]), ts / max(1e-9, s["s"])
    # ⚠️ WALL CLOCK ALONE IS MISLEADING HERE. The arms do not emit the same number
    # of tokens: the surgery arm hits EOS earlier and generated 37% fewer tokens in
    # the first full run, which made a 1.008x throughput change read as "1.605x".
    # Normalise by tokens or the headline is a length artefact, not a speedup.
    print(f"    tokens emitted   {tb} -> {ts}  ({100 * (ts - tb) / max(1, tb):+.1f}%)")
    print(f"    THROUGHPUT       {tps_b:.2f} -> {tps_s:.2f} tok/s  =  {tps_s / tps_b:.3f}x   <- the real number")
    print(f"    per-step time    {1000 * b['s'] / max(1, b['stats']['steps']):.1f} -> "
          f"{1000 * s['s'] / max(1, s['stats']['steps']):.1f} ms  =  "
          f"{(b['s'] / max(1, b['stats']['steps'])) / (s['s'] / max(1, s['stats']['steps'])):.3f}x")
    print(f"    raw wall clock   {b['s']:.2f} s -> {s['s']:.2f} s  =  {speedup:.3f}x  "
          "(NOT a speedup if token counts differ)")
    print(f"    identical token sequences: {exact}/{len(prompts)}")
    tau_b = b["stats"]["accepted"] / max(1, b["stats"]["steps"])
    tau_s = s["stats"]["accepted"] / max(1, s["stats"]["steps"])
    print(f"    tau  baseline {tau_b:.3f}  surgery {tau_s:.3f}")
    if abs(tau_s - tau_b) > 0.15:
        print("    ⚠️  tau MOVED — the repaired state is changing draft acceptance,")
        print("        which means the states are not equivalent.")
    print("\n    Token equality is a diagnostic, NOT the pass criterion: this stack")
    print("    already changes text on 33% of prompts from kernel numerics alone with")
    print("    no measurable quality cost. If the arms differ, the quality harness decides.")

    report = {
        "device": dev, "k": args.k, "tokens": args.tokens, "n_prompts": len(prompts),
        "baseline_s": b["s"], "surgery_s": s["s"],
        "raw_wall_clock_ratio_MISLEADING": speedup,
        "tokens_baseline": tb, "tokens_surgery": ts,
        "throughput_tok_s_baseline": tps_b, "throughput_tok_s_surgery": tps_s,
        "throughput_speedup": tps_s / tps_b,
        "identical_sequences": exact,
        "tau_baseline": tau_b, "tau_surgery": tau_s,
        "baseline_stats": b["stats"], "surgery_stats": s["stats"],
    }
    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
