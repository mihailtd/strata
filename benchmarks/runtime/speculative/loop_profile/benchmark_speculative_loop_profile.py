"""Where does the speculative loop actually spend its time — and what could a fused kernel win?

THE QUESTION
------------
"Fused Speculative Verification (SonicSampler-style)" sits on the roadmap as a
1–2 week ROCm/Triton project, justified by the claim that standard speculative
decoding "wastes critical time bouncing draft tokens between CPU and GPU". That
claim has never been measured on this stack. If the bounce is 3% of the loop,
fusing it wins 3% and the project is dead on arrival; if it is 30%, it is the
best remaining item on the board.

This profiles the real loop from `benchmark_mtp_indomain_speculation_matrix.py`
phase by phase and reports the Amdahl ceiling for each fusable component, so the
decision is sized before any kernel is written.

THE SPECIFIC SUSPECT
--------------------
Verification currently reads accept-length back through Python:

    for i in range(k):
        if draft[0, i].item() == target[i].item():   # <-- 2 GPU->CPU syncs per token
            n_acc += 1

At K=4 that is up to **8 synchronising reads per speculative step**, each one
draining the pipeline. The same result is computable in one GPU op with a single
sync:

    match = (draft[0, :k] == target[:k]).int()
    n_acc = int(torch.cumprod(match, 0).sum())      # 1 sync

Both are implemented and timed here, so the "partial fusion" win is a MEASUREMENT
rather than a projection. The remaining gap to a fully fused kernel (zero syncs,
argmax + compare + commit in one launch) is then reported as an Amdahl bound.

READING THE NUMBERS
-------------------
Per-phase timing requires a `cuda.synchronize()` around each phase, which itself
costs time and inflates the total. Both totals are reported — instrumented and
clean — along with the inflation factor, so the phase percentages are known to be
shares of an inflated whole rather than silently treated as wall clock.

    uv run --env-file .env python \
        benchmarks/runtime/speculative/loop_profile/benchmark_speculative_loop_profile.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

if torch.cuda.is_available():  # fla's device probe is @cache'd at import
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.mtp_draft import (  # noqa: E402
    Qwen35MTPDraftHead,
    restore_state,
    snapshot_state,
)
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

PROMPT_FILE = "data/astral/evaluation_data.jsonl"
ADAPTER = "results/adapters/m2_astral_r8a128"

# Phases attributed to a fused-verification kernel. argmax over a 248320-vocab
# logit tensor is a real kernel, not bookkeeping, so it is listed separately:
# a fused kernel absorbs it, but it does not vanish.
FUSABLE = ["argmax", "accept_calc"]


class Timer:
    """Per-phase accumulator. Every phase is synchronised, so totals inflate."""

    def __init__(self, enabled: bool = True):
        self.t: dict[str, float] = defaultdict(float)
        self.n: dict[str, int] = defaultdict(int)
        self.enabled = enabled

    def __call__(self, name: str):
        return _Phase(self, name)

    def add(self, name: str, dt: float) -> None:
        self.t[name] += dt
        self.n[name] += 1


class _Phase:
    def __init__(self, timer: Timer, name: str):
        self.timer, self.name = timer, name

    def __enter__(self):
        if self.timer.enabled:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        if self.timer.enabled:
            torch.cuda.synchronize()
        self.timer.add(self.name, time.perf_counter() - self.t0)
        return False


@torch.no_grad()
def speculative_profiled(
    model, tokenizer, head, prompt: str, n_new: int, k: int,
    timer: Timer, gpu_accept: bool,
) -> tuple[list[int], float, dict]:
    """The speculation matrix's loop, phase-instrumented.

    `gpu_accept` swaps the Python .item() accept loop for a single-sync GPU
    reduction. Everything else is identical, so the difference between the two
    arms is exactly the cost of the CPU round trip.
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    eos = tokenizer.eos_token_id

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    with timer("prompt_prefill"):
        out = model(ids, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    hids = [out.hidden_states[-1]]
    seq = ids
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    pos = ids.shape[1]
    stats = {"steps": 0, "drafted": 0, "accepted": 0, "syncs": 0}
    done = toks[0] == eos

    while len(toks) < n_new and not done:
        with timer("hidden_cat"):
            H = torch.cat(hids, dim=1)
        with timer("draft_prefill"):
            dcache = head.prefill(H, seq)
        with timer("draft_gen"):
            draft = head.draft(H[:, -1:, :], nxt, k=k, start_pos=pos - 1, cache=dcache)
        with timer("snapshot"):
            snap = snapshot_state(cache)
        with timer("chunk_build"):
            chunk = torch.cat([nxt, draft], dim=-1)
        with timer("verify_fwd"):
            out = model(chunk, past_key_values=cache, use_cache=True, output_hidden_states=True)
        with timer("argmax"):
            target = torch.argmax(out.logits[0], -1)

        with timer("accept_calc"):
            if gpu_accept:
                # One GPU reduction, one sync. cumprod over the match mask gives
                # the length of the leading all-match run.
                match = (draft[0, :k] == target[:k]).int()
                n_acc = int(torch.cumprod(match, 0).sum())
                stats["syncs"] += 1
            else:
                n_acc = 0
                for i in range(k):
                    stats["syncs"] += 2
                    if draft[0, i].item() == target[i].item():
                        n_acc += 1
                    else:
                        break

        stats["steps"] += 1
        stats["drafted"] += k
        stats["accepted"] += n_acc

        with timer("commit_build"):
            committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
        if n_acc < k:
            with timer("restore"):
                restore_state(cache, snap)
            with timer("commit_fwd"):
                out = model(committed, past_key_values=cache, use_cache=True,
                            output_hidden_states=True)
            new_h = out.hidden_states[-1]
        else:
            new_h = out.hidden_states[-1][:, : n_acc + 1, :]

        with timer("bookkeeping"):
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
    return toks[:n_new], time.perf_counter() - t_start, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/speculative_loop_profile.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  SPECULATIVE LOOP PROFILE — where the time goes, and the fusion ceiling")
    print("=" * 100)
    print(f"  device={dev}  K={args.k}  tokens={args.tokens}  prompts={args.n_prompts}\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    head = Qwen35MTPDraftHead(model, args.model_name)
    expert = FoldableExpert.from_dir(REPO_ROOT / ADAPTER, "astral")
    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
    engine.activate(expert)

    rows = [json.loads(x) for x in (REPO_ROOT / PROMPT_FILE).read_text().splitlines() if x.strip()]
    prompts = [f"### Question:\n{r['prompt']}\n\n### Answer:\n" for r in rows[: args.n_prompts]]

    warm = Timer(enabled=False)
    _ = speculative_profiled(model, tok, head, prompts[0], 8, args.k, warm, False)

    report: dict = {"device": dev, "k": args.k, "tokens": args.tokens,
                    "n_prompts": len(prompts), "arms": {}}

    for arm, gpu_accept in (("cpu_item_loop", False), ("gpu_single_sync", True)):
        timer = Timer(enabled=True)
        clean_total = 0.0
        tot_stats = defaultdict(int)
        for p in prompts:
            _, dt, st = speculative_profiled(
                model, tok, head, p, args.tokens, args.k, timer, gpu_accept
            )
            clean_total += dt
            for kk, vv in st.items():
                tot_stats[kk] += vv

        # Uninstrumented run: the honest wall clock, with no per-phase syncs.
        noinstr = Timer(enabled=False)
        t0 = time.perf_counter()
        for p in prompts:
            _ = speculative_profiled(model, tok, head, p, args.tokens, args.k, noinstr, gpu_accept)
        true_total = time.perf_counter() - t0

        instrumented = sum(timer.t.values())
        print(f"\n{'=' * 100}\n  ARM: {arm}\n{'=' * 100}")
        print(f"  steps={tot_stats['steps']}  tau={tot_stats['accepted'] / max(1, tot_stats['steps']):.2f}"
              f"  GPU->CPU syncs in accept={tot_stats['syncs']}"
              f"  ({tot_stats['syncs'] / max(1, tot_stats['steps']):.1f}/step)")
        print(f"  instrumented total {instrumented:.2f} s   clean total {true_total:.2f} s   "
              f"instrumentation inflation {instrumented / max(1e-9, true_total):.2f}x")
        print(f"\n  {'phase':<18}{'total s':>10}{'% instr':>10}{'calls':>9}{'ms/call':>10}")
        print("  " + "-" * 96)
        phases = {}
        for name in sorted(timer.t, key=lambda x: -timer.t[x]):
            pct = 100.0 * timer.t[name] / instrumented
            per = 1000.0 * timer.t[name] / max(1, timer.n[name])
            mark = "  <- fusable" if name in FUSABLE else ""
            print(f"  {name:<18}{timer.t[name]:>10.3f}{pct:>9.1f}%{timer.n[name]:>9}{per:>10.3f}{mark}")
            phases[name] = {"total_s": timer.t[name], "pct_instrumented": pct,
                            "calls": timer.n[name], "ms_per_call": per}

        fus = sum(timer.t[n] for n in FUSABLE if n in timer.t)
        fus_pct = 100.0 * fus / instrumented
        print(f"\n  fusable ({'+'.join(FUSABLE)}): {fus:.3f} s = {fus_pct:.1f}% of instrumented")
        print(f"  Amdahl ceiling if fusion made them FREE: {1 / (1 - fus_pct / 100):.3f}x")

        report["arms"][arm] = {
            "phases": phases, "instrumented_total_s": instrumented,
            "clean_total_s": true_total,
            "inflation": instrumented / max(1e-9, true_total),
            "stats": dict(tot_stats),
            "fusable_pct": fus_pct,
            "amdahl_ceiling_if_fusable_free": 1 / (1 - fus_pct / 100),
        }

    a, b = report["arms"]["cpu_item_loop"], report["arms"]["gpu_single_sync"]
    win = 100.0 * (a["clean_total_s"] - b["clean_total_s"]) / a["clean_total_s"]
    print("\n" + "=" * 100)
    print("  MEASURED: what removing the CPU round trip actually buys")
    print("=" * 100)
    print(f"    accept via Python .item() loop : {a['clean_total_s']:.2f} s  "
          f"({a['stats']['syncs']} syncs)")
    print(f"    accept via one GPU reduction   : {b['clean_total_s']:.2f} s  "
          f"({b['stats']['syncs']} syncs)")
    print(f"    -> {win:+.2f}% wall clock, for a 4-line change and no kernel")
    print(f"\n    Remaining fusable share (argmax + accept) in the GPU arm: {b['fusable_pct']:.1f}%")
    print(f"    A fully fused kernel making that FREE caps at {b['amdahl_ceiling_if_fusable_free']:.3f}x"
          " on the loop,")
    print("    before the prefill share of a real agent turn is applied on top.")
    report["measured_cpu_roundtrip_win_pct"] = win

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
