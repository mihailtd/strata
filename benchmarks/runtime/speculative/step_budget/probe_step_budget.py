"""Where does the 133 ms speculative step actually go, at REALISTIC length?

Measured: the B arm runs 19.91 tok/s at tau=1.65 -> 133.1 ms/step.
Measured components: verify 31.27 + commit 31.78 + draft ~15 = 78 ms.
So ~55 ms/step -- 41% -- is unaccounted, and that is LARGER than the commit
re-forward everyone is aiming at.

§18's lesson is the reason this runs before any optimisation: component
micro-benchmarks do not predict in-loop cost. §18 predicted 52.9 ms/step from a
component saving and measured 71.8, because the repair's real cost only appeared
inside the loop.

Every phase is device-synced, so a phase's time is its own, and head.prefill is
timed separately because it rebuilds a DynamicCache over the WHOLE sequence every
step (O(n) per step, O(n^2) over a generation) and is the prime suspect.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/mihai/gnn-experiment")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

T0 = time.perf_counter()


def stage(m): print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)


import torch  # noqa: E402

torch.zeros(1, device="cuda"); torch.cuda.synchronize()
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gnn_experiment.bucketed_speculative import BucketedSpeculativeDecoder  # noqa: E402
from gnn_experiment.mtp_draft import Qwen35MTPDraftHead  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

MODEL, ADAPTER = "Qwen/Qwen3.5-4B", "results/adapters/m2_astral_r8a128"
K, MAX_SEQ = 4, 2048
N = int(os.environ.get("N", "512"))


def sync_t():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def profile(dec, ids, n_new, stop_ids):
    k = dec.k
    dec.cache.reset()
    cur = ids.shape[1]
    ph = dict.fromkeys(
        ("cat_hids", "head_prefill", "head_draft", "snapshot",
         "verify", "accept_sync", "commit", "bookkeep"), 0.0)
    early: dict[str, float] = {}
    t0 = sync_t()
    out = dec.model(ids, past_key_values=dec.cache,
                    cache_position=torch.arange(0, cur, device=dec.device),
                    use_cache=True, output_hidden_states=True)
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    dec._counter_pos = cur
    hids = [out.hidden_states[-1]]
    seq, toks, pos = ids, [nxt.item()], cur
    steps = accepted = 0
    done = toks[0] in stop_ids

    while len(toks) < n_new and not done:
        a = sync_t()
        H = torch.cat(hids, dim=1)
        b = sync_t(); ph["cat_hids"] += b - a
        dcache = dec.head.prefill(H, seq)
        c = sync_t(); ph["head_prefill"] += c - b
        draft = dec.head.draft(H[:, -1:, :], nxt, k=k, start_pos=pos - 1, cache=dcache)
        d = sync_t(); ph["head_draft"] += d - c
        dec._snapshot_ssm()
        e = sync_t(); ph["snapshot"] += e - d
        chunk = torch.cat([nxt, draft], dim=-1)
        logits, hidden = dec._replay(k + 1, chunk, pos)
        f = sync_t(); ph["verify"] += f - e
        target = torch.argmax(logits[0], -1)
        n_acc = 0
        for i in range(k):
            if draft[0, i].item() == target[i].item():
                n_acc += 1
            else:
                break
        g = sync_t(); ph["accept_sync"] += g - f
        if n_acc < k:
            dec._restore_ssm()
            committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
            _, hidden = dec._replay(n_acc + 1, committed, pos)
            new_h = hidden
        else:
            new_h = hidden[:, : n_acc + 1, :]
            committed = chunk
        h = sync_t(); ph["commit"] += h - g
        steps += 1
        accepted += n_acc
        bonus = target[n_acc].item()
        for t in draft[0, :n_acc].tolist() + [bonus]:
            if len(toks) >= n_new:
                break
            toks.append(t)
            if t in stop_ids:
                done = True
                break
        hids.append(new_h.clone())
        seq = torch.cat([seq, committed], dim=-1)
        pos += n_acc + 1
        nxt = torch.tensor([[bonus]], device=dec.device)
        i2 = sync_t(); ph["bookkeep"] += i2 - h
        if steps == 25:
            early = {kk: v for kk, v in ph.items()}
            early["_steps"] = steps
    total = sync_t() - t0
    return toks, total, ph, steps, accepted, early


def main():
    set_hard_vram_cap(22.0)
    stage("loading")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    head = Qwen35MTPDraftHead(model, MODEL)
    expert = FoldableExpert.from_dir(REPO_ROOT / ADAPTER, "astral")
    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
    engine.activate(expert)
    stop = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid > 0:
            stop.add(tid)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Design a production vector search system for "
          "100M documents. Cover the storage layer, index construction, sharding, the "
          "query path, recall/latency tradeoffs, and how you would benchmark it."}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    dec = BucketedSpeculativeDecoder(model, tok, head, k=K, max_seq_len=MAX_SEQ)
    dec.capture(ids); dec.generate(ids, max_new_tokens=8)
    stage(f"captured; profiling {N} tokens")

    toks, total, ph, steps, acc, early = profile(dec, ids, N, stop)
    tau = acc / max(1, steps)
    stage("=" * 76)
    stage(f"  emitted={len(toks)} steps={steps} tau={tau:.3f} "
          f"{len(toks)/total:.2f} tok/s  step={total/steps*1000:.2f} ms")
    stage(f"  {'phase':16s} {'total ms':>9} {'ms/step':>8} {'% step':>7}")
    tot_ms = total * 1000
    for kk, v in sorted(ph.items(), key=lambda x: -x[1]):
        stage(f"  {kk:16s} {v*1000:9.1f} {v*1000/steps:8.2f} {v/total*100:6.1f}%")
    acct = sum(ph.values())
    stage(f"  {'ACCOUNTED':16s} {acct*1000:9.1f} {acct*1000/steps:8.2f} {acct/total*100:6.1f}%")
    stage(f"  {'prefill+resid':16s} {tot_ms-acct*1000:9.1f} "
          f"{(tot_ms-acct*1000)/steps:8.2f} {(1-acct/total)*100:6.1f}%")
    if early:
        es = early.pop("_steps")
        stage(f"  --- head_prefill growth: first {es} steps "
              f"{early['head_prefill']*1000/es:.2f} ms/step vs whole run "
              f"{ph['head_prefill']*1000/steps:.2f} ms/step ---")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
