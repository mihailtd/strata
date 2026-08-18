"""Lockstep: where does the static head cache first diverge from a rebuilt one?

Established:
  * single-step, fresh cache -> graphed head is BIT-EXACT (max|dlogit| 0.00000)
  * across a generation      -> tau collapses 1.803 -> 0.910

So the fault is cache LIFETIME, not the capture, mask, positions or counters.

This drives ONE trajectory (the reference one, so both sides see identical
inputs) and at every step computes the head's draft logits twice:

    REF     head.prefill() rebuilds the whole head cache from scratch, as shipped
    STATIC  persistent StaticCache: extend(committed) then draft(speculative)

Up to the first divergence both write identical drafts, so the caches are fed
identically and any difference is purely accumulated state. Reports the first
diverging step together with what the cache looked like there -- pos, hfilled,
the previous step's n_acc, and how far the stale speculative tail extended.
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

sys.path.insert(0, str(Path(__file__).parent))
from probe_graphed_head_v2 import GraphedHead  # noqa: E402

MODEL, ADAPTER = "Qwen/Qwen3.5-4B", "results/adapters/m2_astral_r8a128"
K, MAX_SEQ = 4, 2048
N = int(os.environ.get("N", "160"))


@torch.no_grad()
def ref_draft_logits(head, H_all, seq, nxt, k, pos):
    """Shipped path: rebuild the head cache from scratch, then draft k."""
    T = seq.shape[1]
    cache = head.prefill(H_all[:, :T, :], seq)
    h, t2, logs, toks = H_all[:, pos - 1:pos, :], nxt, [], []
    for i in range(k):
        fused = head._fuse(h, t2)
        p = torch.arange(pos - 1 + i, pos + i, device=H_all.device)
        h = head._run_layer(fused, p, cache)
        lg = head._lm_head(h)[:, -1, :]
        logs.append(lg.float().clone())
        t2 = torch.argmax(lg, dim=-1, keepdim=True)
        toks.append(t2)
    return logs, torch.cat(toks, dim=-1)


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
    WeightFoldingEngine(model, [expert], keep_pristine=True).activate(expert)
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
    gh = GraphedHead(head, model)
    gh.capture(at_pos=ids.shape[1] - 1)
    stage("captured")

    dec.cache.reset(); gh.cache.reset()
    cur = ids.shape[1]
    with torch.no_grad():
        out = dec.model(ids, past_key_values=dec.cache,
                        cache_position=torch.arange(0, cur, device=dec.device),
                        use_cache=True, output_hidden_states=True)
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    dec._counter_pos = cur
    H_all = torch.zeros((1, MAX_SEQ, out.hidden_states[-1].shape[-1]),
                        dtype=out.hidden_states[-1].dtype, device=dec.device)
    H_all[:, :cur, :] = out.hidden_states[-1]
    seq, toks, pos, hfilled = ids, [nxt.item()], cur, 0
    steps = 0
    prev_nacc, prev_draft_end = None, None
    first_div = None
    stage(f"  {'step':>4} {'pos':>5} {'hfilled':>7} {'prev_nacc':>9} "
          f"{'stale_tail':>10} {'max|dlogit|':>12}")

    while len(toks) < N and steps < 60:
        T = seq.shape[1]
        with torch.no_grad():
            if T - 1 > hfilled:
                gh.extend(H_all[:, hfilled:T - 1, :], seq[:, hfilled + 1:T], hfilled)
                hfilled = T - 1
            ref_logs, ref_toks = ref_draft_logits(head, H_all, seq, nxt, K, pos)
            _, st_logs = gh.draft(H_all[:, pos - 1:pos, :], nxt, K, pos - 1, want_logits=True)
        md = max((a - b).abs().max().item() for a, b in zip(st_logs, ref_logs))
        stale = (prev_draft_end - hfilled) if prev_draft_end is not None else 0
        if steps < 20 or md > 0.01:
            stage(f"  {steps:>4} {pos:>5} {hfilled:>7} "
                  f"{prev_nacc if prev_nacc is not None else -1:>9} {stale:>10} {md:12.5f}")
        if md > 0.01 and first_div is None:
            first_div = (steps, pos, hfilled, prev_nacc, stale, md)
            stage(f"  >>> FIRST DIVERGENCE at step {steps}")
            break

        # advance the trajectory using the REFERENCE draft
        draft = ref_toks
        dec._snapshot_ssm()
        chunk = torch.cat([nxt, draft], dim=-1)
        with torch.no_grad():
            logits, hidden = dec._replay(K + 1, chunk, pos)
        target = torch.argmax(logits[0], -1)
        n_acc = 0
        for i in range(K):
            if draft[0, i].item() == target[i].item():
                n_acc += 1
            else:
                break
        if n_acc < K:
            dec._restore_ssm()
            committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
            with torch.no_grad():
                _, hidden = dec._replay(n_acc + 1, committed, pos)
            new_h = hidden
        else:
            new_h, committed = hidden[:, : K + 1, :], chunk
        bonus = target[n_acc].item()
        for t in draft[0, :n_acc].tolist() + [bonus]:
            if len(toks) >= N:
                break
            toks.append(t)
        w = n_acc + 1
        H_all[:, pos:pos + w, :] = new_h[:, :w, :]
        seq = torch.cat([seq, committed], dim=-1)
        prev_nacc, prev_draft_end = n_acc, pos - 1 + K
        pos += w
        nxt = torch.tensor([[bonus]], device=dec.device)
        steps += 1

    stage("=" * 74)
    if first_div is None:
        stage(f"  NO DIVERGENCE over {steps} steps -- static cache tracks the rebuild")
    else:
        s, p, hf, pn, stale, md = first_div
        stage(f"  FIRST DIVERGENCE step={s} pos={p} hfilled={hf} prev_n_acc={pn} "
              f"stale_tail={stale} max|dlogit|={md:.5f}")
        stage(f"  stale_tail = how many positions the PREVIOUS draft wrote beyond what")
        stage(f"  extend() has since rewritten. >0 means rejected speculative entries")
        stage(f"  are still live in the cache at positions <= the current draft head.")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
