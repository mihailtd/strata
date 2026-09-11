"""Kill head_prefill + cat_hids by making the draft head's cache incremental.

Profiled step at 512 tokens (tau=1.803, 118.00 ms/step):
    verify 34.34 | commit 27.16 | head_draft 23.91 | head_prefill 11.69
    accept_sync 8.24 | bookkeep 7.04 | snapshot 4.71 | cat_hids 0.66

head_prefill rebuilds a DynamicCache over the WHOLE sequence every step, and
cat_hids re-concatenates every hidden state every step. Both are O(n) per step
for information that changed by only n_acc+1 tokens.

The head is a SINGLE full_attention layer with no SSM, so unlike the backbone its
cache can simply be cropped -- the reason this is safe here and not there.

Two changes, both index arithmetic:
  * H_all preallocated once; new hidden states are written in place, never concatenated
  * the head cache persists across steps, cropped back to the committed length and
    extended by only the newly committed positions

ARM A is the current path, ARM B the incremental one. They must emit IDENTICAL
tokens -- anything else is a bug, not an optimisation.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/mihai/gnn-experiment")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

T0 = time.perf_counter()


def stage(m): print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)


import torch  # noqa: E402

torch.zeros(1, device="cuda"); torch.cuda.synchronize()
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.bucketed_speculative import BucketedSpeculativeDecoder  # noqa: E402
from runtime.mtp_draft import Qwen35MTPDraftHead  # noqa: E402
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

MODEL, ADAPTER = "Qwen/Qwen3.5-4B", "results/adapters/m2_astral_r8a128"
K, MAX_SEQ = 4, 2048
N = int(os.environ.get("N", "512"))


def sync_t():
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def run(dec, ids, n_new, stop_ids, incremental: bool):
    from transformers.cache_utils import DynamicCache

    k = dec.k
    dec.cache.reset()
    cur = ids.shape[1]
    ph = dict.fromkeys(("hidden_mgmt", "head_prefill", "head_draft", "snapshot",
                        "verify", "accept_sync", "commit", "bookkeep"), 0.0)
    t0 = sync_t()
    out = dec.model(ids, past_key_values=dec.cache,
                    cache_position=torch.arange(0, cur, device=dec.device),
                    use_cache=True, output_hidden_states=True)
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    dec._counter_pos = cur

    H_all = None
    hids = None
    if incremental:
        H_all = torch.zeros((1, MAX_SEQ, dec.model.config.get_text_config().hidden_size),
                            dtype=out.hidden_states[-1].dtype, device=dec.device)
        H_all[:, :cur, :] = out.hidden_states[-1]
        hcache, hfilled = DynamicCache(), 0
    else:
        hids = [out.hidden_states[-1]]

    seq, toks, pos = ids, [nxt.item()], cur
    steps = accepted = 0
    done = toks[0] in stop_ids

    while len(toks) < n_new and not done:
        a = sync_t()
        if incremental:
            last_h = H_all[:, pos - 1:pos, :]
        else:
            H = torch.cat(hids, dim=1)
            last_h = H[:, -1:, :]
        b = sync_t(); ph["hidden_mgmt"] += b - a

        if incremental:
            # drop last step's speculative entries, then extend by ONLY the
            # newly committed positions. cache slot j fuses h_j with x_{j+1}.
            T = seq.shape[1]
            if hcache.get_seq_length() > hfilled:
                hcache.crop(hfilled)
            if T - 1 > hfilled:
                s, e = hfilled, T - 1
                fused = dec.head._fuse(H_all[:, s:e, :], seq[:, s + 1:e + 1])
                dec.head._run_layer(fused, torch.arange(s, e, device=dec.device), hcache)
                hfilled = e
            dcache = hcache
        else:
            dcache = dec.head.prefill(H, seq)
        c = sync_t(); ph["head_prefill"] += c - b

        draft = dec.head.draft(last_h, nxt, k=k, start_pos=pos - 1, cache=dcache)
        d = sync_t(); ph["head_draft"] += d - c
        dec._snapshot_ssm()
        e2 = sync_t(); ph["snapshot"] += e2 - d
        chunk = torch.cat([nxt, draft], dim=-1)
        logits, hidden = dec._replay(k + 1, chunk, pos)
        f = sync_t(); ph["verify"] += f - e2
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
        h2 = sync_t(); ph["commit"] += h2 - g
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
        w = n_acc + 1
        if incremental:
            H_all[:, pos:pos + w, :] = new_h[:, :w, :]
        else:
            hids.append(new_h.clone())
        seq = torch.cat([seq, committed], dim=-1)
        pos += w
        nxt = torch.tensor([[bonus]], device=dec.device)
        i3 = sync_t(); ph["bookkeep"] += i3 - h2
    total = sync_t() - t0
    return toks, total, ph, steps, accepted


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
    stage("captured")

    res = {}
    for label, inc in (("A current", False), ("B incremental", True)):
        toks, total, ph, steps, acc = run(dec, ids, N, stop, inc)
        res[label] = (toks, total, ph, steps, acc)
        stage(f"{label}: {len(toks)} toks {steps} steps tau={acc/max(1,steps):.3f} "
              f"{len(toks)/total:.2f} tok/s step={total/steps*1000:.2f} ms")

    a_toks, b_toks = res["A current"][0], res["B incremental"][0]
    same = a_toks == b_toks
    stage(f"  TOKEN-IDENTICAL: {same}" + ("" if same else
          f"  first diff at {next(i for i,(x,y) in enumerate(zip(a_toks,b_toks)) if x!=y)}"))
    stage("=" * 72)
    stage(f"  {'phase':14s} {'A ms/step':>10} {'B ms/step':>10} {'delta':>8}")
    for kk in res["A current"][2]:
        pa = res["A current"][2][kk] * 1000 / res["A current"][3]
        pb = res["B incremental"][2][kk] * 1000 / res["B incremental"][3]
        stage(f"  {kk:14s} {pa:10.2f} {pb:10.2f} {pb - pa:+8.2f}")
    sa = res["A current"][1] / res["A current"][3] * 1000
    sb = res["B incremental"][1] / res["B incremental"][3] * 1000
    stage(f"  {'STEP':14s} {sa:10.2f} {sb:10.2f} {sb - sa:+8.2f}")
    ta = len(a_toks) / res["A current"][1]
    tb = len(b_toks) / res["B incremental"][1]
    stage(f"  throughput {ta:.2f} -> {tb:.2f} tok/s  ({tb/ta:.3f}x)")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
