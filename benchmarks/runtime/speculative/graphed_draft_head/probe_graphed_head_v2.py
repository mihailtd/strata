"""Graph-capture the draft head using the formulation measured BIT-EXACT.

Isolation result (probe_head_equiv.py), no generation, no graph:
    A DynamicCache (reference)      max|dlogit| 0.00000
    B StaticCache, mask=None        max|dlogit| 0.40625   <- wrong
    C StaticCache, explicit mask    max|dlogit| 0.00000   <- exact

So the StaticCache design is sound and mask=None was the fault. The earlier
tau collapse (1.803 -> 1.311 -> 0.910) was the graph WIRING, not the cache.

This rebuilds the capture to mirror arm C exactly:
  * position is a real shape-(1,) int64 buffer, as `torch.arange(p, p+1)` produces
    -- not a 0-dim scalar viewed to (1,)
  * the additive mask is a materialised (1,1,1,max_seq) buffer written before each
    replay, as arm C materialised it -- not computed inside the graph
  * the head cache is reset per generation, and the draft chain starts from the
    backbone's last hidden state

GATE: logits are checked against the eager DynamicCache reference FIRST. A full
generation only runs if the graphed head is bit-exact, because debugging numerics
inside a 512-token loop is what wasted the last two attempts.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

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


class GraphedHead:
    def __init__(self, head, model, max_seq=MAX_SEQ):
        self.head = head
        self.dev = next(head.parameters()).device
        self.dt = next(head.parameters()).dtype
        cfg = model.config.get_text_config()
        hcfg = type(cfg).from_dict(cfg.to_dict())
        hcfg.layer_types, hcfg.num_hidden_layers = ["full_attention"], 1
        self.max_seq = max_seq
        self.cache = StaticCache(config=hcfg, max_batch_size=1, max_cache_len=max_seq,
                                 device=self.dev, dtype=self.dt)
        self.counters = [c for c in
                         (getattr(l, "cumulative_length", None) for l in self.cache.layers)
                         if isinstance(c, torch.Tensor)]
        self.ar = torch.arange(max_seq, device=self.dev)
        self.zero = torch.zeros((), dtype=self.dt, device=self.dev)
        self.ninf = torch.full((), torch.finfo(self.dt).min, dtype=self.dt, device=self.dev)
        # STATIC INPUTS -- shapes chosen to match the bit-exact eager arm exactly.
        self.s_h = torch.zeros((1, 1, head.hidden_size), dtype=self.dt, device=self.dev)
        self.s_tok = torch.zeros((1, 1), dtype=torch.long, device=self.dev)
        self.s_pos = torch.zeros(1, dtype=torch.long, device=self.dev)      # shape (1,)
        self.s_mask = torch.zeros((1, 1, 1, max_seq), dtype=self.dt, device=self.dev)
        self.graph = None

    def _layer(self, fused, p, mask):
        pos_ids = p.unsqueeze(0)
        pe = self.head._rotary(fused, pos_ids.unsqueeze(0).expand(3, -1, -1))
        o = self.head.layer(fused, position_embeddings=pe, attention_mask=mask,
                            position_ids=pos_ids, past_key_values=self.cache,
                            cache_position=p)
        return self.head.norm(o)

    def _write_pos(self, p: int):
        """Everything the step reads that depends on position, written together."""
        self.s_pos.fill_(p)
        for c in self.counters:                      # §26
            c.fill_(p)
        self.s_mask.view(-1).copy_(torch.where(self.ar <= p, self.zero, self.ninf))

    def _body(self):
        fused = self.head._fuse(self.s_h, self.s_tok)
        out = self._layer(fused, self.s_pos, self.s_mask)
        return out, torch.argmax(self.head._lm_head(out)[:, -1, :], dim=-1, keepdim=True)

    def capture(self, at_pos: int, warm: int = 3):
        self._write_pos(at_pos)
        st = torch.cuda.Stream(device=self.dev)
        st.wait_stream(torch.cuda.current_stream(device=self.dev))
        with torch.cuda.stream(st), torch.no_grad():
            for _ in range(warm):
                self._body()
        torch.cuda.current_stream(device=self.dev).wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st), torch.no_grad():
            self.o_h, self.o_tok = self._body()
        torch.cuda.current_stream(device=self.dev).wait_stream(st)
        self.graph = g

    @torch.no_grad()
    def extend(self, hidden, ids, start):
        n = ids.shape[1]
        if n <= 0:
            return
        fused = self.head._fuse(hidden, ids)
        p = torch.arange(start, start + n, device=self.dev)
        for c in self.counters:
            c.fill_(start)
        m = torch.where(self.ar.view(1, -1) <= p.view(-1, 1), self.zero, self.ninf)
        self._layer(fused, p, m.view(1, 1, n, -1))

    @torch.no_grad()
    def draft(self, h, tok, k, start_pos, want_logits=False):
        drafted, logits = [], []
        cur_h, cur_tok = h, tok
        for i in range(k):
            self.s_h.copy_(cur_h)
            self.s_tok.copy_(cur_tok)
            self._write_pos(start_pos + i)
            self.graph.replay()
            if want_logits:
                logits.append(self.head._lm_head(self.o_h)[:, -1, :].float().clone())
            cur_h, cur_tok = self.o_h, self.o_tok
            drafted.append(self.o_tok.clone())
        return (torch.cat(drafted, dim=-1), logits) if want_logits else torch.cat(drafted, dim=-1)


@torch.no_grad()
def reference_draft(head, H, ids, nxt, k):
    cache = head.prefill(H, ids)
    T = ids.shape[1]
    h, t2, logs = H[:, -1:, :], nxt, []
    for i in range(k):
        fused = head._fuse(h, t2)
        p = torch.arange(T - 1 + i, T + i, device=H.device)
        h = head._run_layer(fused, p, cache)
        lg = head._lm_head(h)[:, -1, :]
        logs.append(lg.float().clone())
        t2 = torch.argmax(lg, dim=-1, keepdim=True)
    return logs


@torch.no_grad()
def run(dec, ids, n_new, stop_ids, gh):
    k = dec.k
    dec.cache.reset()
    cur = ids.shape[1]
    ph = dict.fromkeys(("head_prefill", "head_draft", "snapshot", "verify",
                        "accept_sync", "commit", "bookkeep"), 0.0)
    t0 = sync_t()
    out = dec.model(ids, past_key_values=dec.cache,
                    cache_position=torch.arange(0, cur, device=dec.device),
                    use_cache=True, output_hidden_states=True)
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    dec._counter_pos = cur
    H_all = torch.zeros((1, MAX_SEQ, out.hidden_states[-1].shape[-1]),
                        dtype=out.hidden_states[-1].dtype, device=dec.device)
    H_all[:, :cur, :] = out.hidden_states[-1]
    if gh is not None:
        gh.cache.reset()
    hcache, hfilled = DynamicCache(), 0
    seq, toks, pos = ids, [nxt.item()], cur
    steps = accepted = 0
    done = toks[0] in stop_ids
    while len(toks) < n_new and not done:
        a = sync_t()
        T = seq.shape[1]
        if gh is not None:
            if T - 1 > hfilled:
                gh.extend(H_all[:, hfilled:T - 1, :], seq[:, hfilled + 1:T], hfilled)
                hfilled = T - 1
        else:
            hcache = dec.head.prefill(torch.cat([H_all[:, :T, :]], dim=1), seq)
        b = sync_t(); ph["head_prefill"] += b - a
        last_h = H_all[:, pos - 1:pos, :]
        draft = (gh.draft(last_h, nxt, k, pos - 1) if gh is not None
                 else dec.head.draft(last_h, nxt, k=k, start_pos=pos - 1, cache=hcache))
        c = sync_t(); ph["head_draft"] += c - b
        dec._snapshot_ssm()
        d = sync_t(); ph["snapshot"] += d - c
        chunk = torch.cat([nxt, draft], dim=-1)
        logits, hidden = dec._replay(k + 1, chunk, pos)
        e = sync_t(); ph["verify"] += e - d
        target = torch.argmax(logits[0], -1)
        n_acc = 0
        for i in range(k):
            if draft[0, i].item() == target[i].item():
                n_acc += 1
            else:
                break
        f = sync_t(); ph["accept_sync"] += f - e
        if n_acc < k:
            dec._restore_ssm()
            committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
            _, hidden = dec._replay(n_acc + 1, committed, pos)
            new_h = hidden
        else:
            new_h = hidden[:, : n_acc + 1, :]
            committed = chunk
        g2 = sync_t(); ph["commit"] += g2 - f
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
        H_all[:, pos:pos + w, :] = new_h[:, :w, :]
        seq = torch.cat([seq, committed], dim=-1)
        pos += w
        nxt = torch.tensor([[bonus]], device=dec.device)
        h3 = sync_t(); ph["bookkeep"] += h3 - g2
    return toks, sync_t() - t0, ph, steps, accepted


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
    with torch.no_grad():
        o = model(ids, use_cache=True, output_hidden_states=True)
    H, T = o.hidden_states[-1], ids.shape[1]
    nxt0 = torch.argmax(o.logits[:, -1, :], -1, keepdim=True)

    stage("=== NUMERIC GATE: graphed head vs eager DynamicCache reference ===")
    ref = reference_draft(head, H, ids, nxt0, K)
    gh = GraphedHead(head, model)
    gh.extend(H[:, :T - 1, :], ids[:, 1:T], 0)
    gh.capture(at_pos=T - 1)
    gh.cache.reset()
    gh.extend(H[:, :T - 1, :], ids[:, 1:T], 0)
    _, got = gh.draft(H[:, -1:, :], nxt0, K, T - 1, want_logits=True)
    md = max((a - b).abs().max().item() for a, b in zip(got, ref))
    am = sum(int(a.argmax(-1).item() == b.argmax(-1).item()) for a, b in zip(got, ref))
    stage(f"  max|dlogit| = {md:.5f}   argmax match {am}/{K}")
    if md > 0.01:
        stage("  GATE FAILED -- not running generation. Wiring still wrong.")
        stage("ALL DONE")
        return
    stage("  GATE PASSED -- graphed head is numerically equivalent")

    dec = BucketedSpeculativeDecoder(model, tok, head, k=K, max_seq_len=MAX_SEQ)
    dec.capture(ids); dec.generate(ids, max_new_tokens=8)
    res = {}
    for label, g in (("A eager head", None), ("B graphed head", gh)):
        toks, total, ph, steps, acc = run(dec, ids, N, stop, g)
        res[label] = (toks, total, ph, steps, acc)
        stage(f"{label}: {len(toks)} toks {steps} steps tau={acc/max(1,steps):.3f} "
              f"{len(toks)/total:.2f} tok/s step={total/steps*1000:.2f} ms")
    a, b = res["A eager head"], res["B graphed head"]
    stage(f"  TOKEN-IDENTICAL: {a[0] == b[0]}")
    stage(f"  {'phase':14s} {'A ms/step':>10} {'B ms/step':>10} {'delta':>8}")
    for kk in a[2]:
        stage(f"  {kk:14s} {a[2][kk]*1000/a[3]:10.2f} {b[2][kk]*1000/b[3]:10.2f} "
              f"{b[2][kk]*1000/b[3] - a[2][kk]*1000/a[3]:+8.2f}")
    stage(f"  {'STEP':14s} {a[1]/a[3]*1000:10.2f} {b[1]/b[3]*1000:10.2f} "
          f"{b[1]/b[3]*1000 - a[1]/a[3]*1000:+8.2f}")
    stage(f"  throughput {len(a[0])/a[1]:.2f} -> {len(b[0])/b[1]:.2f} tok/s "
          f"({(len(b[0])/b[1])/(len(a[0])/a[1]):.3f}x)")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
