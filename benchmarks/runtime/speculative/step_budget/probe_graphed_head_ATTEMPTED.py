"""Graph-capture the MTP draft head's per-token step.

Established three times today: this rig is kernel-LAUNCH bound, not work bound.
    0.8B drafter   5.3x fewer params  -> only 2.14x faster
    §18 replay     24 eager launches  -> ate 19 of a 24.75 ms saving
    head_prefill   180x less work     -> only 22% faster
Only fewer launches pay, i.e. graph capture.

head_draft costs 25.30 ms/step for FOUR single-layer forwards -- 6.3 ms each,
when a full 32-layer graph-replayed forward is 34 ms. One layer costing 18% of a
32-layer model is ~all launch overhead, and it is the most compressible thing in
the step.

This captures ONE width-1 graph for a draft step (fuse -> layer -> lm_head ->
argmax) over a StaticCache and replays it K times.

Two hazards, both already paid for elsewhere in this repo:
  * the head's StaticLayer keeps cumulative_length in a DEVICE tensor and the
    graph captures its add_ (§26). It must be pinned per replay or the head walks
    off its own cache exactly as the backbone did.
  * fla/Triton autotunes on first sight of a shape, which is a host sync and is
    illegal during capture (§25), so the shape is warmed before capturing.

ARM A is the current eager head, ARM B the graphed one. They MUST emit identical
tokens.
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


class GraphedHeadDrafter:
    """One captured width-1 graph for the head, replayed K times per step."""

    def __init__(self, head, model, max_seq=MAX_SEQ):
        self.head, self.dev = head, next(head.parameters()).device
        cfg = model.config.get_text_config()
        hcfg = type(cfg).from_dict(cfg.to_dict())
        hcfg.layer_types = ["full_attention"]
        hcfg.num_hidden_layers = 1
        self.cache = StaticCache(config=hcfg, max_batch_size=1, max_cache_len=max_seq,
                                 device=self.dev, dtype=next(head.parameters()).dtype)
        self.counters = [c for c in
                         (getattr(l, "cumulative_length", None) for l in self.cache.layers)
                         if isinstance(c, torch.Tensor)]
        self.graph = None
        dt = next(head.parameters()).dtype
        self.ar = torch.arange(max_seq, device=self.dev)
        self.pos_buf = torch.zeros((), dtype=torch.long, device=self.dev)
        self.zero = torch.zeros((), dtype=dt, device=self.dev)
        self.neg_inf = torch.full((), torch.finfo(dt).min, dtype=dt, device=self.dev)

    def _layer_masked(self, fused, p, mask):
        """_run_layer, but with an explicit mask over the static cache.

        The head's _run_layer hardcodes attention_mask=None. Over a tight
        DynamicCache that is correct -- every cached slot is valid. Over a
        2048-slot StaticCache it lets the head attend to UNWRITTEN ZERO slots,
        which silently degrades drafting (measured: tau 1.803 -> 1.311) without
        ever erroring.
        """
        pos_ids = p.unsqueeze(0)
        rope_ids = pos_ids.unsqueeze(0).expand(3, -1, -1)
        pe = self.head._rotary(fused, rope_ids)
        out = self.head.layer(fused, position_embeddings=pe, attention_mask=mask,
                              position_ids=pos_ids, past_key_values=self.cache,
                              cache_position=p)
        return self.head.norm(out)

    def _mask_for(self, pos_buf):
        """Additive mask built ON DEVICE from the pinned position.

        Computed inside the captured graph, so keeping it correct costs no extra
        kernel launches per replay -- only the single fill_ of pos_buf that the
        counter pinning already needs.
        """
        valid = self.ar <= pos_buf
        return torch.where(valid, self.zero, self.neg_inf).view(1, 1, 1, -1)

    def _step_eager(self, h, tok, pos):
        fused = self.head._fuse(h, tok)
        p = torch.arange(pos, pos + 1, device=self.dev)
        for c in self.counters:
            c.fill_(pos)
        self.pos_buf.fill_(pos)
        out = self._layer_masked(fused, p, self._mask_for(self.pos_buf))
        return out, torch.argmax(self.head._lm_head(out)[:, -1, :], dim=-1, keepdim=True)

    def capture(self, warm=3):
        self.s_h = torch.zeros((1, 1, self.head.hidden_size),
                               dtype=next(self.head.parameters()).dtype, device=self.dev)
        self.s_tok = torch.zeros((1, 1), dtype=torch.long, device=self.dev)
        self.s_pos = 64
        st = torch.cuda.Stream(device=self.dev)
        st.wait_stream(torch.cuda.current_stream(device=self.dev))
        with torch.cuda.stream(st), torch.no_grad():
            for _ in range(warm):
                self._step_eager(self.s_h, self.s_tok, self.s_pos)
        torch.cuda.current_stream(device=self.dev).wait_stream(st)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st), torch.no_grad():
            fused = self.head._fuse(self.s_h, self.s_tok)
            p = self.pos_buf.view(1)
            out = self._layer_masked(fused, p, self._mask_for(self.pos_buf))
            self.o_h = out
            self.o_tok = torch.argmax(self.head._lm_head(out)[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.current_stream(device=self.dev).wait_stream(st)
        self.graph = g

    @torch.no_grad()
    def draft(self, h, tok, k, start_pos):
        drafted = []
        cur_h, cur_tok = h, tok
        for i in range(k):
            self.s_h.copy_(cur_h)
            self.s_tok.copy_(cur_tok)
            for c in self.counters:      # §26: the graph's add_ must not own this
                c.fill_(start_pos + i)
            self.pos_buf.fill_(start_pos + i)   # drives BOTH position and mask
            self.graph.replay()
            cur_h, cur_tok = self.o_h, self.o_tok
            drafted.append(self.o_tok.clone())
        return torch.cat(drafted, dim=-1)

    def _layer_masked_eager(self, fused, p, mask):
        return self._layer_masked(fused, p, mask)

    @torch.no_grad()
    def extend(self, hidden, ids, start):
        """Eager, variable width: fill committed positions into the head cache."""
        n = ids.shape[1]
        if n <= 0:
            return
        fused = self.head._fuse(hidden, ids)
        for c in self.counters:
            c.fill_(start)
        p = torch.arange(start, start + n, device=self.dev)
        m = torch.where(self.ar.view(1, -1) <= p.view(-1, 1), self.zero, self.neg_inf)
        self._layer_masked_eager(fused, p, m.view(1, 1, n, -1))


@torch.no_grad()
def run(dec, ids, n_new, stop_ids, gd):
    from transformers.cache_utils import DynamicCache
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
    hid = out.hidden_states[-1].dtype
    H_all = torch.zeros((1, MAX_SEQ, out.hidden_states[-1].shape[-1]), dtype=hid, device=dec.device)
    H_all[:, :cur, :] = out.hidden_states[-1]
    if gd is not None:
        gd.cache.reset()
    hcache, hfilled = DynamicCache(), 0
    seq, toks, pos = ids, [nxt.item()], cur
    steps = accepted = 0
    done = toks[0] in stop_ids

    while len(toks) < n_new and not done:
        a = sync_t()
        T = seq.shape[1]
        if gd is not None:
            if T - 1 > hfilled:
                gd.extend(H_all[:, hfilled:T - 1, :], seq[:, hfilled + 1:T], hfilled)
                hfilled = T - 1
        else:
            if hcache.get_seq_length() > hfilled:
                hcache.crop(hfilled)
            if T - 1 > hfilled:
                s, e = hfilled, T - 1
                fused = dec.head._fuse(H_all[:, s:e, :], seq[:, s + 1:e + 1])
                dec.head._run_layer(fused, torch.arange(s, e, device=dec.device), hcache)
                hfilled = e
        b = sync_t(); ph["head_prefill"] += b - a
        last_h = H_all[:, pos - 1:pos, :]
        if gd is not None:
            draft = gd.draft(last_h, nxt, k, pos - 1)
        else:
            draft = dec.head.draft(last_h, nxt, k=k, start_pos=pos - 1, cache=hcache)
        c = sync_t(); ph["head_draft"] += c - b
        dec._snapshot_ssm()
        d = sync_t(); ph["snapshot"] += d - c
        chunk = torch.cat([nxt, draft], dim=-1)
        logits, hidden = dec._replay(k + 1, chunk, pos)
        e2 = sync_t(); ph["verify"] += e2 - d
        target = torch.argmax(logits[0], -1)
        n_acc = 0
        for i in range(k):
            if draft[0, i].item() == target[i].item():
                n_acc += 1
            else:
                break
        f = sync_t(); ph["accept_sync"] += f - e2
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
    stage("backbone captured")

    stage("capturing head draft graph")
    gd = GraphedHeadDrafter(head, model)
    gd.capture()
    stage("head graph captured")

    res = {}
    for label, g in (("A eager head", None), ("B graphed head", gd)):
        toks, total, ph, steps, acc = run(dec, ids, N, stop, g)
        res[label] = (toks, total, ph, steps, acc)
        stage(f"{label}: {len(toks)} toks {steps} steps tau={acc/max(1,steps):.3f} "
              f"{len(toks)/total:.2f} tok/s step={total/steps*1000:.2f} ms")

    a, b = res["A eager head"], res["B graphed head"]
    same = a[0] == b[0]
    stage(f"  TOKEN-IDENTICAL: {same}")
    if not same:
        d = next((i for i, (x, y) in enumerate(zip(a[0], b[0])) if x != y), None)
        stage(f"  first divergence at index {d}")
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
