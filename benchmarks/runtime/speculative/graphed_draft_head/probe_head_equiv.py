"""Is it the StaticCache, or the graph capture, that breaks the draft head?

Graph-capturing the head hit its predicted speed (head_draft 24.02 -> 15.74 ms)
but tau fell 1.803 -> 1.311, and adding a device-built attention mask made it
WORSE (0.910). Two hypotheses tested inside a full generation, both wrong, and a
generation loop is a terrible place to debug numerics.

This isolates the variable with NO generation and NO graph:

    A  DynamicCache, head._run_layer as shipped        <- reference
    B  StaticCache,  head._run_layer (mask=None)
    C  StaticCache,  explicit additive mask
    D  StaticCache,  mask=None, graph-captured

A vs B  -> StaticCache semantics alone
B vs C  -> whether masking is the issue at all
B vs D  -> graph capture alone

Compares raw logits, not sampled tokens, so a small numeric drift is visible
before it turns into a different argmax.
"""

from __future__ import annotations

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

from runtime.mtp_draft import Qwen35MTPDraftHead  # noqa: E402
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

MODEL, ADAPTER = "Qwen/Qwen3.5-4B", "results/adapters/m2_astral_r8a128"
K, MAX_SEQ = 4, 2048


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
    dev, dt = next(head.parameters()).device, next(head.parameters()).dtype

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Design a production vector search system for "
          "100M documents. Cover storage, indexing, sharding and the query path."}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model(ids, use_cache=True, output_hidden_states=True)
    H = out.hidden_states[-1]
    T = ids.shape[1]
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    stage(f"prefix T={T}")

    cfg = model.config.get_text_config()
    hcfg = type(cfg).from_dict(cfg.to_dict())
    hcfg.layer_types, hcfg.num_hidden_layers = ["full_attention"], 1

    def static_cache():
        c = StaticCache(config=hcfg, max_batch_size=1, max_cache_len=MAX_SEQ,
                        device=dev, dtype=dt)
        return c, [x for x in (getattr(l, "cumulative_length", None) for l in c.layers)
                   if isinstance(x, torch.Tensor)]

    ar = torch.arange(MAX_SEQ, device=dev)
    zero = torch.zeros((), dtype=dt, device=dev)
    ninf = torch.full((), torch.finfo(dt).min, dtype=dt, device=dev)

    def masked_layer(fused, p, mask):
        pos_ids = p.unsqueeze(0)
        pe = head._rotary(fused, pos_ids.unsqueeze(0).expand(3, -1, -1))
        o = head.layer(fused, position_embeddings=pe, attention_mask=mask,
                       position_ids=pos_ids, past_key_values=CUR_CACHE,
                       cache_position=p)
        return head.norm(o)

    results = {}

    # ---- A: DynamicCache, exactly as shipped -------------------------------
    with torch.no_grad():
        cache = head.prefill(H, ids)
        h, t2, logs = H[:, -1:, :], nxt, []
        for i in range(K):
            fused = head._fuse(h, t2)
            p = torch.arange(T - 1 + i, T + i, device=dev)
            h = head._run_layer(fused, p, cache)
            lg = head._lm_head(h)[:, -1, :]
            logs.append(lg.float().clone())
            t2 = torch.argmax(lg, dim=-1, keepdim=True)
    results["A dynamic (reference)"] = logs
    stage("A done")

    # ---- B / C: StaticCache, eager, mask off / on --------------------------
    for label, use_mask in (("B static, mask=None", False), ("C static, masked", True)):
        CUR_CACHE, counters = static_cache()
        globals()["CUR_CACHE"] = CUR_CACHE
        with torch.no_grad():
            fused = head._fuse(H[:, :T - 1, :], ids[:, 1:T])
            p = torch.arange(0, T - 1, device=dev)
            for c in counters:
                c.fill_(0)
            m = (torch.where(ar.view(1, -1) <= p.view(-1, 1), zero, ninf).view(1, 1, T - 1, -1)
                 if use_mask else None)
            masked_layer(fused, p, m)
            h, t2, logs = H[:, -1:, :], nxt, []
            for i in range(K):
                fu = head._fuse(h, t2)
                pp = torch.arange(T - 1 + i, T + i, device=dev)
                for c in counters:
                    c.fill_(T - 1 + i)
                mm = (torch.where(ar <= (T - 1 + i), zero, ninf).view(1, 1, 1, -1)
                      if use_mask else None)
                h = masked_layer(fu, pp, mm)
                lg = head._lm_head(h)[:, -1, :]
                logs.append(lg.float().clone())
                t2 = torch.argmax(lg, dim=-1, keepdim=True)
        results[label] = logs
        stage(f"{label} done")

    ref = results["A dynamic (reference)"]
    stage("=" * 72)
    stage(f"  {'arm':24s} {'max|dlogit|':>12} {'argmax match':>13}")
    for label, logs in results.items():
        md = max((l - r).abs().max().item() for l, r in zip(logs, ref))
        mt = sum(int(l.argmax(-1).item() == r.argmax(-1).item()) for l, r in zip(logs, ref))
        stage(f"  {label:24s} {md:12.5f} {mt:>9}/{K}")
    stage("ALL DONE")


CUR_CACHE = None

if __name__ == "__main__":
    main()
