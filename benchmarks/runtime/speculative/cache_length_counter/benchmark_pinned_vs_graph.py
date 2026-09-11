"""Honest bucketed-speculative numbers, measured with CORRECT KV offsets.

Every speculative figure in this repo so far was measured on a decoder whose
captured graph advanced transformers' device-side `cumulative_length` on every
replay. Speculation replays OVERLAPPING ranges, so the counter never rewound: the
commit re-forward wrote at drifting offsets from the very first partial accept,
and past 2048/width replays it ran off the cache and wedged the GPU.

Arms:
    A  graph autoregressive  -- what server.py actually runs today
    B  bucketed speculative, counter pinned    (CORRECT)
    C  bucketed speculative, counter unpinned  (the OLD, WRONG path; attribution
                                                only, kept under the wedge limit)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/mihai/gnn-experiment")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

import torch  # noqa: E402

torch.zeros(1, device="cuda"); torch.cuda.synchronize()
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.cuda_graph import FoldedCudaGraphDecoder  # noqa: E402
from runtime.mtp_draft import Qwen35MTPDraftHead  # noqa: E402
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

MODEL, K, MAX_SEQ = "Qwen/Qwen3.5-4B", 4, 2048
ADAPTER = "results/adapters/m2_astral_r8a128"
N = int(os.environ.get("N", "512"))
REPEATS = int(os.environ.get("REPEATS", "3"))
T0 = time.perf_counter()


def stage(m): print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)


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
        [{"role": "user", "content":
          "Design a production vector search system for 100M documents. Cover the "
          "storage layer, index construction, sharding, the query path, recall/latency "
          "tradeoffs, and how you would benchmark it. Be thorough and specific."}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

    ctl = FoldedCudaGraphDecoder(model, tok, max_seq_len=MAX_SEQ, device=model.device)
    ctl.capture(ids); ctl.generate_with_graph(ids, max_new_tokens=8)
    stage("graph autoregressive ready")

    from runtime.bucketed_speculative import BucketedSpeculativeDecoder
    res = {}
    for _ in range(REPEATS):
        _, _, tps, _ = ctl.generate_with_graph(ids, max_new_tokens=N)
        res.setdefault("A graph autoregressive", []).append(tps)
    stage(f"A done: {res['A graph autoregressive']}")

    for label, pin in (("B bucketed spec (PINNED, correct)", "1"),
                       ("C bucketed spec (unpinned, WRONG)", "0")):
        os.environ["SPECULATIVE_PIN_CACHE_LEN"] = pin
        dec = BucketedSpeculativeDecoder(model, tok, head, k=K, max_seq_len=MAX_SEQ)
        dec.capture(ids)
        dec.generate(ids, max_new_tokens=8)
        for _ in range(REPEATS):
            toks, el, st = dec.generate(ids, max_new_tokens=N, stop_ids=stop)
            res.setdefault(label, []).append(len(toks) / el)
            res.setdefault(label + " tau", []).append(st["tau"])
        stage(f"{label} done: {[f'{x:.2f}' for x in res[label]]}")
        del dec
        torch.cuda.empty_cache()

    def med(v): return sorted(v)[len(v) // 2]
    stage("=" * 74)
    a = med(res["A graph autoregressive"])
    for k2 in ("A graph autoregressive", "B bucketed spec (PINNED, correct)",
               "C bucketed spec (unpinned, WRONG)"):
        m = med(res[k2])
        tau = f"  tau={med(res[k2 + ' tau']):.2f}" if k2 + " tau" in res else ""
        stage(f"  {k2:38s} {m:7.2f} tok/s   {m / a:5.3f}x vs A{tau}")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
