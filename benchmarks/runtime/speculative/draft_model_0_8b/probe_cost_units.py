"""Measure the per-step cost of a 0.8B-drafted speculative step. No estimates.

tau is now measured (2.323 with the astral expert folded). The projected speedup
straddled 1.0 only because the DRAFT COST was estimated. This measures every
component on the real hardware, on the paths that would actually run:

    baseline   width-1 CUDA graph replay on the 4B   (what server.py runs today)
    verify     width-K+1 graph replay on the 4B
    commit     width-3 graph replay on the 4B        (n_acc+1 at tau=2.323)
    draft      K sequential eager forwards on the 0.8B

speedup = (tau + 1) * baseline_ms / (verify + commit + draft) ms
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

from runtime.bucketed_speculative import BucketedSpeculativeDecoder  # noqa: E402
from runtime.cuda_graph import FoldedCudaGraphDecoder  # noqa: E402
from runtime.mtp_draft import Qwen35MTPDraftHead  # noqa: E402
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

TARGET, DRAFT = "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-0.8B"
ADAPTER = "results/adapters/m2_astral_r8a128"
K, MAX_SEQ, REPS = 4, 2048, 60
TAU = float(os.environ.get("TAU", "2.323"))


def timed(fn, reps=REPS, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1000.0


def main():
    set_hard_vram_cap(22.0)
    stage("loading")
    tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    dmodel = AutoModelForCausalLM.from_pretrained(
        DRAFT, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    expert = FoldableExpert.from_dir(REPO_ROOT / ADAPTER, "astral")
    engine = WeightFoldingEngine(target, [expert], keep_pristine=True)
    engine.activate(expert)
    head = Qwen35MTPDraftHead(target, TARGET)

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Design a production vector search system for "
          "100M documents. Cover storage, indexing, sharding and the query path."}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(target.device)
    stage(f"loaded, prompt={ids.shape[1]}  alloc={torch.cuda.memory_allocated()/2**30:.2f}G")

    ctl = FoldedCudaGraphDecoder(target, tok, max_seq_len=MAX_SEQ, device=target.device)
    ctl.capture(ids); ctl.generate_with_graph(ids, max_new_tokens=8)
    dec = BucketedSpeculativeDecoder(target, tok, head, k=K, max_seq_len=MAX_SEQ)
    dec.capture(ids); dec.generate(ids, max_new_tokens=8)
    stage("graphs captured")

    # baseline: one token via the width-1 graph, as the server decodes today
    _, _, base_tps, _ = ctl.generate_with_graph(ids, max_new_tokens=128)
    base_ms = 1000.0 / base_tps
    stage(f"baseline width-1 graph: {base_tps:.2f} tok/s -> {base_ms:.2f} ms/token")

    pos = 600
    c5 = torch.full((1, K + 1), 100, dtype=torch.long, device=target.device)
    c3 = torch.full((1, 3), 100, dtype=torch.long, device=target.device)
    verify_ms = timed(lambda: dec._replay(K + 1, c5, pos))
    commit_ms = timed(lambda: dec._replay(3, c3, pos))
    stage(f"verify (width {K+1}) {verify_ms:.2f} ms | commit (width 3) {commit_ms:.2f} ms")

    # draft: K sequential eager forwards on the 0.8B over a realistic prefix
    dprefix = torch.full((1, pos), 100, dtype=torch.long, device=dmodel.device)
    with torch.no_grad():
        out = dmodel(dprefix, use_cache=True)
        dcache = out.past_key_values
    nxt = torch.full((1, 1), 100, dtype=torch.long, device=dmodel.device)

    def draft_step():
        with torch.no_grad():
            dmodel(nxt, past_key_values=dcache, use_cache=True)

    draft1_ms = timed(draft_step)
    draft_ms = draft1_ms * K
    stage(f"draft 0.8B: {draft1_ms:.2f} ms/token x K={K} = {draft_ms:.2f} ms")

    step_ms = verify_ms + commit_ms + draft_ms
    emitted = TAU + 1
    speedup = emitted * base_ms / step_ms
    stage("=" * 76)
    stage(f"  baseline               {base_ms:6.2f} ms/token")
    stage(f"  verify  (width {K+1})     {verify_ms:6.2f} ms   = {verify_ms/base_ms:.2f} units")
    stage(f"  commit  (width 3)      {commit_ms:6.2f} ms   = {commit_ms/base_ms:.2f} units")
    stage(f"  draft   (K x 0.8B)     {draft_ms:6.2f} ms   = {draft_ms/base_ms:.2f} units")
    stage(f"  step total             {step_ms:6.2f} ms   = {step_ms/base_ms:.2f} units")
    stage(f"  emitted/step (tau={TAU})  {emitted:.3f} tokens")
    stage(f"  => PROJECTED {speedup:.3f}x vs width-1 graph autoregressive")
    be = step_ms / base_ms - 1
    stage(f"  => break-even tau = {be:.3f}  (measured {TAU}: "
          f"{'CLEARS' if TAU > be else 'MISSES'})")
    stage(f"  if the commit re-forward were free: "
          f"{emitted * base_ms / (verify_ms + draft_ms):.3f}x")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
