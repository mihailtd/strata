"""What does the 0.8B drafter cost GRAPH-CAPTURED, not eager?

The cost probe measured a graph-replayed 4B against an EAGER 0.8B and got
30.79 vs 35.51 ms/token -- only 1.15x, which is a launch-overhead artifact, not a
property of the model. Graph replay is exactly what removes that overhead, and
the draft path is width-1, the shape that is safe.

This is the deciding number:
    0.8B at ~7 ms/token  -> step 0.88+0.89+0.79 = 2.56 units, tau 2.323 -> ~1.30x
    0.8B at ~30 ms/token -> 0.63x, dead
"""

from __future__ import annotations

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

from runtime.cuda_graph import FoldedCudaGraphDecoder  # noqa: E402
from runtime.novel_peft import set_hard_vram_cap  # noqa: E402

TARGET, DRAFT = "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-0.8B"
MAX_SEQ, TAU, K = 2048, 2.323, 4


def main():
    set_hard_vram_cap(22.0)
    tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Design a production vector search system for "
          "100M documents. Cover storage, indexing, sharding and the query path."}],
        tokenize=False, add_generation_prompt=True)

    stage("loading 0.8B")
    d = AutoModelForCausalLM.from_pretrained(
        DRAFT, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    ids = tok(prompt, return_tensors="pt").input_ids.to(d.device)

    stage("eager 0.8B baseline")
    with torch.no_grad():
        out = d(ids, use_cache=True); cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)

    def eager_step():
        with torch.no_grad():
            d(nxt, past_key_values=cache, use_cache=True)

    for _ in range(10):
        eager_step()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(60):
        eager_step()
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t) / 60 * 1000
    stage(f"  eager 0.8B: {eager_ms:.2f} ms/token")

    stage("capturing width-1 graph for the 0.8B")
    dg = FoldedCudaGraphDecoder(d, tok, max_seq_len=MAX_SEQ, device=d.device)
    dg.capture(ids)
    dg.generate_with_graph(ids, max_new_tokens=8)
    _, _, dtps, _ = dg.generate_with_graph(ids, max_new_tokens=192)
    graph_ms = 1000.0 / dtps
    stage(f"  GRAPHED 0.8B: {dtps:.2f} tok/s -> {graph_ms:.2f} ms/token "
          f"({eager_ms / graph_ms:.2f}x faster than eager)")

    # target-side numbers already measured on this rig
    base_ms, verify_ms, commit_ms = 35.51, 31.27, 31.78
    for label, dms in (("eager draft", eager_ms), ("GRAPHED draft", graph_ms)):
        step = verify_ms + commit_ms + dms * K
        stage(f"  {label:14s} draft={dms * K:6.2f} ms  step={step:6.2f} ms  "
              f"=> {(TAU + 1) * base_ms / step:5.3f}x   break-even tau="
              f"{step / base_ms - 1:.2f}")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
