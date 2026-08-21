"""What does the STREAMING serving path cost per token, on top of the GPU?

§28 established this rig is host-bound, and streaming is the path with the most
host work per token. server.py + cuda_graph.py do ALL of this per token:

    next_token.item()                  GPU sync
    tokenizer.decode([tok_id])         Python BPE
    "<think>" in piece                 string scans
    ChatCompletionChunkResponse(...)   THREE nested pydantic models
    chunk.model_dump()                 pydantic -> dict
    json.dumps(...)                    encode
    await asyncio.sleep(0)             event-loop tick

Part 1 is pure Python and needs no GPU -- it prices the serialisation tail
directly. Part 2 compares the decoder's batch path against its streaming path on
the same graph, which isolates .item() + per-token decode.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/mihai/gnn-experiment")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

T0 = time.perf_counter()


def stage(m): print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)


def bench(fn, n=2000, warm=200):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e6      # microseconds


def main():
    from transformers import AutoTokenizer

    from runtime.server import (
        ChatCompletionChunkChoice,
        ChatCompletionChunkDelta,
        ChatCompletionChunkResponse,
    )

    stage("=== PART 1: per-token HOST cost (no GPU) ===")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    tid = tok.convert_tokens_to_ids("the")
    if not isinstance(tid, int) or tid < 0:
        tid = 1000
    piece = tok.decode([tid])

    def mk_chunk():
        return ChatCompletionChunkResponse(
            id="chatcmpl-x", model="qwen3.5-4b-astral",
            choices=[ChatCompletionChunkChoice(
                index=0, delta=ChatCompletionChunkDelta(content=piece))])

    costs = {
        "tokenizer.decode([id])": bench(lambda: tok.decode([tid])),
        "pydantic construct": bench(mk_chunk),
        "construct + model_dump": bench(lambda: mk_chunk().model_dump()),
        "construct + dump + json": bench(lambda: json.dumps(mk_chunk().model_dump())),
        "'<think>' in piece": bench(lambda: "<think>" in piece),
    }
    for k, v in costs.items():
        stage(f"  {k:26s} {v:8.2f} us")
    per_tok = costs["construct + dump + json"] + costs["tokenizer.decode([id])"]
    stage(f"  --> host serialisation tail ~= {per_tok:.1f} us/token "
          f"= {per_tok/1000:.3f} ms/token")
    stage(f"  at 35.51 ms/token GPU that is {per_tok/1000/35.51*100:.2f}% of a token")

    stage("=== PART 2: decoder batch vs streaming (GPU) ===")
    import torch
    torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    from transformers import AutoModelForCausalLM

    from runtime.cuda_graph import FoldedCudaGraphDecoder
    from runtime.novel_peft import (
        FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
    )
    set_hard_vram_cap(22.0)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map="cuda:0",
        trust_remote_code=True).eval()
    expert = FoldableExpert.from_dir(REPO_ROOT / "results/adapters/m2_astral_r8a128", "astral")
    eng = WeightFoldingEngine(model, [expert], keep_pristine=True)
    eng.activate(expert)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Design a production vector search system for "
          "100M documents. Cover storage, indexing, sharding and the query path."}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    dec = FoldedCudaGraphDecoder(model, tok, max_seq_len=2048, device=model.device)
    dec.capture(ids)
    dec.generate_with_graph(ids, max_new_tokens=8)

    N = 256
    _, _, tps_batch, _ = dec.generate_with_graph(ids, max_new_tokens=N)
    stage(f"  generate_with_graph (batch decode at end): {tps_batch:6.2f} tok/s")

    torch.cuda.synchronize(); t = time.perf_counter()
    n = 0
    for _piece in dec.generate_tokens_stream(ids, engine=eng, expert=expert,
                                             max_new_tokens=N):
        n += 1
    torch.cuda.synchronize()
    tps_stream = n / (time.perf_counter() - t)
    stage(f"  generate_tokens_stream (.item + decode/token): {tps_stream:6.2f} tok/s "
          f"over {n} tokens")

    # streaming + the FULL server-side serialisation tail, minus ASGI/network
    torch.cuda.synchronize(); t = time.perf_counter()
    n = 0
    for p in dec.generate_tokens_stream(ids, engine=eng, expert=expert, max_new_tokens=N):
        if p:
            c = ChatCompletionChunkResponse(
                id="chatcmpl-x", model="qwen3.5-4b-astral",
                choices=[ChatCompletionChunkChoice(
                    index=0, delta=ChatCompletionChunkDelta(content=p))])
            _ = f"data: {json.dumps(c.model_dump())}\n\n"
        n += 1
    torch.cuda.synchronize()
    tps_full = n / (time.perf_counter() - t)
    stage(f"  stream + pydantic + json (server tail):      {tps_full:6.2f} tok/s")

    stage("=" * 70)
    stage(f"  batch                {tps_batch:6.2f} tok/s   1.000x")
    stage(f"  streaming            {tps_stream:6.2f} tok/s   {tps_stream/tps_batch:.3f}x")
    stage(f"  streaming + server   {tps_full:6.2f} tok/s   {tps_full/tps_batch:.3f}x")
    stage(f"  cost of streaming    {1000/tps_stream - 1000/tps_batch:+6.2f} ms/token")
    stage(f"  cost of server tail  {1000/tps_full - 1000/tps_stream:+6.2f} ms/token")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
