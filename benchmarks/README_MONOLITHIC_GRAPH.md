# Monolithic Full-Model Decode Graph (`FullDecodeGraph`) for Native 27B Triton Engine

## 1. Overview & Motivation

Autoregressive decode executes 64 layers (48 SSM + 16 Full Attention) plus the LM Head for each generated token:
- In eager execution or fragmented chunk graphing, the Python interpreter dispatches **~240 individual GPU kernel launches per token**, interleaving Python object allocations, RoPE slicing, dictionary lookups, and D2H synchronization.
- On AMD ROCm (`gfx1100`), 240 host enqueues per token introduce **~12–15 ms of pure CPU dispatch latency** per token.
- At 21.3 tok/s (47 ms/token), host dispatch overhead constitutes **>25% of the total decode time**.

### The Solution: Monolithic Full-Model Decode Graph
By capturing all 64 layers into **ONE SINGLE monolithic CUDAGraph**:
1. **Dynamic RoPE**: Precomputed static rotation frequency buffers indexed via GPU tensors.
2. **Dynamic KV Cache Indexing**: In-place GPU index writes using `index_copy_(dim=2, index=static_pos, source=k_new)`.
3. **Dynamic Attention Mask**: Pre-allocated `(1, 1, 1, max_seq_len)` mask where position $p$ is activated prior to each replay.
4. **Fused Top-1 Token Extraction**: `torch.argmax(logits, dim=-1, out=static_next_token)` executed inside the graph.

At decode time:
- The host CPU submits **1 single command** (`graph.replay()`).
- All 64 layers, LM Head, and argmax execute as a continuous, uninterrupted hardware pipeline on the GPU.
- Host dispatch drops to **<0.1 ms**, unlocking theoretical hardware saturation (>30 tokens/sec).

---

## 2. Technical Mechanics & Dynamic Invariants

```
                FRAGMENTED DECODE (16 Python Hops / Token)
CPU: [Replay 0..2] -> [Eager Attn 3] -> [Replay 4..6] ... -> [Eager LM Head] -> [.item() Sync]
GPU:   [Compute]         [Wait]           [Compute]               [Compute]        [Halt & Flush]

                MONOLITHIC FULL-MODEL GRAPH (1 Launch / Token)
CPU: [Update static_pos & static_mask] -> [graph.replay()] (<0.1 ms!)
GPU: [================ Pipelined 64-Layer + LM Head Hardware Buffer ================]
```

### Invariants:
- `static_pos`: `torch.zeros((1,), dtype=torch.long, device=device)`
- `static_mask`: `torch.full((1, 1, 1, max_seq_len), -1e4, dtype=torch.bfloat16, device=device)`
- `static_next_token`: `torch.zeros((1,), dtype=torch.long, device=device)`
- Pointers to all weights and pre-allocated caches remain strictly immutable.

---

## 3. Risks & Mitigations

1. **Static Attention Mask Overhead**:
   - Running SDPA over `max_seq_len` (e.g. 4096) could add compute overhead if done naively.
   - *Mitigation*: For short conversations ($<512$ tokens), size `max_seq_len` dynamically to the request's context limit or capture graphs for standard bucket lengths (128, 512, 2048).
2. **LoRA Adapter Switching**:
   - In-place mutation of static low-rank adapter factor buffers preserves graph memory validity without requiring re-capture.

---

## 4. Reproduction Command

```bash
uv run python benchmarks/benchmark_monolithic_graph.py
```
Raw metrics saved to `results/benchmarks/monolithic_graph_synthetic_benchmark.json`.
