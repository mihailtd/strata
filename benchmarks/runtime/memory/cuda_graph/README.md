# ⭐ Pointer-Stable CUDA Graph Decode Replay

> **Tier Classification**: **⭐ Industry Standard** (Graph Capture) / **🔥 Applied Practice** (Pointer-Stable Mutation Synergy)  
> **Concept Origin**: **Static execution graph recording and fixed-buffer hardware replay applied to in-place LLM weight mutations.**

---

### Classification Breakdown: What is Standard vs. What is Innovative
* **⭐ Industry Standard Baseline**: CUDA / HIP Graph capture using `StaticCache` to eliminate CPU dispatch overhead during autoregressive token generation (standard practice in vLLM, TensorRT-LLM).
* **🔥 Our Innovative Applied Practice**: Achieving **Pointer Stability across live adapter swaps**. Standard PEFT layers create dynamic tensor allocations that invalidate captured graphs. By performing in-place low-rank arithmetic (`torch.addmm`) directly on frozen VRAM pointers (`data_ptr()`), our engine enables continuous multi-expert swapping without ever invalidating the static CUDA Graph.

---

## 1. The Challenge with CUDA Graphs
CUDA Graphs freeze a sequence of GPU kernel launches and their physical memory addresses (`data_ptr()`) to eliminate CPU launch overhead. If an inference server dynamically swaps LoRA layers using standard wrappers, PyTorch creates new tensor allocations, causing memory corruption or forcing full graph recompilation.

---

## 2. The Solution: Pointer Stability under In-Place Mutation
Inside `FoldedCudaGraphDecoder` ([`src/runtime/cuda_graph.py`](file:///home/mihai/gnn-experiment/src/runtime/cuda_graph.py)):
1. Pre-allocates fixed static buffers for `static_input_ids`, `static_position_ids`, `static_cache_position`, `static_attention_mask`, and `StaticCache`.
2. Employs in-place tensor mutations (`copy_()`, `+= 1`) during generation.
3. Swaps domain experts via `WeightFoldingEngine` directly into the static weight memory addresses without triggering reallocation.

---

## 3. Correctness Gate (`verify_against_eager()`)
Any benchmark cited must pass the `verify_against_eager()` gate, ensuring graph replay matches eager decode token-for-token over prefix contexts before reporting throughput.

## 4. Scripts
- **`benchmark_folded_cuda_graph.py`**: Benchmarks Base Eager vs Folded Eager vs Folded CUDA Graph Replay over full 256-token horizons.
