# RDNA3 W4A16 GEMV Memory Coalescing & Tile Tuning

## 1. Overview & Motivation

In single-token autoregressive decode ($M=1$), LLM inference is strictly memory-bandwidth bound:
- The GPU must stream all 13.5 GB of Qwen 3.8-27B model parameters from VRAM to compute cores for every single generated token.
- The **AMD Radeon RX 7900 XTX** (`gfx1100`) features a 384-bit GDDR6 memory bus delivering **960 GB/s theoretical peak bandwidth**.
- If a custom Triton GEMV kernel achieves only 150 GB/s bandwidth efficiency (~15% bus saturation), token decode will take ~90 ms per token, capping throughput at ~11 tok/s regardless of how many CPU kernel launches are eliminated.
- Achieving **30–40 tokens/sec** requires streaming 13.5 GB in $<30\text{ ms}$, which translates to **>450 GB/s effective memory bandwidth** (>47% bus saturation).

---

## 2. Technical Mechanics on RDNA3 (`gfx1100`)

### Memory Architecture Specifics
1. **Dual-Issue SIMD & Wave32/Wave64**:
   - RDNA3 compute units run Wave32 natively for vector workloads.
   - Vector loads must be coalesced into 128-bit transactions (`global_load_dwordx4`) to saturate the GDDR6 memory controllers and Infinity Cache.
2. **Tile Geometry**:
   - `BLOCK_N`: Number of output columns computed per threadblock.
   - `BLOCK_K`: Depth of the inner reduction tile along the hidden dimension.
   - Tuning the ratio between `BLOCK_N` and `BLOCK_K` determines L1 cache residency and memory transaction coalescing.
3. **Warp Count and Pipeline Stages**:
   - Too few warps (`num_warps=2`) fails to hide memory load latency.
   - Too many warps (`num_warps=8`) increases register pressure (VGPR spilling).
   - `num_stages=2` double-buffers global memory loads into LDS/registers.

---

## 3. Risks & Quality Degradation Guardrails

- **Zero Numerical Deviation**:
  - Quantized INT4 weight values and scales are identical regardless of tile geometry.
  - Verification requires bit-identical output ($1.000000$ cosine similarity) against the unoptimized baseline.
- **Hardware Register Spilling**:
  - Overly aggressive unrolling can spill VGPRs to scratch memory.
  - The benchmark verifies that latency improves monotonically with bandwidth increases without register spill penalties.

---

## 4. Reproduction Command

```bash
uv run python benchmarks/benchmark_rdna3_gemv_coalescing.py
```
Raw metrics saved to `results/benchmarks/rdna3_gemv_tuning_benchmark.json`.
