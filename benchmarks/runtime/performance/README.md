# Triton WMMA: Native RDNA3 Matrix Acceleration

Kernel-level matrix multiplication optimizations and Wave Matrix Multiply-Accumulate (WMMA) compilation tuning on **AMD RDNA3 (`gfx1100` / Radeon RX 7900 XTX)**.

---

## 💡 What is Triton WMMA in Plain English? (Layman's Terms)

Every modern AMD Radeon RX 7000 graphics card has two ways of doing math:
1. **Regular Compute Units:** Doing calculations like standard CPU/GPU cores (one vector of numbers at a time).
2. **Dedicated AI Matrix Cores (WMMA):** Special hardware accelerators built directly into the silicon designed to multiply entire grids (matrices) of numbers in a single clock cycle.

### The Problem: Why Regular Software Failed to Use It
The popular AI kernel compiler **Triton** (made for writing high-performance GPU code) was built primarily for **$15,000+ datacenter GPUs** (like AMD MI300 or Nvidia H100). 

Those enterprise chips use wide 64-thread computing waves. But consumer Radeon cards use **32-thread waves (Wave32)** and a different $16 \times 16 \times 16$ tile layout. When regular Triton code runs on an RX 7900 XTX, the compiler gets confused by the shapes, gives up, and **turns off the AI matrix cores**—falling back to slow regular math.

### What We Did:
* **We Fixed the Shape Translation:** We tuned Triton's tile configurations (`BLOCK_M=64`, `BLOCK_N=64`, `BLOCK_K=32`) so the compiler natively recognizes the card's 32-thread AI cores.
* **We Forced Hardware Emission:** When the code compiles, the graphics card now directly fires its dedicated **`v_wmma_f32_16x16x16_bf16`** hardware instructions.
* **We Built "Fused LoRA Branching":** When a specialized AI expert adapter is active, instead of calculating the base AI model, saving it to memory, and then running the expert, our kernel computes both the base model and the expert in a **single pass inside the processor's ultra-fast internal registers**.

### What This Means For You:
* **⚡ 73+ Trillion Calculations per Second (73.3 TFLOPS):** Raw sustained matrix computing power on a desktop GPU.
* **⏩ 1.48× Faster Multi-Token Guess Verification:** Speculative draft checking ($K=2$) drops to **0.075 ms**.
* **🎯 100% Zero-Loss Precision:** Bit-exact agreement with standard PyTorch (1.000000 cosine similarity, 0.0000 max error).

---

## 🔬 Technical Deep-Dive: Hardware Mechanics & Compilation

### 1. The RDNA3 ISA Challenge: Wave32 vs. Wave64
Datacenter AMD architectures (CDNA / MI300) execute in 64-thread wavefronts with `v_mfma_*` ($32 \times 32 \times 8$) instructions. RDNA3 (`gfx1100`) executes in native 32-thread wavefronts (Wave32) with dual-issue VOPD and $16 \times 16 \times 16$ WMMA units.

By specifying Wave32 layout and block dimensions that are exact integer multiples of 16, the LLVM AMDGPU backend emits native `v_wmma` instructions:

```asm
v_wmma_f32_16x16x16_bf16 v[25:32], v[55:62], v[63:70], v[25:32]
v_wmma_f32_16x16x16_bf16 v[9:16],  v[55:62], v[71:78], v[9:16]
v_wmma_f32_16x16x16_bf16 v[17:24], v[55:62], v[63:70], v[17:24]
v_wmma_f32_16x16x16_bf16 v[1:8],   v[55:62], v[71:78], v[1:8]
```

### 2. Fused Base GEMM + Dynamic LoRA Branch (Action 2.2)
Computes the base model projection and dynamic adapter rank addition without intermediate global VRAM memory round-trips:

$$C = A \cdot W + \alpha (A \cdot L_A) \cdot L_B$$

```python
from runtime.triton_wmma import fused_wmma_lora_matmul

# Executed in a single fused Triton pass with WMMA hardware tensor core acceleration
out = fused_wmma_lora_matmul(x, base_w, lora_a, lora_b, alpha=alpha)
```

---

## 📊 Empirical Benchmark Results

Measured on **AMD Radeon RX 7900 XTX** (`gfx1100`, ROCm 7.2.4, CachyOS, PyTorch 2.13.0):

```
┌────────────────┬────────────────┬──────────────┬──────────────┬─────────┬───────────────┬──────────────┐
│ Matrix Shape   │ Workload / Reg │ PyTorch (ms) │ Triton (ms)  │ Speedup │ Triton TFLOPS │ Torch TFLOPS │
├────────────────┼────────────────┼──────────────┼──────────────┼─────────┼───────────────┼──────────────┤
│    2x4096x4096 │ MTP Spec (K=2) │       0.1121 │       0.0758 │   1.48x │          0.89 │         0.60 │
│    4x4096x4096 │ MTP Spec (K=4) │       0.0497 │       0.0770 │   0.64x │          1.74 │         2.70 │
│   16x4096x4096 │ Micro-batch 16 │       0.0474 │       0.0774 │   0.61x │          6.93 │        11.33 │
│   64x4096x4096 │ Prefill Chunk  │       0.0467 │       0.0748 │   0.62x │         28.72 │        46.00 │
│  256x4096x4096 │ Prefill Medium │       0.1503 │       0.1549 │   0.97x │         55.47 │        57.14 │
│  512x4096x4096 │ Prefill Large  │       0.2482 │       0.2614 │   0.95x │         65.72 │        69.23 │
│ 1024x4096x4096 │ Serving Batch  │       0.4656 │       0.4687 │   0.99x │         73.31 │        73.80 │
└────────────────┴────────────────┴──────────────┴──────────────┴─────────┴───────────────┴──────────────┘
```

Raw telemetry artifact: [`results/benchmarks/triton_wmma_perf.json`](file:///home/mihai/Projects/gnn-experiment/results/benchmarks/triton_wmma_perf.json).

### ⚠️ CORRECTION (2026-08-24) — honest reading of the table

The headline claims ("73+ TFLOPS", "1.48× faster") are contextually misleading when read as general performance wins:

1. **The 1.48× speedup applies only to M=2** (the K=2 speculative verification case). At M≥4, **PyTorch is faster** (0.61×–0.99×). The custom kernel is a narrow-M specialist, not a general replacement.

2. **73.31 TFLOPS is at M=1024**, where PyTorch also hits **73.80 TFLOPS**. Our kernel *matched* PyTorch at the largest shape — it did not beat it. The TFLOPS figure demonstrates that WMMA hardware is being utilized, not that our kernel is superior.

3. **The real contribution is ISA verification**: proving that `v_wmma_f32_16x16x16_bf16` instructions actually compile and fire on `gfx1100` under Triton's Wave32 backend. This is platform enablement (making the hardware work at all), not a general performance win over PyTorch's already-tuned rocBLAS.

**What to claim**: "We verified native WMMA emission on RDNA3 and wrote a Triton kernel that wins at M=2 (the speculative verification hot path). At larger shapes, PyTorch's tuned BLAS is as fast or faster."

---

## 📂 Subdirectory Index

| Directory | Scope |
| :--- | :--- |
| **[`prefill_vs_decode/`](prefill_vs_decode/)** | Characterizes latency share between prompt ingestion (5.9%) and autoregressive decoding (94.1%). |
| **[`fla_triton_kernels/`](fla_triton_kernels/)** | `flash-linear-attention` Triton bindings on RDNA3 for GatedDeltaNet SSM layers. |
| **[`batch_scaling/`](batch_scaling/)** | Multi-request batch scaling frontiers from $B=1$ to $B=64$ (saturating at 611 tok/s). |
| **[`serving_path/`](serving_path/)** | Async zero-copy Server-Sent Events (SSE) token streaming benchmarks. |

---

## 🛠️ How to Reproduce & Test

```bash
# 1. Run the Triton WMMA benchmark and ISA disassembler inspection
uv run python benchmarks/runtime/performance/benchmark_triton_wmma.py

# 2. Run the Triton WMMA unit test suite (accuracy, 3D batching, fused LoRA, ISA validation)
uv run pytest tests/test_triton_wmma.py -v
```
