# High-Batch Scaling Frontier ($B=1 \dots 64$)

> **Tier Classification**: **⭐ Industry Standard**  
> **Focus**: **High-throughput continuous batching characterization and verification cost scaling on bare-metal CachyOS + AMD ROCm 7.2.4.**

---

## Layman's Terms: What Is Batch Scaling & Why Does Weight Folding Win?

* **What is Batching?** 
  * When a GPU generates words for 1 user ($B=1$), it has to read the entire 8 GB model from VRAM for every single word. The GPU's computation engines sit mostly idle waiting for memory.
  * When we batch requests together ($B=4, 8, 16$), the GPU reads the 8 GB model *once* and generates words for 4, 8, or 16 users simultaneously.
  * Notice how at $B=4$, it takes only **36.82 ms** (almost the same as 33.53 ms at $B=1$), but delivers **108.64 total words/sec** instead of 29.83 words/sec.
* **Why Does Weight Folding Beat PEFT Wrappers?**
  * Standard adapter frameworks (like PEFT) run extra separate matrix operations alongside the main model.
  * Our **Weight Folding Engine** fuses the adapter deltas directly into the base weights in-place.
  * Across all batch sizes, folding eliminates wrapper overhead, delivering a **1.80x speedup at $B=1$** and maintaining a **1.15x–1.75x speedup** across all batch sizes!

---

## 1. Empirical Scaling Measurements on Native CachyOS (RX 7900 XTX)

### 1. Decode Step Scaling ($k=1$): Is Batching "Free"?
Tested on `Qwen/Qwen3.5-4B` in native `bfloat16` with deterministic SDPA attention on AMD RX 7900 XTX:

| Batch Size ($B$) | ms/step | Aggregate Throughput (tok/s) | Per-Request Throughput (tok/s) | Latency Penalty vs $B=1$ |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 33.53 ms | 29.83 tok/s | 29.83 tok/s | 1.00x |
| **2** | 36.17 ms | 55.30 tok/s | 27.65 tok/s | **1.08x (Near-Free!)** |
| **4** | 36.82 ms | 108.64 tok/s | 27.16 tok/s | **1.10x** |
| **8** | 42.96 ms | 186.20 tok/s | 23.28 tok/s | 1.28x |
| **16** | 55.25 ms | 289.61 tok/s | 18.10 tok/s | 1.65x |
| **32** | 87.19 ms | 367.01 tok/s | 11.47 tok/s | 2.60x |
| **64** | 149.83 ms | **427.14 tok/s** | 6.67 tok/s | 4.47x |

**Key Takeaway**: For interactive real-time multi-agent serving, **$B \le 4$ is the Goldilocks zone**—it quadruples aggregate throughput (**108.64 tok/s**) while keeping per-user latency virtually unchanged (27.16 tok/s vs 29.83 tok/s).

---

### 2. In-Place Weight Folding vs. Wrapped PEFT
Does the weight folding latency advantage survive under batched compute density?

| Batch Size ($B$) | Wrapped PEFT (ms) | Folded Weights (ms) | Folding Speedup Win |
| :---: | :---: | :---: | :---: |
| **1** | 60.67 ms | 33.65 ms | **1.80x** |
| **2** | 62.03 ms | 36.31 ms | **1.71x** |
| **4** | 64.50 ms | 36.94 ms | **1.75x** |
| **8** | 71.01 ms | 43.11 ms | **1.65x** |
| **16** | 80.87 ms | 55.04 ms | **1.47x** |
| **32** | 110.48 ms | 87.05 ms | **1.27x** |
| **64** | 172.44 ms | 149.98 ms | **1.15x** |

**Finding**: Weight folding maintains a **1.65x–1.80x** advantage for interactive serving ($B \le 8$) and continues delivering a **+15% to +27%** speedup even under heavy saturated batches ($B=32 \dots 64$).

---

## 2. Scripts
- **`benchmark_batch_scaling.py`**: Benchmarks decode latency, aggregate tokens/sec, chunked verification ratios, and wrapper vs. folding speedups across configurable batch sizes.
