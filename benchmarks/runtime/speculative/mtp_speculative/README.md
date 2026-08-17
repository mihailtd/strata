# 🔥 Native MTP Speculative Decoding Engine (K-Sweep Frontier)

> **Tier Classification**: **🔥 Applied Practice**  
> **Concept Origin**: **Transactional Memory State Checkpointing (from database write-ahead logging / game state snapshots) applied to the LLM runtime.**

---

### Classification Breakdown: What is Standard vs. What is Innovative
* **⭐ Industry Standard Component**: The general concept of EAGLE-style multi-token drafting and utilizing vendor-shipped Multi-Token Prediction (MTP) weights.
* **🔥 Our Innovative Applied Practice**: Adapting **Transactional Memory State Checkpointing** to the LLM runtime via a **52.5 MB Fixed-Size Recurrent State Buffer** (`snapshot_state()` / `restore_state()`). This solves an active, unsolved blocker in mainstream LLM frameworks (like HuggingFace `transformers`), which explicitly refuse assisted generation on stateful hybrid recurrent architectures (GatedDeltaNet).

---

## 1. The Active LLM Runtime Problem & Solution
Standard `transformers` throws a hard error when attempting speculative decoding on hybrid models:
> `"ValueError: assisted generation is not supported with stateful models, such as Qwen3_5ForCausalLM"`

In Qwen 3.5, 24 of 32 layers are GatedDeltaNet. When candidate draft tokens are rejected, the recurrent hidden state cannot be rolled back by simple KV-cache slicing. 

* **The Transactional Rollback Engine**: The engine captures a fixed-size **52.5 MB** state snapshot before verification and restores it in-place in **$<0.5\text{ ms}$** via bit-exact tensor copies upon partial acceptance ($\tau < K$).
* **Tied Draft Context**: The auxiliary draft head (`Qwen35MTPDraftHead`) drafts $K$ candidate tokens autoregressively from committed hidden states without stepping the base model's recurrent state.

---

## 2. Hardware Verification Costs on AMD ROCm (`gfx1100`)
With `flash-linear-attention` 0.5.2 and `triton-rocm` 3.7.1 active (`fla_active: true` in `results/fla_verification_profile.json`), verification costs scale smoothly:

| Verification Length ($K$) | Latency (ms) | Scaling vs $K=1$ | Status |
| :---: | :---: | :---: | :---: |
| **$K=1$** | 27.87 ms | 1.00x | Base single-token step |
| **$K=2$** | 38.06 ms | 1.37x | Chunked forward |
| **$K=4$** | **33.27 ms** | **1.19x** | Chunked forward (optimal efficiency) |
| **$K=8$** | 34.23 ms | 1.23x | Chunked forward |

* **Break-Even Analysis**: Because $K=4$ verification costs only **$1.19\times$** a single-token step (replacing the old un-fused $2.84\times$ penalty), the theoretical break-even threshold is **$\tau \ge 1.39$ accepted tokens/step**.

---

## 3. Empirical Steady-State Speculative Velocity (256-Token Horizon)

Benchmarked on `Qwen/Qwen3.5-4B` in `bfloat16` across full 256-token sequences:

| Draft Length ($K$) | Decode Velocity (tok/s) | Net Speedup vs Greedy | Acceptance Rate (%) | Accepted Tokens / Step ($\tau$) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Plain Greedy)** | **25.31 tok/s** | **1.00x** | — | 1.00 | Baseline |
| **$K=2$** | **39.92 tok/s** | **1.58x** | 85.6% | 1.71 | ✅ WIN |
| **$K=4$** | **50.06 tok/s** | **1.98x** | 71.0% | 2.84 | ✅ WIN (2x speed!) |
| **$K=6$ (Sweet Spot 🔥)** | **55.71 tok/s** | **2.20x (+120%)** | 60.4% | **3.62** | 🚀 **PEAK VELOCITY** |
| **$K=8$** | **53.84 tok/s** | **2.13x** | 52.1% | 4.17 | ✅ WIN |

---

## 4. Understanding Numerical Exactness in `bfloat16`
On hybrid recurrent architectures, the multi-token chunked kernel (used by verification) and the single-token recurrent kernel (used by plain decode) disagree numerically after $\approx 20\text{--}30$ tokens due to `bfloat16` accumulation order. Controls show that speculative decode diverges from plain greedy at the exact same rate as two non-speculative eager paths diverge from each other. The speedup numbers reflect valid, highly coherent text generation.

---

## 5. Usage & Scripts
- **`benchmark_mtp_speculative.py`**: Runs the $K$-sweep speculative scaling benchmark across $K \in [2, 4, 6, 8]$ at 256-token horizons:
  ```bash
  uv run --env-file .env python benchmarks/runtime/speculative/mtp_speculative/benchmark_mtp_speculative.py --tokens 256 --k 2 4 6 8
  ```
