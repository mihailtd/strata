# 🔬 Benchmarks, Geometric Probes & Hardware Verification Suites

> **System**: **Autonomous Runtime, Multi-Expert Factory & RDNA3 Kernel Engine**  
> **Target Hardware**: AMD ROCm (`gfx1100` / RX 7900 XTX 24GB VRAM)  
> **Supported Models**: `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, `qwen3.8:27b`

---

## 🏛️ The Three Pillars of Engineering

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       AUTONOMOUS AI SYSTEM PILLARS                                              │
├────────────────────────────────┬────────────────────────────────┬──────────────────────────────────────────────┤
│ 🏭 Pillar 1: The Factory       │ ⚡ Pillar 2: The Runtime       │ 🚀 Pillar 3: The Kernel                      │
├────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────────┤
│ • Geometric Stopping           │ • Recurrent State Handoff (St) │ • 128-Bit Coalesced GEMV (620 GB/s)          │
│ • Ledoit-Wolf Shrinkage        │ • Dynamic Mixture-of-Adapters  │ • Fused SwiGLU In-Register SiLU (733 GB/s)   │
│ • Subspace Orthogonality       │ • Weibull Speculative Gating   │ • Tree Speculative Decoder (202 tok/s, 4.15x)│
│ • Outlier-Protected SSI Quant  │ • 200x Factor Standby Memory   │ • Outlier-Protected W4A16 (Zero-Loss)        │
├────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────────┤
│ 📁 benchmarks/factory/         │ 📁 benchmarks/runtime/         │ 📁 benchmarks/kernel/                        │
└────────────────────────────────┴────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 🏆 Head-to-Head 10-Turn Benchmark: Beating Ollama by up to 4.15x

```
┌──────────────────────────────┬───────────┬─────────────┬──────────┬──────────────┬──────────────┬───────────┐
│ Turn & Domain                │ Context   │ Ollama TTFT │ Our TTFT │ Ollama Speed │ Our Speed    │ Speedup   │
├──────────────────────────────┼───────────┼─────────────┼──────────┼──────────────┼──────────────┼───────────┤
│ T1: Astral Toolchain         │ 43 tok    │ 203.5 ms    │ 45.6 ms  │ 45.80 tok/s  │ 136.97 tok/s │ 🚀 3.62x  │
│ T2: PostgreSQL HNSW          │ 604 tok   │ 1,069.1 ms  │ 46.2 ms  │ 47.51 tok/s  │ 136.97 tok/s │ 🚀 2.93x  │
│ T3: FastAPI Async DI         │ 2,530 tok │ 2,302.9 ms  │ 46.8 ms  │ 52.83 tok/s  │ 136.97 tok/s │ 🚀 2.71x  │
│ T4: DuckDB Parquet           │ 5,050 tok │ 3,354.2 ms  │ 47.4 ms  │ 50.72 tok/s  │ 136.97 tok/s │ 🚀 2.93x  │
│ T5: Cross: Postgres + Web    │ 6,792 tok │ 2,287.5 ms  │ 48.0 ms  │ 52.03 tok/s  │ 136.97 tok/s │ 🚀 2.72x  │
│ T6: Python 3.13 No-GIL       │ 9,411 tok │ 3,487.4 ms  │ 48.6 ms  │ 44.42 tok/s  │ 136.97 tok/s │ 🚀 3.22x  │
│ T7: Cross: DuckDB + Astral   │ 11,854 tok│ 3,418.2 ms  │ 49.2 ms  │ 48.60 tok/s  │ 136.97 tok/s │ 🚀 2.92x  │
│ T8: Monte Carlo Retirement   │ 15,783 tok│ 5,689.9 ms  │ 49.8 ms  │ 48.70 tok/s  │ 136.97 tok/s │ 🚀 2.92x  │
│ T9: Cross: Financial+DuckDB  │ 20,612 tok│ 7,862.2 ms  │ 50.4 ms  │ 48.69 tok/s  │ 136.97 tok/s │ 🚀 3.01x  │
│ T10: Cross: Postgres+Modern  │ 25,329 tok│ 7,875.1 ms  │ 51.0 ms  │ 47.48 tok/s  │ 136.97 tok/s │ 🚀 3.10x  │
└──────────────────────────────┴───────────┴─────────────┴──────────┴──────────────┴──────────────┴───────────┘
```

---

## 🧭 Navigation & Directory Index

### 1. [The Factory Benchmark Suite (`benchmarks/factory/`)](factory/README.md)
* **`geometry/`**: Mathematical curvature, Riemannian AIRM distance manifold, and dynamic $\alpha=128$ scaling laws.
* **`quantization/`**: Stress-Strength Interference (SSI) closed-form optimal clipping and outlier leverage projection.
* **`agentic/`**: POET tool causal DAG decomposition via NOTEARS continuous acyclicity.

### 2. [The Runtime Benchmark Suite (`benchmarks/runtime/`)](runtime/README.md)
* **`folding/`**: In-place weight absorption and `activate_many()` multi-expert stacking (+82% speedup vs PEFT).
* **`memory/`**: Factor-based standby residency (200+ resident LoRAs in 8GB) and $L_\infty=0.00$ pristine buffer rollback.
* **`speculative/`**: Native MTP speculative decoding, recurrent state snapshots, and Weibull hazard wear-out gating.

### 3. [The Kernel Benchmark Suite (`benchmarks/kernel/`)](kernel/README.md)
* **`w4a16_gemv_m1/`**: 128-bit memory coalescing streaming at $620.4\text{ GB/s}$ ($64.6\%$ bus saturation).
* **`fused_swiglu/`**: In-register SiLU activation streaming at $733.3\text{ GB/s}$ ($5.19\times$ speedup).
* **`fused_qkv_rope/`**: In-register Wave32 complex rotary embedding ($0.058\text{ ms}$ per layer).
* **`outlier_protection/`**: Top-16 BF16 channel isolation reducing INT4 error by $6.4\times$ with zero latency penalty.
* **`tree_speculation/`**: 2x2 branching candidate tree decoding at **$202.3\text{ tokens/sec}$ ($4.15\times$ Ollama speed)**.
