# ⚡ Pillar 2: The Runtime Engine (Serving Dynamics, State Retention & Speculation)

```
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                          THE RUNTIME SERVING ENGINE & SPECULATION                            ║
║                                                                                              ║
║  1. Recurrent State Handoff ($S_t$) : Zero multi-turn lag ($O(1)$ constant 48ms TTFT)        ║
║  2. In-Register Mixture-of-Adapters : Instant multi-expert dynamic routing in +0.030ms       ║
║  3. Weibull Hazard Speculative Gating: Intelligent draft truncation saving compute on divergence║
║  4. Factor VRAM Standby Residency   : 200x memory reduction enabling 200+ resident LoRAs     ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

## 🎯 High-Level Overview (For Humans / Layman's Terms)
Traditional LLM servers like Ollama forget the internal state after every single question. When you ask question #5 in a conversation, Ollama has to read your entire conversation history from scratch, causing a frustrating $5\text{--}8\text{ second}$ pause before it starts typing.

**The Runtime Engine makes conversation instantaneous:**
1. **$S_t$ Recurrent State Memory:** We keep the compressed conversational brain state ($54.97\text{ MB}$) warm in GPU memory. When you send a new question, typing begins in **$48\text{ milliseconds}$ flat**, even after 25,000 tokens of history!
2. **Instant Hot-Swappable Experts:** Our engine swaps or blends specialist adapters (Postgres, DuckDB, Python, Financial) in **under 30 microseconds** without reloading weights or stalling the GPU.
3. **Hazard Gating:** Like an anti-lock braking system (ABS), our runtime detects when the model is uncertain and gracefully truncates speculative predictions to prevent hallucination.

---

## 🔬 Subsystem Suite

| Directory | Sub-Benchmark | Key Metric | Layman's Analogy |
| :--- | :--- | :--- | :--- |
| [`folding/`](./folding/) | **In-Place Weight Absorption & MoA** | $+82.1\%$ Speedup vs PEFT | Morphing the tool in your hand instead of putting it away and opening a toolbox. |
| [`memory/factor_residency/`](./memory/factor_residency/) | **Factor-Based VRAM Residency** | $200.6\times$ Memory Reduction | Storing flat-packed IKEA furniture in the closet instead of fully assembled tables. |
| [`memory/pristine_state_buffer/`](./memory/pristine_state_buffer/) | **Zero-Drift Pristine Buffer ($W_0$)** | $L_\infty = 0.00$ Bit-Exact Drift | Database Transaction Rollbacks ensuring the base weights never accumulate rust or errors. |
| [`speculative/`](./speculative/) | **Speculative Graph Replay** | $2.20\times$ Speedup at $K=6$ | Autocomplete for sentences that verifies whole paragraphs at once. |
| [`speculative/weibull_hazard_gating/`](./speculative/weibull_hazard_gating/README.md) | **Weibull Hazard Gating** | 4-Phase Hazard Truncation | ABS brakes that automatically slow down when driving on icy roads. |
| [`agentic/CUT_SET_README.md`](../agentic/CUT_SET_README.md) | **Minimal Cut-Set DAG Reliability** | $k$-out-of-$n$ Hedging | Dual backup generators ensuring the hospital power never goes out. |

---

## 📐 Mathematical Formalism

### 1. Recurrent State Handoff ($S_t$)
In Qwen 3.x Gated DeltaNet layers, sequence history is compressed into recurrent state $S_t \in \mathbb{R}^{B \times H \times d_k \times d_v}$:
$$S_t = \alpha_t S_{t-1} + K_t^T (V_t - K_t S_{t-1})$$
$$O_t = Q_t S_t$$
Between conversation turns, $S_t$ is preserved in static GPU memory buffers. For a new prompt of length $L$, prefill complexity is $O(L)$ instead of $O(N_{\text{history}} + L)$, eliminating quadratic re-prefill stalls.

### 2. In-Register Mixture-of-Adapters (MoA) Stacking
For $E$ active domain adapters with routing weights $w_i \ge 0$:
$$Y = X W_{\text{base}} + \sum_{i=1}^{E} w_i \cdot \frac{\alpha_i}{r_i} (X A_i) B_i$$
Executed in-place within Triton register accumulators with zero dynamic tensor allocation.
