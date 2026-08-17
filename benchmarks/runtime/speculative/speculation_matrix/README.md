# ⭐ In-Domain & Cross-Domain Speculative Decoding Matrix (3x3 Grid)

> **Tier Classification**: **⭐ Industry Standard** (Matrix Benchmarking) / **🔥 Applied Practice** (Transactional Recurrent Rollback Engine)  
> **Focus**: **Rigorous 3x3 audit (180 runs, 3 interleaved repeats) measuring empirical acceptance ($\tau$) and net speedup across all domain adapters on M2 (`r8a128` in native `bf16`).**

---

### Classification Breakdown: What is Standard vs. What is Innovative
* **⭐ Industry Standard Baseline**: Characterizing speculation acceptance rates across domain tasks to build heuristic routing tables.
* **🔥 Our Innovative Applied Practice**: Executing all 2160 timed generations with the **52.5 MB Recurrent State Snapshot & Restore Engine** under live **In-Place Low-Rank Weight Folding**. This allows the router to dynamically enable or disable speculative drafting per expert domain on the fly (`{"astral": true, "postgresql": true, "financial_planning": false}`).

---

## 1. The 3x3 Evaluation Design
The audit tests 9 distinct conditions (3 experts $\times$ 3 prompt domains), with **40 prompts per domain** across 3 interleaved repeats and 2 arms — **2160 timed generations** plus 1080 chunked-reference runs:
- **On-Diagonal (In-Domain)**: The adapter active matches the prompt domain (e.g., `astral` expert answering Astral prompts).
- **Off-Diagonal (Cross-Domain)**: The adapter active differs from the prompt domain (e.g., `financial` expert answering Postgres prompts).

---

## 2. Empirical Results on M2 Regime (`m2_r8a128` in native `bf16`)

| Folded Expert | Prompt Set | Mean $\tau$ | Accept % | Speedup (median) | 3-Repeat Range | Predicted | Exact vs Chunked | Router Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **astral** | **astral (DIAG 🔥)** | **1.93** | 48.3% | **1.09x** | 1.090–1.133 | 1.102 | 78% | **ENABLED ✅** |
| astral | postgresql | 1.85 | 46.3% | **1.10x** | 1.093–1.101 | 1.071 | 80% | — |
| astral | financial_planning | 1.93 | 48.2% | **1.12x** | 1.099–1.135 | 1.100 | 90% | — |
| postgresql | astral | 2.19 | 54.7% | **1.26x** | 1.263–1.269 | 1.197 | 90% | — |
| **postgresql** | **postgresql (DIAG 🔥)** | **1.94** | 48.5% | **1.14x** | 1.132–1.152 | 1.105 | 95% | **ENABLED ✅** |
| postgresql | financial_planning | 1.78 | 44.5% | **1.07x** | 1.053–1.077 | 1.045 | 95% | — |
| financial_planning | astral | 2.24 | 56.0% | **1.30x** | 1.295–1.305 | 1.217 | 82% | — |
| financial_planning | postgresql | 1.90 | 47.6% | **1.11x** | 1.109–1.121 | 1.091 | 90% | — |
| **financial_planning** | **financial_planning (DIAG 🔥)** | **1.70** | 42.5% | **1.011x** | 0.996–1.012 ⚠️ | 1.014 | 82.5% | **DISABLED ❌** (unresolved) |

### Dynamic Router Configuration
The gate is **measured speedup > 1.0 AND $\tau \ge 1.39$ AND the repeats do not straddle 1.0**. `financial_planning`'s diagonal has a median of 1.011x but repeats spanning 0.996–1.012, so its win is *unmeasured* rather than absent — it is gated off for lack of a resolved result, not because it is slow:
```json
{
  "astral": true,
  "postgresql": true,
  "financial_planning": false
}
```

---

## 3. Scripts
- **`benchmark_mtp_indomain_speculation_matrix.py`**: Executes the full 9-cell matrix audit with bootstrap confidence intervals and writes results to `results/mtp_indomain_speculation_matrix.json`.
