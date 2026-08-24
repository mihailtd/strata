# ⚡ Single-Pass Range Statistic Speculative Early-Exit Gating

> **Grounding:** Chapter 8 (*Outlier Detection Based on Range Statistic Empirical Evaluation and Comparisons* — Dania Dallah, Hana Sulieman, Ayman Alzaatreh).  
> **Target Problem:** Pruning low-confidence draft tokens dynamically before verification, eliminating wasted GPU verification passes without 2-pass entropy overhead.  
> **Raw Telemetry Artifact:** [`results/benchmarks/range_speculative_gating.json`](../../../../results/benchmarks/range_speculative_gating.json)

---

## 🧭 Executive Summary & Core Discovery

In multi-token speculative decoding, the draft head typically generates a fixed window of $K=4$ or $K=6$ tokens regardless of uncertainty. When the drafter encounters ambiguous tokens, continuing to draft steps $i=2..K$ produces low-quality guesses that inevitably fail verification, wasting GPU decode latency.

Traditional confidence gating requires evaluating full logit entropy ($-\sum p \log p$) or variance ($\sigma^2$), which demands 2-pass reductions, exponentiations, and shared memory synchronization across 152,000 vocabulary words.

By replacing entropy with **Single-Pass Extreme Range Statistics** ($R_8 = z_{(1)} - z_{(8)}$ on top-8 candidate logits):
1. **$O(1)$ Register Execution**: Range spread is calculated in a single fast partial sort directly in GPU registers with **zero shared-memory reductions**.
2. **+18.8% Net Speedup**: Delivers **68.49 tok/s** (vs. **57.65 tok/s** for blind fixed-width speculation), pruning **36.4% of wasted draft tokens** while preserving full acceptance yield ($\tau = 2.92$).

---

## 📊 Measured Benchmark Telemetry (GPU: AMD RX 7900 XTX)

Evaluating across 500 speculative verification rounds on mixed-entropy token streams:

```
┌────────────────────────────────────────────────────────┬──────────────────────┬────────────────────────┬────────────────────────┐
│ Gating Strategy                                        │ Acceptance Yield (τ) │ Wasted Drafts Pruned   │ Decode Velocity (tok/s)│
├────────────────────────────────────────────────────────┼──────────────────────┼────────────────────────┼────────────────────────┤
│ Arm A: Blind Fixed-Width Speculation (K=4, No Gate)    │            2.92 tok  │                  0.0%  │              57.65     │
│ Arm B: Softmax Entropy Gating (2-Pass Softmax + Log)   │            1.75 tok  │                 74.4%  │              51.08     │
│ Arm C: Single-Pass Range Statistic Gate (Chapter 8)    │        **2.92 tok**  │             **36.4%**  │          **68.49 🔥**  │
└────────────────────────────────────────────────────────┴──────────────────────┴────────────────────────┴────────────────────────┘
```

---

## 🔬 Mathematical Breakdown & Analysis

### 1. Extreme Range Spread Statistic ($R_M$)
For ordered top-$M$ candidate logits $z_{(1)} \ge z_{(2)} \ge \dots \ge z_{(M)}$:

$$R_M = z_{(1)} - z_{(M)}$$

* **Peaked Distribution (High Confidence)**: Top candidate dominates ($z_{(1)} \gg z_{(M)}$), yielding $R_M \ge \tau_R$. Drafting proceeds without interruption.
* **Flat Distribution (Uncertainty / Outlier)**: Multiple candidates have similar scores ($z_{(1)} \approx z_{(M)}$), yielding $R_M < \tau_R$. The gate aborts the draft chain immediately at step $i$, returning a truncated chunk of width $k_{\text{actual}} < K$.

### 2. Eliminating Wasted Speculative Chunk Overhead
Because our speculative decoder ([`src/runtime/bucketed_speculative.py`](file:///home/mihai/Projects/gnn-experiment/src/runtime/bucketed_speculative.py)) maintains **pre-captured discrete bucket graphs** for widths $1 \dots K+1$, early-exited chunks of width $k_{\text{actual}}$ execute directly on the $k_{\text{actual}}+1$ graph bucket, reducing base model verification latency from $33.5\text{ ms} \to 28.7\text{ ms}$.

---

## 🛠️ Usage in Codebase

Import `RangeStatisticGate` from [`src/runtime/range_statistic_gate.py`](file:///home/mihai/Projects/gnn-experiment/src/runtime/range_statistic_gate.py):

```python
from runtime.range_statistic_gate import RangeStatisticGate
from runtime.mtp_draft import Qwen35MTPDraftHead

# Initialize Chapter 8 single-pass range gate
gate = RangeStatisticGate(top_m=8, threshold=5.0, mode="extreme_range")

# Draft up to K tokens with dynamic early-exit uncertainty gating
draft_tokens = draft_head.draft(
    hidden=H[:, -1:, :],
    next_token=nxt,
    k=4,
    start_pos=pos - 1,
    cache=dcache,
    gate=gate,  # <-- single-pass range gating enabled
)
```
