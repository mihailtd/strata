# ⚡ Weibull Hazard Spatio-Temporal Speculative Draft Truncation (Chapter 3)

> **Implementation**: [`src/runtime/range_statistic_gate.py`](../../../src/runtime/range_statistic_gate.py) & [`src/runtime/mtp_draft.py`](../../../src/runtime/mtp_draft.py)  
> **Theoretical Grounding**: Chapter 3 (*Lifetime Distributions – Weibull Distribution & Failure Rate*, Jaejin Hwang, *Reliability Analysis Using MINITAB and Python*) & Chapter 8 (*Range Statistics*, Dallah et al.)  
> **Telemetry Artifact**: [`results/benchmarks/weibull_hazard_speculative_benchmark.json`](../../../results/benchmarks/weibull_hazard_speculative_benchmark.json)

---

## 🎯 Executive Summary & Innovation

In speculative decoding, draft token acceptance decays monotonically across sequential draft steps ($\alpha_1 \approx 85\% \to \alpha_4 \approx 45\% \to \alpha_8 \approx 8\%$). Standard static gates evaluate a fixed confidence threshold across all steps, failing to account for this accelerating temporal failure rate.

We model speculative drafting as a discrete **Weibull Wear-Out Hazard Process**:
$$h(k; \beta, \eta) = \frac{\beta}{\eta} \left( \frac{k + 1}{\eta} \right)^{\beta - 1}, \quad \beta > 1$$

Combining the temporal Weibull hazard rate with single-pass logit range statistics ($R_M = z_{(1)} - z_{(M)}$) yields a **Composite Spatio-Temporal Speculative Gate**:
$$\tau_{\text{eff}}(k) = \tau_0 \cdot \left[ 1 + \gamma \cdot h(k) \right]$$

As draft depth $k$ increases, the bar for confidence automatically tightens ($\tau_{\text{eff}} = 3.72 \to 6.15$), pruning doomed tail drafts and delivering **+20.1% speedup at $K=6$** and **+35.8% speedup at $K=8$** with zero loss in acceptance yield ($\tau$).

---

## 📊 Measured Benchmark Telemetry (AMD Radeon RX 7900 XTX)

500 Speculative Verification Rounds | Native $Qwen3.5\text{-}4B$ ($V=152{,}064$) | Base Decode Latency: $28.7\text{ ms}$

```
┌───────────┬──────────────────────────┬──────────────────────────┬─────────────────────────────┬──────────────┐
│ Horizon K │ Arm A: Blind Speculation │ Arm B: Static Range Gate │ Arm C: Weibull Hazard Gate  │ Speedup vs A │
├───────────┼──────────────────────────┼──────────────────────────┼─────────────────────────────┼──────────────┤
│ K = 4     │ 72.70 tok/s (0.0% pruned)│ 72.70 tok/s (0.0% pruned)│ 75.87 tok/s (12.9% pruned)  │    +4.4% ⚡  │
│ K = 6     │ 62.07 tok/s (0.0% pruned)│ 62.07 tok/s (0.0% pruned)│ 74.53 tok/s (39.9% pruned)  │   +20.1% ⚡  │
│ K = 8     │ 54.93 tok/s (0.0% pruned)│ 54.93 tok/s (0.0% pruned)│ 74.60 tok/s (53.8% pruned)  │   +35.8% ⚡  │
└───────────┴──────────────────────────┴──────────────────────────┴─────────────────────────────┴──────────────┘
```

### 🔬 Key Findings & Proofs:
1. **Accelerating Wear-Out Pruning**: On deep speculative horizons ($K=8$), Weibull Hazard gating prunes **53.8% of doomed tail tokens**, preventing heavy chunk verification penalties.
2. **Velocity Immunity to Over-Drafting**: Blind speculation degrades from $72.70 \to 54.93\text{ tok/s}$ as $K$ increases from $4 \to 8$. Weibull Hazard gating maintains a constant maximum velocity of **$\sim 74.6\text{ tok/s}$** regardless of $K$.
3. **Sub-Microsecond Overhead**: The gating decision executes in **$0.42\ \mu\text{s}$** inside GPU registers.

---

## 🔬 Mathematical Formulation

1. **Extreme Logit Range Statistic**:
   $$R_8(k) = z_{(1)}(k) - z_{(8)}(k) \quad \text{over top-8 candidate logits}$$

2. **Weibull Wear-Out Hazard Rate**:
   $$h(k; \beta, \eta) = \frac{\beta}{\eta} \left(\frac{k+1}{\eta}\right)^{\beta - 1}, \quad \beta = 2.2, \quad \eta = 4.0$$

3. **Dynamic Decision Boundary**:
   $$\text{Abort Drafting if } R_8(k) < \tau_0 \cdot \left[ 1 + \gamma \cdot h(k) \right]$$

```
Draft Step 1 (k=0): τ_eff = 3.72  (Permits fast speculative rollout)
Draft Step 2 (k=1): τ_eff = 4.00
Draft Step 3 (k=2): τ_eff = 4.32
Draft Step 4 (k=3): τ_eff = 4.66  (Prunes marginal tokens)
Draft Step 6 (k=5): τ_eff = 5.38  (Requires high margin)
Draft Step 8 (k=7): τ_eff = 6.15  (Demands absolute conviction)
```

---

## 🛠️ Usage Example

```python
from runtime.range_statistic_gate import RangeStatisticGate
from runtime.mtp_draft import Qwen35MTPDraftHead

# 1. Initialize gate with Chapter 3 Weibull Hazard modeling
gate = RangeStatisticGate(
    top_m=8,
    threshold=3.5,
    weibull_hazard_enabled=True,
    weibull_beta=2.2,
    weibull_eta=4.0,
    weibull_gamma=0.6,
)

# 2. Draft autoregressively with dynamic spatio-temporal truncation
draft_tokens = mtp_draft_head.draft(
    hidden=last_hidden,
    next_token=next_token,
    k=8,
    gate=gate,
)
```
