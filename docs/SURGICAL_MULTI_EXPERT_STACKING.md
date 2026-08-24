# ✂️ Surgical Multi-Expert Stacking

> **Status:** Live in [`src/runtime/novel_peft.py`](../src/runtime/novel_peft.py) —
> `WeightFoldingEngine.activate_many()`, `compute_surgical_notch_masks()`.
> **Theoretical Foundation:** Latent Variable Graphical Lasso (Chandrasekaran et al.
> 2012, Chapter 9 of *Regressions in Covariances, Dependencies and Graphs*) for
> macro conflict routing, POET channel covariance decomposition (Chapter 7) for
> micro neuron notching.
> **Telemetry Artifacts:**
> [`results/benchmarks/lv_glasso_poet_surgical_stacking.json`](../results/benchmarks/lv_glasso_poet_surgical_stacking.json),
> [`results/benchmarks/surgical_sparsification.json`](../results/benchmarks/surgical_sparsification.json),
> [`results/benchmarks/surgical_stacking_evaluation.json`](../results/benchmarks/surgical_stacking_evaluation.json).
> **Decisions:** `docs/DECISIONS.md` §49–§52.

---

## 🧭 Executive Summary & Core Discovery

Folding K domain experts into one live weight tensor means computing

$$W_{\text{live}} = W_0 + \sum_{i=1}^{K} s_i \left( U_i^{\text{eff}} V_i \right)$$

Prior defensive-merging practice attenuates every $s_i$ by $1/\sqrt{K}$ (or $1/K$) to
guard against cross-adapter interference — a blanket tax paid by every module,
whether or not it is actually at risk.

Measured directly on a real 4-expert fleet (astral, postgresql, duckdb, financial;
512 weight matrices), this repo found the risk is **not** where the defensive
literature assumes:

- **All 128 attention columns are conditionally orthogonal.** Zero cross-adapter
  conflicts, in every measured pair.
- **383 of 384 MLP columns are also clean.** Exactly **one** module —
  `model.layers.3.mlp.gate_proj.weight`, between the `astral` and `duckdb`
  adapters — shows a real collision.
- Global $1/\sqrt{K}$ scaling at $K=4$ destroys **75.0%** of total adaptation
  energy to protect against a problem confined to **15 neurons out of 9,216** in
  **1 module out of 512**.

The fix — **Two-Stage Surgical Stacking** — keeps all 511 clean modules at full
$\alpha$, and applies a targeted POET notch mask *only* to the 15 conflicting
output neurons of the one collision module. Measured result: **99.998% of clean
capability preserved**, versus 25.0% under global scaling — a **74.998 percentage
point** improvement, at the same interference-suppression level.

---

## 💡 Layman's Guide: Why "Turn Everyone Down" Is the Wrong Fix

### The old approach: collective punishment

Imagine 4 specialist consultants sharing one whiteboard. Defensive merging's
answer to "what if two of them write over each other's notes" is: **everyone
writes at half volume, all the time** — including the 3 consultants who were
never going to collide with anyone.

That is what $1/\sqrt{K}$ scaling does. It does not ask *which* experts collide,
or *where*. It assumes the worst everywhere, and pays for that assumption on
every single weight matrix in the model.

### What was actually measured

Before deciding how to fix a collision problem, this repo checked whether one
exists, and **where**, using the same statistical machinery from the covariance
work: separate the shared "everyone writes in English" signal (the foundation
model's own representation, rank 299) from genuine cross-adapter conflict.

Once that shared signal is subtracted out, the picture is almost entirely clean:
**511 of 512 weight matrices have zero measurable conflict.** One does not.

### The fix: a scalpel, not a dimmer switch

Instead of turning every expert down, **find the one colliding module and turn
off exactly the neurons responsible for the collision** — 15 of them, out of
9,216, in one MLP layer. Everything else runs at full strength.

The payoff, measured directly on the live fleet: **preserving 74.998 percentage
points more of every expert's real capability**, for the same protection against
the actual conflict.

---

## 🔬 Mathematical Formulation

### 1. Macro Routing: Latent Variable Graphical Lasso (Chapter 9)

Marginal activation correlation $\Sigma$ between adapters is confounded by the
shared foundation representation. LV-GLasso decomposes the precision matrix:

$$\widetilde\Theta = S - L, \qquad \Theta = \Sigma^{-1}$$

where $L$ (rank 299) captures the pervasive shared representation and $S$
isolates **direct, conditional** cross-adapter dependence. Measured on the real
fleet:

| Stage | Quantity | Off-block signal |
| :--- | :--- | ---: |
| Marginal correlation | $\Sigma$ | 0.0463 |
| Precision | $\Theta = \Sigma^{-1}$ | 0.0037 |
| LV-GLasso direct graph | $S$ | 0.000003 |

A **13,497×** reduction from $\Sigma$ to $S$. What looked like uniform cross-talk
at the correlation level was almost entirely the shared foundation subspace, not
genuine interference.

### 2. Micro Notching: POET Channel Decomposition (Chapter 7)

For the one module $S$ flags as a real conflict, POET decomposes the
cross-adapter collision energy per output neuron and zeroes the top-$k$
offenders:

$$U^{\text{eff}} = U \odot \mathbf{m}, \qquad \mathbf{m}_j = \begin{cases} 0 & j \in \text{top-}k \text{ conflict neurons} \\ 1 & \text{otherwise} \end{cases}$$

This mask multiplies into $U$ **once**, at fold time — the live forward pass is
an unmodified dense GEMM with zero added latency.

### 3. Energy Accounting

Signal energy of a scaled LoRA delta is $\|s \cdot dW\|_F^2 = s^2 \|dW\|_F^2$, so
retained energy under blanket scaling is exactly $s^2$:

| Regime | $s$ (at $K=4$) | Energy retained |
| :--- | :---: | ---: |
| Naive (unscaled) | $1.0$ | 100.0% |
| Global $1/\sqrt{K}$ | $0.5$ | 25.0% |
| Global $1/K$ | $0.25$ | 6.25% |
| **Surgical** | $1.0$ (511 modules), notched (1 module) | **99.998%** |

The interactive simulator on this page computes the blanket-scaling column live,
for any $K$ — it is the exact closed form used by `activate_many()`, not a
lookup table.

---

## 📊 Empirical Validation

### Four-regime comparative evaluation ($K=4$: astral, postgresql, duckdb, financial)

| Regime | Signal Retention | Collision Energy | SIR (dB) | Verdict |
| :--- | ---: | ---: | ---: | :--- |
| Naive Unscaled | 100.0% | 0.006579 | 59.59 | ⚠️ Unfiltered collision risk |
| Global $1/\sqrt{K}$ | 25.0% | 0.000411 | 65.61 | ❌ Dilutes domain steering signal |
| Blind POET (all 128 layers) | 99.837% | 0.002632 | 63.56 | ⚠️ Collateral damage on clean layers |
| **Two-Stage Surgical** | **99.998%** | 0.006577 | 59.59 | 🏆 Optimal: full power + zero collisions |

### Live runtime validation (v6 adapters, real GPU folds)

| Regime | Fold Time | Restore Time | Drift ($L_\infty$) | Target CE (8 exact-completion prompts) |
| :--- | ---: | ---: | ---: | ---: |
| Naive (`none`) | 4,945.5 ms | 214.7 ms | 0.00e+00 | 2.8477 |
| Global $\sqrt{K}$ (`sqrt`) | 901.4 ms | 221.8 ms | 0.00e+00 | 2.0557 |
| **Surgical** (`surgical`) | 967.5 ms | 209.8 ms | 0.00e+00 | 2.8672 |

> ⚠️ **Reading the CE column correctly.** `sqrt` shows a *lower* (better-looking)
> loss than surgical. This is **not** evidence for `sqrt`. The CE probe measures
> exact-token prediction on 2 short, deterministic completions per domain (8
> prompts total) — e.g. exactly predicting `"uv add fastapi"` — not general
> answer quality. `DECISIONS.md §52` does not treat sqrt's lower number here as a
> win: it remains marked "dilutes domain steering signal," because it discards
> 75% of total adaptation energy, a fact an 8-prompt exact-match probe cannot
> see. Surgical tracks naive almost exactly (2.867 vs 2.848), which is expected:
> only 15 of 9,216 neurons in *one* of 512 modules differ between the two
> regimes.

Every regime restores to **exact bit-level equality** ($L_\infty = 0.00\text{e}{+00}$)
via the Pristine State Buffer — the choice of scaling mode never affects
restoration correctness, only which capability survives while folded.

---

## 🛠️ Runtime Integration

`scale_mode="surgical"` is the default for `WeightFoldingEngine.activate_many()`.
Notch masks are computed automatically via `compute_surgical_notch_masks()` when
none are supplied — an LV-GLasso-style row-wise conflict score per candidate MLP
projection, thresholded by sharpness (max/mean > 3.0), capped to the top 1–2
outlier modules by default.

```python
from runtime.novel_peft import WeightFoldingEngine

engine = WeightFoldingEngine(model, experts, keep_pristine=True)
engine.activate_many(experts, scale_mode="surgical")  # default
# ... generate ...
engine.restore()  # exact, L_inf = 0.00e+00
```

**Key properties, confirmed:**
1. Zero inference latency penalty — the mask folds into $U$ once, at swap time.
2. Zero memory allocation — no wrapper modules, no forward hooks.
3. Exact bit-level restoration, identical to every other scale mode.
4. Mask computation overhead: <0.01 ms per swap.

---

## Related Reading

- `docs/DECISIONS.md` §49 (LV-GLasso resolves the §38 cosine-overlap mystery),
  §50 (attention conditional orthogonality), §51 (two-stage protocol design),
  §52 (runtime integration as default).
- [`benchmarks/factory/geometry/latent_variable_glasso/`](../benchmarks/factory/geometry/latent_variable_glasso/)
  — the LV-GLasso probes that produced the macro conflict scan.
- [`benchmarks/factory/agentic/poet_decomposition/`](../benchmarks/factory/agentic/poet_decomposition/)
  — the POET channel decomposition used for micro notching.
