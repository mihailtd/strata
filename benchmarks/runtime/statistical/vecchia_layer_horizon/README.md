# Vecchia Horizon Windowing (Ch.12 §12.2.1)

**Status: ⚠️ PARTIALLY VALIDATED — the method survives, the proposed `m = 2` does not.**
Ranked P1 in the original proposal; measured P3.

**Scope boundary: only worth running at L ≥ 64 layers.** On the current 32-layer 4B model
it is **2× slower** than a dense inverse. It earns its place on the 70B/72B streaming path
(L = 80), not on the model in production today.

---

## 💡 The problem in plain English

Tracking how activations co-vary across an L-layer stack means estimating an L×L
covariance and inverting it — O(L³) — every time you update. **Vecchia** says: condition
each layer only on the handful of layers immediately before it. The precision matrix
becomes *banded*, and every update is O(L·m²).

Mathematically this is a modified Cholesky decomposition with a banded triangular factor:

$$T \Sigma T' = D, \qquad \Theta = \Sigma^{-1} = T' D^{-1} T$$

Fitting is L independent ridge regressions of layer ℓ on its m predecessors. No L×L
inverse ever forms.

**The speedup is arithmetic and not in question. The assumption is.** Whether dependence
beyond m layers back is negligible is an empirical claim about *this model's residual
stream*, and it is what this benchmark exists to test.

---

## 🔬 What was measured

**41,535 real token positions**, captured from a `Qwen3.5-4B` forward pass over 288
passages spanning all six domain corpora, using the true per-layer residual delta
‖h_{ℓ+1} − h_ℓ‖.

Held-out negative log-likelihood decides `m`. In-sample likelihood falls monotonically
with `m` by construction and would "prove" any answer you like.

---

## 📊 Results

### 1. Is `m = 2` enough? (24,921 train / 16,614 held out) — **No.**

```
┌─────┬───────────────┬──────────────┬─────────────┬────────────┬─────────┐
│  m  │ held-out NLL  │ gain vs m=0  │ KL to dense │ band mass  │ params  │
├─────┼───────────────┼──────────────┼─────────────┼────────────┼─────────┤
│  0  │     44.5440   │        0.0%  │    13.2169  │      0.0%  │     32  │
│  1  │     35.3876   │       77.7%  │     3.1398  │     35.7%  │     63  │
│  2  │     34.9232   │       81.6%  │     2.6056  │     43.0%  │     93  │ ← proposed
│  4  │     34.2792   │       87.1%  │     1.8680  │     58.3%  │    150  │
│  8  │     33.3905   │       94.6%  │     0.6972  │     74.0%  │    252  │ ← recommended
│ 16  │     32.9700   │       98.2%  │     0.2498  │     90.3%  │    408  │
│ 24  │     32.8191   │       99.5%  │     0.0764  │     98.0%  │    500  │
│dense│     32.7586   │      100.0%  │     0.0000  │    100.0%  │    528  │
└─────┴───────────────┴──────────────┴─────────────┴────────────┴─────────┘
```

`m = 2` captures 81.6% of the held-out likelihood gain but only **43.0% of the precision
mass** — the majority of conditional dependence lies *outside* the band. The curve does
not flatten (within 0.05 nats) until **m = 24**.

**`m = 8` is the defensible operating point**: 94.6% of the gain, KL 0.70, 252 stored
parameters against the dense 528.

### 2. Why: the dependence decays, but not to zero

Mean |partial correlation| by layer separation:

```
lag 1: 0.324   lag 2: 0.093   lag 3: 0.060   lag 4: 0.137   lag 5: 0.060   lag 6: 0.057
```

The banding premise is **directionally right** — a sharp drop after lag 1 — it just does
not decay to nothing. Residual skip connections carry signal much further than 2 layers.

> 🔎 **Unexplained: the lag-4 bump (0.137).** It breaks an otherwise monotone decay. Four is
> also the attention-adapter stride in the v7 adapters — but this arm measures the **base
> model with no adapters loaded**, so that cannot be the cause here. Worth a look.

### 3. Is the speedup real? (banded fit vs dense covariance + inverse, m = 2, n = 64)

```
┌──────┬───────────────┬─────────────────┬───────────┬────────────────┬──────────────────┐
│  L   │ dense fit ms  │ Vecchia fit ms  │  speedup  │ dense quad µs  │ Vecchia quad µs  │
├──────┼───────────────┼─────────────────┼───────────┼────────────────┼──────────────────┤
│    8 │       0.0100  │         0.0396  │    0.25x  │         1.12   │          5.37 ⚠️ │
│   32 │       0.0221  │         0.0440  │    0.50x  │         1.18   │          5.43 ⚠️ │ ← 4B model
│   64 │       0.0610  │         0.0489  │    1.25x  │         1.37   │          5.45 ⚠️ │
│   80 │       0.0954  │         0.0533  │    1.79x  │         1.54   │          5.49 ⚠️ │ ← 70B model
│  128 │       0.2476  │         0.0615  │    4.03x  │         1.98   │          5.53 ⚠️ │
│  256 │       0.9065  │         0.0851  │   10.65x  │         4.87   │          5.73 ⚠️ │
│  512 │       5.4681  │         0.1345  │   40.66x  │        20.02   │          6.54 🔥 │
└──────┴───────────────┴─────────────────┴───────────┴────────────────┴──────────────────┘
```

Measured complexity exponents: **dense L^1.54, Vecchia L^0.28**. Not L³, and not L.

**Two corrections to the O(L³) → O(L) framing, both measured:**

1. Forming the covariance costs **O(n·L²)** and dominates the O(L³) inverse until L ≳ 256.
   At real stack depths you are paying the quadratic term, not the cubic one.
2. **Below L = 64 the banded fit loses outright** to one LAPACK call on a small dense
   matrix — 0.50× at L = 32.

**The apply side is a loss at every real depth.** A dense Θ·y matvec beats the banded
quadratic form until L = 512.

> The proposed "<0.4 ms in a single Triton fused pass" is met on **CPU** — 0.053 ms at
> L = 80 — but so is the dense baseline, at 0.095 ms. No Triton/GPU figure is claimed:
> these are 32–80 element problems where a kernel launch costs more than the arithmetic.

### 4. Null control — the estimator does not manufacture structure

A surrogate stream built from real v7 adapter factors driven by **Gaussian** inputs shows
flat partial correlations (~0.03 at every lag) and a dense fit that does **not** beat the
diagonal model on held-out data. When there is no cross-layer structure, this benchmark
reports none — which is what makes the real arm's numbers mean something.

---

## ✅ Verdict and deployment rules

| | |
| :--- | :--- |
| **Ship it, conditionally** | `src/runtime/vecchia_precision.py` → `fit_vecchia_batched` |
| **Use `m = 8`** | never `m = 2`. m=2 misses 57% of the precision mass. |
| **Only at L ≥ 64** | at L = 32 (the 4B model) it is **0.50×** — a regression. Use the dense inverse there. |
| **Fit side only** | the banded apply loses to a dense matvec below L = 512. Use it to *build* Θ, not to apply it. |
| **Buys at L = 80** | **1.79×** on the refit |

---

## 🛠️ Reproduce

```bash
# GPU used ONLY for the activation capture (39 s vs ~10 min on CPU); estimators stay on CPU
uv run python benchmarks/runtime/statistical/vecchia_layer_horizon/benchmark_vecchia_horizon.py --gpu-capture

# Fully CPU-bound, safe while the GPU is training
uv run python benchmarks/runtime/statistical/vecchia_layer_horizon/benchmark_vecchia_horizon.py

uv run pytest tests/test_statistical_estimators.py -k "vecchia or horizon or band" -v
```

Runtime ≈ 6 s with the activation cache warm. **Run `--serial` via `run_all.py` if you care
about the timing table** — concurrent processes contend and inflate latency figures.

Raw telemetry: [`results/benchmarks/vecchia_layer_horizon.json`](../../../../results/benchmarks/vecchia_layer_horizon.json)
