# Riemannian metric on the SPD cone — and what it says about our adapters

> Ch.3 §3.3 / §3.5 (norms, log of a covariance matrix), Ch.8 §8.1.4 (Ledoit-Wolf).
> Result: **the metric is now correct, and its answer is a refutation.**
> `DECISIONS.md` §59. Supersedes the §55 write-up, which was wrong.

```
CUDA_VISIBLE_DEVICES="" uv run python \
    benchmarks/factory/geometry/riemannian_metric/benchmark_riemannian_domain_distance.py
```

CPU only, ~16 s, 6 experts × 128 weight matrices. The script **refuses to run**
with a GPU visible.

---

## The instrument

For a LoRA delta `dW = s·U@V` the output-side operator `Σ = dW dWᵀ` is a function
of the adapter alone. It is 2560×2560 of rank 8, so 2552 eigenvalues are zero and
any regulariser you add to invert it dominates the answer. So both adapters are
projected into **one shared orthonormal basis** `Q` of
`span(range dW_a ∪ range dW_b)`, `k ≤ 2r = 16` dimensions:

```
Σ̃ = (QᵀU) [s² V Vᵀ] (QᵀU)ᵀ          k × k, rank ≤ r
d_R(Σ̃_a, Σ̃_b) = ‖log(Σ̃_a^{-1/2} Σ̃_b Σ̃_a^{-1/2})‖_F
```

AIRM is congruence-invariant and the spherical shrinkage target `μI` is invariant
under orthogonal conjugation, so `Q`'s arbitrary orientation cancels exactly. That
is *why* the number is well defined, and it is checked at runtime: the benchmark
**aborts** unless a random rank-basis rotation moves `d_R` by less than `1e-6`
(measured `1.3e-12`) and `d_R(X, X) < 1e-6` (measured `5.7e-15`).

`d_R` is also split into the two things it confounds:

| component | meaning |
| :--- | :--- |
| **scale** = \|tr L\|/√k | one adapter's delta is simply *bigger* |
| **shape** = ‖L − (tr L/k)I‖_F | the deltas point *elsewhere* |

`total² = scale² + shape²`. Only *shape* is a statement about the domain.

---

## What it measures — v6, δ = 0.05, mean over 128 weight matrices

| | astral | postgresql | duckdb | financial | python_modern | python_web |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **astral** | · | 14.219 | 14.198 | 14.204 | 14.190 | 14.185 |
| **postgresql** | 14.219 | · | 14.161 | 14.169 | 14.159 | 14.155 |
| **duckdb** | 14.198 | 14.161 | · | 14.148 | 14.131 | 14.127 |
| **financial** | 14.204 | 14.169 | 14.148 | · | 14.144 | 14.140 |
| **python_modern** | 14.190 | 14.159 | 14.131 | 14.144 | · | 14.108 |
| **python_web** | 14.185 | 14.155 | 14.127 | 14.140 | 14.108 | · |

Whole off-diagonal range: **14.108 – 14.219**. That is a spread of **0.8% of the
mean**. Ordering is stable under the regulariser (Spearman ≥ +0.96 across
δ ∈ {0.02, 0.05, 0.10, 0.20}), so the flatness is real and not a δ artefact.

**`mean k = 16.00` for every pair, in all 128 weight matrices.** The subspaces
never intersect anywhere. `d_R` is therefore pinned at its fully-disjoint value
for every pair, and the 0.8% residue is spectral shape, not domain relatedness.

### The control that settles it

| comparison | `d_R` | mean `k` |
| :--- | ---: | ---: |
| **same domain**, astral v4 vs v6 | 14.478 | 16.00 |
| **same domain**, postgresql v4 vs v6 | 14.461 | 16.00 |
| **same domain**, duckdb v4 vs v6 | 14.396 | 16.00 |
| different domains, astral + postgresql | 14.234 | 16.00 |
| different domains, astral + duckdb | 14.207 | 16.00 |
| different domains, postgresql + duckdb | 14.181 | 16.00 |

**One domain trained twice is *farther apart* than two different domains.** The
ordering is not weak — it is backwards. Two adapters trained on the *same corpus
lineage* still land in fully disjoint 8-dimensional subspaces, because `lora_B`
starts at zero and `lora_A` is random: the subspace is chosen by the
initialisation, not by the data.

This agrees with two instruments we already had — `times_above_chance`
(1.10–1.28× chance) and the measured cross-adapter cosine of 0.0206. Three
independent probes now say the same thing.

### Consistency against measured stacking (n = 5)

| pair | `d_R` | synergy | collateral |
| :--- | ---: | ---: | ---: |
| financial+postgresql | 14.162 | −8.52 | +6.58 |
| postgresql+duckdb | 14.209 | **+13.90** | −7.29 |
| financial+astral | 14.210 | −2.63 | −3.27 |
| astral+postgresql | 14.261 | +1.83 | **−11.17** |
| astral+duckdb | 14.261 | −0.82 | **+6.31** |

Spearman +0.30 / −0.30 at n = 5 — no signal, and n = 5 could not establish one
anyway. The last two rows are the sharp version: **identical `d_R` to three
decimals, opposite collateral damage** (−11.17 vs +6.31). Whatever decides
whether two experts hurt each other, this is not it.

---

## Consequences

1. **`d_R` on weight Gramians is not a domain-affinity metric** for rank-8 LoRA on
   this model. It is well defined; it is empty. Do not route on it.
2. **`RiemannianTeamRouter`'s harmony term is inert.** Replacing its distance
   matrix with a constant changes the selected team in **0 of 4000** random
   relevance draws. The router is a relevance argmax with a size penalty; the
   geometry contributes nothing. (`DECISIONS.md` §56 needs this correction.)
3. **`DECISIONS.md` §55's readings do not survive.** It called `financial` "the
   most geometrically isolated domain" — corrected, `financial` is mid-pack and
   `astral` is the outlier. It reported a "tight cluster" of
   `python_modern`/`postgresql`/`duckdb` — corrected, the closest pair is
   `python_modern`/`python_web`, and by 0.8%.

## Where this maths still has a real job

Not on weight Gramians — on **activation covariance** `Σ_h = E[h hᵀ]`, where
`n` tokens against `p = 2560` channels is the genuine `n < p` regime Ledoit-Wolf
was built for, and where `d_R(Σ_base, Σ_current)` is scale-invariant under the
RMSNorm rescalings that distort a Frobenius meter. `ledoit_wolf_from_samples()`
is implemented and tested for exactly that. It costs a forward pass, so it is a
GPU item and it is not run here.

## Files

- `benchmark_riemannian_domain_distance.py` — this benchmark, with the abort gate
- `src/runtime/riemannian_covariance.py` — primitives, shrinkage, shared-subspace construction
- `tests/test_riemannian_covariance.py` — 19 tests; `test_airm_invariant_to_rank_basis` is the regression guard
- `results/benchmarks/riemannian_domain_geodesics.json` — artifact
