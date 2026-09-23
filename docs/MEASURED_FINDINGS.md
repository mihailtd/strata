# Measured Findings — In-Place Weight Folding, Speculative Decoding & Adapter Stacking

Deep empirical results behind the headline numbers in the root [`README.md`](../README.md). Moved here so the README can stay a map/quickstart rather than a research log — nothing here was rewritten, only relocated, and stale paths from before the `apps/` monorepo restructuring were fixed in place.

**Before citing any number below, check [`EXPERIMENT_REAUDIT_2026-09.md`](EXPERIMENT_REAUDIT_2026-09.md)** — a systematic re-audit found a handful of shipped defaults and `docs/DECISIONS.md` entries resting on synthetic-data-as-real substitution. Nothing in *this* file was flagged by that audit (everything here is measured on real hardware, with retractions already inline where an earlier version was wrong), but the audit is the current source of truth for what's verified vs. not across the wider repo.

**Measured Steady-State Performance (RX 7900 XTX / ROCm 7.2 / Qwen 3.5 4B):**

| Metric | Measured Result | Architectural Significance |
| :--- | :---: | :--- |
| **In-Place Weight Folding vs Wrapped PEFT** | **33.26 tok/s (+82.1% speedup)** | Completely eliminates PEFT wrapper overhead (runs at 99.6% raw base speed) |
| **Native MTP Speculative Decoding ($K=6$)** | **55.71 tok/s (2.20x net speedup)** | 52.5 MB recurrent rollback unlocks speculation on hybrid linear-attention models |
| **Expert Hot-Swap Latency** | **18.08 ms (566 GB/s bandwidth)** | In-place tensor absorption into pristine buffer ($L_\infty = 0.00$ drift) |
| **Factor Standby Residency** | **42.47 MB/expert (200.6x compression)** | Holds up to 216 concurrent domain experts in standby on 24GB VRAM |
| **CUDA Graph Swapping Stability** | **Single capture (`count == 1`, 0 ms penalty)** | Preserves frozen `data_ptr()` VRAM pointers across multi-turn swaps |
| **Batch Scaling Throughput** | **611.04 tok/s at $B=64$ (17.52x of a perfect 64x = 27.4% efficiency)** | Per-request throughput falls 34.87 → 9.55 tok/s; speculation's break-even $\tau$ *rises* 1.15 → 1.45 |

---

## 🚀 Key Performance Specs

All figures below are gated on a correctness check first: the folded model must
reproduce the wrapped adapter's tokens, and graph replay must reproduce eager
greedy decode token-for-token. Speed numbers taken without that gate passing are
not reported, because a graph that decodes against a stale mask is *faster*
precisely because it is doing the wrong thing.

| Optimization | Decode | Notes |
| :--- | :---: | :--- |
| adapter wrapped (`NovelLoraLinear`) | $25.09\text{ tok/s}$ | 256 extra kernel launches/token at batch 1 |
| **adapter folded into base weights** | **$30.38\text{ tok/s}$ ($+21.1\%$)** | reaches unadapted base speed; does not exceed it |
| folded + CUDA graph replay | $\approx 1.01\times$ over folded alone | decode here is kernel-execution bound, not launch bound |
| in-place expert swap | $18.8-19.3\text{ ms}$, $0$ bytes transient churn | break-even after ~3 generated tokens |

> **Retracted.** An earlier version of this table claimed $32.89\text{ tok/s}$
> ($1.21\times$) for "AITER + Folded CUDA Graph". That arm was confounded four
> ways: only it had an adapter folded, it used a different generation loop, its
> timing window excluded prefill, and the graph was decoding incorrectly so it
> never hit EOS and ran to the token cap. AITER was also never active -- see
> `apps/runtime/fused_norm.py` for the measurements.

---

## 🔬 Measured Findings

All figures below come from runs on this machine (RX 7900 XTX / gfx1100, ROCm 7.2,
Qwen3.5-4B, bf16, greedy decode). Every result is gated on a correctness check
first — a folded model must reproduce the wrapped adapter's tokens, and graph
replay must reproduce eager decode token-for-token — because the failure mode
here is that a *wrong* implementation looks *fast*.

### 1. The bf16 merge-absorption law — replicated across 3 domains

Folding an adapter into bf16 base weights quantises the delta: `W0 + dW` is
rounded to 8 mantissa bits, so wherever `dW` is small relative to `W0` it is
partially or entirely absorbed. Swept over **three independent domains x five
alphas** (15 adapters; identical architecture `id_kron` r 8x8, 150 steps, batch
2, lr 2e-4 — alpha is the only variable):

| α | scaling | \|dW\|/\|W\| | merge err | absorbed |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.25 | ~0.023 | ~7.3% | ~8.0% |
| 32 | 0.50 | ~0.045 | ~3.7% | ~4.1% |
| 64 | 1.00 | ~0.090 | ~1.85% | ~2.0% |
| 128 | 2.00 | ~0.181 | ~0.93% | ~1.0% |
| 256 | 4.00 | ~0.358 | ~0.49% | ~0.51% |

The product `(|dW|/|W|) x merge_err` is constant to ~6% across a **15.7x range**
of the ratio, and **the same constant appears in all three domains**:

| domain | constant range | spread |
| --- | --- | ---: |
| financial_planning | 0.1664 – 0.1770 | 6.3% |
| astral | 0.1667 – 0.1769 | 6.0% |
| postgresql | 0.1664 – 0.1769 | 6.2% |

```
merge_rel_err ≈ 0.167 / (|dW|/|W|)
absorbed_frac ≈ 0.183 / (|dW|/|W|)
```

The `1/ratio` form follows from bf16 having fixed relative precision
(ULP ≈ |W|·2⁻⁸). The **constant is stable across domains** for adapters sharing
an architecture and training recipe. It is *not* universal: dense random deltas
give 0.234, and one adapter trained on degenerate data (940 copies of a single
templated prompt) gave 0.085. Calibrate per adapter family, not per domain.

### 2. The "don't fold low-scale adapters" rule — NOT SUPPORTED 🔴

The obvious corollary is a routing rule: fold high-scale adapters, keep
low-scale ones wrapped to avoid truncation. **Fifteen points across three
domains do not support it.**

| domain | corr(merge_err, folded−wrapped) | \|max Δ\| |
| --- | ---: | ---: |
| financial_planning | −0.113 | 2.50pp |
| astral | **+0.224** | 7.86pp |
| postgresql | **−0.909** | 5.00pp |
| **pooled (n=15)** | **−0.144**  (t=−0.53, df=13) | — |

The per-domain correlations run from **+0.224 to −0.909** — they do not even
agree on sign, which is what noise looks like at n=5. Pooled over all 15 points
the correlation is −0.144 with t=−0.53: nowhere near significant. Postgresql's
apparently strong −0.909 collapses to −0.376 when its single α=16 point is
removed, so it rests on one observation. And astral's *largest* penalty
(−7.86pp) occurs at its second-*lowest* merge error, directly against the rule.

Average cost of folding across all 15 adapters: **−0.231pp**.

**Verdict: fold by default.** The rule would trade a measured **+21% decode
speedup** for an effect that is smaller than this eval can resolve. Honest
bounds: folding *does* change outputs materially (deltas reach ±7.9pp), but
unpredictably and with a mean near zero — so it is not damage that merge error
lets you anticipate. At n=20 questions this rules out a large systematic
penalty; it does not prove the effect is zero.

### 3. Optimal α is domain-specific, and adapter quality is sharply non-monotonic

| domain | base | best α | best wrapped | gain over base |
| --- | ---: | ---: | ---: | ---: |
| astral | 12.20% | **64** | 60.20% | **+47.99pp** |
| postgresql | 49.67% | **64** | 74.67% | **+25.00pp** |
| financial_planning | 78.33% | **32** | 83.33% | **+5.00pp** |

All three domains have adapters that clearly beat base — **at the right α**.
The peak is *not* in the same place: α=64 for two domains, α=32 for the third,
so α=32 does not transfer (assuming it for astral would have cost ~13pp).

Training loss rose monotonically with α in **all three** domains
(astral 1.05→2.04, postgresql 1.34→2.12, financial 1.65→2.51) while quality
peaked in the middle. **Do not select adapters on loss** — it is anti-correlated
with adherence here.

This also invalidated an earlier conclusion in this repo. Sampling only α=16
and α=64 for financial — both in troughs — produced "this domain has too little
headroom for the adapter to demonstrate value." That was wrong. **Sweep α before
concluding an adapter does not work.**

### 4. Cross-task subspaces are orthogonal (with calibrated controls)

Whether adapters for different tasks share a low-rank subspace decides if
shared-basis compression (Tucker / LoKr / MasterBasis) can work at all.
**Read the ratio to chance, not the raw percentage**: projecting onto a
k-dimensional subspace of an (in·out)-dimensional matrix space captures
k/(in·out) by chance — ~0.0005% at k=32 — so a "retention < 1%" test can never
fail and carries no information.

| pair | retention | × chance | reading |
| --- | ---: | ---: | --- |
| any adapter vs itself | 12.5–19.6% | 26,569–41,447× | ceiling (control) |
| astral a256 vs astral a128 | 0.00338% | **7.15×** | same task → real shared structure |
| astral vs financial | 0.00061% | 1.28× | orthogonal |
| astral vs postgres | 0.00058% | 1.22× | orthogonal |
| financial vs postgres | 0.00052% | 1.10× | orthogonal |

**Every cross-task pair sits at chance.** Shared-basis compression cannot work
across these tasks — corroborating the 0.00% cross-task retention recorded
independently in [`TODO.md`](../TODO.md). The same-task pair at 7.15× shows the probe *can*
detect real structure, so the null is informative rather than a broken metric.

### 5. Speculative decoding — UNBLOCKED by `fla` on gfx1100 🟢

This was recorded in [`TODO.md`](../TODO.md) as **architecturally blocked**. That was wrong,
and the error is worth naming precisely because it cost real speedup.

**The mistake:** treating `fla`/`causal_conv1d` as one dependency. They are two,
with *independent* fallbacks in the modeling code:

```python
self.chunk_gated_delta_rule = chunk_gated_delta_rule or torch_chunk_gated_delta_rule  # from fla
self.causal_conv1d_fn       = causal_conv1d_fn                                        # from causal_conv1d
```

`causal_conv1d` genuinely cannot build here (needs `nvcc`). **`fla` is
Triton-based and runs natively on gfx1100.** `is_fast_path_available` reports
`False` because `causal_conv1d` is absent — but that flag only gates a *warning*,
while `chunk_gated_delta_rule` is wired in separately and does the work.

**Gotcha:** fla's device probe is `@cache`d and runs at import. Import it before
a GPU context exists and it latches to CPU for the whole process, warning
"Triton is not supported on current platform". Touch CUDA first, then import.

#### The multi-token wall collapsed

Cost of one forward over K tokens (the quantity that decides whether
speculation can pay):

| K | torch fallback | with `fla` |
| ---: | ---: | ---: |
| 1 | 33.56 ms (1.00x) | 29.27 ms (1.00x) |
| 2 | 95.44 ms (**2.84x**) | 37.18 ms (**1.27x**) |
| 4 | 94.33 ms (2.81x) | 34.33 ms (**1.17x**) |
| 8 | 91.25 ms (2.72x) | 40.18 ms (1.37x) |

Plain greedy decode also gains **1.073x** (29.39 -> 31.54 tok/s) for free.

#### Residual overhead audit (K=4 verification path)

⚠️ **An earlier version of this section reported a 0.94x ratio and a break-even
of ~0.94 accepted tokens, i.e. that multi-token verification is CHEAPER than
single-token decode. That was a measurement error and is retracted.** The script
called `model(tokens[:, :k], use_cache=False)` -- a standalone K-token forward
with no KV cache and no context. Real verification appends K tokens to a warm
cache. Without a cache the cost is dominated by the fixed ~8 GB weight read and
is flat in K *by construction*. Measured side by side:

| method | K=1 | K=2 | K=4 | K=8 |
| --- | ---: | ---: | ---: | ---: |
| `use_cache=False` (wrong) | 1.00x | 1.03x | 0.99x | 0.96x |
| warm cache (correct) | 1.00x | 1.53x | 1.52x | 1.46x |

Corrected profile, appending K tokens to a warm 128-token cache
([`benchmarks/runtime/performance/fla_triton_kernels/profile_mtp_verification_path.py`](../benchmarks/runtime/performance/fla_triton_kernels/profile_mtp_verification_path.py)):

| K | latency | vs K=1 | required accepted tokens to break even |
| ---: | ---: | ---: | ---: |
| 1 | 27.87 ms | 1.00x | 1.02 |
| 2 | 38.06 ms | 1.37x | 1.71 |
| **4** | **33.27 ms** | **1.19x** | **1.39** |
| 8 | 34.23 ms | 1.23x | 1.45 |

**Break-even is not the cost ratio.** Accepting M drafts yields M+1 tokens, and a
partial acceptance costs a further state re-advance on the same chunked path:

```
required E[M]  =  draft_ms/t1 + ratio * (1 + P_partial) - 1
```

At K=4 that is **1.39 accepted tokens**; measured acceptance is **2.27**, which
is why the end-to-end result below is a win. Quoting tau = ratio understates the
requirement by roughly 40%.

**Run-to-run variance is real:** the K=4/K=1 ratio has measured 1.17x, 1.19x,
1.31x and 1.52x across separate runs on this box. Treat it as **~1.2-1.5**, not
a pinned constant, and re-measure on an idle machine before quoting it.

#### Short-conv cost, actually measured

`causal_conv1d` is not installed (it needs `nvcc`), so `conv1d` runs the torch
fallback. The previous audit claimed "zero residual short-conv bottleneck" from a
profiler table that was **entirely zeros** -- on this torch/ROCm build
`evt.device_time_total` exists but is always 0, so `getattr(evt,
"device_time_total", <fallback>)` reads the zero and never falls back.
`torch.profiler` cannot attribute GPU kernel time here at all. Using CUDA events
on module hooks instead:

| component (K=4) | time | share |
| --- | ---: | ---: |
| wall | 30.90 ms | 100% |
| `linear_attn` total | 12.49 ms | 40.4% |
| of which `conv1d` | **1.18 ms** | **3.8%** of wall, 9.5% of `linear_attn` |

So the conv fallback is **small but not zero**. The old conclusion was
directionally lucky and evidentially void; the ceiling from installing
`causal_conv1d` is about **3.8%**.

## Two Execution Regimes: 4-bit NF4 vs bf16

**The repo is not one stack. It is two, and the seam runs through the middle of
a single pipeline rather than between "old" and "new" work.**

Every adapter in `results/adapters/` was trained against a **4-bit NF4** base;
every folding, speculation, and sweep benchmark loads a **bf16** base and folds
those deltas in. A bf16 delta cannot be folded into packed 4-bit weights, so the
two regimes cannot be mixed at inference -- but the training->folding handoff
mixes them *by construction*, because the adapter learned a correction to
quantized weights and is then applied to unquantized ones.

### Regime by script (historical — these scripts predate the `apps/factory` consolidation and mostly live in `apps/factory/legacy/` now)

| 4-bit NF4 (`load_in_4bit=True`) | bf16 |
| :--- | :--- |
| `finetune_novel_adapter.py` **(trainer)** | `train_financial_adapter.py` **(trainer)** |
| `export_adapter.py` **(trainer)** | `train_mtp_adapter.py` **(trainer)** |
| `evaluate_novel_adapter.py` | `benchmark_weight_folding.py` |
| `benchmark_adapter_swap.py` | `benchmark_mtp_*.py` (all 4) |
| `benchmark_inference_velocity.py` | `benchmark_batch_scaling.py` |
| `benchmark_speculative_decode.py` | `benchmark_stacked_experts.py` |
| `run_micro_probe_benchmark.py` | `evaluate_folded_vs_wrapped.py`, `eval_controlled_headtohead.py` |
| | `measure_fold_precision.py`, `benchmark_alpha_absorption_sweep.py`, + 12 more |

**No adapter records its own regime.** 0 of 69 adapter directories carry a
quantization marker in `adapter_config.json`, `novel_adapter_config.json`, or
`training_args.bin`. Provenance is currently recoverable only from which script
wrote the directory -- `export_adapter.py` leaves no `checkpoints/` subdir,
`train_financial_adapter.py` does. That is a fragile way to track a variable this
consequential and should be fixed by writing the regime into the adapter config
at save time.

### Consequences that are already visible

* **Every "folded" result in this README rests on a 4-bit-trained delta applied
  to a bf16 base.** That includes the corrected in-domain speculation matrix,
  the folding win, and the alpha/absorption sweeps. The measurements are
  internally consistent -- both arms of each comparison share the mismatch -- but
  the absolute quality numbers are not what a bf16-trained adapter would give.
* **The `bf16` absorption law is a bf16-only statement.** `merge_rel_err ~
  0.167/(|dW|/|W|)` was derived from bf16 rounding behaviour and does not
  describe NF4 at all.
* **The financial expert's harmfulness is confounded with regime.**
  `ctl_lora_fin_a128` was 4-bit-trained when it measured 78.33% against an
  83.33% base (i.e. worse than no adapter). It has since been retrained in bf16
  by `train_financial_adapter.py`. The stacking and speculation results in this
  README used the **4-bit** version; the adapter now on disk is a **different
  artifact** and those results do not describe it.

### Rule going forward

Tag every experiment with its regime, never compare a 4-bit number to a bf16
number, and re-run rather than port any result across the seam.

#### In-Domain Speculative Decoding Audit ($3 \times 3$ Grid, $N=20\text{--}40$ Prompts/Cell)

Direct 9-cell empirical audit ([`benchmarks/runtime/speculative/speculation_matrix/benchmark_mtp_indomain_speculation_matrix.py`](../benchmarks/runtime/speculative/speculation_matrix/benchmark_mtp_indomain_speculation_matrix.py)) measuring EAGLE-style MTP speculative decoding ($K=4$) vs $K=1$ autoregressive baseline across 3 folded Stock LoRA experts on the **M2 regime** (`m2_r8a128`, trained in native `bfloat16` with Liger kernels) and 3 domain prompt sets (**3 interleaved repeats**, median reported):

| Folded Expert | Prompt Set | $\tau$ | Speedup (median) | 3-repeat range | Predicted from $\tau$ | Exact vs chunked | Gate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **astral** | **astral (DIAG)** | **1.93** | **1.116x** | 1.102-1.118 | 1.102 | 77.5% | ENABLED ✅ |
| astral | postgresql | 1.85 | 1.079x | 1.077-1.089 | 1.071 | 80.0% | — |
| astral | financial_planning | 2.11 | 1.195x | 1.194-1.200 | 1.170 | 85.0% | — |
| postgresql | astral | 2.19 | 1.255x | 1.253-1.261 | 1.197 | 90.0% | — |
| **postgresql** | **postgresql (DIAG)** | **1.94** | **1.133x** | 1.120-1.168 | 1.105 | 95.0% | ENABLED ✅ |
| postgresql | financial_planning | 1.99 | 1.145x | 1.138-1.205 | 1.125 | 92.5% | — |
| financial_planning | astral | 2.24 | 1.294x | 1.276-1.303 | 1.217 | 82.5% | — |
| financial_planning | postgresql | 1.90 | 1.110x | 1.105-1.114 | 1.091 | 90.0% | — |
| **financial_planning** | **financial (DIAG)** | **1.70** | **1.011x** | 0.996-1.012 ⚠️ | 1.014 | 82.5% | DISABLED ❌ |

**Key M2 Finding**: 40 prompts/domain, 3 interleaved repeats, 2 arms = **2160 timed generations**. Eight cells resolve as clean wins, up to **1.294x** off-diagonal. The in-domain `financial_planning` diagonal does **not resolve**: its median is 1.011x but the three repeats span 0.996–1.012, straddling 1.0. It is gated off because the win is *unmeasured*, not because it is slow.

**The production gate is `measured speedup > 1.0` AND `τ ≥ 1.39` AND `repeats do not straddle 1.0`.** It is not the analytic break-even of the predicted-speedup model (τ ≈ 1.66); an earlier version of this section attributed the disable to that threshold when the speedup arm was what fired. Router: `{"astral": true, "postgresql": true, "financial_planning": false}`.

> ⚠️ **The speedups above are measured on a decoder that does not always reproduce its own verifier's output.** The exact-vs-chunked column is a correctness gate, and it fails on 5–22.5% of generations per cell (worst: `astral|astral` at 77.5%, i.e. 27 of 120 generations emit different text). See the [module README](../benchmarks/runtime/speculative/speculation_matrix/) for why 100% exactness is unreachable here.
>
> ✅ **The divergent text is not worse — measured, n=100.** [`speculation_quality/`](../benchmarks/runtime/speculative/speculation_quality/) scored three arms on the canonical eval sets with the repo's own scorer: speculative − autoregressive = **+0.19pp, CI [−0.77, +1.17]** (nothing significant, on full or length-matched answers). A control arm running the identical machinery with **every draft rejected** diverges on 28% of prompts against speculation's 33%, so the chunked kernel accounts for nearly all of it; 30 of the 33 diverging prompts scored *identically*. For scale, adapter stacking was retired at −10.96pp on this same metric. **The speedup is not paid for in quality, and the gate needs no quality term.**
>
> ⚠️ **`financial_planning` sits exactly at break-even and its sign is prompt-dependent.** On the 20 hand-curated prompts alone τ = 1.627 and an earlier run measured 0.973x; adding 20 held-out generated prompts to level the set to n=40 moved it to τ = 1.699 / 1.011x. Those held-out prompts are systematically *easier to draft* for every expert (+0.15 to +0.46 τ), so the two halves measure different difficulty distributions. Treat this domain as at-break-even, not as resolved either way.

#### High-Batch Scaling Frontier ($B=1 \dots 64$)

Direct batch scaling audit ([`benchmarks/runtime/performance/batch_scaling/benchmark_batch_scaling.py`](../benchmarks/runtime/performance/batch_scaling/benchmark_batch_scaling.py)) probing decode throughput, chunked verification penalties, and weight folding speedup scaling across batch sizes $B \in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]$ on AMD Radeon RX 7900 XTX (24 GB VRAM):

| Batch Size ($B$) | Step Latency (ms) | Aggregate Throughput | Per-Req Throughput | Speculation Break-Even | Weight Folding Speedup Win |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 28.68 ms | 34.87 tok/s | 34.87 tok/s | 1.15 | **1.80x** (53.3 ms vs 29.6 ms) |
| **2** | 28.93 ms | 69.12 tok/s | 34.56 tok/s | 1.23 | **1.78x** |
| **4** | 32.31 ms | 123.81 tok/s | 30.95 tok/s | 1.14 | **1.86x** |
| **8** | 37.92 ms | 211.00 tok/s | 26.37 tok/s | 1.18 | **1.73x** |
| **12** | 41.64 ms | 288.16 tok/s | 24.01 tok/s | 1.17 | **1.73x** |
| **16** | 48.92 ms | 327.09 tok/s | 20.44 tok/s | 1.11 | **1.52x** |
| **24** | 56.38 ms | 425.70 tok/s | 17.74 tok/s | 1.27 | **1.49x** |
| **32** | 71.27 ms | 449.00 tok/s | 14.03 tok/s | 1.20 | **1.33x** |
| **48** | 86.11 ms | 557.41 tok/s | 11.61 tok/s | 1.43 | **1.29x** |
| **64** | 104.74 ms | **611.04 tok/s** | 9.55 tok/s | 1.45 | **1.21x** |

**Batch Scaling Takeaways**:
1. **Aggregate Throughput**: Scales **17.52x** from 34.87 tok/s ($B=1$) to **611.04 tok/s** ($B=64$) — **27.4% of the perfect 64x**, which is the number the benchmark itself prints. $B=2$ is close to free (+0.9% latency for 2x tokens); the efficiency loss is concentrated above $B=16$.
2. **Per-request cost**: throughput per request falls **34.87 → 9.55 tok/s (−73%)**. At $B=64$ a single user sees under 10 tok/s, which is at the edge of interactive usability — $B=64$ is a throughput operating point, not a latency one.
3. **Speculation Break-Even**: the column is the $\tau$ you must *exceed* to profit, and it **rises with batch** (1.15 at $B=1$ → 1.45 at $B=64$), so speculation's margin erodes as batch grows. It does not vanish: measured in-domain $\tau$ is 1.93 (astral) and 1.94 (postgresql), still clearing 1.45 — but headroom shrinks from +0.78 to +0.48. **No speculative decoding was actually run at $B>1$**; this is a K=1-vs-K=4 forward-cost proxy.
4. **Weight Folding Win**: **1.73x–1.86x** over PEFT wrappers for interactive serving ($B \le 12$), decaying monotonically to **1.21x** at $B=64$ (1.52x @16, 1.33x @32, 1.29x @48). The win is real but shrinking with batch, not flat.

**On the exact-vs-chunked column (70-95%): this is NOT a speculation defect.** An
earlier revision of this section claimed the loop "diverges from the path its own
verifier computes" and called the speedups upper bounds pending a fix. That was
wrong, and it is retracted. The gate function `chunked_reference()` runs a
*teacher-forced single forward* over the whole sequence, while the verifier runs
*incremental chunked forwards against a KV cache* -- different numerical paths, so
the gate never measured speculation correctness. Controls (n=60, 20 prompts x 3
domains, 32 tokens):

| Comparison | Match |
| :--- | :---: |
| greedy vs greedy (determinism control) | **100.0%** |
| **speculative vs plain greedy** | **80.0%** |
| plain greedy vs teacher-forced (kernel-divergence control) | 83.3% |

Speculation diverges from greedy at the **same rate two non-speculative paths
diverge from each other** (80.0% vs 83.3% -- 2 prompts out of 60). Determinism is
100%, so none of this is sampling noise. The ~20% floor is bf16 kernel
path-dependence: the chunked multi-token kernel and the single-token recurrent
kernel genuinely disagree (measured directly: divergence at tokens 2-4 of 32 on
3/3 prompts, with no speculation involved). Verification is inherently
multi-token, so it cannot use the single-token kernel, and **100% token-exactness
against plain greedy is therefore not reachable on this stack** -- not for the
speculative loop, and not for plain chunked decode either. The speedups stand as
measured. The earlier `r = +0.44` correlation is void along with its premise.

**And the divergent text is not worse — that is now measured, not argued.** The
paragraph above was, until 2026-08-17, an argument from a 3-prompt control. The
experiment that closes it is
[`speculation_quality/`](../benchmarks/runtime/speculative/speculation_quality/):
three arms, n=100 paired triples, 256 tokens, the canonical eval sets and the
repo's own scorer.

| contrast | isolates | Δ | 95% CI |
| :--- | :--- | ---: | :--- |
| autoregressive → forced-reject | the chunked kernel alone | +1.02pp | [−0.98, +3.69] |
| forced-reject → speculative | accepting drafts | −0.83pp | [−3.50, +1.17] |
| **autoregressive → speculative** | **what a user receives** | **+0.19pp** | **[−0.77, +1.17]** |

The middle arm is the load-bearing one: it runs the identical speculative
machinery with `n_acc` pinned to 0, so it is the chunked kernel *without*
speculation. It diverges from plain decode on **28%** of prompts while real
speculation diverges on **33%** — the kernel does nearly all of it, exactly as
claimed above. 33/100 prompts emitted different text; **30 of those scored
identically.** Nothing is significant on either the full answers or with every arm
truncated to the shortest arm's length. See `docs/DECISIONS.md` §7.

#### Exactness is not achievable against plain decode, and that is not a bug

The chunked kernel (used by verification) and the single-token recurrent kernel
(used by plain decode) **disagree numerically**: 1 of 3 prompts diverges at
token 9/32 with no speculation involved at all. So the honest reference for a
speculative decoder is the path its verifier actually computes. Both are
reported: `vs 1tok` and `vs chunk`. Un-adapted and astral pass 4/4 against the
chunked reference; with postgres or financial folded it drops to 3/4.

#### Rollback on a recurrent model

`transformers` refuses assisted generation for this family outright
("assisted generation is not supported with stateful models") because a
GatedDeltaNet's recurrent state after K tokens is not recoverable from the
state after M<K. The state is **fixed size** (52.5 MB here), so snapshot and
restore is a cheap copy — verified to round-trip exactly. That is what makes
speculation possible at all here.

---

### 6. Eval-harness bug that inflated earlier base numbers

An un-fine-tuned model runs past its answer and fabricates a new
`### Question:` block, which then gets scored. Affected **18/20 astral, 16/20
postgresql and 13/20 financial** base answers; for financial that hallucinated
tail supplied **57%** of the base model's term hits. Fine-tuned adapters learned
to stop (0–4/20), so every adapter was being scored against an inflated base.
Fixed with `stop_strings` plus a regex backstop in `eval_suite.py`.

The size of the distortion is **not predictable from metric type** and must be
re-measured per domain. Measured before vs after the fix: postgresql base
57.57% → **49.67%** (−7.90pp, inflated), astral base 12.20% → **12.20%**
(unchanged), financial severely distorted (the hallucinated tail supplied 57%
of its hits). Astral's ratio happened to survive because its base is dominated
by *bad* hits and the tail carried a similar mix; postgresql's is dominated by
*good* hits and the tail skewed further that way. Treat any pre-fix adherence
figure as suspect until re-measured.

### 7. Multi-Expert Simultaneous Weight Folding & The Three-Ceiling Architectural Model

Simultaneously folding multiple domain adapters into base weights ($W_{\text{live}} \leftarrow W_0 + \sum_{i=1}^N S_i \cdot U_i V_i$) eliminates runtime expert-swapping overhead entirely.

#### Empirical Stacking Benchmark Results (`bfloat16` Stock LoRA $r=8, \alpha=128$)

Direct evaluation ([`benchmark_stacked_experts.py`](../benchmarks/runtime/folding/benchmark_stacked_experts.py) `--experts stock`) across all 3 clean, `bfloat16`-trained domain experts (`ctl_lora_fin_a128`, `ctl_lora_r8_a128`, `ctl_lora_pg_a128`):

| Condition / Stack | Financial Planning | Astral CLI | PostgreSQL | Mean Retention vs Solo |
| :--- | :---: | :---: | :---: | :---: |
| **Base Model (Unadapted)** | 83.33% | 12.08% | 51.33% | — |
| **fin (Solo Financial)** | **87.50%** | 13.33% | 60.92% | 100.0% |
| **ast (Solo Astral)** | 68.33% | **54.82%** | 66.67% | 100.0% |
| **pg (Solo Postgres)** | 78.33% | 7.08% | **60.00%** | 100.0% |
| **fin + ast (Dual Stack)** | **88.33%** | **54.51%** | 55.50% | **109.6%** |
| **fin + pg (Dual Stack)** | **88.33%** | 15.25% | **72.80%** | **183.9%** |
| **ast + pg (Dual Stack)** | 67.50% | **65.01%** | **71.55%** | **178.5%** |
| **fin + ast + pg (Triple Stack 🔥)** | **90.00%** | **50.71%** | **67.17%** | **144.4%** |

> [!WARNING]
> **The "Mean Retention vs Solo" column above is a degenerate statistic and its
> values >100% are not gains.** Retention is $(\text{stacked}-\text{base})/(\text{solo}-\text{base})$.
> Measured solo gains are financial **+4.17pp**, postgres **+8.67pp**, astral
> **+42.74pp** -- on $n=20$ questions that is **0.83**, 1.73 and 8.55
> question-equivalents. Dividing by a sub-question denominator is what produces
> 120%, 160% and 247.7%; it signals the metric has broken down, not that
> stacking adds free performance. The 183.9% and 178.5% means are driven
> entirely by the two small denominators.
>
> On astral -- the only domain whose denominator can support the ratio -- the
> triple stack scores **50.71% vs 54.82% solo = -4.11pp (retention 90.4%)**, a
> loss. Takeaway 2 below inverts once the unmeasurable domains are excluded, and
> the "Constructive Interference" claim in takeaway 3 is contradicted by that
> same -4.11pp. The absorption law governs **merge error** (how faithfully a
> delta survives bf16 rounding), not task quality; a faithfully-represented
> delta can still be the wrong delta.
>
> Superseded by absolute pp deltas with 95% paired-bootstrap CIs -- see
> [Settling the stacking question](#settling-the-stacking-question) below.

#### Key Takeaways

1. **Clean Financial Expert is Positive (+4.17pp)**:
   - Standalone `ctl_lora_fin_a128` (trained on clean, un-contaminated dataset under `bfloat16`) scores **87.50%** on Financial Planning prompts vs Base Model at **83.33%** (**+4.17pp gain**).
   - This officially cures the financial degradation anomaly (previously 78.33% due to 49% `"Hey!"` chatter and 4-bit NF4 training mismatch).

2. **The Triple Stack (`fin + ast + pg`) Achieves 144.4% Mean Retention**:
   - Stacking all 3 orthogonal domain experts simultaneously delivers **90.00%** Financial (+6.67pp over Base), **50.71%** Astral (+38.63pp over Base), and **67.17%** Postgres (+15.84pp over Base).
   - Average performance retention across all 3 domains is **144.4%** relative to single-expert deltas.

3. **VRAM Accounting & Absorption Law**:
   - **Zero Marginal VRAM**: Stacking $N$ experts into a single static fold costs $0\text{ MB}$ extra VRAM per added expert.
   - **Constructive Interference**: Because the low-rank subspaces are statistically orthogonal (subspace overlap ratio to chance $\approx 1.10-1.28\times$), summing deltas increases magnitude $\|\Delta W_{\text{stacked}}\| / \|W_0\|$, which improves `bfloat16` representation fidelity (absorption law).

#### The Architectural Rule: Geometric Orthogonality $\neq$ Functional Independence
Pairwise cosines across all 128 module layers range from $+0.0003$ to $+0.0004$ (max $|\cos| = 0.0026$) — orthogonal to four decimal places.
> **Architectural Rule:** Geometric orthogonality in weight space ($\cos \approx +0.0002$) does **not** grant
> non-linear functional independence in activation space -- that part holds. But the **monotonic-decay claim
> and the $N \le 2$/$N \le 3$ rule are RETRACTED.** Measured on the methodology-matched m2 set (astral,
> $n=40$, paired bootstrap): `ast+fin` loses $-10.96$pp (CI $[-21.50,-1.62]$, resolved) but `ast+fin+pg`
> gains $+5.74$pp (CI $[-6.63,+18.26]$, not resolvable) -- **adding a third expert moves the score back
> above solo.** Decay is not monotonic and no stack-size rule is supported by the data.

#### The Three-Ceiling Architectural Model

| Ceiling Type | Theoretical Limit | Physical Mechanism | Binds First? |
| :--- | :---: | :--- | :---: |
| **1. Empirical Interference** | **$N \approx 4 - 6$** | **Non-linear activation cross-talk** ($\sim 20\text{pp}$ decay/expert) | **YES (Primary Ceiling)** |
| 2. Perturbation Size | $N \approx 10 - 20$ | Quadratic norm accumulation ($\|\Delta W_{\text{total}}\| \to 0.3 \cdot \|W_0\|$) | No (Secondary) |
| 3. Rank Saturation | $N \approx 40$ (at $r=64$) | Pigeonhole overlap in rank-2560 parameter space | No (Theoretical Floor) |

#### Status of In-Place MTP Adapter Folding: ❓ EXPERIMENTAL / UNPROVEN
Fusing low-rank adapters directly into native MTP head parameters ($W_{\text{mtp}} \leftarrow W_{\text{mtp}} + S \cdot U_{\text{mtp}} V_{\text{mtp}}$) executes in **$0.30\text{ ms}$** with zero VRAM churn and $\approx 3.3\text{ ms}$ single-token step time ($K=1$). However, on a micro-eval ($N=41$ tokens across 4 prompts), 30-step quick-tuned adapters degraded accuracy from 21.95% (un-adapted MTP) down to 12.20%–14.63%. MTP Adapter Folding remains **❓ Experiment / Unproven** until evaluated with full domain training ($N \ge 1000$ tokens) and end-to-end acceptance benchmarks.

### 8. id_kron vs Stock LoRA — controlled head-to-head: PARITY

An earlier "strict 1-to-1 head-to-head" concluded that default `id_kron` loses to
Stock LoRA and that Kronecker's win is rank compression at `r=16`. **Both claims
are refuted, in opposite directions.** That audit compared four PRE-EXISTING
adapters of unknown provenance; it reported `id_kron` at `r=64, 49.8M` when the
adapter it loaded was `rank_total=8 / 6.26M` (inverting its own headline -- that
adapter used 41% FEWER params than the LoRA that beat it), had no training
records for any arm, and left effective scaling uncontrolled at 32.0 vs 2.0.

#### The controlled version

All 21 adapters trained fresh: astral (815 records), 150 steps, batch 2, lr 2e-4,
bf16. **Effective scaling (`alpha/rank_total`) is matched across architectures,
not alpha** -- comparing `alpha=256 @ r=8` against `alpha=32 @ rank_total=16`
compares scaling 32 against scaling 2, which is what the retracted audit did.
Each architecture is then compared at its own peak.

| scaling | LoRA 10.62M | id_kron rt8 6.26M | id_kron rt16 12.42M |
| ---: | ---: | ---: | ---: |
| 0.25 | 25.00% | 30.42% | 42.92% |
| 0.50 | 26.25% | 34.86% | 37.08% |
| 1.00 | 26.25% | 49.75% | **56.24%** |
| 2.00 | 42.75% | **57.83%** | 50.83% |
| 8.00 | 47.08% | *diverged* | *diverged* |
| 16.00 | **56.03%** | *diverged* | *diverged* |
| 32.00 | 40.31% | *diverged* | *diverged* |

Base is 12.20%. **Peaks: 57.83% / 56.24% / 56.03% -- a 1.80pp spread, which is
0.4 questions on a 20-question rubric. This is parity.** Architecture is not the
lever; tuning is.

* The `72.44%` the retracted audit reported does **not reproduce** -- the best of
  15 controlled adapters is 57.83%, and `rt16` specifically peaks at 56.24%.
* "Kronecker's win is rank compression at r=16" is **backwards**: `rt16`
  (12.42M) does not beat `rt8` (6.26M), 56.24% vs 57.83%.
* **Optimal scaling varies 16x by architecture** (LoRA 16, rt8 2, rt16 1). That
  alone explains how a single-point comparison could produce any verdict: at
  scaling 1.0 id_kron leads by 23.5pp; at scaling 16 LoRA wins outright because
  id_kron has already diverged.

#### The parameter-count advantage does NOT reach swap latency

`id_kron` genuinely stores 41% fewer parameters. That saving **disappears when
folded**, because `V` must be the block-diagonal expansion `blkdiag(w_a x r1)`,
a dense `(2560, 8)` -- exactly the size of LoRA's `A`:

| | stored (on disk) | folded (VRAM factors, what a swap uses) |
| --- | ---: | ---: |
| LoRA r=8 | 21.2 MB | **21.2 MB** |
| id_kron rt8 | **12.5 MB** | **21.2 MB** (identical) |
| id_kron rt16 | 24.8 MB | 42.5 MB (2x worse) |

And even a real payload saving would not move swap latency. A swap moves
**10.24 GB** (read `W0` 5.12 + write `W_live` 5.12); the adapter factors are
**0.021 GB = 0.2%** of that. Swap cost is set by MODEL size, not adapter size --
inherent to folding. Zeroing the payload entirely would save ~38 us of a ~19 ms
swap.

The 41% saving is real for **disk and distribution**, not for swap throughput.
To make it real in VRAM you would have to fold the Kronecker structure without
materialising the block-diagonal (apply `w_a` per chunk), which would halve
resident factors -- 2.1 GB vs 4.25 GB at 100 experts -- but still not speed up
swaps.

#### Stability: LoRA is the safer default

`id_kron` diverges between scaling 2 and 8 at **both** ranks (loss ~7.7 vs ~1.03),
so the threshold tracks effective scaling, not rank. LoRA is stable across the
whole 128x range swept. `rt8`'s optimum sits at **exactly the last stable point**
-- tuned to the edge of a cliff -- while LoRA has a broad plateau (47-56% across
scaling 8-16). The optimum also moved with rank (rt8 at 2.0, rt16 at 1.0), so it
is not a constant that can be pinned once.

**Caveats.** 1.80pp is inside noise at n=20; treat the three as tied on quality.
`id_kron` peaks are *cut off by divergence, not turned over*, so 57.83% is its
ceiling at lr 2e-4 and possibly not its ceiling. `rt16`'s curve is jagged
(42.92 -> 37.08 -> 56.24 -> 50.83), evidence of real per-point variance. Single
seed per point. And matching *scaling* may still not be the right control -- if
id_kron's effective `|dW|/|W|` per unit alpha is larger, matched `|dW|/|W|` would
be, which would explain both its low-scaling efficiency and its divergence.

### 9. Batching is nearly free — the largest unclaimed win

Every performance result in this repo was measured at batch 1. At batch >= 2 the
arithmetic changes qualitatively, and nobody had measured it.

| B | ms/step | aggregate tok/s | per-request tok/s | latency vs B=1 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32.82 | 30.47 | 30.47 | 1.00x |
| 2 | 32.29 | **61.94** | 30.97 | **0.98x** |
| 4 | 32.28 | **123.90** | 30.97 | **0.98x** |
| 8 | 38.42 | **208.23** | 26.03 | 1.17x |

**At B=4 four concurrent users each get the same speed as a solo user** (30.97 vs
30.47 tok/s). Step latency is flat because it is dominated by reading ~8 GB of
weights, and that read produces 4 tokens instead of 1 -- the signature of being
memory-bandwidth bound. Aggregate scales **6.83x from B=1 to B=8** (85% of
perfect); B=8 is where it bends (+17% latency), so the sweet spot is **B~4**.

*(The common intuition is inverted here: latency DOUBLING at B=2 would mean
compute-bound and batching buying nothing. Flat latency means batching is free.)*

Two knock-on results:

* **Speculation's break-even improves with batch** -- the K=4/K=1 ratio falls
  from 1.31 to ~1.15, so speculation and batching compose. But they are
  substitutes under load: batching gives 4x, speculation 1.38x.
* **The folding win survives batching** (1.10x / 1.11x / 1.16x / 1.04x at
  B=1/2/4/8), so the wrapper tax does not amortise away.

**The REST gateway serves batch-1**, leaving up to **6.8x aggregate throughput**
unclaimed -- more than folding and speculation combined, and requiring no new
math, only continuous batching. Caveat: these are synthetic uniform-length
sequences, so real serving loses some of it to padding waste.

### 10. MTP head adaptation does not improve draft acceptance -- CLOSED, do not retry

`train_mtp_adapter.py` produces LoRA adapters for the draft head itself
(`mtp.fc`, `mtp.layer.self_attn.*`, `mtp.layer.mlp.*`). They were never loaded by
anything: `mtp_draft.py` reads bare `mtp.*` tensors from the checkpoint. The
open question was whether domain-adapted drafting raises acceptance ($\tau$),
which drives net speculative speedup far harder than draft latency does.

Measured directly ([`benchmarks/runtime/speculative/mtp_head_folding/benchmark_mtp_head_adapter_acceptance.py`](../benchmarks/runtime/speculative/mtp_head_folding/benchmark_mtp_head_adapter_acceptance.py)):
40 astral prompts x 4 offsets = **160 draft events per condition**, $K=6$,
accepted-prefix scoring, un-adapted baseline re-measured in the same process.

| Condition | Scaling | $\tau$ | $\Delta$ vs bare | 95% CI (paired bootstrap) |
| :--- | :---: | :---: | :---: | :---: |
| **un-adapted (bare checkpoint head)** | — | **2.456** | — | baseline |
| `mtp_astral_sweep_r64_a16` | 0.25 | 1.875 | $-0.581$ | $[-0.819, -0.344]$ |
| `mtp_astral_sweep_r64_a32` | 0.50 | 1.931 | $-0.525$ | $[-0.794, -0.263]$ |
| `mtp_astral_sweep_r64_a64` | 1.00 | 1.531 | $-0.925$ | $[-1.194, -0.650]$ |
| `mtp_astral_sweep_r64_a128` | 2.00 | 1.819 | $-0.637$ | $[-0.912, -0.362]$ |
| `mtp_astral_lora_r64_a64` | 1.00 | 1.988 | $-0.469$ | $[-0.688, -0.263]$ |

**Every adapted variant is worse, and every CI excludes zero.** The bare
checkpoint head drafts best. Do not wire the MTP adapters into the speculative
path.

**No scaling trend is readable.** The two independently-trained adapters at
*identical* scaling 1.0 differ by **0.456** ($1.531$ vs $1.988$) -- as large as
most between-scaling differences. Training-run variance swamps any dose-response
curve, so the $\tau$ ordering across alphas above must not be interpreted. The
headline survives this because even the *best* adapted variant has a CI clear of
zero.

**Speculation is not at risk.** Every condition, including the worst at $1.531$,
stays above the $\tau \ge 1.39$ break-even. This was an upside test that came
back empty, not a regression.

**Why, mechanistically.** A draft head's job is not to be domain-fluent -- it is
to *agree with the backbone it speculates against*. `train_mtp_adapter.py`
optimises next-token loss on domain text, which pulls the head away from the
target it must match. That predicts the same outcome for any domain, not just
astral. Revisiting this means changing the trainer's objective to distil the
backbone's outputs; re-tuning rank or alpha will not help.

**This also retires the earlier `benchmark_mtp_folding_sweep.py` result** (21.95%
-> 12.20-14.63% top-1), which reached the same conclusion but could not support
it: it trained throwaway adapters inline at **30 steps** and never loaded the
150-step adapters on disk, used $n=41$ tokens over 4 prompts, and scored
single-token top-1 rather than accepted prefix. Same answer, now properly earned.

**Related dead end:** folding the MTP head to reduce draft *latency* is also not
worth building. Drafting is only ~27% of a speculative round (4 x 3.3 ms against
a 35.6 ms verify), so eliminating it entirely is bounded at ~$1.14\times$ by
Amdahl -- and there is nothing wrapped to fold in any case, since the head loads
bare checkpoint tensors and uses no `peft` layer.

**A directly related, later, and more thorough experiment on the real EAGLE MTP head reached the same conclusion with a different loss and a bigger sweep — see `docs/DECISIONS.md` §63.** That result is the one to cite going forward; this section is kept for the mechanistic explanation ("why, mechanistically" above), which still holds.

### 11. Settling the stacking question {#settling-the-stacking-question}

Scoring here is greedy and deterministic: two independent runs returned
**bit-identical** means for every condition not involving the one adapter that
changed (`base`, `ast`, `pg`, `ast+pg`). So run-to-run variance is exactly zero
and the only uncertainty is **which questions were sampled**. That makes a
paired bootstrap over per-question scores the right -- and cheap -- instrument.

`benchmark_stacked_experts.py` now keeps per-question scores (it previously
averaged them away at the point of measurement, which is why this could not be
settled from the stored results) and reports, for every stack/domain cell,
`stacked - solo` in **percentage points with a 95% paired-bootstrap CI**. The
retention ratio is no longer reported at all.

**Decision rule, fixed in advance:** a cell counts as real interference only if
its CI excludes zero. Cells whose CI spans zero cannot distinguish stacking
interference from question-sampling noise, however large the point estimate.

**Known limit:** financial (+4.17pp) and postgres (+8.67pp) have too little
headroom to resolve regardless of statistics -- the base model already scores
83.33% and 51.33%. Astral (base 12.08%) is currently the only domain that can
answer the question. Making the other two testable requires eval sets on which
the base model scores low, which is a data task, not a benchmarking one.

---

### 12. Opinionated adapters: measured on both axes for the first time — style works, correctness pays for it 🔴

The `python_modern` / `agentic_coding` adapters were built to make the model
write modern, idiomatic Python without a large system prompt. Until this
session nobody had measured **either** axis of that claim on a real benchmark:
not the correctness cost, and not the style benefit they exist to deliver.
Measured both, on real hardware, 4B, `runtime-next` over real HTTP.

Two harness bugs had to be fixed first; both silently moved numbers, so
everything below post-dates the fixes.

#### 12a. Two harness bugs that were corrupting results

**`benchmarks/humaneval/executor.py` graded rejected drafts.** `clean_code`
returned the **first** markdown block containing `def <entry_point>`. For a
thinking model that block is typically a scratch draft inside `<think>`, not
the answer. On `HumanEval/5`, `python_modern` wrote a correct final solution
and was scored on a broken comprehension it had explicitly abandoned
mid-thought ("Actually, I need to be more careful here"). The bug penalised the
*more exploratory* arms hardest, i.e. exactly where the comparison lived. Now
grades what follows `</think>`, preferring the last block defining the entry
point. Effect: base 147→**150**, python_modern 101→**107**, agentic_coding
95→**98**. The pre-fix numbers are void, not merely noisy.
`runner.py` now also stores the raw completion so a scoring change can be
re-evaluated offline instead of re-running 164 generations on the GPU.

**`SnapshotClient.delete` never deleted anything.** It sent `DELETE` to
`/v1/state/snapshot` (singular); the engine routes `DELETE` on
`/v1/state/snapshot**s**`. The 404 was swallowed as "best-effort", with a
comment claiming a TTL would clean up. There was no TTL — the engine
deliberately never evicts. So snapshots leaked for the server's lifetime, and
once `MAX_SNAPSHOTS = 8` accumulated, `create()` failed and `run_task`'s
fallback re-sent the **initial prompt** every turn instead of the pytest
feedback. The repair loop silently degraded into asking the same question five
times. Only tasks that fail turn 1 leak a snapshot (the `passed` break precedes
snapshot creation), so a single 10-task arm stayed under the cap, but any run
of two arms in one server process crossed it partway through the second.
**Any multi-arm aider result produced in one server process before this fix
should be treated as contaminated.**

#### 12b. Correctness: the adapters are a large, one-directional regression

HumanEval, all 164 problems, single-shot, temperature 0, deterministic:

| arm | pass@1 | mean tokens | mean thinking |
| :--- | :---: | :---: | :---: |
| base | **150/164 (91.5%)** | 914 | 1934 chars |
| `python_modern` | 107/164 (65.2%) | 434 | 823 chars |
| `agentic_coding` | 98/164 (59.8%) | 184 | 226 chars |

Almost perfectly one-directional: `python_modern` loses 45 problems base
passed and wins back 2 (**net −43**); `agentic_coding` loses 55 and wins 3
(**net −52**). 26 problems fail under both adapters and pass on base. At n=164
this is not sampling noise.

#### 12c. Style: the adapters deliver exactly what they were built for

Same solutions, scored with `ruff check --select UP,C4,SIM,PTH,PERF,RET`, and
**only on solutions that pass their tests** so pretty-but-broken earns nothing:

| arm | ruff violations per passing solution | correct **and** style-clean |
| :--- | :---: | :---: |
| base | 0.60 | **98/164** |
| `python_modern` | **0.20** (3× cleaner) | 92/164 |
| `agentic_coding` | **0.06** (10× cleaner) | 92/164 |

So the style training genuinely works — this is the first positive evidence for
these adapters, on the axis they were designed for, which had never been
measured. But base still wins the combined metric, because its correctness edge
outweighs its style deficit.

**The prize is quantified by this table**: an adapter with base's correctness
and `agentic_coding`'s style rate scores ≈**142/164** against base's 98. The
style half is solved; correctness preservation is the whole remaining problem.

Base's style deficit is also shallower than "style" suggests: `UP006` (24×,
`List[int]`→`list[int]`), `UP035` (21×, deprecated `typing` imports), `RET505`
(14×, unnecessary `else` after `return`). That is legacy typing syntax and a
return-shape nit — token-level lexical choices, not algorithmic style. The
current adapters rewrite all 32 MLPs to achieve it.

#### 12d. Root cause is in the training data, not the serving path

Audited the actual datasets rather than inferring from behaviour:

| | `python_modern` | `agentic_coding` |
| :--- | :---: | :---: |
| records | 1418 | 1193 |
| completions containing `<think>` | **0** | **0** |
| completions emitting SEARCH/REPLACE | 0 | 605 (51%) |
| user turns mentioning pytest/traceback | **0** | 257 |
| median completion | 351 chars | 214 chars |

With `completion_only_loss: true`, those completions *are* the entire training
signal. Observed thinking length tracks trained completion length almost
exactly (`agentic_coding`: 214 chars trained → 226 chars of thinking observed).
The model learned the response *format* of its data, and that format has no
reasoning in it. This is format collapse, not destroyed capability — 143/164
and 153/164 responses still emit a `</think>`, so the machinery is intact and
merely suppressed, which is why it is likely recoverable without retraining.

Two distribution mismatches also invalidate much of how these adapters have
been benchmarked historically:

- `python_modern` has **zero** examples of "here is a failing test, fix it",
  yet turns 2–5 of the aider loop are nothing but error-feedback repair. Turn 1
  is in-distribution (100% markdown fence = `whole` format); nothing after it
  is. This explains the session's oddest result — "LoRA drafts, base repairs"
  (8/10) beating "base drafts, LoRA repairs" (6/10). That was the distribution
  boundary, found by accident.
- `agentic_coding` emits SEARCH/REPLACE in **51%** of its data and is 29%
  shell / 20% regex, yet every run this session used `edit_format="whole"` on
  pure-Python tasks. Most of its training distribution has never been tested.

#### 12e. Damage is proportional to adapter scale

Same harness, same 5 attempts, same feedback; only the fold scale differs
(`swap_to_adapter` supports `name@scale`, applied as
`target_scale · (α/r) · BA`):

| arm | 20 problems | 32 problems | mean code attempts | 1st-attempt solves |
| :--- | :---: | :---: | :---: | :---: |
| `python_modern@1.0` | 10/20 | 17/32 (53%) | 1.65 | 11/17 |
| `python_modern@0.5` | **18/20** | **28/32 (88%)** | **1.21** | 23/28 |
| base thinks → `@0.5` writes | 17/20 | 27/32 (84%) | 1.41 | 21/27 |
| base alone | 17/20 | **29/32 (91%)** | 1.24 | 23/29 |

Halving the scale recovers 11 of 32. Cost asymmetry is stark too: `@1.0` burned
75 turns on 15 unsolved problems, base 15 turns on 3. The `@0.5`-vs-base
ordering **flipped** between the 20- and 32-problem sets, which is the useful
result: a diluted adapter is statistically indistinguishable from base, i.e.
the reward for no longer hurting is parity, not gain.

The "base deliberates, adapter writes" arm (turn 1 = base, prompted to analyse
only, never graded) did **not** beat base. Discounting its think turn it needed
*more* code attempts (1.41 vs 1.21/1.24) and its first code attempt succeeded
21/27 vs `@0.5`'s 23/28 — plausibly because a context full of base's verbose
deliberation is off-distribution for an adapter trained on short prompt → short
patch.

#### 12f. Cross-adapter tensor handoff forks the output immediately — and lower scale is NOT safer

`diagnose_cross_adapter_handoff_divergence_against_full_reprefill` (server.rs).
Identical conversation, same folded adapter for the generated turn; the two
routes differ only in how the prefix arrived — a base-captured KV/GDN snapshot
restored under the adapter, versus a full re-prefill under it:

| scale | identical prefix | first divergence |
| :--- | :---: | :--- |
| `python_modern@0.5` | **1/32 tokens** | position 1 — `"The test"` vs `"The issue"` |
| `python_modern@1.0` | 8/32 tokens | position 8 — `"1. Parse"` vs `"1. Split"` |

**Retracts an intuition offered earlier in the same session**: that `@0.5` is
safer because the weight bases are ~96% aligned (`dW/W = 0.072` from
`regime.json`, halved at scale 0.5). It is not. `@0.5` diverges *earlier*.
Divergence position is a threshold effect on how close the top-2 logits happen
to be, not a smooth function of perturbation size, so there is no "small enough
delta" that makes cross-adapter handoff safe.

It also reframes why this was dangerous: both continuations are *plausible*, so
the corruption produces a **different valid answer**, not garbage — invisible
in aggregate pass rates (arm B scored the same with and without it) and total
in actual behaviour.

Mechanism, verified against the adapter's own tensors rather than assumed:
`mlp.{gate,up,down}_proj` in **all 32** layers, `self_attn.{q,k,v,o}_proj` in
**8** (the full-attention layers), and **zero** GDN or `lm_head` tensors. So
contamination has two channels — direct (adapted `k_proj`/`v_proj`) and, larger
because the MLP carries most of the parameter mass, indirect: shifted hidden
states reach even the 24 GDN layers whose own weights are never adapted.

**Engine fix.** Snapshots now record the folded configuration they were
captured under (`StoredSnapshot.adapter_sig`, with scale part of the identity).
A mismatched resume is refused with a loud error naming both signatures and the
stale snapshot is dropped so its VRAM is not stranded. Idle TTL of 60s measured
from **last use, not creation** — an absolute TTL would kill active sessions,
since a 5-turn task routinely exceeds a minute. `harness_direct.py` now
accumulates real conversation history and, on an adapter switch, deletes the old
snapshot and re-prefills everything under the new adapter; handoff is used only
within a run of same-adapter turns. Covered by
`real_resume_across_an_adapter_swap_is_refused_and_same_adapter_resume_still_works`.

**Consequence for architecture.** Exact KV invariance is impossible for any
adapter that touches the trunk: the residual stream globally couples layers, so
any delta anywhere changes every downstream layer's K/V — even a query-only
LoRA, via the attention output. And approximate correction cannot rescue it,
because greedy decoding forks wherever top-2 logits sit inside the residual
error. Exact cross-LoRA handoff therefore requires the adapter to live
**outside** the trunk (output head, or a parallel side tower for real depth);
everything in-trunk buys a later fork, never no fork.

#### 12g. Spec-gated best-of-N does not rescue the residual — but the test was weak

On exactly base's 14 HumanEval failures, best-of-8 (greedy, then temp 0.8),
accepting the first candidate satisfying the problem's **own docstring
examples** (visible spec, never the hidden tests): **0/14 converted, 0/14
accepted by the gate.**

Reported with its limitation, which is large: 9/14 of those problems have no
doctests at all, and the gate itself mis-parsed 3–4 more (HumanEval's docstring
formatting makes `doctest` capture the closing `"""` as expected output). Only
`HumanEval/55` had a working gate — and there it correctly rejected all 8
samples (`fib(10) → 34`, want `55`, the same off-by-one every time). So the
mechanism is largely **untested**, not disproven; what *is* established is that
this particular residual is a capability wall rather than sampling variance,
which best-of-N cannot move.

Also note the methodological limit: the 14 targets were selected *because* the
hidden tests said base failed them. That makes this a mechanism diagnostic, not
a pass@1 — and the projection printed by the script is invalid for the same
reason, since applied to all 164 the gate could also reject a correct greedy
answer and accept a wrong resample.

#### 12h. What this says to do next

- **Don't serve these adapters at `@1.0`.** It costs 26–32 points of pass@1 for
  a style gain worth ~6 points on the combined metric.
- **Fix the recipe, not the serving path.** Rejection sampling is the direct
  fix for the measured failure: generate candidates with *base* over the ~300
  verifiable problems already in the repo, keep only those that pass their tests
  **and** score cleaner on ruff, and train on those. Correctness-preserving by
  construction (every target passed), thinking-preserving for free (targets are
  base's own generations, so they already contain `<think>`), and style-directed
  by a programmatic reward instead of hand-written demonstrations.
- **Style belongs at the head.** The measured deficit is token-level lexical
  choice, so an `lm_head`-only adapter is expressively sufficient, cannot damage
  reasoning (the trunk is untouched), and is KV-invariant by construction so
  cross-adapter handoff becomes exactly sound. `fold_adapter_into` currently
  only iterates `weights.layers`, so `lm_head` is not yet a fold target — a
  small additive change, since it is already a `LinearWeight`.
- **Generalise `adapter_sig` to a compatibility class** keyed on *which modules*
  an adapter touches rather than which adapter it is. Two head-only adapters
  would then be mutually resumable, making the guard precise rather than
  conservative.
- **Report both axes from now on.** Pass rate alone cannot see a style adapter
  succeeding, and ruff alone cannot see it failing. 142/164 is the number to beat.
- **Stop tuning on HumanEval and the 10 core aider tasks.** Both are now burnt
  as instruments — not by model contamination but by us designing against their
  known failure modes all session. The 128 aider tasks outside the core suite
  are untouched and carry real headroom; every future mechanism claim should be
  a delta measured there against a base-only baseline.

---

### 13. At fixed compute budget the optimum is a MIX — pure breadth and pure depth both lose 🟢

First mechanism this session to beat base on ground neither the author nor the
agent had inspected, with paired significance tests rather than raw score
comparison — and a live example of why testing the endpoint matters.

**Setup.** The 128 aider tasks outside the 10-task core suite: never run
before, so not shaped by anyone's knowledge of their failure modes. 4B, no
adapter, `runtime-next` over real HTTP, on the binary with §14's decode bound
already fixed. Every arm makes **exactly five model calls**, so this is a pure
*allocation* comparison, not extra compute.

| allocation | pass rate |
| :--- | :---: |
| K=1 draft + up to 4 sequential repairs (the default loop) | 67/128 (52.3%) |
| **K=3 drafts + up to 2 repairs on the best** | **78/128 (60.9%)** |
| K=5 drafts + 0 repairs (pure sampling) | 58/128 (45.3%) |

Paired (McNemar exact, two-sided):

| comparison | gained | lost | net | p |
| :--- | :---: | :---: | :---: | :---: |
| K=1 → K=3 | 13 | 2 | **+11** | **0.0074** |
| K=1 → K=5 | 6 | 15 | −9 | 0.078 |
| K=5 → K=3 | 22 | 2 | **+20** | **0.00004** |

**The optimum is interior, and that is the finding.** Sequential repair alone
(67) beats parallel sampling alone (58); the mix beats both by a wide,
highly-significant margin. Breadth and depth are complementary here, not
substitutes.

**Retraction of an intermediate claim.** After K=3 came back at +11, this
section was first written up as *"sampling diversity beats iterative repair"*,
with a proposed post titled "Breadth Beats Depth". K=5 falsified it: strip the
repair turns entirely and the score falls **below** the original baseline.
Had the endpoint not been run, this repo would have published a claim that its
own next experiment disproves. Test the endpoint of any allocation curve before
naming the trend.

**Why this was tried — the diagnosis, not a hunch.** The baseline's own failure
distribution pointed at it:

- **All 61 failures exhausted the full 5-turn budget.** None failed early.
- Passes by turn were **47 / 13 / 3 / 1 / 3** — 90% of wins land on turns 1–2,
  while turns 3–5 rescued 7 tasks out of 128 and consumed most of the compute.
- Failures generate ~7× the tokens of passes (median 3126 vs 452), with 7 tasks
  past 6000 tokens and one at 13,381 — a doom-loop signature, not convergence.

So the *marginal* return on repair collapses after turn 2 — which is a
different statement from repair being worthless, as K=5 proves. In the winning
arm, 56 of 78 solves came from drafting and 22 from repair; both phases carry
real weight.

**Selection signal.** Candidates are ranked by their pytest result (collection
error worse than assertion failures; fewer failures better). In aider-style
evaluation the harness already feeds test failures back between turns — test
feedback *is* the benchmark's defined iteration signal — so this reads the same
signal the baseline already had. The hidden grader is never consulted for
selection, and the test file is never shown to the model.

**Stated limitations.**
- Drafts 2..K sample at temperature 0.8, so these arms are stochastic; a rerun
  draws different candidates. McNemar establishes that an allocation beat
  another *on these tasks with these samples*; it does not bound run-to-run
  variance in effect size. The 13-vs-2 and 22-vs-2 lopsidedness is what makes
  the result credible, more than the p-values alone.
- Only K ∈ {1,3,5} were measured. K=2 and K=4 are unmeasured, so "K=3 is
  optimal" is not established — only that the optimum lies strictly inside the
  range.

**Why it plausibly generalises.** Nothing here is aider-specific or
benchmark-shaped: "generate a few candidates, keep the one that passes the most
tests, then fix it" is what a developer does, and it needs only a test command.
It also runs against where effort usually goes — the field builds ever-deeper
agentic repair loops, and on a 4B at fixed budget, spending the whole budget on
depth costs 11 tasks against a balanced split.

**Prior context that now reads differently.** §12e found that adapter ordering,
scale dilution, and a base-deliberates/adapter-writes handoff all failed to beat
base. Every one of those was a variation on *how to spend sequential turns*. The
allocation between drafting and repair was the unexamined variable, and moving
it was worth more than every adapter configuration tested combined.

---

### 14. Decode was unbounded: a GPU page fault reachable from an ordinary HTTP request 🔴

`start_request` validated the **prompt** against `MAX_SEQ_LEN` (8192):

```rust
if base_position + prompt_ids.len() >= MAX_SEQ_LEN { return Err(...) }
```

Nothing validated **decode**. `step()` → `forward_one_token` advanced
`state.position` with no bound, so a long enough generation walked straight off
the end of the KV cache that `DecodeState::new(MAX_SEQ_LEN)` had allocated:

```
Memory access fault by GPU node-1 (Agent handle: 0x...) on address 0x7f54d9c00000.
Reason: Page not present or supervisor privilege.
```

An out-of-bounds GPU write reachable from a normal `/v1/chat/completions`
request. It takes down the whole server process, not just the request.

**How it surfaced.** The first attempt at the §13 held-out baseline. The server
died on task 12 of 128 and the harness kept recording failures against a dead
endpoint, producing **6/128 (4.7%)** — a number that looks like a
catastrophically hard benchmark and is in fact a crashed process. Only tasks
1–11 were real (6/11). The integrity check added for §12's snapshot bug flagged
113 tasks as having lost tensor handoff, which is what identified this as
systemic rather than a hard benchmark.

**Trigger.** `beer_song` — a 3,022-token prompt that burned all 5 turns. With
tensor handoff, position accumulates *across* turns within a task, so a large
prompt plus repeated generations against `max_tokens=4096` blows through 8192.
Any sufficiently long multi-turn session hits this.

**Fix.** `Engine::context_exhausted()`, consulted by both generation loops so
they terminate with `finish_reason="length"` (semantically correct — generation
did stop because the context ran out), plus a hard refusal inside `step()` so no
other caller can fault the GPU. Regression test
`real_decode_stops_at_context_limit_instead_of_faulting_the_gpu` prefills to
8091, decodes exactly 100 tokens, stops at 8191, and asserts the engine still
serves a fresh request afterwards.

**Standing note on proportion.** `max_tokens` defaults to 4096 against an 8192
window — half the context per turn. For multi-turn work that is badly
proportioned: a few held-out tasks legitimately need a larger window rather than
a better model. Raising `MAX_SEQ_LEN` costs KV-cache VRAM and is a deliberate
trade, not a free fix.

**Methodological point.** Three result-corrupting bugs were found in a single
session — this one, the HumanEval executor grading rejected drafts (§12a), and
the 404'd snapshot delete that silently killed the repair loop (§12a). All three
produced *plausible numbers*. None announced themselves. The habit that caught
all three was the same: when a number is surprising, verify the apparatus before
believing the number.

---

### 15. Long context is blocked by LDS in one decode kernel, not by KV size 🔴

Asked to port `apps/runtime/long_context_engine.py`'s "4-bit quantized KV /
32k context" into `runtime-next`. Two things were found before any port
happened, and both changed the task.

**There is nothing to port.** That module's entire import list is `json`,
`time`, `urllib.request`, `typing` — no torch, no HIP, nothing that can touch
a cache. `PinnedPrefixCache` stores a string, estimates tokens as
`len(split())*2`, and reports `"status": "PINNED_IN_GPU_VRAM"` as a hardcoded
literal. `LongContextVRAMManager` is arithmetic about a cache it never
allocates. `context_shift_if_needed` truncates a Python **message list**, not a
KV cache. `stream_chat` POSTs to `127.0.0.1:11434` — Ollama. The docstring
describes a system the file does not contain; porting it would have
manufactured exactly the facade `AGENTS.md`'s Zero-Mock Invariant forbids.

**KV quantization would solve a problem this architecture does not have.**
Only 8 of 32 layers are full attention; the other 24 are GDN with a FIXED-size
recurrent state that does not grow with sequence length. Measured cost per
token on 4B:

| component | per token |
| :--- | ---: |
| KV cache (8 layers x 2 x 4 heads x 256 dim x 2 B) | 32 KB |
| `attn_scores` (`MAX_PREFILL_CHUNK=256` x 2 B) | 0.5 KB |
| **total** | **~32.5 KB** |

So the whole KV cache is ~268 MB at 8192 on a 24 GB card running a model that
measures 14.4 GB. A conventional 32-layer transformer would be ~512 KB/token
here — 16x more — which is why KV quantization matters elsewhere and not here.

**The real ceiling is shared memory in `attention_decode_split.hip`**, which
sizes its LDS allocation by the full kv_stride:

```c
size_t shmem_bytes = (head_dim + kv_stride + threads + head_dim*kv_split) * sizeof(float);
```

With `threads = ATTN_HEAD_DIM * ATTENTION_DECODE_KV_SPLIT = 1024` (already the
AMD workgroup maximum), gfx1100's 64 KB LDS per workgroup gives

```
max_seq_len <= 16384 - 256 - 2*256*4 = 14080
```

Found the hard way: the window was raised to 32768, prefill to 32661 worked
fine, and then the first decode step failed with HIP `invalid argument` —
a 132 KB LDS request against a 64 KB budget. `KV_LEN_BUCKETS`'s top entry of
12288 was never arbitrary; it was this limit all along.

**Shipped:** window raised 8192 -> **12288** (largest bucket-aligned value that
fits), made runtime-configurable via `RUNTIME_NEXT_MAX_SEQ_LEN` with a startup
clamp and a loud message rather than an opaque mid-generation HIP failure, and
the decode state's real cost is now **measured** at startup via
`hip::mem_info()` rather than computed:

```
[runtime-next] context window 12288 tokens; decode state cost 530 MB measured (15.3 GB of 24.0 GB free remaining)
```

**A test bug is retracted here too.** §14's regression test accepted ANY `Err`
from `step()` as a clean refusal, so the 32768 attempt went green on a real HIP
launch failure. It now asserts the error is the engine's own `context
exhausted`, and fails on anything else. A test that passes on a crash is worse
than no test.

**The actual unlock for long context** is rewriting
`attention_decode_split.hip` to TILE the KV row through shared memory instead
of holding all of it — then the window is bounded by VRAM (~32.5 KB/token, so
32k costs ~1 GB) rather than by 64 KB of LDS. That is a real kernel rewrite and
is not done.

---

### 16. Batched decode: built, proven exact, and worth 2.7x — not the 5.6x the proxy promised 🟡

§9 called continuous batching "the largest unclaimed win ... requiring no new
math", measured on the *Python* runtime. This is that claim tested on
`runtime-next`'s own kernels, then built, then measured on the real decode
path rather than a proxy.

#### 16a. The proxy said build it

Batched decode at B=N has the same GEMM shapes as a prefill of N tokens, so
timing the EXISTING batched prefill path bounds what batching could win before
writing any of it (`diagnose_batching_headroom_via_batched_prefill_scaling`):

| rows | ms/call | ms/row | vs B=1 |
| ---: | ---: | ---: | ---: |
| 1 | 19.73 | 19.73 | 1.00x |
| 2 | 27.14 | 13.57 | 1.45x |
| 4 | 27.37 | 6.84 | 2.88x |
| 8 | 28.07 | 3.51 | 5.63x |
| 16 | 28.13 | 1.76 | **11.23x** |

`ms/call` flat from 2 to 16 (+3.6% for 8x the work) — textbook memory-bound,
and better scaling than §9's Python numbers, which bent at B=8 with +17%.

#### 16b. What was built

`BatchedDecodeState` (N independent `SequenceSlot`s, each reusing `LayerState`
unchanged) plus `forward_batched_decode`. **No new kernels.** The insight that
made it small: `rope_prefill` and `kv_cache_append_prefill` already read a
PER-ROW position out of `position_buf`, which is exactly what B unrelated
sequences need. So norms, qkv/in_proj, fused qkv prep, RoPE, gating, o_proj and
the whole MLP are one batched launch each; only 5 kernels are per-sequence
(attention: kv append + attention read; GDN: conv1d, gate/beta, recurrent),
because those touch that slot's own cache/recurrent state.

#### 16c. Gated on exactness, not plausibility

`real_batched_decode_matches_sequential_single_sequence_decode`: 4 DIFFERENT
prompts (identical ones would pass even if every slot secretly read slot 0's
cache), 24 tokens each, asserting batched output is **token-for-token
identical** to the same prompts decoded alone on the trusted path. It passes.
Batched decode re-derives every pointer offset at batch width, and a single
wrong row would produce fluent, plausible, wrong text — the failure mode this
session hit three separate times.

#### 16d. First cut: correct, but only 2.7x

The first working version batched the projections and MLP but kept the five
state-touching kernels (attention: kv append + attention read; GDN: conv1d,
gate/beta, recurrent) in a per-sequence loop, and applied `lm_head` per row.

| B | ms/step | tok/s aggregate | tok/s per sequence |
| ---: | ---: | ---: | ---: |
| 1 (graphed, production path) | 11.94 | 83.7 | 83.7 |
| 1 (eager, batched path) | 12.63 | 79.2 | 79.2 |
| 2 | 22.27 | 89.8 | 44.9 |
| 4 | 27.39 | 146.0 | 36.5 |
| 8 | 37.47 | 213.5 | 26.7 |

2.70x against the honest eager-vs-eager baseline — less than half the proxy's
5.63x. The HIP graph was NOT the difference (11.94 vs 12.63 ms, ~6%). The
B=1->B=2 step (+76% ms/step) was the tell.

#### 16e. Closing the gap: batched kernels, then the real culprit

**Step 1 — batch the five per-sequence kernels.** `causal_conv1d_update` was
*already* batch-aware (`blockIdx.y`, `conv_state[batch, ...]`); the other four
gained the same `blockIdx.y` dimension plus an explicit stride, and per-layer
state moved to one batch-contiguous allocation so a sequence is addressed by
stride rather than a per-slot pointer. Result: 213.5 -> **246.5 tok/s** at B=8
(2.70x -> 3.54x). Real, but still short.

**Two stride bugs were caught here by the equality gate, not by inspection**,
and both were invisible at slot 0 (the only slot whose offset is zero):
`causal_conv1d_update` indexes its source with stride `conv_dim`, but was being
handed the combined in_proj output whose stride is `GDN_IN_PROJ_COMBINED_DIM`;
and `gdn_recurrent` derived q/k/v strides from head dims when those tensors
live inside a `GDN_CONV_DIM`-wide row. Both produced fluent, wrong text from
slot 1 onward. This is precisely the class of bug the gate exists for.

**Step 2 — the actual bottleneck was `lm_head`.** It was still applied once per
row. It is the widest matrix in the model (`HIDDEN_SIZE x VOCAB_SIZE` = 2560 x
248320, ~1.27 GB in bf16), so a per-row apply re-reads all of it per sequence:
at B=8 that is more memory traffic than the entire rest of the decode step
combined. One batched GEMM instead:

| B | ms/step | tok/s aggregate | tok/s per sequence |
| ---: | ---: | ---: | ---: |
| 1 (graphed, production path) | 12.27 | 81.5 | 81.5 |
| 1 (eager, batched path) | 13.97 | 71.6 | 71.6 |
| 2 | 22.16 | 90.3 | 45.1 |
| 4 | 22.40 | 178.6 | 44.6 |
| 8 | **23.27** | **343.8** | 43.0 |

**ms/step is now flat from B=2 to B=8** (22.16 -> 23.27, +5% for 4x the work) —
the memory-bound signature 16a predicted. Against eager-vs-eager:
**1.26x / 2.49x / 4.80x** at B=2/4/8, against the proxy's 5.63x ceiling.

Progression, all gated on token-identical output: **2.70x -> 3.54x -> 4.80x**.

#### 16f. What remains, and the honest tradeoff

Per-sequence latency still degrades: 71.6 -> 43.0 tok/s at B=8 (1.67x slower,
much improved from the first cut's 3x). Aggregate throughput is bought with
per-user latency; B~4 remains the balanced point (2.49x aggregate for 1.6x
per-sequence slowdown).

The residual gap to 5.63x is the B=1->B=2 step (13.97 -> 22.16 ms). That is the
GEMV->GEMM crossover: B=1 uses a matrix-VECTOR fast path that batching gives
up, visible in 16a's proxy too (19.73 -> 27.14 at rows 1->2). It is a property
of the shape change, not of this implementation.

For the repo's own best-of-K drafting (§13, K=3 generations from one prompt),
the realistic win is the B=4 column: **~2.5x wall-clock**.

**Not done:** batched PREFILL across different-length prompts (each slot is
still prefilled through the single-sequence path, then copied in via
`load_slot`), a scheduler, and HIP-graph capture of the batched path.

#### 16g. Shipped as OpenAI `n` — best-of-K in one call

Batched decode is only useful if something can reach it. The natural wire-level
fit was OpenAI's `n` parameter (N completions for one prompt), because that is
*exactly* the shape of §13's best-of-K drafting, which currently pays for N
separate full generations.

`POST /v1/chat/completions` with `"n": 4` now prefills the prompt ONCE and fans
that single prefilled state out to 4 batch slots via `load_slot`, then decodes
them together with an independent RNG per slot. Measured over real HTTP, same
prompt, `temperature=0.8`, `max_tokens=120`:

| | wall clock | tokens | throughput |
| :--- | ---: | ---: | ---: |
| 4 x sequential `n=1` calls (today's best-of-K) | 6.44 s | 480 | 74.5 tok/s |
| one `n=4` call | **3.08 s** | 480 | **157.4 tok/s** |

**2.09x wall-clock**, four genuinely distinct completions. Slightly under the
2.49x the B=4 microbenchmark shows, the difference being the shared prefill and
HTTP framing — still the real end-to-end number a caller sees.

Two deliberate restrictions, both returning a real 400 rather than doing
something surprising:
- **`n > 1` requires `temperature > 0`.** All slots start from the identical
  prefilled state, so at temperature 0 every completion would be the same
  greedy continuation. Diversity comes from per-slot RNG draws, nothing else.
- **`n > 1` is non-streaming only.** Interleaving N token streams over one SSE
  connection is not something the OpenAI wire format expresses.

The `BatchedDecodeState` is cached on the `Engine` and reused across requests
(rebuilt only when `n` changes): its KV allocation is ~400 MB per slot at a
12288 window, far too expensive to build per request. No zeroing is needed
between requests because `load_slot` fully overwrites every layer and attention
only ever reads up to each slot's own position — the same reasoning
`DecodeState::reset` already documents for the single-sequence KV caches.

#### 16h. Multi-prompt batch endpoint — and the limit it exposed 🔴

`POST /v1/chat/completions/batch` takes N INDEPENDENT prompts (different
content, different lengths) and decodes them together. Each slot is prefilled
through the ordinary single-sequence path and copied in via `load_slot`;
prefill is a small fraction of a generation, so the batched decode still
carries the win. Measured over real HTTP, 4 prompts of 17-31 prompt tokens,
`max_tokens=100`:

| | wall clock | throughput |
| :--- | ---: | ---: |
| 4 separate sequential calls | 5.08 s | 78.7 tok/s |
| one batched call | **2.34 s** | **170.7 tok/s** |

**2.17x wall-clock.** This is static batching, deliberately: the caller already
holds all N prompts (a benchmark sweep, an offline eval), so there is nothing
to schedule.

**But verifying it against sequential output found a real limit, and it is not
an indexing bug.** At `temperature=0`, batched output does NOT always match the
same prompt run alone. Isolated:

| check | result |
| :--- | :--- |
| batch-of-1 vs sequential | **identical** |
| batch-of-4 slot 0 vs sequential | **diverged** |
| all 4 slots, identical prompts, equal to each other | **yes** |
| repeatable across runs | **yes** |

Per-slot indexing is therefore exactly right and the path is deterministic; the
divergence tracks the BATCH SHAPE. Changing the batch changes the GEMM's `M`,
hipBLAS reduces in a different order, the logits differ in their last bits, and
a near-tied argmax flips. Same non-associativity class as §136.

**It is common, not marginal.** At B=8, `max_tokens=120`, temperature 0:

```
identical to sequential : 3/8
diverged                : 5/8
  first difference at 66%, 8%, 80%, 10%, 38% through the output
```

**Consequence a caller must know:** batching is *not* a transparent speedup for
a temperature-0 A/B comparison. Running §13's 128-task sweep through the batch
endpoint would produce results that are **not directly comparable** to the
67/128 sequential baseline, because some tasks would take a different (equally
valid, equally deterministic) decode path. Use batching to generate, not to
re-measure an existing baseline — or re-run the baseline at the same batch size.

For the intended uses this is a non-issue: best-of-K drafting (§13) runs at
`temperature=0.8` and wants diversity, and `n>1` already requires
`temperature>0` for exactly that reason.

**`real_batched_decode_matches_sequential_single_sequence_decode` still asserts
exact equality** at B=4 / 24 tokens, and is kept that way on purpose: indexing
and stride bugs diverge at the FIRST token of the affected slot (both bugs in
16e did), so exact equality at a short horizon is the sharp detector. Its doc
now states plainly that it does not prove bit-identity in general.

**Still not done:** true continuous batching -- dynamic admission of concurrent
clients into a running batch. The server's accept loop is single-threaded
(`for request in server.incoming_requests()`, handled inline), so that needs a
threaded scheduler with a request queue and per-request channels, not another
kernel. It is the one remaining item whose failure modes are threading bugs
rather than numerical ones, and it is worth far less to a single-user workload
than the static batching above.
