# ⚡ In-Place Folding OpenAI-Compatible Engine & REST Server

An OpenAI-compatible local REST server built on **in-place weight folding**: micro-expert adapters are merged into the base weights in VRAM so inference runs with no adapter wrappers in the execution path.

Exposes local micro-experts (`postgresql`, `astral`, `financial_planning`) through standard
`/v1/chat/completions` and `/v1/models` endpoints. A swap is an in-place mutation
$W_{\text{live}} \leftarrow W_0 + s \cdot U V$ measured at **18.8-19.3 ms** with **0 bytes**
transient VRAM churn (peak-above-baseline).

**Measured, on an RX 7900 XTX (gfx1100) / ROCm 7.2:**

| | |
| --- | --- |
| decode, adapter wrapped | 25.09 tok/s |
| decode, adapter folded | **30.38 tok/s** (+21.1%, = unadapted base speed) |
| expert swap (factors resident) | 18.8-19.3 ms; pays for itself after ~3 tokens |
| putting the swap behind an API boundary | +0.012 ms (a ~20 byte command) |
| CUDA graph replay on top of folding | **~1.01x** -- decode here is kernel-execution bound, not launch bound |
| AMD AITER | **not used**: its rmsnorm silently returns zeros at hidden=2560 on gfx1100, and the one correct kernel is 2-6x slower than ATen |


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
> `src/gnn_experiment/fused_norm.py` for the measurements.

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
independently in `TODO.md`. The same-task pair at 7.15× shows the probe *can*
detect real structure, so the null is informative rather than a broken metric.

### 5. Speculative decoding — UNBLOCKED by `fla` on gfx1100 🟢

This was recorded in `TODO.md` as **architecturally blocked**. That was wrong,
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
([`scripts/profile_mtp_verification_path.py`](scripts/profile_mtp_verification_path.py)):

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

#### Measured speculative decode, K=4

Using the checkpoint's own shipped MTP head (`mtp.*`, 15 tensors, EAGLE-style),
which `transformers` never loads — see `src/gnn_experiment/mtp_draft.py`.

| backbone | base | speculative | speedup | accept% | gate |
| --- | ---: | ---: | ---: | ---: | :---: |
| un-adapted | 32.43 | 42.66 | **1.32x** | 56.7% | 4/4 ✅ |
| astral folded | 32.02 | 44.64 | **1.39x** | 60.8% | 4/4 ✅ |
| postgres folded | 31.68 | 37.02 | 1.17x | 48.1% | 3/4 ⚠️ |
| financial folded | 32.78 | 33.81 | **1.03x** | 44.9% | 3/4 ⚠️ |

**The yield is workload-dependent, not a fixed number.** An adapter degrades
drafting most on *its own* domain (financial-folded: 2.08 -> 1.58 accepted on
financial prompts), so a prompt mix weighted toward the active expert's domain
loses most of the win. These four prompts are financial-heavy, which is why
financial-folded collapses to 1.03x while astral-folded reaches 1.39x.

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

#### The Practical Win
Dual-folding `astral` + `postgresql` delivers a zero-latency, **0-byte marginal VRAM** multi-expert model maintaining **+45.06pp** Astral gain ($12.08\% \to 57.14\%$) and **+18.47pp** PostgreSQL gain ($51.33\% \to 69.80\%$).

#### Relative Delta Norms & True Mechanism
$\|\Delta W_{\text{ast}}\| / \|W_0\| = 0.0902$ vs. $\|\Delta W_{\text{pg}}\| / \|W_0\| = 0.0898$ — they are virtually identical in relative norm size. The "larger delta dominates" hypothesis is disproved. Astral's resilience ($101.4\% \to 92.8\%$ retention) stems from its large headroom above base ($12.08\% \to 56.52\%$, $+44.44\text{pp}$), whereas PostgreSQL ($51.33\% \to 75.42\%$, $+24.09\text{pp}$) sits closer to the base model's ceiling, making its marginal gain far more sensitive to non-linear feature shifts.

#### VRAM Accounting
- **Zero Marginal VRAM**: Stacking $N$ experts into a single static fold costs $0\text{ MB}$ extra VRAM per added expert.
- **Fixed Memory Footprint**: Static resident factors ($\sim 467\text{ MB}$ for 3 experts) and pristine buffer ($5.12\text{ GB}$) remain constant regardless of $N$.

#### The Architectural Rule: Geometric Orthogonality $\neq$ Functional Independence
Pairwise cosines across all 128 module layers range from $+0.0003$ to $+0.0004$ (max $|\cos| = 0.0026$) — orthogonal to four decimal places.
> **Architectural Rule:** Geometric orthogonality in weight space ($\cos \approx +0.0002$) does **not** grant non-linear functional independence in activation space. Expect monotonic decay ($\sim 20\text{pp}$ retention loss per added expert for fragile/low-headroom tasks, $\sim 8\text{pp}$ for resilient high-headroom tasks). Stack high-gain experts only ($N \le 2$ or $N \le 3$).

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

## 💻 Quick Start: Launching the REST Server

To launch the production OpenAI-compatible REST server daemon on `http://127.0.0.1:8000`:

```bash
uv run --env-file .env python3 scripts/run_openai_api_server.py --host 127.0.0.1 --port 8000
```

The server initializes `Qwen/Qwen3.5-4B`, swaps in exact PyTorch RMSNorm stand-ins (CUDA-graph friendly; not a speedup), pre-loads the factor micro-experts, captures the CUDA/HIP graph once, and listens for HTTP requests.

---

## 🔌 Connecting Tools & Clients

### 1. Using OpenCode CLI

With `opencode.json` configured, you can select micro-experts directly using `opencode --model imb/<expert>`:

#### Bash / Linux / macOS
```bash
# Launch interactive session with Astral expert
opencode --model imb/astral

# One-off command execution with PostgreSQL expert
opencode run --model imb/postgresql "Design a PostgreSQL 18 schema for embeddings using pgvector"
```

#### Windows PowerShell
```powershell
# Launch interactive session with Astral expert
opencode --model imb/astral

# One-off command execution with PostgreSQL expert
opencode run --model imb/postgresql "Design a PostgreSQL 18 schema for embeddings using pgvector"
```

---

### 2. Using `curl` (HTTP REST Calls)

#### List Available Models (`GET /v1/models`)
```bash
curl http://127.0.0.1:8000/v1/models
```

#### Non-Streaming Chat Completion (`POST /v1/chat/completions`)
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "postgresql",
    "messages": [{"role": "user", "content": "Design a PostgreSQL schema for storing vector embeddings."}],
    "max_tokens": 64
  }'
```

#### SSE Token Streaming (`stream: true`)
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "astral",
    "messages": [{"role": "user", "content": "Write a FastAPI similarity search endpoint."}],
    "max_tokens": 64,
    "stream": true
  }'
```

---

### 3. Using Official OpenAI Python SDK

```python
from openai import OpenAI

# Point client to local IMB REST server
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")

# Instant zero-copy in-place expert swap to postgresql expert
response = client.chat.completions.create(
    model="postgresql",
    messages=[{"role": "user", "content": "Explain pgvector HNSW index creation."}],
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## 🎯 Pre-Loaded Micro-Experts

| Expert ID | Primary Domain & Target Specialization | In-Place Activation Latency |
| :--- | :--- | :---: |
| `postgresql` (or `postgres`) | Database schema design, `pgvector`, HNSW indexes, SQL queries | **$19.29\text{ ms}$** |
| `astral` | Python FastAPI, standard libraries, high-performance async APIs | **$19.29\text{ ms}$** |
| `financial_planning` (or `fin`) | Financial planning strategy, sequence-of-returns risk, wealth modeling | **$19.29\text{ ms}$** |
| `base` (or `qwen3.5`) | Standard Qwen 3.5 4B base model without expert weights | **$0.00\text{ ms}$** |

---

## 🧪 Testing & Verification Suite

Run the full GPU integration test suite:

```bash
# Full REST API server integration test suite
uv run --env-file .env python3 scripts/test_openai_api_server.py


# Zero-recapture expert swapping synergy benchmark
uv run --env-file .env python3 scripts/benchmark_zero_recapture_swapping.py
```

---

## 🛠️ Project Structure

```text
gnn-experiment/
├── src/gnn_experiment/
│   ├── server.py               # Production FastAPI OpenAI REST server & IMB gatekeeper
│   ├── fused_norm.py           # Exact PyTorch RMSNorm stand-ins for Qwen3.5
│   ├── cuda_graph.py           # FoldedCudaGraphDecoder locked graph replay engine
│   └── novel_peft.py           # WeightFoldingEngine & FoldableExpert mechanics
├── scripts/
│   ├── run_openai_api_server.py # Server CLI launcher daemon
│   ├── test_openai_api_server.py# Complete REST server integration test suite
│   └── benchmark_zero_recapture_swapping.py # Synergy benchmark
└── README.md                   # System documentation
```
