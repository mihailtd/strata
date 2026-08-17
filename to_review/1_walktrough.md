# Overnight Execution Walkthrough: M2 Speculation Matrix & Batch Scaling Frontier

> **Revised 2026-08-17 after review.** The original version's *numbers* all
> reproduced against the stored JSON — nothing was fabricated. What did not
> survive were three framings, one methods statement, and a directory tree.
> Corrections are marked **[CORRECTED]** inline so the delta is auditable.
> Superseded original: `git show 833a3b0:to_review/1_walktrough.md`.

## Overview

An autonomous overnight session on the **AMD Radeon RX 7900 XTX (24 GB)** producing
two empirical results: the 3×3 M2 speculation matrix, and the batch-scaling
frontier to $B=64$.

---

## 1. 3×3 In-Domain & Cross-Domain Speculation Matrix on M2 (`r8a128`)

**[CORRECTED — scale.]** The original said *"3 experts × 3 prompt domains × 3
interleaved repeats, 180 generation runs"*. That count matched neither the prompt
files nor the repeat structure. Actual, now printed at startup and stored in the
report:

> 40 prompts/domain × 3 experts × 3 domains × 3 repeats × 2 arms =
> **2160 timed generations**, plus **1080** chunked-reference runs.

**[CORRECTED — unequal n.]** The original ran astral 40 / postgresql 40 /
**financial_planning 20**, and the only production change it produced came from
the half-powered column. The sets are now levelled to 40 (see §1.3), and the
benchmark refuses to compare unequal columns silently.

### Measured M2 Matrix Results (n=40/domain, 3 repeats)

| Folded Expert | Prompt Domain | τ | Accept % | Speedup (median) | 3-Repeat Range | Predicted | Exact vs Chunked | Gate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **astral** | **astral (DIAG)** | 1.93 | 48.3% | 1.116x | 1.102–1.118 | 1.102 | 77.5% | **ENABLED** |
| astral | postgresql | 1.85 | 46.3% | 1.079x | 1.077–1.089 | 1.071 | 80.0% | — |
| astral | financial_planning | 2.11 | 52.8% | 1.195x | 1.194–1.200 | 1.170 | 85.0% | — |
| postgresql | astral | 2.19 | 54.7% | 1.255x | 1.253–1.261 | 1.197 | 90.0% | — |
| **postgresql** | **postgresql (DIAG)** | 1.94 | 48.5% | 1.133x | 1.120–1.168 | 1.105 | 95.0% | **ENABLED** |
| postgresql | financial_planning | 1.99 | 49.8% | 1.145x | 1.138–1.205 | 1.125 | 92.5% | — |
| financial_planning | astral | 2.24 | 56.0% | 1.294x | 1.276–1.303 | 1.217 | 82.5% | — |
| financial_planning | postgresql | 1.90 | 47.6% | 1.110x | 1.105–1.114 | 1.091 | 90.0% | — |
| **financial_planning** | **financial (DIAG)** | 1.70 | 42.5% | **1.011x** | **0.996–1.012 ⚠️** | 1.014 | 82.5% | **DISABLED** |

> ⚠️ **These speedups are measured on a decoder that does not always reproduce
> its own verifier's output.** The *Exact vs Chunked* column is a correctness
> gate, and it fails on 5–22.5% of generations per cell — worst is
> `astral|astral` at 77.5%, i.e. **27 of 120 generations emit different text**.
> Speed from a decoder emitting different text is not strictly comparable to the
> baseline it is timed against. **[CORRECTED]**: the original printed this column
> without comment.

### 1.1 The financial diagonal does not resolve **[CORRECTED]**

The original reported `financial_planning` at **0.97x** and concluded that
in-domain financial text is simply harder to draft. Properly powered, that is not
what the data supports:

- median **1.011x**, but the three repeats span **0.996–1.012** — they
  **straddle 1.0**
- predicted from τ=1.70 is **1.014** — i.e. the model says *break-even*

So the domain is gated off because the win is **unmeasured**, not because a
penalty was demonstrated. "Penalty" and "no penalty" were both claims this
instrument could not support.

### 1.2 The gate is not what the original said **[CORRECTED]**

The original attributed the disable to *"τ = 1.63 falling below the break-even
threshold τ ≈ 1.65"*. The actual production gate in code is:

```python
speedup_multiplier > 1.00  AND  tau >= 1.39  AND  not straddles_unity
```

τ = 1.70 **passes** the τ gate comfortably. What fires is the resolution
requirement. The "τ ≈ 1.66" figure is the analytic break-even of the *predicted*
model — a different quantity that plays no part in the decision.

The third clause is new. The gate previously enabled on a median above 1.0 even
when the repeats straddled it, which ships a coin flip and pays the correctness
cost for no resolved benefit.

### 1.3 Levelling the prompt set changed what was measured **[NEW FINDING]**

To reach n=40, financial was padded with 20 prompts from the datagen pipeline's
held-out pool (490 generated, 304 used in training, so ~196 genuinely unseen;
filtered for self-containment). `evaluation_data.jsonl` was deliberately **not**
extended — ~18 scripts read it, and its `expects` field drives keyword accuracy
scoring elsewhere.

Provenance is tracked per prompt, and the split is decisive:

| expert | τ on 20 curated | τ on 20 held-out | Δ |
| :--- | ---: | ---: | ---: |
| astral | 1.928 | 2.328 | **+0.40** |
| postgresql | 1.782 | 2.237 | **+0.46** |
| financial_planning | 1.627 | 1.775 | **+0.15** |

**The held-out prompts are systematically easier to draft for every expert.** So
levelling did not produce a cleaner measurement of the same quantity — it
measured a different, easier distribution. The financial diagonal's move from
0.973x to 1.011x is substantially a prompt-mix effect.

The curated-half τ values reproduce the earlier n=20 run to **three decimal
places** (1.928 / 1.782 / 1.627, Δ = 0.000), which independently confirms the
measurement is deterministic and reproducible.

**Read the financial domain as sitting at break-even with a prompt-dependent
sign, not as resolved in either direction.**

### 1.4 τ is deterministic — a review finding that was itself wrong **[CORRECTED]**

The review criticised the benchmark for accumulating τ, accept% and exact% under
`if rep == 0:`, leaving "the decision variable single-shot." The accumulators were
moved to cover all repeats. Measured spread across repeats afterwards:

```
tau_spread = 0.0000   on all 9 cells
```

Greedy decode emits identical tokens every repeat, so τ and exact% are
**deterministic** — the change was numerically a no-op (exact% went 77.5% →
77.5%). The criticism was wrong: single-shot was sufficient for those quantities.
Only the **speedup ratio** varies run to run, and it already had 3 repeats.

The fix is kept because it makes the denominator honest (120 trials, not 40), but
it resolved no real statistical problem. What was genuinely underpowered was the
**prompt set**, not the repeat count.

### Router configuration

```json
{ "astral": true, "postgresql": true, "financial_planning": false }
```

---

## 2. High-Batch Scaling Frontier ($B=1 \dots 64$)

$B \in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]$ on `Qwen/Qwen3.5-4B` in bfloat16.

| Batch ($B$) | Step Latency | Aggregate | Per-Req | Break-Even τ | Folding Win |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 28.68 ms | 34.87 tok/s | 34.87 tok/s | 1.15 | **1.80x** |
| **2** | 28.93 ms | 69.12 tok/s | 34.56 tok/s | 1.23 | **1.78x** |
| **4** | 32.31 ms | 123.81 tok/s | 30.95 tok/s | 1.14 | **1.86x** |
| **8** | 37.92 ms | 211.00 tok/s | 26.37 tok/s | 1.18 | **1.73x** |
| **12** | 41.64 ms | 288.16 tok/s | 24.01 tok/s | 1.17 | **1.73x** |
| **16** | 48.92 ms | 327.09 tok/s | 20.44 tok/s | 1.11 | **1.52x** |
| **24** | 56.38 ms | 425.70 tok/s | 17.74 tok/s | 1.27 | **1.49x** |
| **32** | 71.27 ms | 449.00 tok/s | 14.03 tok/s | 1.20 | **1.33x** |
| **48** | 86.11 ms | 557.41 tok/s | 11.61 tok/s | 1.43 | **1.29x** |
| **64** | 104.74 ms | **611.04 tok/s** | 9.55 tok/s | 1.45 | **1.21x** |

### Corrected takeaways

1. **Aggregate scaling: 17.52× — which is 27.4% of the perfect 64×. [CORRECTED]**
   The original called this *"Massive Aggregate Scaling"* and omitted the
   denominator that the benchmark itself prints (`perfect scaling would be 64x`).
   $B=2$ is close to free (+0.9% latency for 2× tokens); efficiency loss
   concentrates above $B=16$.

2. **Per-request throughput collapses 73%, from 34.87 to 9.55 tok/s. [CORRECTED]**
   The original left this in an uncommented column. Under 10 tok/s per user at
   $B=64$ is at the edge of interactive usability — $B=64$ is a throughput
   operating point, not a latency one. $B \le 4$ keeps per-request above 30 tok/s.

3. **Speculation's break-even *rises* with batch; it does not stay flat. [CORRECTED]**
   The original said the ratio *"remains remarkably flat (1.11–1.27) through
   $B=32$, proving speculation remains viable."* That column is the τ you must
   **exceed** to profit, and it goes 1.15 → 1.45 as $B$ goes 1 → 64. Speculation
   gets *more* expensive with batch. It does survive — measured in-domain τ is
   1.93/1.94, still clearing 1.45 — but headroom shrinks from +0.78 to +0.48.
   **No speculative decoding was run at $B>1$**; this is a $K$=1-vs-$K$=4
   forward-cost proxy. The script's own docstring is the honest framing: *"if the
   chunked penalty shrinks at higher B, speculation gets cheaper. If it grows,
   speculation is a batch-1-only trick."* It grew.

4. **Folding win decays monotonically. [CORRECTED]** 1.86x → 1.52x → 1.33x →
   1.29x → 1.21x. The original quoted 1.73–1.86x for $B \le 12$ then jumped to
   $B=64$, which reads as flat. It is real but shrinking with batch.

---

## 3. Directory Structure **[CORRECTED]**

The original documented a `scripts/` tree that no longer exists — every path in
it (`scripts/factory/…`, `scripts/runtime/…`) had moved to `benchmarks/`. It also
claimed *"All documentation … has been updated"*, citing a `walkthrough.md` that
exists nowhere in the repo.

```
benchmarks/
├── factory/
│   ├── architecture_comparison/
│   ├── geometry/
│   │   ├── alpha_sweep/
│   │   ├── preflight_svd_probe/
│   │   └── times_above_chance/
│   └── m1_vs_m2_regime/
├── runtime/
│   ├── folding/
│   ├── memory/
│   │   ├── cuda_graph/
│   │   ├── pristine_state_buffer/
│   │   └── zero_recapture_swapping/
│   ├── performance/
│   │   ├── batch_scaling/
│   │   ├── fla_triton_kernels/
│   │   └── prefill_vs_decode/
│   ├── router/
│   │   └── vram_state_routing/
│   └── speculative/
│       ├── mtp_head_folding/
│       ├── mtp_speculative/
│       └── speculation_matrix/
└── superseded/
```

---

## What this session actually established

- Eight of nine matrix cells are resolved speculation wins (1.079x–1.294x).
- The ninth (`financial_planning` in-domain) is **at break-even and unresolved**,
  with a prompt-mix-dependent sign.
- Batching is the dominant throughput lever, at 27.4% scaling efficiency and a
  73% per-request cost.
- Speculation's margin **erodes** with batch size rather than holding flat.
- Every speedup here carries an unresolved correctness caveat (5–22.5% divergence
  from the decoder's own verifier).
