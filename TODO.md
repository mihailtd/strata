# Project Execution Plan

> **Revisit Billboard Impostors when having 10–20 distinct domain adapters for "hot-swapping"!** It requires building a fixed-slot tensor copy infrastructure from scratch, and without a fleet of distinct domain adapters (SQL, Rust, YAML), you would be testing infrastructure without a real workload.

---

## ⚠️ Status of This Document (2026-08-11)

This plan was written before several of its premises were measured. Corrections,
all from real runs on this repo's hardware — see `## Measured Reality` below for
the full numbers:

- **Velocity-Masked SFT is dead as a pillar.** The "59.95% → 83.70%" figure this
  document is built on came from runs where the gate was provably inert (0% of
  layers ever masked, due to a hook bug). Under a controlled A/B it neither
  improves adherence nor saves compute. Sections describing it as a load-bearing
  component are struck through below rather than deleted, so the reasoning trail
  survives.
- **The ~3 MB micro-adapter target is real as arithmetic, unmet in practice.**
  Kronecker factorization really does give ~3.2 MB on this model's shapes, but
  every adapter measured at that scale collapses in quality (LoKr 1.2-2.1 MB →
  14-19% adherence vs base 11.94%). Nothing yet reaches ~3 MB with usable
  quality; the best small result is `id_kron_r8` at **12.6 MB** (4x over target).
- **⭐ THE UNIFYING RESULT — why every compression scheme here has failed.**
  Measured directly (CPU, no training): fine-tuning updates on this model are
  **~one independent direction per layer, mutually near-orthogonal across layers
  AND across tasks.** Projecting the 60.80% winner onto its own SVD basis
  saturates at exactly k = layer count (k=32 → 98.23%, k=64 → no gain), and
  projecting it onto a *different task's* basis retains **0.00%** — cosine
  3.25e-4 vs 2.06e-4 expected for random directions. There is no shared
  low-dimensional adaptation manifold to exploit, across depth or across tasks.
  This single property predicted, in advance, the failure of Tucker (15.66%),
  LoKr (14.42%) and master_basis (13.40%). **Measure subspace overlap before
  building any further shared-basis scheme** — `scripts/extract_svd_basis.py`
  plus the retention projection answers it on CPU in ~15 s.
- **The architecture benchmark is confounded — twice.** Baseline on-disk sizes
  were inflated ~2x (fp32 assumed vs real bf16), and the custom-path variants
  carry an **~18x inflated effective update** from a kaiming fan_in bug. Fixing
  only the init drops `custom_standard` from 58.17% to 37.55%, i.e. to peft
  LoRA's level. **No custom-vs-peft architecture claim in this document is
  currently supported.** See `## Measured Reality`.
- **id_kron vs Stock LoRA: PARITY (controlled, 21 fresh adapters).** Peaks
  57.83% (id_kron rt8, 6.26M) / 56.24% (rt16, 12.42M) / 56.03% (LoRA, 10.62M) --
  a 1.80pp spread = 0.4 questions at n=20. Architecture is not the lever; tuning
  is. Optimal scaling varies 16x by architecture (LoRA 16, rt8 2, rt16 1), which
  is why the earlier single-point audit could reach any verdict. That audit is
  **retracted**: it used pre-existing adapters with no training records, reported
  id_kron at 49.8M params when the loaded adapter had 6.26M (inverting its own
  headline), and left scaling uncontrolled at 32.0 vs 2.0. Its 72.44% does not
  reproduce (best of 15 controlled adapters: 57.83%). id_kron's 41% parameter
  saving is real on DISK but vanishes when folded (block-diagonal expansion makes
  V dense) and is irrelevant to swap latency anyway (payload is 0.2% of the
  10.24 GB a swap moves). LoRA is the safer default: id_kron diverges above
  scaling 2 at both ranks. See README section 8.
- **Training loss is anti-correlated with adherence on this benchmark** — the
  best-scoring run has nearly the worst loss. Do not use loss as a quality proxy.
  The same caution applies to *reconstruction energy*: a k=32 basis retaining
  **98.23%** of the winner's energy delivered **54.70%** adherence, not the
  **59.7%** that `energy x score` predicts — a 5.0 pp overestimate. **Never
  convert a retention/loss figure into a quality claim without running the eval.**
- **The entry point is NOT a HIP kernel.** `peft` 0.20.0 already ships DoRA
  (`LoraConfig(use_dora=True)`) and Kronecker (`LoKrConfig`). The core
  quality-vs-size question is answerable in a few PyTorch training runs, before
  committing to weeks of C++/HIP work.
- **❓ In-Place MTP Adapter Folding: EXPERIMENTAL / UNPROVEN.** Fusing low-rank adapters directly into native MTP head parameters ($W_{\text{mtp}} \leftarrow W_{\text{mtp}} + S \cdot U_{\text{mtp}} V_{\text{mtp}}$) executes in **0.30 ms** with zero VRAM churn and ~3.3 ms single-token step time ($K=1$). However, on a micro-eval ($N=41$ tokens across 4 prompts), 30-step quick-tuned adapters degraded accuracy from 21.95% (un-adapted MTP) down to 12.20%-14.63%. Due to the small sample size ($N=41$) and lack of end-to-end acceptance rate measurement, MTP Adapter Folding remains **❓ Experiment / Unproven** until evaluated with full domain training and $N \ge 1000$ token evaluation.
- **A local peft patch was found and vendored.** peft's LoKr has no 4-bit
  dispatch and crashes on a quantized base. The fix had been applied by editing
  `.venv/.../peft/tuners/lokr/layer.py` in place — which, because uv *hardlinks*
  its global cache into venvs (verified: same inode, 4 links), had silently
  mutated `~/.cache/uv` and propagated to `training/.venv` as well. Now restored
  to pristine upstream (peft matches its install-record hashes) with the fix
  living in `src/gnn_experiment/peft_compat.py`.
- **✅ A working hot-swap primitive now exists** (`swap_adapter_weights`), 3.7x
  faster than the re-wrap it replaces and — unlike the incumbent — safe to call
  repeatedly without corrupting the model. **This unblocks Step 3 (Billboard
  Impostor).** But the measured agentic-loop payoff it was meant to justify is
  **~2% of wall-clock**, not "stuttering → instantaneous": decode at ~31 tok/s
  makes a 200-token step cost 6.4 s, dwarfing any swap. See `Hot-swap
  primitive` below.
- **🟢 CORRECTED: speculative decoding works here, at 1.03-1.39x.** This bullet
  previously read "architecturally unavailable". The error was treating
  `fla`/`causal_conv1d` as one dependency — they are two, with independent
  fallbacks. `causal-conv1d` genuinely needs `nvcc` and cannot build; **`fla` is
  Triton-based and runs natively on gfx1100**. Installing it collapses the
  multi-token forward penalty from **2.84x to ~1.2-1.5x**, dropping speculation's
  break-even from ~2.8 to **~1.4 accepted tokens** at K=4 (measured on a warm
  cache: 33.27 ms vs 27.87 ms at K=1, a 1.19x ratio; break-even also accounts for
  draft cost and the state re-advance, so it exceeds the raw ratio).
  ⚠️ An earlier entry here claimed a **0.94x ratio / 0.94-token break-even** from
  a script that used `use_cache=False` — a context-free forward whose cost is
  flat in K by construction. That is retracted; see README "Residual overhead
  audit". `transformers` still refuses its
  built-in assisted generation, but a custom loop with snapshot/restore of the
  fixed-size 52.5 MB recurrent state works. See README "Speculative decoding —
  UNBLOCKED" for the full table.
- **⚠️ An earlier decode figure was contaminated and is now corrected.** The
  17.77 tok/s used in prior analysis is not reproducible — the same config
  re-measured on an idle machine gives 25.25 tok/s. The original was taken
  while 8 stray MLflow workers and exhausted swap were starving the host.
  Batch-1 decode is CPU-bound on kernel launches; **benchmark it only on an
  idle machine.**
- **⭐ The one result that got BETTER than its source.** An 8.00 KB coefficient
  adapter — CPU least-squares projection of the 60.80% LoRA onto a k=32 SVD
  basis, then 150 steps tuning only 4,096 scalars at lr=1e-2 — scores
  **61.96%**, the highest in the project. Caveats: it *requires* that LoRA as
  input (2x total training budget), +1.16pp is inside unmeasured seed noise, and
  it still needs the 61.3 MB task-specific bank. Real single-task
  distillation; not the multi-task architecture.

---

## 📐 Measured Reality (supersedes claims below)

Everything in this section is from real runs/arithmetic on Qwen3.5-4B on this machine, greedy-decoded eval (`do_sample=False`), 20-question astral benchmark:

### Architecture Benchmark & Payload Sizes

> ⚠️ **This table is CONFOUNDED. Do not draw architecture conclusions from it.**
> Two systematic errors were found in it, both flattering the custom
> architectures over the peft baselines. Corrected analysis directly below.

Adherence values below are correct (verified against MLflow). **On-disk sizes
and all custom-vs-peft comparisons are not.**

| Scheme | Trainable Params | On-Disk (as claimed) | **On-Disk (measured)** | Adherence % |
| :--- | :---: | :---: | :---: | :---: |
| **Base Model** | 0 | — | — | **11.94%** |
| `id_kron_r16` | 12.4M | 24.9 MB | **24.92 MB** ✅ | **59.32%** |
| `custom_standard` | 10.6M | 40.0 MB | **21.32 MB** ❌ 88% inflated | **58.17%** |
| `id_kron_r8` | 6.26M | 12.0 MB | **12.61 MB** ✅ | **51.44%** |
| PEFT LoRA (r=8, α=16) | 10.6M | 40.47 MB | **21.27 MB** ❌ 90% inflated | **40.47%** |
| **`custom_standard_fixedinit`** | 10.6M | — | **21.32 MB** | **37.55%** |
| PEFT DoRA | 11.4M | 40.98 MB | **22.83 MB** ❌ 79% inflated | **33.92%** |
| KroTucker (r=64) | 85.0M | 170.0 MB | **170.07 MB** ✅ | **19.70%** |
| PEFT LoKr (r=32) | 1.04M | 2.12 MB | **2.12 MB** ✅ | **18.92%** |
| PEFT LoKr (r=8) | 585K | 1.22 MB | **1.22 MB** ✅ | **14.42%** |
| `master_basis` (k=8, cold random basis, α=4096) | 1,024 (c_i) | 4.67 KB | **4.67 KB** ✅ | **13.40%** 🔴 |
| **`mbproj_k32`** (k=32, SVD basis, projected — no training) | 4,096 (c_i) | — | **8.00 KB** + 61.3 MB bank | **54.70%** |

#### 🔴 Error 1 — baseline sizes inflated ~2x (mixed dtype conventions)

Every *new* architecture got its real bf16 on-disk size; every *baseline* got an
fp32-assumed size (10.6M x 4 bytes ~= 42 MB instead of the actual 10.6M x 2 =
21.3 MB). Note PEFT LoRA's "40.47 MB" is literally its adherence value (40.47%)
copy-pasted into the size column.

Consequence: **"id_kron_r16 beats custom_standard while saving 40.7% payload
(24.9 vs 40.0 MB)" inverts.** Real: 24.92 vs 21.32 MB — id_kron_r16 is **17%
LARGER**, and has more parameters (12.4M vs 10.6M). Beating a smaller-capacity
adapter by 1.15 pp while using 17% more capacity is not architectural evidence.
Likewise `id_kron_r8`'s "70.4% payload reduction" is really **41%** (12.61 vs
21.32 MB), for **-6.73 pp** adherence — a legitimate size/quality tradeoff point,
but not a free win, and 12.6 MB is 4x over the ~3 MB micro-adapter target.

#### 🔴 Error 2 — the custom path has an ~18x inflated update (init bug)

`NovelLoraLinear` stores its down-projection **transposed** vs peft: `lora_a` is
`(in_features, rank)` where peft's `lora_A` is `(rank, in_features)`.
`nn.init.kaiming_uniform_` derives fan_in from `tensor.size(1)`, so on this
layout it reads fan_in = **rank (8)** instead of **in_features (2560)** and
over-initialises by `sqrt(2560/8)`. Three independent measurements agree:

| | ratio |
| --- | --- |
| Predicted `sqrt(in/rank)` | 17.889x |
| Measured at init | 17.824x |
| Measured in trained ΔW (custom vs peft) | **18.294x** |

`lora_b` is zero-init so the starting delta is 0 and nothing errors — but
`∂L/∂B ∝ A`, so B grows ~18x faster. The two implementations also converge to
**essentially orthogonal solutions**: mean cosine(ΔW_peft, ΔW_custom) = 0.0012,
with |cos| > 0.1 on **0 of 128** modules.

**Controlled test** — `custom_standard_fixedinit` is identical to
`custom_standard` except the init uses the true fan_in:

| variant | init | final loss | adherence |
| --- | --- | --- | --- |
| `custom_standard` | buggy (fan_in=8) | 1.1396 | **58.17%** |
| `custom_standard_fixedinit` | correct (fan_in=2560) | 1.0074 | **37.55%** |
| PEFT LoRA | correct (fan_in=2560) | 1.0083 | **40.47%** |

Fixing one line drops custom_standard **58.17% → 37.55%**, landing next to peft
LoRA, and its training loss moves to within **0.0009** of peft's (from a 0.13
gap). **The two implementations are equivalent once init matches; the entire
17.7 pp "custom beats peft" advantage was the inflated update.**

All custom-path variants inherit this — verified: `id_kron` (`w_a` as
`(in_sub, r2)`) and `krotucker` (`tucker_a` as `(in_f, rank)`) use the identical
transposed pattern.

#### ✅ What survives

- Comparisons **within** the custom family (`id_kron_r16` vs `custom_standard`)
  are fair — same inflated init. But 1.15 pp with 17% more params is inside
  seed noise, which has never been measured here.
- **DoRA underperforms LoRA on this task** (33.92% vs 40.47%) at 77% more wall
  time (327 s vs 185 s) — contradicts this document's "DoRA ~90% quality" premise.
- **LoKr's collapse is real, not a bug.** Investigated directly: peft's LoKr has
  no 4-bit dispatch and crashed on the quantized base (`RuntimeError: shape
  '[11796480, 1]' is invalid for input of size 23592960`; 11796480 = 9216*2560/2,
  the packed-nibble byte count). That was patched before the recorded runs, and
  the saved adapter is fully trained with a delta **1.8x larger** than LoRA's
  (0.738 vs 0.410 mean Frobenius) applied at the correct orientation. So the low
  score is a structural-expressiveness limit, **not** the "3-factor gradient
  vanishing" claimed — gradients that vanish do not produce a larger delta than
  the baseline. (Patch now vendored: `src/gnn_experiment/peft_compat.py`.)

#### The real finding

An ~18x larger effective update is worth **+20.6 pp** adherence at 150 steps /
lr 2e-4. That means the peft-default configuration is badly under-powered for
this task and step budget, and **update magnitude (α/r or LR) — not
factorization — is the dominant lever measured so far.** Until the baseline is
magnitude-tuned, any architecture comparison mostly measures which arm
accidentally had the larger effective learning rate.

### 🏁 The honest baseline: magnitude-tuned peft LoRA wins outright

Same peft LoRA, r=8, only `lora_alpha` swept (nothing else changed — no custom
code, no init change, stock `peft`):

| Variant | α | scaling | Train loss | **Adherence** |
| :--- | ---: | ---: | ---: | ---: |
| **PEFT LoRA** | **256** | **32x** | 1.0746 | **60.80%** 🥇 |
| `id_kron_r16` (custom path, inflated init) | — | — | 1.1301 | 59.32% |
| `custom_standard` (custom path, inflated init) | 16 | 2x | 1.1396 | 58.17% |
| `id_kron_r8` (custom path, inflated init) | — | — | 1.1301 | 51.44% |
| PEFT LoRA | 128 | 16x | 1.0187 | 50.23% |
| PEFT LoRA | 16 | 2x | 1.0083 | 40.47% |
| `custom_standard_fixedinit` | 16 | 2x | 1.0074 | 37.55% |
| PEFT DoRA | 16 | 2x | 1.0065 | 33.92% |

**Stock LoRA with one tuned hyperparameter (60.80%) beats every custom
architecture in this project, including `id_kron_r16` (59.32%) — the variant
previously crowned 👑 — at the same 21.27 MB payload and no new code.** The
custom architectures were never beating LoRA; they were beating an
*under-tuned* LoRA whose effective update was ~18x too small.

Two consequences:

1. **The bar for any new architecture is 60.80%, not 40.47%.** Anything claimed
   to beat LoRA must beat magnitude-tuned LoRA at equal payload, with the init
   scales matched.
2. **α is not yet optimised** — 16 → 128 → 256 is still climbing (+9.8 pp then
   +10.6 pp). The true LoRA ceiling on this benchmark is above 60.80% and
   unmeasured. A proper sweep (α=512, 1024, and LR variation) should be run
   before any factorization work continues.

#### ⚠️ Training loss is anti-correlated with adherence here

Note the loss column above: the **best** adherence (60.80%) has nearly the
**worst** loss (1.0746), and the best loss (1.0065, DoRA) has the worst
adherence (33.92%). Across the whole sweep, lower training loss tracks *worse*
tool adherence. Fitting the SFT distribution more closely does not produce more
uv/ruff/ty preference — so **training loss must not be used as a proxy for
benchmark quality in this project**, and any "our loss is lower" claim says
nothing about the metric that is actually being optimised for.

### Master Basis Set Reduction — CLAIM FALSIFIED 🔴

`master_basis` freezes a shared bank of `k` basis factors and trains only `k`
scalar coefficients per wrapped Linear (1,024 total = **4.67 KB** payload). The
pitch: keep one bank in VRAM, hot-swap tasks as sub-KB "mixture recipes",
blend tasks by adding coefficient vectors.

**Result: the architecture cannot work on this model, for a measured reason.**

#### 1. Cold random basis fails, and it is NOT the alpha trap

Calibrated *before* running (this is the mistake we made with peft LoRA and did
not repeat): with 8 zero-init scalars per module and Adam moving each ~`lr`/step,
`|c| <= lr*steps = 0.03` after 150 steps, capping reachable `||dW||_F` at
**0.094** vs the 60.80% winner's **5.82** — a 62x shortfall. So alpha was swept:

| α | scaling | loss | adherence |
| ---: | ---: | ---: | ---: |
| 256 | 32x | 1.3889 | 12.43% (+0.50 pp) |
| 1024 | 128x | 1.3461 | 12.64% (+0.70 pp) |
| 4096 | 512x | 1.2473 | 13.40% (+1.46 pp) |

A **16x magnitude increase bought +0.97 pp.** Flat, not a scaling curve — unlike
peft LoRA, where 16x alpha bought +20.3 pp. This is not an under-powered update;
8 random directions in a 2560x9216 space span nothing task-relevant.

#### 2. Adaptation does not compress across layers (~1 direction per layer)

Projecting the 60.80% winner onto its OWN SVD basis, least-squares optimal
coefficients, no training (this is the *ceiling* — training cannot beat it):

| k | overall retained energy | median per-module | bank size |
| ---: | ---: | ---: | ---: |
| 8 | 43.07% | **0.0%** | 15.3 MB |
| 16 | 60.81% | 99.7% | 30.7 MB |
| **32** | **98.23%** | 100.0% | 61.3 MB |
| 64 | 98.23% (**zero gain**) | 100.0% | 122.7 MB |

Retention tracks **k / (layers in the shape group)**: the 8-layer attention
groups hit ~100% at k=8, the 32-layer MLP groups hit ~25%. It saturates exactly
at k = 32 = the layer count, and k=64 adds *nothing*. **Each layer contributes
its own independent adaptation direction; they do not share a subspace.** The
k=8 / 4.67 KB configuration was mathematically incapable of representing the
task before a single training step ran.

#### 3. The transfer claim: 0.00%, at random-chance level

The systems story needs ONE bank serving MANY tasks. Projecting the astral
winner onto a **postgres**-derived k=32 bank:

| basis source | retained energy |
| --- | ---: |
| astral (in-task) | 98.23% |
| **postgres (cross-task)** | **0.00%** (min = median = max = 0.0%, every group) |

Verified not a bug: both banks are real and unit-normalised (655,360 nonzero
elements each), and cosine between their leading basis elements is **3.25e-4**
versus **2.06e-4** expected for *randomly oriented* directions in a
23,592,960-dim space. **The two tasks' adaptation subspaces are statistically
indistinguishable from random relative to each other.**

#### The economics invert

Each task needs its own **61 MB** bank to hold an **18.7 KB** coefficient
vector. That bank is **~3x larger than the 21.3 MB LoRA it replaces** — worse
for one task, and no better for many, because banks do not share. The sub-50 µs
hot-swap and scalar-blending wins were downstream of "quality holds with a
*shared* bank"; that premise is measured false, so they do not land.

#### ✅ What survives from master_basis — MEASURED, not extrapolated

**k=32 in-task compression is real: 54.70% adherence from 8.00 KB of
coefficients** (`mbproj_k32`, built by least-squares projection of the
`lora_a256` winner onto its own k=32 bank — *no training*, so this is the
provable ceiling for that bank).

| | value |
| --- | ---: |
| source (`lora_a256`, 21.3 MB) | 60.80% |
| **projected k=32 (8.00 KB coefficients)** | **54.70%** |
| base model | 11.94% |
| adherence retained (raw ratio) | 90.0% |
| adherence retained (gain-over-base) | 87.5% |

At 8 KB this beats several fully-trained 21 MB adapters outright — peft LoRA
α=16 (40.47%), DoRA (33.92%), `id_kron_r8` (51.44%).

> ⚠️ **Energy retention overestimates behaviour retention.** The naive
> extrapolation `98.23% energy x 60.80%` predicts **59.7%**; the measured value
> is **54.70%**, i.e. **5.0 pp optimistic**. 98.23% of the Frobenius energy
> survives but only 90% of the adherence does. This is the same lesson as the
> loss/adherence anti-correlation above: **energy-style proxies must not be
> converted into quality claims on this benchmark.** Any number of the form
> "retention x score" is an estimate, not a measurement, and must be labelled
> as such until an eval is run.

#### 🔴 …but the storage accounting still does not work

| | |
| --- | ---: |
| coefficients | 8.00 KB |
| + basis bank (task-specific — cross-task retention **0.00%**) | 61.3 MB |
| **total for one task** | **61.3 MB** |
| vs the LoRA it replaces | 21.3 MB |
| | **2.9x WORSE** |

The "~1100x compression" framing compares the 8 KB coefficient payload against
the 21.3 MB LoRA while omitting the bank those coefficients are meaningless
without. The ratio only holds if one bank serves many tasks; cross-task
retention is 0.00%, so it never does. **Single-task compression: real and
measured. Multi-task deployment story: still dead.**

#### The unifying explanation

- **Why all three compression schemes failed.** Fine-tuning updates on this
  model are ~one independent direction per layer, mutually near-orthogonal
  across layers *and* across tasks. That single measured property predicts, in
  advance, that **any** shared-subspace compression scheme will fail here —
  which is exactly what Tucker (15.66%), LoKr (14.42%) and master_basis (13.40%)
  all did. Do not build another one without first measuring subspace overlap;
  `scripts/extract_svd_basis.py` + the retention projection does it on CPU in
  ~15 s, with no training.

### Velocity-Masked SFT — CLAIM FALSIFIED 🔴

Controlled three-arm comparison, identical everything except layer selection.

> ⚠️ Adherence in THIS table was measured with the old **sampled** decoding and
> is therefore NOT comparable to the greedy numbers in the table above (where
> `custom_standard` = 58.17%, not 63.82%). The three arms here are still
> comparable *to each other* — all sampled, all same config. All three also
> share the inflated-init bug described above.

| Arm | Train loss | Wall s | Quiet % | Set churn % | Adherence (sampled) |
| --- | --- | --- | --- | --- | --- |
| `custom_standard` (no masking) | 1.1396 | 182.6 | 0 | — | **63.82%** |
| `velocity` (bottom-25% by Δh_l) | 1.1135 | 198.1 | 25.0 | 0.0 | 61.73% |
| `random_mask` (random 25%, control) | 1.1040 | 180.7 | 25.0 | 76.2 | 60.85% |

1. **The velocity ranking carries no signal.** Random selection lands within 0.9 pp
   of velocity — i.e. choosing layers by measured Δh_l is not distinguishable from
   choosing them by coin flip.
2. **It was never stochastic-depth regularization.** Churn is 0.013: the *same* 8
   layers (`[12,13,14,16,17,24,25,29]`) are masked on every single step. It is a
   static capacity cut to 24/32 layers, not a dynamic regularizer.
3. **No speedup is physically available.** The adapter is 0.437% of parameters;
   masking 25% of its layers skips ~0.1% of compute, while the frozen base still
   runs full forward+backward through every layer. Measured: velocity was *slower*
   (198.1 s vs 182.6 s) because the measurement hooks cost more than the skipped
   math saves. It therefore cannot "recover DoRA's ~20% overhead".
4. **The original 83.70% never demonstrated the technique.** It was produced with
   `skip_recompute=True` under non-reentrant gradient checkpointing, where the
   hook's `torch.is_grad_enabled()` test misfires and records nothing — the gate
   was inert for that entire run.

### Eval harness — FIXED ⚠️

The benchmark previously used sampled decoding (`do_sample=True, temperature=0.2`).
Two *mechanically identical* configs scored 76.96% and 63.05% on the postgres set —
a 13.9 pp swing of pure decoding noise, larger than any real effect measured. Eval
is now greedy (`do_sample=False`) everywhere. **All adherence numbers recorded
before this change are not comparable to numbers after it.**

### Hot-swap primitive — BUILT ✅ / agentic-loop claim — NOT SUPPORTED 🔴

`swap_adapter_weights()` + `AdapterPayload` + `prepare_swap_slots()` in
`novel_peft.py`; benchmarked by `scripts/benchmark_adapter_swap.py`.

**What it replaces.** `load_novel_adapter` calls `apply_novel_lora`, which walks
the module tree and replaces every target Linear with a fresh wrapper. Calling
it twice on one model double-wraps and corrupts it (confirmed by a real crash
earlier), and its cost is Python-side module surgery, not bytes moved. The new
primitive resolves payload keys to the model's *existing* VRAM tensors once,
then a swap is a fixed set of `copy_` calls into static addresses — no
allocation, no graph mutation, and safe to call repeatedly. **This is what was
blocking Billboard Impostors.**

Measured, 50 trials, `torch.cuda.synchronize()` on both sides:

| mechanism | p50 | vs incumbent |
| --- | ---: | ---: |
| `load_novel_adapter` (re-wrap, incumbent) | 117.1 ms | — |
| LoRA in-place, blocking | 60.3 ms | 1.9x |
| **LoRA in-place, non_blocking + pinned** | **32.1 ms** | **3.7x** |
| coefficient in-place, blocking | 17.3 ms | |
| **coefficient in-place, non_blocking + pinned** | **0.64 ms** | **183x** |

`pin_memory=True` on the host payload is load-bearing: without page-locked
memory `non_blocking=True` is silently synchronous (17.3 ms → 0.64 ms for the
coefficient arm is almost entirely this).

**A failed optimisation, recorded so it is not retried.** Effective bandwidth is
0.66 GB/s against ~25 GB/s of PCIe, which looked launch-overhead-bound, so the
swap was rewritten to issue through `torch._foreach_copy_`. It changed nothing:
**32.08 ms vs 32.13 ms.** The cost is per-transfer DMA setup on this WSL2/ROCm
paravirtualised path, which batching cannot fuse. The `batched=` flag is kept
(correct, harmless) but is not a lever. Reaching the frequently-quoted "<30 µs"
would need adapter parameters allocated as slices of one contiguous VRAM buffer
so a swap is a *single* DMA rather than 128 — a real architectural change, not a
flag. We measure **644 µs**, 21x above that figure.

#### The agentic-loop payoff is ~1-2%, not "massive"

> ⚠️ **Decode-rate correction.** Earlier analysis used **17.77 tok/s** from
> `astral_inference_benchmark`. That figure is **not reproducible** and should
> not be cited. Re-measuring the *identical* configuration (velocity adapter,
> 256 tokens, sampled) now gives **25.25 tok/s**. The original was recorded
> while the machine was under heavy contention — 8 stray MLflow `huey` workers
> plus fully-exhausted swap, immediately before the OOM kills documented
> elsewhere in this session. Batch-1 decode is CPU-bound on per-layer kernel
> launches, so host contention degrades it directly. **Benchmark decode only on
> an idle machine, and re-measure rather than reusing an old number.**

Clean re-measurement (greedy, idle machine):

| configuration | tok/s |
| --- | ---: |
| base model, 128 new tokens | 29.95 |
| base model, 256 new tokens | 31.07 |
| base model, 512 new tokens | 32.07 |
| **base model + custom adapter** | **25.32** (**-18.7%**) |
| sampling vs greedy | no measurable difference |

**New finding: the custom adapter costs 18.7% of decode throughput.** At 0.437%
of parameters that is not a FLOPs effect — it is 256 extra small matmuls per
token (2 per wrapped Linear x 128) each paying kernel-launch overhead at batch 1.
A merged adapter (`W + dW` folded into the base weights) would avoid it entirely
for single-adapter serving. **This prediction was subsequently tested and holds
— see "Weight folding" below — but merging is not free, and the cost is not the
one anyone expected.**

Swap overhead against the corrected rate (31.07 tok/s base):

| swaps | tok/step | pure decode | rewrap | lora-fast | coef-fast |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 50 | 48.3 s | 7.27% | 1.99% | **0.04%** |
| 30 | 200 | 193.1 s | 1.82% | 0.50% | **0.01%** |

A 200-token agent step costs **6.4 s** of decode; even the slow 117 ms swap is
~2% of one step. Going from the incumbent all the way to 8 KB coefficient swaps
saves **~2% of wall-clock** at realistic step lengths. The "3-5 second spinner
caused by adapter swapping" does not exist here — 30 swaps at the *claimed*
10 ms is 0.3 s, and the bottleneck is token generation, not weight movement.

#### Speculative decoding — ~~ARCHITECTURALLY BLOCKED~~ SUPERSEDED 🟢

> **This section is retained for the record but its conclusion is WRONG.**
> Installing `fla` (Triton-based, runs on gfx1100) collapses the multi-token
> penalty 2.84x -> ~1.2x and makes speculative decoding work at 1.03-1.39x
> depending on workload. The mistake below is conflating `fla` with
> `causal_conv1d`: only the latter requires `nvcc`. See README section 5.


The standard way to break a batch-1 decode ceiling does not work on this model
at all. Both variants fail immediately:

```text
ValueError: assisted generation is not supported with stateful models,
such as Qwen3_5ForCausalLM
```

Cause: 24 of 32 layers are `Qwen3_5GatedDeltaNet` linear-attention layers
carrying a fixed-size **recurrent state** instead of a growing KV cache.
Speculative decoding must roll that state back whenever a drafted token is
rejected — trivial for a KV cache (truncate), unimplemented for a recurrent
one — so `transformers` refuses the entire family up front.

This blocks **both**:

- **prompt-lookup decoding** (`prompt_lookup_num_tokens`) — needs no draft model
  and no extra VRAM, still refused;
- **draft-model speculation** with `Qwen/Qwen3.5-0.8B`, even though that draft is
  fully compatible on every other axis (tokenizer verified identical: 248,077
  vocab, same EOS, byte-identical encodings; 1.7 GB downloaded and cached).

The same hybrid architecture also blocks the *other* obvious decode fix: the
`fla`/`causal_conv1d` fast path for those layers is missing (`linear_attn` is
32.9% of decode time in the profile) and **cannot be installed on this rig** —
`causal-conv1d`'s build requires `nvcc`, which is CUDA-only.

**Net (SUPERSEDED — see the banner at the top of this section): both ARE
available via `fla`. What remains true is that `causal_conv1d` cannot build
here, and that `transformers`' built-in assisted generation still refuses this
model family.** The original, now-incorrect conclusion follows:

~~Net: on this model + hardware, neither speculative decoding nor the linear-
attention fast path is available.~~ Decode stays ~31 tok/s. Any plan that
assumes 2-3x decode speedup needs a different base model, not a different
adapter scheme. `scripts/benchmark_speculative_decode.py` re-checks this in one
run if the situation changes (e.g. a transformers release adds stateful-model
support).

Also note p99 >> p50 throughout (32 ms → 109 ms): the WSL2 GPU path is jittery,
and **that jitter alone is larger than the entire coefficient-swap saving**.

#### Multi-tenant VRAM: the argument inverts on measured data

| 100 tasks pinned in VRAM | |
| --- | ---: |
| standard LoRA | 2.13 GB |
| coefficients *if one bank were shared* | 62 MB (34x better) |
| **coefficients as measured** (cross-task retention 0.00% → one bank per task) | **6.13 GB — 2.9x worse** |

The density win requires bank sharing, which is exactly what the 0.00%
cross-task retention rules out. **Verdict: the primitive is a genuine
engineering win worth keeping; the latency and density stories it was meant to
justify are not supported on this hardware.**

---

### Weight folding — WORKS ✅ / not novel ⚪ / not free ⚠️

`FoldableExpert` + `WeightFoldingEngine` + `unwrap_novel_lora` in
`novel_peft.py`; benchmarked by `scripts/benchmark_weight_folding.py`,
`scripts/evaluate_folded_vs_wrapped.py`, `scripts/measure_fold_precision.py`.

Folding removes the adapter wrappers from the execution path by writing
`W_live = W0 + scaling * (U @ V)` into the base weights. It recovers the
wrapper tax predicted above.

| arm (bf16, greedy, one process, same weights) | tok/s |
| --- | ---: |
| base | 29.49 |
| **wrapped** | **25.09** (−15.6%) |
| **folded** | **30.38** (+21.1% vs wrapped) |
| base again (drift control) | 29.97 |

Folding reaches base speed. It does **not** exceed it — the printed "103% of
base" is inside the 1.6% drift between the two base arms, and base's first
sample was cold. Swap latency **p50 17.76 ms / p95 21.48 ms**, break-even at
**2.6 generated tokens**, so folding wins for any turn longer than a few words.

#### ⚠️ Three claims that were reported here earlier and are false

A previous `benchmark_multi_expert_chain.py` reported 11.246 ms swaps, 31.15
tok/s and 0.00000000 weight drift. **Those numbers were never produced by that
script** (its own `results/*_runs.jsonl` was absent, the summary JSON was
missing a key the script always writes). The script has been deleted along with
`benchmark_in_place_folding.py`, `evaluate_financial_planning.py`, and the
162-line dense-payload API they used. What was wrong, for the record:

- **Dense deltas cannot be held.** Materialising `dW` densely is a **812x**
  expansion of a 12.6 MB adapter → 10.2 GB fp32 each, 30.7 GB pinned for three,
  on a 23 GB box. Keeping the `(out,r)x(r,in)` factors on-device instead: **467
  MB for all three.**
- **11.2 ms was below the bus floor.** A dense swap moves 2 x 10.2 GB over a
  *measured* 12.6 GB/s PCIe link = **~1617 ms**. 11.2 ms implies 1821 GB/s.
- **bf16 add/sub is not reversible.** Measured **4.9e-4** L_inf drift after 400
  add/sub cycles. Our 0.0 is real but is *not* a numerical result — `activate`
  writes `W0 + dW` and `restore` is a `copy_`, so the drifting arithmetic never
  happens. It costs **5.12 GB VRAM** for the pristine copy.

Also fixed: scaling must be `alpha / rank_total`, derived from the tensors, not
a constant. A hardcoded 2.0 is **8x too large** for the `rank_in=8, rank_out=8`
financial adapter (correct: 0.25).

#### Folding preserves aggregate quality, not determinism

20 questions/domain, greedy, three arms, one model load:

| domain | base | wrapped | folded | delta | exact string match |
| --- | ---: | ---: | ---: | ---: | ---: |
| astral | 12.20% | 51.33% | 52.58% | +1.25pp | 13/20 |
| postgresql | 57.57% | 72.75% | 73.40% | +0.65pp | 6/20 |
| financial_planning | 60.00% | 35.00% | 35.00% | +0.00pp | 13/20 |

**32/60 exact matches — 47% of prompts produce different text.** Direction of
change is inconsistent (good hits 91→101, 102→117, 44→36), so this is re-rolled
generations around an unchanged mean, not degradation. **Fold for serving; do
not fold if anything downstream needs byte-reproducible output.**

#### 🔬 The one genuinely new finding: merge fidelity scales with |dW|/|W|

Folding stores the adapter *inside* a bf16 weight, so `W0 + dW` is rounded to 8
mantissa bits and small deltas are absorbed. The fp32 control is exactly 0.00%,
confirming precision as the sole cause:

| expert | \|dW\|/\|W\| | rel. error | delta absorbed | fp32 control |
| --- | ---: | ---: | ---: | ---: |
| astral (scaling 2.0) | ~0.08 | 0.96% | 3.22% | 0.00% |
| postgresql (scaling 2.0) | ~0.08 | 0.94% | 3.10% | 0.00% |
| **financial_planning (scaling 0.25)** | **~0.012** | **7.06%** | 7.71% | 0.00% |

financial's delta is ~7x smaller relative to `W` and its error is ~7.3x larger;
worst modules are consistently the late-layer `v_proj`/`o_proj` (layers 27, 31)
where the relative delta is smallest. **Rule: the smaller an adapter's delta
relative to the base weights, the more merging costs it.** A low-`scaling`
adapter should stay wrapped, or the base should be held at higher precision.

#### Novelty: none in the mechanism

`peft` 0.20.0 `LoraLinear.merge()`/`.unmerge()` already do
`base_layer.weight.data += delta_weight` in place, and the single-tenant-merge
vs multi-tenant-multiplex trade-off is the stated premise of S-LoRA and Punica.
**What is ours: correct scaling derivation for `id_kron` (which peft cannot
load), exact restore, and the |dW|/|W| fidelity relationship above.** The
speed win is a property of merging, not of this implementation.

#### 🔴 Unrelated: the financial_planning adapter is worse than base

Visible in the *wrapped* arm, so not a folding artifact: it drops domain-term
coverage from 12/20 prompts to 7/20 (**60% → 35%**). Its metric is also
degenerate — `bad_hits` is **0 across all 60 generations**, so
`GENERAL_FILLER_TERMS` never fires and "adherence" collapses to a binary
did-it-mention-anything indicator. Retrain the adapter and redesign that eval
before citing any financial number.

---

## ✅ Execution Plan (gated, cheapest-decisive-first)

The original Step 1 below opens with 2–3 weeks of C++/HIP kernel work. That is the
wrong entry point: `peft` 0.20.0 already ships **DoRA** (`LoraConfig(use_dora=True)`)
and **Kronecker** (`LoKrConfig`), so the core quality-vs-size question is answerable
in a handful of PyTorch training runs first. Each step below has an explicit gate —
if the gate fails, the steps after it are not worth doing.

| Step | Work | Cost | Gate to proceed |
| --- | --- | --- | --- |
| **A. Re-baseline (greedy)** | astral base + LoRA `custom_standard` | ~10 min | none — mandatory, every pre-greedy number is incomparable |
| **B. DoRA baseline** | `LoraConfig(use_dora=True)`, no new code | ~10 min | does DoRA actually beat plain LoRA here? (this doc claims ~90% vs ~80%) |
| **C. LoKr baseline** ⭐ | `LoKrConfig`, no new code; **measure real saved adapter size** | ~10 min | **decisive**: does Kronecker hold adherence at ~3 MB, or collapse the way Tucker did? |
| **D. KronA + DoRA** | `LoKrConfig` has no `use_dora` field — add a magnitude vector to LoKr in PyTorch | ~1 day | only if C holds quality but loses some to plain LoRA/DoRA |
| **E. Fused HIP kernel** | `krona_dora_fused.hip` (Step 1 below) | 2–3 weeks | only if C/D prove quality at ~3 MB. Reframed honestly as a **speed/VRAM optimization**, not a capability unlock — C/D already deliver the small adapters |
| **F. Billboard Impostor** | Step 3 below. ✅ **Unblocked** — `swap_adapter_weights()` built and benchmarked (3.7x faster than re-wrap, repeat-safe) | done | ⚠️ **Gate now fails on value, not feasibility**: the latency it removes is worth ~1% of loop wall-clock, and the VRAM-density win needs bank sharing that cross-task retention (0.00%) rules out. Build only if a use case survives those two numbers |

**A + C together cost ~20 minutes of GPU time and test the premise that the entire
2–3 week kernel effort rests on.** If LoKr's adherence collapses at 3 MB, the
micro-adapter thesis dies there for the price of two training runs.

---

## 🗺️ System Overview

The three-component architecture that resolves the multi-tenant LLM serving trilemma:

$$\underbrace{\cancel{\text{Velocity-Masked SFT}}}_{\text{FALSIFIED — see Measured Reality}} + \underbrace{\text{KronA + DoRA}}_{\text{\~{}3 MB High-Rank Payload}} + \underbrace{\text{Billboard Impostor Router}}_{\text{0ms In-Flight Handoff}}$$

```text
                            TRAINING PHASE
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. [STRUCK] Velocity-Masked SFT (Dynamic Gradient Regularizer)             │
│    FALSIFIED: ranking carries no signal (random control matches it), the   │
│    masked set is static (churn 0.013 -> not stochastic depth), and no      │
│    speedup is possible (adapter is 0.437% of params). See Measured Reality.│
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 2. Fused KronA + DoRA HIP Kernel (RDNA3 / gfx1100)                         │
│    - Combines Kronecker high-rank updates (A ⊗ B) + DoRA magnitude (m)     │
│    - Computes B · X · A^T in GPU registers (Zero VRAM allocation spikes)   │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
                        [~3 MB High-Rank Micro-Adapters]
                                      │
                               INFERENCE PHASE
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. Billboard Impostor Engine (Zero-Stall Routing)                          │
│    - 100 KB VRAM Centroid Proxy (0 ms Token 1 Launch)                      │
│    - ~10 ms PCIe Background Stream of ~3 MB Micro-Payload                  │
│    - In-Flight Slot Swap in Captured Execution Graph                       │
└────────────────────────────────────────────────────────────────────────────┘
```

### Why This Combination Is Novel

Parts of these techniques exist in isolation across research literature, but combining them into a unified, zero-stall execution engine on local/consumer hardware has never been done before:

| Component | What Exists in Literature / Industry | What THIS System Achieves | Novel? |
| --- | --- | --- | --- |
| **Adapter Compression** | Standard DoRA (heavy ~30 MB files) OR Tucker (ultra-tiny ~1.5 MB, but **collapsed adherence** due to global basis sharing). | **KronA + DoRA (~3 MB payload)**: Decouples magnitude ($m$) from high-rank Kronecker direction ($A \otimes B$), maintaining **full fine-tuning adherence** while cutting file size by 80%+. | 🟢 **Yes.** Solves Tucker's mathematical failure mode at micro-adapter scales. |
| ~~**Training Regularization**~~ | Standard drop-out or layer-dropping. | ~~**Velocity-Masked SFT**: Bypasses backward passes on quiet layers ($\Delta h_l$)~~ | 🔴 **FALSIFIED.** A random-selection control matches it within 0.9 pp; masked set is static (churn 0.013); no speedup possible. The 83.70% came from a run where the gate was inert. |
| **GPU Execution Kernel** | Eager PyTorch (unrolls Kronecker matrices into VRAM, causing memory spikes). | **Fused C++/HIP Kernel (`gfx1100`)**: Computes $B \cdot X \cdot A^T$ directly in GPU registers/LDS with zero intermediate VRAM allocations. | 🟢 **Yes.** High-rank register-level Kronecker execution on RDNA3. |
| **Adapter Hot-Swapping** | vLLM / SGLang (PCIe transfer stalls or heavy VRAM caching). | **Billboard Impostor Engine**: Generates Token 1 instantly via a 100 KB VRAM proxy (0 ms launch) while streaming the ~3 MB payload in ~10 ms in the background for an in-flight slot swap. | 🟢 **Yes.** Completely masks the physical PCIe transfer window. |

### The Serving Trilemma This Solves

Until now, the entire AI serving industry operated under a strict **trilemma** — forced to choose two and suffer the third:

```text
              THE SERVING TRILEMMA

              [1. High Quality]
               (Full-Rank Rules)
                    /\
                   /  \
                  /    \
                 /  ??  \
                /        \
[2. Tiny Payload] ──────── [3. Zero Hot-Swap Latency]
 (2-4 MB / Edge)            (Instant Token 1 Launch)
```

**Why existing solutions failed:**

1. **Option A: High Quality + Zero Latency (vLLM / SGLang / S-LoRA):** Keep standard ~30 MB adapters permanently resident in GPU VRAM. Failure: requires **gigabytes of VRAM caching**, starving the KV cache on consumer hardware.
2. **Option B: Tiny Payload + Zero Latency:** Compress adapters to ~1.5 MB using global factor sharing. Failure: **catastrophic adherence collapse** — global shared basis destroys layer-specific subspace rotations.
3. **Option C: High Quality + Tiny Payload:** Offload small adapters to host RAM/disk, stream over PCIe on demand. Failure: **execution stalls** — every domain switch pays a PCIe streaming and graph allocation penalty before Token 1 can generate.

**The new reality:**

```text
             THE NEW REALITY (THIS SYSTEM)

               [1. High Fine-Tuning Quality]
                 (DoRA Magnitude + High-Rank KronA)
                             │
                             │  Achieved Simultaneously!
                             ▼
  [2. ~3 MB Micro-Payload] ───── [3. 0ms Initial Launch Delay]
    (Stores 50+ Experts in <150MB)  (Impostor Proxy + In-Flight Swap)
```

---

## 📊 Comparison: This System vs. Bleeding Edge

| Architectural Axis | Bleeding Edge (vLLM + DoRA / LoRA) | Unified Architecture |
| --- | --- | --- |
| **Adapter Artifact Size** | **~20 MB – 60 MB** per adapter | 🟢 **~2 MB – 4 MB** (80%+ reduction via A⊗B) |
| **Training Speed** | Baseline (DoRA adds ~20% norm overhead) | ⚪ **Unproven** — ~~Velocity-Masking~~ cannot recover this (falsified; adapter is 0.437% of compute). DoRA's real overhead on this rig is still unmeasured. |
| **Training VRAM Allocation** | High (PyTorch unrolls temporary matrices) | 🟢 **Zero Allocation Spikes** (Fused HIP kernel in registers) |
| **Initial Token Launch Delay** | **~100 ms – 200 ms** (PCIe streaming stall) | 🟢 **0 ms** (Instant execution via VRAM Centroid Impostor) |
| **PCIe Transfer Latency** | ~100 ms – 150 ms | 🟢 **~10 ms – 15 ms** (Micro-payload streams invisibly) |
| **Multi-Tenant VRAM Density** | ~1 GB – 2.5 GB for 50 domain adapters | 🟢 **<150 MB** for 50 domain adapters |
| **Downstream Adherence / Quality** | High (~83%–90%+) | 🟢 **High** (Matches full fine-tuning via DoRA magnitude m) |

### Why 200 ms Actually Matters (Not Just a Micro-Optimization)

In production agentic workflows, a ~200 ms stall creates a massive **systemic wall**:

- **Compound Stutter in Agent Loops:** Modern AI pipelines run multi-step agent loops (_Router → SQL Generator → Validator → YAML Formatter → Python Executor_). Every domain switch in a standard framework pays the PCIe penalty. Across a 10-step agent execution: **2 to 3 seconds of raw idle GPU time**. This system eliminates it entirely.
- **VRAM Multi-Tenancy Wall:** To avoid PCIe stalls, frameworks like vLLM/SGLang keep 30–60 MB adapters permanently in VRAM. Holding 50–100 adapters consumes gigabytes, directly starving the **KV Cache** and crushing batch throughput. This system stores 50 experts in **<150 MB**.

The final breakthrough in 3 sentences:

> 1. **Compression without adherence collapse:** By using **DoRA magnitude vectors (m)** alongside **Kronecker directional products (A⊗B)**, full fine-tuning quality is maintained on strict domain rules at a **~3 MB file size**.
> 2. ~~**Training overhead eliminated:** Velocity-Masking (Δh_l) skips backward passes on quiet layers — a regularizer that boosts adherence to 83.70% while cutting DoRA's training time penalty.~~ **🔴 FALSIFIED — see Measured Reality.** The claim requires masking adapter layers to save meaningful compute; the adapter is 0.437% of parameters, so it cannot. Measured: masking made training *slower*.
> 3. **Physical latency masked:** Token 1 executes instantly via a **100 KB VRAM Impostor Proxy** while the ~3 MB micro-payload streams over PCIe (~10 ms) and swaps directly into a pre-allocated **fused HIP execution slot (gfx1100)** — making hot-swapping completely invisible.

---

---

## Step 1: Build the `krona_dora_fused.hip` Kernel

**Goal:** A fused C++/HIP kernel for RDNA3 (`gfx1100`) that computes the KronA + DoRA forward and backward passes in GPU registers/LDS without any intermediate VRAM allocation.

### 1.1 — Why `w4a16.hip` Cannot Be Reused

The existing `w4a16.hip` kernel is a **W4A16 GEMV (Matrix-Vector) Decode Kernel**:
- **Input:** Packed 4-bit weights + FP16 activation vector (B=1).
- **Operation:** Dequantize 4-bit nibbles in GPU registers → fused dot product → output vector.
- **Direction:** Forward-pass single-token decode **only**.

KronA + DoRA Training requires three entirely different GPU operations:

1. **The Kronecker Pass (A⊗B):** Computing Y = (A⊗B)X, which reshapes inputs into higher-dimensional tensors.
2. **The DoRA Magnitude Norm (||W||_c):** Calculating L2 norms across matrix columns dynamically during every forward pass.
3. **The Backward Pass (∂L/∂A, ∂L/∂B, ∂L/∂m):** Computing gradients for A, B, and m through autograd hooks.

> `w4a16.hip` solved a **VRAM-read bandwidth bottleneck for 4-bit inference**. KronA + DoRA training needs a **fused matrix-transformation kernel for 16-bit/32-bit gradients**. They are two completely different HIP C++ codebases.

### 1.2 — The Core Mathematical Identity

Instead of PyTorch expanding (A⊗B) into a temporary high-rank matrix in VRAM, the fused HIP kernel exploits the Kronecker product identity:

$$(A \otimes B) \vec{x} \equiv \text{vec}\left( B \cdot \text{mat}(\vec{x}) \cdot A^T \right)$$

This means the full Kronecker expansion **never materializes in VRAM**. Instead, the kernel:

1. **Reads** A ∈ ℝ^(p×q), B ∈ ℝ^(r×s), X ∈ ℝ^(qs) from VRAM into thread-block registers / LDS.
2. **Reshapes** X into a (s × q) matrix in shared memory.
3. **Computes** B · X_mat · A^T in-place across wave fronts — output shape (r × p) — directly in registers.
4. **Writes** the flattened result Y ∈ ℝ^(rp) to VRAM.

### 1.3 — What the Fused Kernel Does (3 Operations, 1 Pass)

1. **Zero Tensor Allocation:** Computes B·X·A^T in small GPU shared memory (LDS) or registers without expanding A⊗B into VRAM.
2. **Fused DoRA Normalization:** Computes column-wise norms and directional updates in the same pass as the GEMM, bypassing PyTorch's intermediate tensor allocations.
3. **Fused Backward Pass:** Evaluates partial derivatives for A, B, and m directly in GPU registers.

### 1.4 — KronA + DoRA Architecture (Why Not DoRA Alone)

NVIDIA's **DoRA** (Weight-Decomposed Low-Rank Adaptation) splits a weight update into scalar magnitude (m) and directional matrix (V), parameterized by standard LoRA:

$$W = m \cdot \frac{W_0 + A \cdot B}{\Vert{}W_0 + A \cdot B\Vert{}}$$

- **What DoRA gives you:** Higher accuracy/adherence. Separating direction from magnitude behaves closer to full fine-tuning.
- **What DoRA does NOT give you:** Smaller file sizes. DoRA still uses standard low-rank A·B for direction. Parameter count is virtually identical to standard LoRA (**~20 MB to 60 MB** per adapter on 8B models).

> DoRA alone gives the _quality_, but leaves the exact same heavy **~25 MB payload** per adapter. Streaming across PCIe in 10 ms is impossible, and storing 50 DoRA adapters still eats gigabytes of VRAM.

**The hybrid construction — KronA + DoRA (m + A⊗B):**

- **KronA alone (A⊗B):** Compresses parameter footprints by 80%+ (~3 MB per adapter), but can suffer from optimization/scaling instability during SFT.
- **DoRA alone (m + A·B):** Excellent accuracy, heavy parameter footprints (~25 MB).
- **KronA + DoRA (m + A⊗B):** Kronecker products for the directional update (A⊗B) + DoRA's magnitude vector (m) for scaling. **Full fine-tuning adherence at a ~3 MB payload size.**

The merge formula for production deployment:

$$W_{\text{final}} = m \cdot \frac{W_0 + (A \otimes B)}{\Vert{}W_0 + (A \otimes B)\Vert{}_F}$$

**Structural comparison:**

| Architectural Dimension | Standard LoRA | Tucker (DO NOT USE) | KronA + DoRA Hybrid |
| --- | --- | --- | --- |
| **Tool Adherence & Quality** | High (~83%) | 🔴 **Collapsed** (Shared Basis Bottleneck) | 🟢 **Matches Full Fine-Tuning** |
| **Expressive Matrix Rank** | Low Rank (rank r) | Restricted Low Rank | 🟢 **High / Full Rank** (A⊗B) |
| **Adapter Checkpoint Size** | ~21 MB | ~1.5 MB | 🟢 **~2 MB – 5 MB** |
| **Layer Independence** | 100% Independent | 🔴 Forced Global Shared Basis | 🟢 **100% Layer-Independent** |
| **Inference Overhead** | 0 ms (Merged) | 0 ms (Merged) | 🟢 **0 ms (Merged)** |

Tucker Factorization is **off the table and will not be mentioned again**. All benchmarks and architectural comparisons are strictly evaluated against: **Standard LoRA (A·B)**, **DoRA (NVIDIA)**, and **vLLM/SGLang multi-adapter serving frameworks**.

### 1.5 — ~~Velocity-Masked SFT as the Training Regularizer~~ 🔴 FALSIFIED

> **This entire subsection is retained only as a record of a disproven hypothesis.**
> Do not build on it. The measurements that killed it are in `## Measured Reality`
> at the top of this document. Summary of why it fails, in the order the failures
> were found:
>
> 1. **The gate was inert when the headline number was produced.** The 59.95% →
>    83.70% result came from runs with `skip_recompute=True` under non-reentrant
>    gradient checkpointing, where the hook's `torch.is_grad_enabled()` recompute
>    test is always true and therefore skips every measurement. 0% of layers were
>    ever masked. That run demonstrated nothing about velocity masking.
> 2. **It is not stochastic depth.** With the hook fixed, quiet-set churn is
>    **0.013** — the identical 8 layers (`[12,13,14,16,17,24,25,29]`) are masked
>    every single step. It is a static capacity cut to 24/32 layers wearing a
>    dynamic-regularizer costume.
> 3. **The ranking carries no signal.** A random-selection control arm (same mask
>    count, redrawn each step, churn 76.2%) scored 60.85% vs velocity's 61.73% —
>    within noise of each other and of the unmasked baseline's 63.82%.
> 4. **The compute-saving claim is arithmetically impossible.** The adapter is
>    0.437% of model parameters. Masking 25% of its layers skips ~0.1% of compute
>    while the frozen base still runs full forward+backward through all 32 layers.
>    Measured: velocity training was *slower* (198.1 s vs 182.6 s) because the
>    measurement hooks cost more than the skipped math saves. It cannot offset
>    DoRA's overhead, which was the entire reason it appeared in this plan.
>
> **What survives and is worth keeping** (all in `src/gnn_experiment/novel_peft.py`):
> the self-calibrating percentile gate (can't silently go inert the way a fixed
> threshold did), the warmup-complete-with-zero-observations warning, and the
> quiet-set churn metric that exposes static-vs-stochastic masking. These are
> useful diagnostics regardless of this technique being dead.

<details>
<summary>Original (disproven) text, kept for the reasoning trail</summary>

During training, **KronA + DoRA** defines _how_ parameters are factorized, while **Velocity-Masking** controls _when and where_ gradients are applied across model depth.

- **Regularization & Adherence Boost:** Skipping backward updates on low-velocity layers acts as a dynamic stochastic-depth regularizer (claimed: adherence boosted from ~59.95% up to 83.70%). That same regularization applies directly to the KronA directional matrices (A⊗B) and DoRA magnitude vectors (m).
- **Mitigating DoRA's Training Overhead:** DoRA adds ~20% wall-clock penalty due to dynamic norm calculations. Velocity-Masking completely bypasses the backward pass on quiet layers, eliminating gradient and norm calculations for those layers — clawing back that training time loss directly.

> **"Velocity-Masked SFT acts as the training regularizer that preserves adherence and cuts backward pass compute, while the fused KronA + DoRA HIP kernel compresses those updates into ~3 MB micro-adapters designed for zero-stall Billboard Impostor hot-swapping."**

</details>

The corrected training stack (velocity removed):

```text
       ┌─────────────────────────────────────────────────────────┐
       │                KRONA + DORA (A ⊗ B + m)                 │
       │   - High-rank Kronecker directional updates (A ⊗ B)     │
       │   - Decoupled DoRA magnitude scaling vector (m)         │
       │   - Step C/D below: PyTorch first, HIP kernel only if   │
       │     the quality-at-3MB result justifies it              │
       └────────────────────────┬────────────────────────────────┘
                                │
                                ▼
                    [~3 MB Micro-Adapters]
```

| Component | Layer in the Stack | Concrete Function |
| --- | --- | --- |
| ~~**Velocity-Masked SFT**~~ | 🔴 **REMOVED** | Falsified — see above. |
| **KronA + DoRA** | **Model Factorization & Execution** | Encodes high-rank, layer-independent updates into ultra-compact **~3 MB micro-adapters**. |
| **Billboard Impostor** | **Inference & Routing Engine** | Serves domain experts with **0 ms initial launch delay** and sub-15 ms PCIe hot-swapping. |

### 1.6 — Engineering vs. PyTorch Wrapper Trade-Off

| Metric | Eager PyTorch (KronA + DoRA) | **Custom Fused HIP Kernel (KronA + DoRA)** |
| --- | --- | --- |
| **Adapter File Size** | �� ~3 MB – 5 MB | 🟢 ~3 MB – 5 MB |
| **Adherence / Quality** | 🟢 Matches Full Fine-Tuning | 🟢 Matches Full Fine-Tuning |
| **Training Speed** | 🔴 +25% to 40% slower than LoRA | 🟢 **Matches standard LoRA speed (~3.5 min)** |
| **Training VRAM Spikes** | 🔴 High (Unrolled intermediate tensors) | �� **Zero extra VRAM (Computes in registers)** |
| **Engineering Effort** | 🟢 1–2 days (PyTorch wrapper) | 🔴 **2–3 weeks of low-level C++/HIP/ROCm dev** |

**Build the Custom Fused HIP Kernel IF:**
1. You plan to train dozens of adapters continuously — shaving ~30% off training time and eliminating VRAM spikes saves real GPU compute hours.
2. You want the systems engineering portfolio asset — a fused C++/HIP/ROCm kernel for RDNA3 handling Kronecker matrix algebra (A⊗B) and DoRA magnitude updates in registers is a top-tier systems accomplishment.

**Stick to a Clean PyTorch Wrapper IF:**
1. You only need 2–3 domain experts for testing — a standard PyTorch/PEFT wrapper gets the job done immediately without weeks of C++/HIP kernel development.

---

## Step 2: Dataset Pipeline & Domain Asset Generation

**Goal:** Generalize the training dataset pipeline to support any domain corpus and produce 2–3 distinct high-adherence domain adapters for testing.

### 2.1 — Dataset Pipeline (Already In Progress)

The dataset pipeline (`load_micro_dataset()` in `src/gnn_experiment/micro_probe/dataset.py`) has been updated:

- Use `tokenizer.apply_chat_template()` with `SFTTrainer(assistant_only_loss=True)` so response-only loss is active — LoRA gradients focus 100% on assistant output correctness.
- Persona sanitizer strips informal greeting/sign-off filler from assistant content before training (`sanitize_assistant_content()`).
- Fingerprint bumped to `astral_micro_docs_v2` to bust stale HF cache.
- Negative/rejection examples added (~15% of dataset) — teaches the adapter when NOT to use tools in the described way, preventing hallucination of invalid patterns.
- Default paths updated: `data/astral/raw` + `data/astral/training_data.jsonl`.

### 2.2 — Domain Asset Generation

Build 2–3 synthetic domain datasets to produce distinct, high-adherence test artifacts:

1. **SQL domain** — Strict SQL AST generation, schema validation, rejection of bad queries.
2. **YAML domain** — Structured YAML formatting with schema enforcement.
3. *(Optional)* **Rust/Python AST domain** — Code generation with syntax rules.

Each domain dataset should follow the same SFT format:
- Positive expert Q&A pairs (happy-path tool use)
- Negative/rejection pairs (~15% of total): "Should I do X?" → "No. Do Y instead because..."
- Consistent direct technical tone (no conversational filler)

---

## Step 3: Billboard Impostor Engine

**Goal:** Build the zero-stall multi-adapter routing system that serves domain experts with 0 ms first-token delay and sub-15 ms hot-swap latency.

> **Prerequisite:** Step 1 must be complete (need ~3 MB adapters). The Impostor system only makes physical sense with ultra-compact payloads. Standard 25 MB LoRA adapters would still stall.

### 3.1 — How On-the-Fly Switching Works

```text
Standard SOTA Engine (vLLM / SGLang):
[Request New Adapter] ──► [100-200ms PCIe Transfer Stall] ──► [GPU VRAM Allocation Spike] ──► [Token 1 Delayed]

Unified Architecture:
[Request New Adapter] ──► [Token 1 Generates Instantly via Impostor Proxy (0ms)]
                               │ (~10ms Background PCIe Stream of ~3MB Payload)
                               ▼
                          [In-Flight Slot Swap via Fused HIP Kernel] ──► [Token 2+ Full Quality]
```

### 3.2 — The Three Phases

**Phase 1 — Zero-Delay First Token (The Impostor Proxy)**

When an agent or user switches domains (e.g., jumping mid-chat from general conversation to a strict **SQL AST** or **Rust Clippy** task):
- The engine doesn't pause to wait for adapter weights to load.
- Token 1 generates immediately using the resident **100 KB Centroid Impostor Proxy**, incurring **0 ms of cold-start launch delay**.

**Phase 2 — Sub-15 ms In-Flight PCIe Background Streaming**

KronA + DoRA uses Kronecker factor expansion (A⊗B) with DoRA magnitude vectors (m) while preserving layer independence. The complete payload drops from **~25–30 MB down to ~2–4 MB**:
- Transferring a ~3 MB payload across the PCIe bus takes only **~10 ms to 15 ms** (vs. ~100–150 ms for standard LoRA).
- The true high-rank domain adapter streams into GPU VRAM in the background while the first token is being generated by the proxy.

**Phase 3 — Register-Level In-Flight Slot Swapping (No GPU Stalls)**

Once the ~3 MB payload lands in VRAM, the custom fused C++/HIP kernel executes an in-flight swap directly into a pre-allocated static slot inside the captured execution graph:
- **Zero Tensor Re-allocations:** The kernel handles Y = (A⊗B)X using register/LDS reshapes (B·X·A^T), avoiding PyTorch memory allocation spikes or VRAM fragmentation.
- **Seamless Transition:** The active model cleanly switches from Impostor Proxy to the true, full-rank domain adapter by Token 2 or 3 without dropping a single clock cycle.

### 3.3 — Streaming Latency vs. PCIe Bottleneck

The Impostor Routing algorithm relies on an initial low-latency phase (using lightweight centroid vectors) while streaming full A·B weights into the active slot in the background:

- **Standard LoRA (~25 MB):** Streaming 25 MB per adapter over PCIe consumes ~100–150 ms. If routing swaps adapters frequently across agent turns, this creates visible generation stutters.
- **KronA + DoRA Micro-Adapters (~2 MB – 4 MB):** Kronecker factorization drops the parameter payload by 80%+ while preserving high-rank expressiveness. Streaming a ~3 MB adapter takes **~10–15 ms**. The "background streaming" window becomes practically invisible.

### 3.4 — In-Memory Density for Fixed VRAM Slots

If pre-allocating a fixed-slot tensor engine in GPU VRAM (reserving VRAM for dynamic adapter slots):

- **Standard LoRA:** Storing 20 domain adapters resident in GPU VRAM: **~500 MB to 600 MB** overhead.
- **KronA + DoRA:** Storing 20 domain adapters: **under 60 MB total**. An entire library of domain experts stays permanently resident without fragmenting memory or starving the KV cache.

### 3.5 — Hot-Swap Metrics

| Metric / Experience | Standard Multi-LoRA Serving (vLLM / SGLang) | **Unified System** |
| --- | --- | --- |
| **First Token Launch (Un-cached)** | 🔴 **100 ms – 200 ms stall** (PCIe bottleneck) | �� **0 ms** (Instant launch via VRAM Centroid Impostor) |
| **Adapter Transfer Payload** | 🔴 **~20 MB – 60 MB** per adapter | 🟢 **~2 MB – 4 MB** (80%+ reduction via A⊗B) |
| **Adapter Hot-Swap Latency** | 🔴 ~100 ms – 150 ms transfer window | 🟢 **~10 ms – 15 ms** (Transfers invisibly mid-token) |
| **GPU Execution Allocation** | 🟡 High (Unrolls matrices in VRAM during forward pass) | 🟢 **Zero Allocation Spikes** (Fused register computation) |
| **VRAM Density (50 Adapters)** | 🔴 ~1 GB – 2.5 GB reserved VRAM | 🟢 **<150 MB** reserved VRAM |

---

## Step 4: Integration & Benchmarking

**Goal:** Wire all three components together, benchmark against standard LoRA and DoRA baselines (vLLM / SGLang), and validate the complete pipeline.

### 4.1 — Complete System Pipeline

```text
[Dataset / Prompts]
       │
       ▼ (Fast ~3.5 min SFT)
[KronA + DoRA HIP Kernel] ──► Produces ~3 MB Micro-Adapters
       │
       ▼
[Billboard Impostor Router] ──► Instant Centroid Proxy Execution
       │                       └──► ~10ms PCIe Background Stream
       ▼
[Fixed VRAM Tensor Slots] ──► Zero-Latency Domain-Expert Execution
```

### 4.2 — Benchmark Axes (vs. Standard LoRA and DoRA Baselines)

| Architectural Axis | Baseline (Standard LoRA) | Baseline (DoRA) | Target (This System) |
| --- | --- | --- | --- |
| **Adapter File Size** | ~21 MB – 30 MB | ~21 MB – 30 MB | **~2 MB – 4 MB** |
| **Adherence / Quality** | Moderate (~80%) | 🟢 High (~90%+) | 🟢 **High (~90%+)** |
| **Training Speed** | Baseline | Baseline –20% (unverified on this rig) | ⚪ **Unproven** — ~~Velocity-Masking recovers DoRA overhead~~ falsified; no mechanism currently identified to offset DoRA's cost |
| **Training VRAM Spikes** | Moderate | Moderate | 🟢 **Zero** (Fused HIP kernel in registers) |
| **Initial Token Launch (Un-cached)** | ~100–200 ms stall | ~100–200 ms stall | 🟢 **0 ms** (VRAM Centroid Impostor) |
| **PCIe Transfer Latency** | ~100–150 ms | ~100–150 ms | 🟢 **~10–15 ms** (Micro-payload) |
| **Multi-Tenant VRAM (50 Adapters)** | ~1–2.5 GB | ~1–2.5 GB | 🟢 **<150 MB** |

### 4.3 — What Was Previously Impossible

This system achieves simultaneously:

> **"You can run a fleet of 50+ specialized, high-adherence LLM domain experts on a single consumer GPU without paying for enterprise VRAM caching, without risking accuracy collapse, and without experiencing a single millisecond of hot-swapping latency."**

That is a complete, end-to-end systems engineering innovation spanning training regularizers, custom low-level C++/HIP kernels, and dynamic inference routing. You aren't just tweaking an existing tool; you are assembling a novel, end-to-end systems paradigm for local multi-tenant LLM execution.
