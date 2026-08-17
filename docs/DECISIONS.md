# Architecture decisions — what was removed, and the measurement that removed it

Companion to `NOVELTY.md` (tiering), `CURRENT.md` (live state), `GLOSSARY.md`
(naming). This file records **retirements** and, more importantly, **why** — so
nobody rebuilds a thing that was already measured dead, and so nobody attributes
a death to the wrong cause and over-generalises from it.

Last updated: 2026-08-17.

**Rule for this file: one row, one cause, one measurement.** If a component died
for a reason that was never measured, say "predicted, not measured".

---

## 1. RETIRED — APSP / Floyd-Warshall VRAM state routing

**Replaced by:** direct cost lookup + SLA-bounded cluster scheduling
(`router/vram_state_router.py`). See
[`benchmarks/superseded/apsp_floyd_warshall/`](../benchmarks/superseded/apsp_floyd_warshall/).

**Cause (MEASURED):** transition cost is **destination-only**, `C(u,v) = f(v)`.
Spread across source states 0.23–0.31 ms against 0.66–2.12 ms of noise. Any
detour costs `f(k)+f(v) > f(v)`, so the direct edge is always optimal — 0 pairs
improved at every expert count from 1 to 10.

**Do not rebuild** any graph-pathfinding scheduler unless the cost model becomes
source-dependent. `test_direct_edge_is_always_optimal` fails loudly if it does.

---

## 2. RETIRED — PiSSA and the `W0`-SVD initialisation family

**Applies to:** PiSSA (**measured**), OLoRA / CorDA (**predicted, not measured**).

**Cause (MEASURED, for PiSSA):** the premise fails. The trained update's energy
in `W0`'s top-8 subspace is **1.37× the random-chance floor** (128 modules, 32
layers, exact SVD). The spectral profile is flat throughout — top-8 1.38×,
top-512 1.06×. And `W0` itself has no dominant directions: `s[0]/s[7] = 2.05`.

So "initialise into the *critical* directions" has no critical directions to
initialise into. On this model PiSSA degenerates toward an arbitrary orthogonal
init at scale.

**Extension to OLoRA/CorDA is inference, not measurement.** They are also
`W0`-derived, so the same flat spectrum should apply — but nobody has run them.
Label accordingly.

**Caveat that must travel with this:** the probe falsifies PiSSA's *stated
rationale*, not necessarily the method. PiSSA may still help via optimisation
conditioning (large well-scaled `A,B` at init), which was not tested. If that is
ever chased, `init_lora_weights="orthogonal"` tests it more cheaply and without
the residual-subtraction risk.

**Replaced by:** peft's default init. No change required.
See [`PISSA_ASSESSMENT.md`](../benchmarks/factory/geometry/preflight_svd_probe/PISSA_ASSESSMENT.md).

---

## 3. RETIRED — shared-basis adapter schemes

**Applies to:** `master_basis` / `MasterBasisBank` (VeRA-family).

**Cause (MEASURED):** cross-task subspace overlap is **1.10–1.28× chance** —
statistically orthogonal. There is no shared low-dimensional adaptation manifold
across tasks, so a basis shared between tasks has nothing to share. Scored
**13.40%** against a 11.94% base.

**⚠️ Do not lump Tucker and LoKr into this grave.** They died separately:

| scheme | score | actual cause |
| :--- | ---: | :--- |
| `master_basis` | 13.40% | **shared basis** — killed by cross-task orthogonality |
| Tucker / KroTucker | 15.66% / 19.70% | **structural capacity** — a cross-layer shared core cannot represent per-layer-independent updates |
| LoKr | 14.42% (r=32) | **structural capacity** — Kronecker factorisation at 1.04M params. **LoKr never uses `W0`'s SVD at all**, so the flat-spectrum result is irrelevant to it |

Tucker and LoKr are *factorisation structures*, not *initialisations*. Attributing
them to the flat spectrum attributes the wrong cause and would license
over-generalising the flat-spectrum finding to methods it says nothing about.

---

## 4. RETIRED (from serving) — adapter stacking / `activate_many()`

**Decision:** load exactly one expert at a time in the serving path.

**Cause (MEASURED, one pair):** `ast+fin` costs astral **−10.96pp**, 95% CI
[−21.50, −1.62], n=40 — a resolved loss. The saving it would buy is 10.42 ms
(25.86 ms co-resident vs 2 × 18.14 ms), i.e. **0.55% of a ~1.9 s request**,
against a measured swap-overhead budget of 0.86% of wall clock.

**Do not over-generalise.** That is **one** 2-way pair on **one** eval domain.
The 3-way stack measured **+5.74pp** (CI [−6.63, +18.26], not resolvable), and it
is *not* monotonic — adding a third expert moved astral back above solo. There is
no supported "stack at most N" rule.

**Keep the capability.** `activate_many()` is tiered 🔥 in `NOVELTY.md` and costs
nothing to retain. This is a serving-policy decision, not a code removal.

---

## 5. RETIRED — training the MTP draft head

**Decision:** the shipped `mtp.*` head stays frozen. Train only the `r=8` LoRA
adapters on `q/k/v/o/gate/up/down_proj`.

**Cause (MEASURED):** all five adapters *lower* draft acceptance (τ 2.456 →
1.531–1.988), every CI excluding zero, n=160/condition. Distillation made it
worse (τ 3.50 → 0.95, Δ −2.55, CI [−2.84, −2.21]) and dropped generation to
21.0 tok/s, below plain greedy. A draft head must *agree with its backbone*, not
be domain-fluent.

**Already true in code — no change required.** `train_expert_CURRENT_m2.py`
targets only the seven projections and never references `mtp.*`. The action is
simply not to repeat the `mtp_head_folding` experiment; it is already in
`superseded/`.

---

## 6. OPEN — deterministic tool-graph routing vs the SLA scheduler

**Proposed:** replace request scheduling with a deterministic state machine that
loads the expert dictated by the current tool-graph step.

**This is ❓, not ⭐ — it is not built yet.** ⭐ applies once it exists.

**What it would give up, measured:** the SLA-bounded cluster scheduler reduced
SLA violations against both baselines across the load sweep —

| offered load ρ | FIFO | Greedy | **Router** |
| ---: | ---: | ---: | ---: |
| 0.8 | 43.5% | 38.5% | **33.0%** |
| 0.95 | 62.5% | 51.5% | **47.0%** |
| 1.5 | 98.0% | 96.0% | **79.5%** |

That protection exists because ordering is free under destination-only costs, so
all ordering freedom is spent on deadlines.

**The decision hinges on one question:** is execution genuinely single-threaded?

- **Yes — a deterministic tool graph, one request at a time.** Then there is no
  queue, nothing to schedule, and the scheduler is dead weight. Remove it.
- **No — concurrent users share the engine.** Then a deterministic state machine
  removes the only mechanism protecting minority domains under load, and the
  ρ≥0.95 rows above are what gets given up.

Decide this explicitly before removing the scheduler. Both answers are
defensible; silently picking one is not.

---

## 7. NOT A GRAVE — the speculative "correctness gate" is kernel numerics

**Do not open a bug hunt on this.** The exact-vs-chunked column (77.5–95%) is
frequently misread as "the engine emits wrong output 1 in 4 times."

**Measured control:** speculative-vs-greedy exactness is **80.0%** against a
**non-speculative control of 83.3%**, determinism 100%, n=60. A decoder doing
**zero speculation** diverges at a comparable rate, because chunked and
single-token kernels disagree numerically on this stack (`NOVELTY.md` 🚀).
100% exactness is unreachable here; it is a hard constraint, not a defect, and
`CURRENT.md` already lists the "loop diverges from its verifier" claim as
retracted.

**Different tokens ≠ wrong output.** Greedy decode through two numerically
different kernels can yield different but equally valid text. No hallucination
has been demonstrated.

**The genuinely open question is quality, not exactness:** does speculative
output *score worse* on the domain evals than non-speculative output? That has
never been measured. If speculation correctness is to be investigated, that is
the experiment — not a verifier bug hunt.

---

## 8. OPEN — fixed `max_steps` is wrong, but not only in the direction assumed

**Measured:** at effective batch 4 (`per_device=2 × grad_accum=2`) × 150 steps =
600 samples seen:

| domain | records | epochs seen |
| :--- | ---: | ---: |
| astral | 815 | **0.74** |
| postgresql | 411 | 1.46 |
| financial_planning | 304 | **1.97** |

**Astral never sees a quarter of its training data; financial sees its data
twice.** A shared step budget across domains is incoherent.

**But early stopping only cuts steps** — it cannot give astral the epoch it is
missing. It addresses at most half of this.

**Three unresolved objections before implementing EMA early stopping:**

1. `lr_scheduler_type="cosine"` makes the LR at step *t* a function of
   `max_steps`. A plateau at step 100/150 is partly the schedule decaying, and
   halting there skips the low-LR anneal. If fewer steps are wanted, **set**
   `max_steps` lower so the cosine compresses — do not interrupt a 150-step
   schedule.
2. `logging_steps=10` → 15 loss points per run. A patience of 15 *steps* is 1.5
   logged points. Needs `logging_steps=1` first.
3. No per-step curve has ever been recorded (`mlruns.db` holds 41 `final_loss`
   rows and nothing else), so ε=0.005 and patience=15 are guesses about an
   unobserved curve.

**And the safety issue:** the eval resolves ±10pp, so a real 4pp regression from
under-training would be **invisible**.

**Cheapest next step:** set `logging_steps=1`, train the three domains, look at
the curves (~10 GPU-min). That tests the "different domains converge at different
rates" hypothesis directly and yields ε/patience empirically. A deterministic
alternative may then be better than early stopping: set
`max_steps = ceil(target_epochs × records / effective_batch)` per domain, which
fixes the asymmetry in **both** directions with no callback and no
schedule conflict — at the cost of changing what "matched methodology" means
(comparable in *data exposure* rather than in *compute*), which would be an m3.
