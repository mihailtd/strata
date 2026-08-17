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

**Replaced by:** direct cost lookup. (It was briefly replaced by SLA-bounded
cluster scheduling, which was itself retired in §6 — the engine is single-tenant.)
See
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

## 6. RESOLVED — SLA-bounded cluster scheduler removed (engine is single-tenant)

**Decided 2026-08-17. Execution is single-threaded**: one agent walking a
deterministic tool DAG, one node active at a time. There are no concurrent
callers competing for VRAM, and no minority-domain starvation, because only one
domain is active at any node.

**Cause: not a defect — an absent problem.** The scheduler worked, and it beat
both baselines on deadline misses:

| offered load ρ | FIFO | Greedy | Router |
| ---: | ---: | ---: | ---: |
| 0.8 | 43.5% | 38.5% | **33.0%** |
| 0.95 | 62.5% | 51.5% | **47.0%** |
| 1.5 | 98.0% | 96.0% | **79.5%** |

But those rows only exist under concurrent multi-tenant load. With one caller
there is nothing to reorder, so the scheduler is inert by construction. The
end-to-end A/B had already said as much: swap overhead is 0.86% of wall clock,
mean latency moved −62 ms with 95% CI [−1481, +1417] (not significant), and
batch-order commitment cost **+3.6 s of P95**.

**What was removed:**

- `server.py` — the batching window, queue drain, reordering, `router_config`
  (`enabled` / `batch_window_ms` / `sla_deadline_s`), the `transitions_avoided`
  counterfactual, `total_batches`, `_count_transitions`, and `POST
  /v1/router/config`. Replaced by a strict arrival-order executor.
- `router/vram_state_router.py` — `VRAMStateScheduler`, `PendingRequest`.
  Retired to [`benchmarks/superseded/sla_cluster_scheduler/`](../benchmarks/superseded/sla_cluster_scheduler/)
  with the numbers above and the revival condition.
- Its two benchmarks moved with it.

**What was deliberately kept:**

- `VRAMState` — the server still tracks which expert is resident.
- `TransitionCosts` / `VRAMStateGraph` and their tests — this is the measured
  physics that retired the APSP router (§1), and
  `test_direct_edge_is_always_optimal` must keep failing loudly if the cost model
  ever becomes source-dependent.
- `calibrate_transition_costs.py`, promoted to
  [`benchmarks/runtime/cost_model/`](../benchmarks/runtime/cost_model/) — it
  produces that model and is still the pre-flight for any future routing idea.
- `GET /v1/router/status`, now reporting `mode: sequential-single-tenant`.

**Revive only if the engine becomes multi-tenant** — concurrent callers sharing
one GPU with different target experts. Then the table above is what comes back.

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

## 8. RESOLVED — keep a single `max_steps`; no early stopping, no per-domain budget

**Measured 2026-08-17.** Per-step loss curves captured for all three domains
(`--logging-steps 1`, 150 steps each, `results/loss_curves/*.json`, analysed by
`scripts/analyze_loss_curves.py`). This settles two proposals — **both are
rejected by the data, including the one this file previously leaned toward.**

### The premise was that domains converge at different rates. They do not.

| domain | records | epochs seen | **converges at step** |
| :--- | ---: | ---: | ---: |
| astral | 815 | 0.74 | **107** |
| financial_planning | 304 | 1.97 | **109** |
| postgresql | 411 | 1.46 | **135** |

("converges" = first step whose EMA is within 0.02 of the final EMA.)

**Convergence step does not track dataset size or epochs.** astral has 2.7× the
data of financial and sees 0.38× the epochs, yet both converge at essentially the
same step (107 vs 109). postgresql sits between them on both inputs and converges
*latest*. There is no monotonic relationship, so the epoch asymmetry — real as it
is — does **not** translate into a convergence-rate asymmetry.

That removes the motivation for per-domain step budgets. A single budget is
defensible, and epoch-normalising `max_steps` would be solving a problem the
curves say does not exist.

### EMA early stopping would under-train every domain, not just astral

Where EMA + min-delta + patience would actually fire:

| domain | needs | d=0.005,p=15 | d=0.005,p=25 | d=0.01,p=15 | d=0.02,p=15 | d=0.01,p=30 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| astral | 107 | **55** | 65 | 55 | 54 | 70 |
| financial_planning | 109 | **45** | never | 45 | 42 | never |
| postgresql | 135 | **44** | never | 44 | 44 | never |

The proposed settings (ε=0.005, patience=15) stop at **~30–37% of the steps
actually needed**, on all three. The curves have noisy plateaus around step
42–55 that EMA+patience reads as convergence while ~0.2–0.3 of loss is still to
come.

**The parameter sensitivity is the second reason not to ship it.** For postgresql
and financial, `patience=15` fires at step 44–45 while `patience=25` never fires
at all — the outcome flips between "cut 70% of training" and "change nothing" on
a constant nobody had data for. Combined with a ±10pp eval that could not detect
the resulting quality regression, this was a coin flip that would have silently
degraded every adapter.

### No best-checkpoint selection either

All three end slightly above their minimum EMA (astral +0.006, postgresql +0.022,
financial +0.036), which would suggest saving the best checkpoint rather than the
last. But each gap is **within 3× the smoothed step-to-step noise floor**
(0.015–0.018), so it is not distinguishable from noise. Not worth the complexity.

### What is actually available

A single fixed cut, and it is modest:

- **`max_steps` 150 → 135** keeps all three domains within 0.02 of their final
  loss, for a **~10% saving**. Safe.
- **150 → 110** would save 27% and keeps astral and financial within 0.05, but
  costs postgresql, which needs 135.

`max_steps=150` is roughly right and, for postgresql, arguably short — its tail
slope is still −0.0054/step at step 150. **Training time is not the bottleneck
worth optimising here**; the honest conclusion is to leave the budget alone or
take the 10%.

**Do not implement EMA early stopping. Do not implement per-domain epoch
budgeting.** Both were reasonable hypotheses; the curves rejected both.
