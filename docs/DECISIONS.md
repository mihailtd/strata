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

**Applies to:** PiSSA (**trained and measured**), OLoRA (**trained and measured**),
CorDA (**predicted, not measured**).

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

### TRAINED AND MEASURED (2026-08-17) — the verdict holds, but for none of the above reasons

The caveat this section carried was the right one: *"the probe falsifies PiSSA's
stated rationale, not necessarily the method. PiSSA may still help via optimisation
conditioning (large well-scaled `A,B` at init), **which was not tested**."* It has
now been tested. 10 matched runs, `scripts/compare_init_schemes.py`, same data,
same r=8, same lr, same 150 steps, `logging_steps=1`, only `init_lora_weights`
and `alpha` varying. **OLoRA is no longer "predicted" — it was run.**

**Gate that licenses every number below:** PiSSA at init is mathematically the
base model (`W_res + scaling·B₀A₀ = W0`), as is stock LoRA (`dW = 0`). Step-1
losses agreed across all three schemes to **≤ 0.011**, so the arms are matched.

**The headline: at matched alpha, both schemes are inert.** astral, α=8, only the
init differs (noise floor 0.0171):

| init | final EMA loss | vs stock | |
| :--- | ---: | ---: | :--- |
| stock | 1.104 | — | |
| **pissa** | 1.095 | **−0.009** | 0.5× noise — indistinguishable |
| **olora** | 1.123 | **+0.019** | 1.1× noise — barely resolvable |

**Neither buys an iteration win.** Across all three domains at α=128, neither
scheme ever reaches the stock run's final loss within 150 steps. Best config in
the whole sweep remains **stock LoRA r=8 α=128 at 1.072**.

**⚠️ The large apparent deficits were a SCALING ARTEFACT, not the subspace.** Run
at the repo's α=128 (scaling 16) these schemes look catastrophic; at their
paper-native α=r they are inert. PiSSA's deficit is monotonic in alpha —
α=8 +0.024, α=32 +0.054, α=128 +0.127 — and **96.6% of OLoRA's apparent +0.557
disaster evaporates at matched alpha (+0.019)**. Any future report of these
schemes MUST hold alpha fixed; the α=128 numbers measure scaling, not
initialisation.

This also **retires a tempting wrong explanation**. It is natural to say PiSSA and
OLoRA both initialise on `W0`-derived directions, the task update sits at 1.37×
chance inside that subspace, so they waste early training unlearning it — a story
that agrees with the section above and fits the α=128 numbers perfectly. The
matched-alpha control falsifies it: hold scaling fixed and the init scheme does
nothing at all. **The geometric argument does not explain these curves.** It may
still be true of the geometry; it is not what produced the loss differences.

**⚠️ And PiSSA is NOT free downstream, contrary to `PISSA_ASSESSMENT.md` §2.**
PiSSA trains against a mutated base (`W_res = W0 − scaling·B₀A₀`), while this
engine folds onto a **pristine `W0`** buffer — so a naively-saved PiSSA adapter
would add the principal component twice. peft's conversion fixes it by emitting
`dW = scaling·(B_trained A_trained − B₀A₀)`, and refactorising a *difference of
two rank-r products* gives **rank 2r**. Measured on disk:

| | rank | alpha | size |
| :--- | ---: | ---: | ---: |
| production `m2_financial_r8a128` | 8 | 128 | **41 MB** |
| PiSSA r=8, serving-compatible | **16** | **256** | **82 MB** |

So adopting PiSSA means doubling adapter size and fold FLOPs — or abandoning the
pristine-`W0` design the whole swap architecture rests on — to buy a **−0.009**
loss change. `scripts/train_expert_CURRENT_m2.py --init-lora-weights` performs
this conversion automatically and warns; do not save a PiSSA adapter without it.

**Replaced by:** peft's default init. No change required — now on evidence rather
than inference. Evidence: `results/init_scheme_comparison.json`,
`results/loss_curves/*_{pissa,olora}*.json`.
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

## 7. RESOLVED / NOT A GRAVE — the speculative "correctness gate" is kernel numerics, and it costs no quality

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

### RESOLVED (2026-08-17) — the quality question was measured, and the answer is no

The paragraph that used to sit here called the quality question "genuinely open."
It has now been run: [`benchmarks/runtime/speculative/speculation_quality/`](../benchmarks/runtime/speculative/speculation_quality/).
Three arms on the canonical eval sets with the repo's own scorer, n=100 paired
triples, 256 new tokens:

| arm | token choice via | drafts accepted | mean score |
| :--- | :--- | :--- | ---: |
| A autoregressive | single-token recurrent kernel | — | 61.08% |
| C forced-reject | chunked verifier | 0 (forced) | 62.10% |
| B speculative K=4 | chunked verifier | real (τ 1.67–2.33) | 61.27% |

| contrast | isolates | Δ | 95% CI |
| :--- | :--- | ---: | :--- |
| A → C | chunked kernel alone | +1.02pp | [−0.98, +3.69] |
| C → B | accepting drafts | −0.83pp | [−3.50, +1.17] |
| **A → B** | **what a user receives** | **+0.19pp** | **[−0.77, +1.17]** |

Nothing is significant, on either the full answers or with every arm truncated to
the shortest arm's token count. **The speedup is not paid for in quality.**

**Why the third arm was necessary.** A two-arm design cannot separate "speculation
is lossy" from "the chunked kernel rounds differently." Arm C runs the identical
speculative machinery with every draft rejected, so it is the chunked kernel with
zero speculation. It diverges from A on **28%** of prompts while B diverges on
**33%** — i.e. the chunked kernel accounts for essentially all of the divergence
and accepting drafts adds ~5pp. This is the direct confirmation of the claim §7
made on a 3-prompt control.

**Divergence is real and score-invisible.** 33/100 prompts produced different text
under B than under A; **30 of those 33 scored identically**. Median first
divergence is token 28 (B) / 33 (C).

**Power.** Adapter stacking was retired at −10.96pp, CI [−21.50, −1.62] (§4). The
CI here is roughly ±1pp on the same metric and same eval sets, so this design
would have detected a stacking-scale regression with room to spare. The worst case
consistent with the data is ~1pp of harm against a 2.20× throughput gain.

**Scope, stated honestly.** This shows the divergent text is not worse *on the
repo's own scorer* — rubric coverage and the good/bad term ratio. That is the
metric that killed stacking, so the comparison is apples-to-apples, but it is
coarse and no human judgement was collected. It also does not license the
speculative decoder for production: `server.py` has no draft path today, so the
2.20× remains a benchmark result.

**⚠️ One retracted run, kept as provenance:**
`results/speculative_quality_RETRACTED_eos_overrun.json` reported "+5.71pp pooled,
speculation IMPROVES quality, SIGNIFICANT." That was a harness bug, not a finding.
A speculative step commits up to `n_acc+1` tokens at once, so checking EOS on
`toks[-1]` at the top of the loop let generation run past an EOS that landed
mid-block: arm B averaged 119.2 tokens against arm A's 46.1 on astral. Arms A and
C commit one token per iteration and were unaffected, which is what exposed it.
The good/bad term ratio then rewarded the extra length — and because it scores
`0.0` when an answer contains no domain term at all, a rambling arm turned 0.0
into 92.0 on one prompt without being more correct. **The tell was internal:
financial_planning is the only domain scored by rubric coverage rather than the
ratio, and it was the only domain reporting exactly 0.00pp.** Lesson for any
future arm-vs-arm quality benchmark here: **report mean token count per arm, and
score length-matched as well as full** — both are now in the harness.

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

---

## 9. RESOLVED — the financial expert was never inert; its eval was blind

**Symptom:** the financial expert scored **83.33%** on its own domain and the base
model scored **83.33%** — an adapter effect of exactly **+0.00pp**, against
astral's +44.44pp and postgresql's +24.08pp. One of three production experts
appeared dead, and a third of the evaluation grid with it.

**Cause 1 — the instrument.** 68.2% of the eval's rubric terms appeared verbatim
in their own question, and 9 of 20 questions gave away *every* term they asked
for. `fin_01` asks about "money scripts... money avoidance and money status" and
scores `['money script','money avoidance','money status']`. The rubric measured
whether the model echoes the prompt. **Base sat at 100% on 14 of 20 items.**

Run per-question, the adapter **changed 20/20 answers** and the rubric scored
**0** of those changes. Alive adapter, blind instrument.

The relationship holds across all three domains and is now checkable statically
by `scripts/audit_eval_rubrics.py`, with no GPU:

| domain | giveaway | adapter effect |
| :--- | ---: | ---: |
| astral | 5.0% | +44.44pp |
| postgresql | 55.0% ⚠️ | +24.08pp |
| financial_planning | 68.2% ⚠️ | +0.00pp |

⚠️ **postgresql's +24.08pp is therefore an understatement**, and any future null
result must be checked against this audit before being reported as an adapter
failure.

**Cause 2 — the corpus taught recitation.** With a 0.0%-giveaway eval the adapter
measured +10.00pp, CI [−2.50, +22.50] — visible but underpowered. The reason was
in the training data: **235 "How…", 64 "What…", 5 scenarios, and 0 questions
asking the model to classify a described client.** Asked definitionally it named
"money scripts" correctly; shown a vignette it invented "Financial Identity
Framework", "Financial Enmeshment", "autobiographical memory".

**Fix:** 78 applied classification records (taxonomy, flashpoint→script→behaviour,
bias, risk tolerance vs capacity) → **+35.00pp, CI [+15.00, +55.00], excludes
zero**. All four confabulations corrected.

**Two guard-caught mistakes worth keeping.** (a) The generator twice produced
prefix diversity below the ~97% house standard, including two framings with 40
characters of fixed template before the variable text — the same failure that
killed the 940-record corpus. Fixed the generator, not the threshold. (b) Making
money scripts 64% of the applied set caused the adapter to over-apply that
framework, answering a mental-accounting vignette with "money vigilance" (**50 →
0, worse than base**). **Class balance of applied examples is a hyperparameter.**

**Not promoted.** `m2_financial_r8a128`, `evaluation_data.jsonl` and
`training_data.jsonl` are untouched; the new work sits alongside as `*_v2`.
Swapping the canonical eval would break comparability with every historical
financial number in the repo — a human decision, not a side effect.

**Residual:** `v2_18` still over-applies the money-script frame (−50), and the two
control categories that received no new data slipped from +25.0 to +12.5.

Evidence: `benchmarks/factory/eval_instrument/`, `results/eval_rubric_audit.json`,
`results/diagnose_financial_planning{,_v2,_v2corpus}.json`.

---

## 10. RETIRED (before being built) — SonicSampler-style fused speculative verification

**Roadmap claim:** speculative decoding "wastes critical time bouncing draft
tokens between CPU and GPU"; fusing verification was scoped at 1–2 weeks of
ROCm/Triton work. **Never measured on this stack. It is false here.**

Profiled at K=4, 89 speculative steps, τ=2.02
([`benchmarks/runtime/speculative/loop_profile/`](../benchmarks/runtime/speculative/loop_profile/)):

| phase | % of loop | ms/call |
| :--- | ---: | ---: |
| `verify_fwd` | 43.9% | 36.36 |
| **`commit_fwd`** | **29.4%** | 34.96 |
| `draft_gen` | 14.4% | 11.88 |
| `accept_calc` | **1.0%** | 0.81 |
| `argmax` | **0.6%** | 0.46 |

**The fusion target is 1.5% of the loop — ceiling 1.016×**, and ~1.014%
end-to-end once the 87% decode share of a 3,000/150 agent turn is applied.

**The negative result that settles it.** The CPU round trip was removed rather
than argued about: the `.item()` accept loop (484 syncs) was replaced by a single
GPU reduction (89 syncs). It came out **1.65% SLOWER** (6.16 s → 6.26 s). The
`.item()` loop short-circuits on first mismatch, averaging 5.4 syncs/step rather
than a K=4 worst case of 8, and `cumprod`+`sum` launch more kernel overhead than
the syncs they save. **There is no CPU-bouncing problem to fuse.**

**What the profile found instead: `commit_fwd` at 29.4%, on nobody's list.** On a
partial accept the chunked forward has already advanced the recurrent state past
the rejected tokens, so the loop restores the snapshot and re-runs a full forward
over the committed prefix — on **62 of 89 steps (70%)** at τ=2.02, costing as much
as the verification it repairs. A speculative step usually costs **two full model
forwards**. Eliminating it caps at **1.42× on the loop**, ~25× the ceiling of the
project it displaces. It requires the GatedDeltaNet chunked scan to emit
intermediate per-position states — a real kernel project, and a different one.
**Lead, not result:** feasibility on gfx1100 without `fla` is unknown.

---

## 11. CORRECTED — "prefill is 5.9%, off the critical path" holds only for 1024-token generations

`benchmark_prefill_share.py` measured prefill at 1.5% @2k / 3.0% @4k / **5.9%
@8k** and the repo concluded decode "governs >94% of user latency". The
measurement is sound; it decodes `max_new_tokens=1024`, which amortises prefill
over 1024 steps. **An agent turn emits ~150 tokens.**

Measured at agent shapes
([`benchmark_agent_turn_split.py`](../benchmarks/runtime/performance/prefill_vs_decode/benchmark_agent_turn_split.py)):

| prompt | out=50 | out=150 | out=300 | out=1024 |
| ---: | ---: | ---: | ---: | ---: |
| 2000 | 25.2% | 10.2% | 5.2% | 1.5% |
| **3000** | 30.8% | **13.0%** | 7.0% | 2.2% |
| 8000 | **55.8%** | **30.3%** | 17.5% | 5.8% |

The two scripts reconcile at out=1024 (5.8% vs 5.9% @8k) — a workload-shape
difference, not a contradiction.

**Decode work is still justified at the stated agent turn** (3,000 in / 150 out:
decode 87%, prefill 651 ms of a 5,001 ms turn) — but a 2× decode win returns
**1.77×** end-to-end, not 2×, and an infinitely fast decode caps at 7.68×. **At
8,000 in / 50 out prefill is 55.8% and decode optimisation is capped at 1.79×
regardless.** Size every decode optimisation against the turn shape, not against
the 1024-token figure.
