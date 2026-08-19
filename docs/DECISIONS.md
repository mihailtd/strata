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
now been tested. 10 matched runs, `scripts/old/compare_init_schemes.py`, same data,
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
loss change. `scripts/train/train_expert.py --init-lora-weights` performs
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
`scripts/audit/analyze_loss_curves.py`). This settles two proposals — **both are
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
by `scripts/audit/audit_eval_rubrics.py`, with no GPU:

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

---

## 12. CONFIRMED (was untested) — Liger is worth 47.5% on the real workload; keep it

**Item:** "Fused Triton Backprop + Sequence Packing (Faster Training on ROCm)".
Status before today: fused kernels were **already adopted** (`liger_fused_kernels:
true` on every m2 adapter) but **never A/B'd on this rig**. Adopted on faith.

**Measured, matched A/B on the real trainer, both arms in-session:**

| arm | train_runtime | final EMA loss | noise floor |
| :--- | ---: | ---: | ---: |
| liger | **127.6 s** | 1.0717 | 0.0173 |
| noliger | 188.2 s | 1.0704 | 0.0172 |

**Liger is 47.5% faster and the losses are equivalent** (Δ −0.0013, well inside
noise). The Liger arm also reproduces the previous session's baseline (124.8 s),
so the comparison is anchored. It additionally saves **4.35 GB** peak VRAM.

**⚠️ A synthetic benchmark said the exact opposite and was wrong.**
`benchmark_training_step_profile.py` at batch 2 × seq 512 with random tokens and
every position labelled measured **liger_on 874.3 ms/step vs liger_off 689.4** —
"Liger is 27% SLOWER". The factory never runs that shape: real batches are
dynamically padded to a **mean of 109 tokens** with the prompt masked. A
microbenchmark shape that does not match the workload produced a sign error, and
the real-data A/B supersedes it.

**Two harness bugs were found and fixed getting here, both of which produced
plausible-looking wrong numbers:**

1. `apply_liger_kernel_to_qwen3_5()` rebinds at **class level** with no unpatch,
   so running both arms in one process measured **Liger against itself** — it
   reported 0.994× and −0.01 GB. Caught by arithmetic, not inspection:
   `fused_linear_cross_entropy` avoids materialising a 2×512×248320 bf16 logits
   tensor (~508 MB plus gradient), so a saving of zero is impossible. Each arm
   now runs in a fresh interpreter.
2. `torch.profiler` returns **zero CUDA events** on this ROCm build (no
   roctracer). The kernel table printed empty under "total GPU self time 0.0 ms"
   and nothing flagged it. Replaced with manual synchronised phase timing.

**Consequence for the roadmap: this RAISES the prior for further fusion work.**
The argument for deprioritising "Fused Triton Backprop" was that Liger's existing
coverage bounds what more fusion could buy. That bound is 47.5%, not a few
percent. Fused kernels matter a great deal on gfx1100. **What remains untested is
whether the ops Liger does NOT cover are a large enough share to be worth
attacking** — notably RoPE (Liger raises `NotImplementedError` for Qwen3.5's
hybrid GatedDeltaNet/attention mix) and the GatedDeltaNet backward. Sizing that
needs a kernel-level trace, which this ROCm build cannot currently provide.

Evidence: `results/training_step_profile.json`,
`results/loss_curves/astral_{liger,noliger}_insession.json`.

---

## 13. REFUTED (measured) — the explicit subspace-orthogonalisation penalty

**Applies to:** "Explicit Subspace Orthogonalization Penalty" and "Explicit
Subspace Orthogonalization for 100+ Experts (Zero Fleet Collision)".

Previously dismissed by *inference* ("overlap is already 1.10–1.28× chance"). Now
measured on both sides, and the inference was half wrong — which is why it was
worth checking.

**Static headroom** (three live experts, 128 common modules,
[`benchmarks/factory/geometry/orthogonality_headroom/`](../benchmarks/factory/geometry/orthogonality_headroom/)):

| pair | INPUT side (read) | OUTPUT side (write) |
| :--- | ---: | ---: |
| astral \| postgresql | 1.031× | **6.866×** (max 21.0×) |
| astral \| financial | 1.023× | **5.996×** (max 17.4×) |
| postgresql \| financial | 1.036× | **7.164×** (max 25.0×) |

**"Statistically orthogonal" holds only for the INPUT side.** The output side —
which directions each expert writes into — sits at 6–7× chance. Note LoRA
initialises `B = 0` and `A` random, so U is entirely gradient-driven while V
retains its random init; the input-side result may partly be an artefact of A
never moving far from initialisation rather than evidence the tasks are
orthogonal.

**Trained with the penalty** (postgresql, frozen astral peer, both sides,
`scripts/old/train_with_orthogonality_penalty.py`):

| λ | overlap IN | overlap OUT | final EMA loss | vs baseline |
| ---: | ---: | ---: | ---: | ---: |
| — | 1.031× | 6.866× | 0.7481 | — |
| 1.0 | **0.006×** | **0.024×** | 0.7741 | **+0.0260** (1.4× noise) |
| 10.0 | **0.000×** | **0.001×** | 0.7968 | **+0.0486** (2.7× noise) |

**The penalty works perfectly and that is the refutation.** Baseline overlap was
already AT chance, so there was no harmful overlap to remove; all the penalty can
do is push past random into anti-correlation (0.006× is ~170× *below* random),
which confers no interference benefit. It charges a monotonic loss cost for it —
a clean dose-response, so causal rather than noise — plus **1.9× training wall
clock** (430.9 s vs ~230 s) for the per-step QR over 128 modules. Its only
consumer, adapter stacking, is retired independently (§4).

**Not run:** solo-quality and stacked-accuracy evals were started and cancelled.
Neither could change the verdict — a technique that costs loss and wall clock, has
nothing to remove, and serves a retired consumer is dead regardless of how those
land. Recorded so nobody assumes they exist.

**Three implementation bugs, two of which would have produced the right verdict
for the wrong reason:**
1. peft yields `base_model.model.model.layers.N...` while FoldableExpert keys
   carry a `.weight` suffix — every lookup missed and the run completed reporting
   `final penalty term: 0.000000`. That reads as "the penalty does nothing" when
   in fact the experiment was switched off. `_assert_matched` now refuses to run.
2. `QR(0)` is degenerate and LoRA initialises `lora_B` to exactly zero, so the
   penalty returned `nan`. **The output-side subspace does not exist at
   initialisation** — a property of LoRA worth knowing, not just a guard to add.
3. λ=0.1 adds ~0.005 to a loss of ~1.5. Too weak to move a subspace; a null there
   would have measured the λ, not the idea.

---

## 14. SIZED AND REJECTED — custom Triton kernels for RoPE and GatedDeltaNet

Liger's 47.5% (§12) reset the prior on fusion, so the two ops it does NOT cover
were sized before committing to a kernel project
([`benchmark_unfused_op_sizing.py`](../benchmarks/factory/training_profile/benchmark_unfused_op_sizing.py)).

Step = 460.9 ms: **forward 187.4 ms (41%), backward+optimizer 273.5 ms (59%)**.
Architecture is 24 GatedDeltaNet + 8 full-attention layers.

| layer type | ms/step | % of forward | % of step |
| :--- | ---: | ---: | ---: |
| LigerQwen3MoeSwiGLUMLP | 43.93 | 23.4% | 9.5% |
| **Qwen3_5GatedDeltaNet** | **23.17** | **12.4%** | **5.0%** |
| Qwen3_5Attention | 9.16 | 4.9% | 2.0% |
| LigerRMSNormForQwen3Next | 4.79 | 2.6% | 1.0% |
| Conv1d | 4.00 | 2.1% | 0.9% |
| **apply_rotary_pos_emb** | **0.99** | **0.5%** | **0.2%** |

**RoPE — REJECTED.** 0.21% of the step, ≤0.53% including a proportional backward.
Perfect fusion caps at **1.005×**. Liger's blanket `NotImplementedError` for
Qwen3.5 costs essentially nothing.

**GatedDeltaNet — REJECTED, though 25× the better candidate.** 5.03% forward,
≤**12.36%** including backward, capping at **1.141×**. That bound is generous
twice over: it assumes the backward scales with the forward AND that a fused
kernel makes the op *free*. A realistic 2–3× on the op yields ~1.08×, for
multi-week Triton work on gfx1100 where `fla` does not build — against 47.5%
already obtained from a config flag. And this is the **training** path only;
§8 already concluded training time is not the bottleneck worth optimising.

**⚠️ THREE MEASUREMENT ATTEMPTS, TWO OF WHICH SILENTLY ERASED THE TARGET OP.**
Recorded because both failures looked like results:

| attempt | method | failure | symptom |
| ---: | :--- | :--- | :--- |
| 1 | leaf-module hooks | GatedDeltaNet is COMPOSITE — its scan is functional code in its own forward, so it was never hooked | 240 of 459 ms attributed; **"GatedDeltaNet = 0.0%"** |
| 2 | all modules − direct children | CONTAINER modules (`ModuleList`) have no forward, never time, so subtracting them removes nothing and parents keep their whole subtree | sum **239%** of step, UNATTRIBUTED **−643 ms** |
| 3 | whitelist of non-nesting sibling classes, inclusive | — | coherent |

Attempt 1's "0.0%" reads as *"already free, do not fuse"* when it means *"never
measured"*. Attempt 2 was caught only because an `UNATTRIBUTED` row had been added
after attempt 1 — without it, a 239% total would have printed as a plausible
ranked table.

Two rules this leaves behind for any future op sizing here:
1. **Print the unattributed remainder.** A coverage hole must surface as a number.
2. **Do not use `register_full_backward_hook` for attribution** — it returned
   `0.00%` for GatedDeltaNet (unreliable on multi-input/output modules). The
   backward split here is measured by **ablation** (forward-only vs full step),
   which cannot silently return zero, and the per-op backward is reported as an
   explicit upper bound rather than a fake measurement.

`torch.profiler` remains unusable on this ROCm build (zero CUDA events, no
roctracer), which is why none of this could be done with standard tooling.

---

## 15. CORRECTED — `fla` IS available on this rig; the missing piece is `causal_conv1d` only

**This retracts a claim the repo has carried, and which I repeated earlier today.**
`mtp_draft.py` states the chunked path "falls back to slow PyTorch" because
`fla`/`causal_conv1d` are "not buildable on this AMD rig", and every run log prints
"The fast path is not available because one of the required library is not
installed". Both readings are wrong.

Measured (crucially, **with `--env-file .env`** — a bare `uv run` reports no CUDA,
which makes `is_flash_linear_attention_available()` return False for the wrong
reason and nearly produced a bogus "package name mismatch" diagnosis here):

| check | result |
| :--- | :--- |
| `flash-linear-attention` / `fla-core` installed | **0.5.2** |
| `_is_package_available("fla")` | `(True, '0.5.2')` |
| `is_flash_linear_attention_available()` | **True** |
| `chunk_gated_delta_rule` bound in `modeling_qwen3_5` | **yes** |
| `is_causal_conv1d_available()` | **False** ← the only gap |

The warning names two libraries and only `causal_conv1d` is absent. Independent
corroboration: `benchmark_unfused_op_sizing.py` hooks **`FusedRMSNormGated`** as a
live module class, and that class is imported from `fla.modules`.

**Consequence: do not write a GatedDeltaNet chunked-scan Triton kernel.** It
already exists, already compiles on gfx1100, and is already executing. Any
hand-written replacement would be reimplementing a specialist research library's
kernel with worse numerics.

**What this does NOT explain.** `mtp_draft.py` measures a 2.84x cost jump from K=1
to K=2 and attributes it to the PyTorch fallback. The jump is real; the attributed
cause is not. **Unexplained — do not cite the fallback as the reason.**

**Still open, and the only remaining sized target:** `commit_fwd` is 29.4% of the
speculative loop (§10) because the chunked scan returns only the FINAL recurrent
state, forcing a snapshot-restore and full re-forward on 70% of steps. Before any
kernel work, check (a) whether `fla`'s API can already expose per-position
intermediate states, and (b) whether `causal_conv1d` builds here. Both are shell
work, not kernel authorship.

---

## 16. MEASURED — postgresql's eval also leaked; the clean instrument moves it +24.08 → +27.92pp

§9 flagged postgresql at **55% giveaway** and predicted its +24.08pp was an
understatement. Built `data/postgresql/evaluation_data_v2.jsonl` (40 scenario
prompts, **0.0% giveaway** on both good AND bad terms — v1 also leaked 12 *bad*
terms, handing the model legacy vocabulary):

| | base | expert | delta | 95% CI |
| :--- | ---: | ---: | ---: | :--- |
| v1 (55% giveaway) | 51.33% | 75.42% | +24.08pp | — |
| **v2 (0% giveaway)** | **18.33%** | **46.25%** | **+27.92pp** | **[+7.92, +47.50]** |

Prediction confirmed but **modest** — +3.84pp, not the transformation the financial
case saw (+0.00 → +35.00pp). Base falls 51.33 → 18.33 on the clean set, which is
the leakage being removed; the *adapter effect* was largely intact all along.

By category: vector_indexing **+80.00pp**, anti_pattern **+20.00pp**,
unified_platform **+11.67pp**, ai_native_features **+0.00pp**.

**No retrain performed, and none indicated.** Financial needed one because two
things were broken — blind eval AND a 98.7%-definitional corpus. Here only the
instrument was at fault: the adapter changes 40/40 answers and posts a
significant effect on the clean eval, so the corpus is not implicated.

**v1 remains canonical.** v2 sits alongside so every historical postgresql number
keeps its meaning. Promoting either is a human decision.

---

## 17. PROMISING (kill-test passed) — recurrent state replay can replace `commit_fwd`

**Status: prototype greenlit, NOT a claimed win.** One micro-benchmark passed; the
end-to-end test has not been run.

`commit_fwd` is 29.4% of the speculative loop (§10): on a partial accept the
chunked scan has already advanced the recurrent state past the rejected tokens, so
the loop restores a 52.5 MB snapshot and runs a full model forward over the
committed prefix purely to re-derive that state.

**fla cannot supply the intermediate state, structurally.**
`chunk_gated_delta_rule` returns `(o, final_state)` — exactly one state, shape
`[N, HV, K, V]` — and its `chunk_size` is **64**, so a K+1 = 5 token verification
window contains **zero chunk boundaries**. Chunked scans compute a block's
aggregate transition and never materialise per-position states; that is what
chunking *is*. This is not an upstream API gap to file.

**But `commit_fwd` recovers three things and only one is expensive:** attention KV
entries are truncatable, hidden states are a slice, and the recurrent SSM state is
the single irrecoverable tensor. `fused_recurrent_gated_delta_rule` takes the
*identical* argument list to the chunked call, so the committed prefix can be
replayed through the recurrent kernel from the snapshot — scan only, no forward.

**Measured** (K=4, n_acc=2, 24 GatedDeltaNet layers,
[`benchmarks/runtime/speculative/state_replay/`](../benchmarks/runtime/speculative/state_replay/)):

| arm | | |
| :--- | ---: | :--- |
| A — full model re-forward over 3 tokens | **34.23 ms** | current cost |
| B — recurrent replay, 24 layers | **1.23 ms** | **27.8× cheaper** |
| worst relative divergence, B vs C | **7.31e-03** | |

Arm C is the *chunked* scan over the same prefix from the same initial state —
included so the numerics comparison isolates recurrent-vs-chunked kernel
divergence instead of confounding it with every other difference in a forward
pass.

**⚠️ THE DIVERGENCE IS MARGINAL, NOT CLEAN.** bf16 machine epsilon is ~3.9e-03, so
7.31e-03 is roughly **2 ULP** — the script's "within slack" verdict used a 1e-2
threshold chosen by hand, and that threshold is doing real work in the conclusion.
The state is carried forward, so the open question is whether this **compounds
across ~85 speculative steps** or is damped by the gated delta rule's forget gate.
Not measured. Do not treat the pass as settled.

**Other limits of this measurement:**
- Isolated micro-benchmark, not the loop. The synthetic chunk repeats one token;
  real drafts differ and the divergence may differ with them.
- The 1.396× ceiling assumes replay fully replaces `commit_fwd` with no new
  overhead. A real implementation must cache per-layer projections during the
  chunked verify, which adds memory traffic not counted here.
- **Serving path only, and `server.py` has no draft path today.**

**The right follow-up is end-to-end, not another micro-benchmark:** run the
speculative loop with replay substituted for `commit_fwd` and check (a) real loop
speedup, and (b) whether generated text changes, using the harness from §7. If the
text holds and the loop speeds up, the interesting consequence is not the 1.4× —
it is that a cheaper step lowers the acceptance break-even, which is what could
move speculation from marginal (measured τ 1.67–2.33) to profitable on this rig.

### §17 follow-up (2026-08-17) — end-to-end run, and a FLAW IN THE TEST ARM

Ran the replay inside the real speculative loop, 8 prompts × 64 tokens, K=4.

| measure | result |
| :--- | ---: |
| identical token sequences | **3/8** (diverged at tokens 4, 5, 13, 19, 20) |
| mean relative divergence, early quartile | 8.70e-02 |
| mean relative divergence, late quartile | 7.59e-02 |
| **late/early ratio** | **0.87×** |
| worst observed | 4.41e-01 |

**RESOLVED — the error does NOT compound.** 0.87× across 25 steps and 1152
samples: the gated delta rule's forget gate damps it rather than accumulating. That
was the main risk §17 left open and it is now settled.

**⚠️ THE TEXT RESULT IS NOT EVIDENCE ABOUT THE REAL DESIGN. The test arm was
mis-specified, and the flaw is mine.** To avoid cache surgery, the arm let
`commit_fwd` run and then overwrote only the recurrent state. That produces a
configuration neither design generates: **hidden states from `commit_fwd`'s
re-forward, recurrent state from the replay.** The real design would use the
chunked forward's hidden states AND the chunked-derived state, consistently.

It also explains the 12× larger divergence (8.70e-02 vs the kill-test's 7.31e-03):
the kill-test compared replay against the chunked scan on IDENTICAL inputs, while
this compares it against `commit_fwd`, which recomputes projections through a
different numerical path whose differences then amplify across 24 layers.

**So the clean number for the real design remains the kill-test's 7.31e-03**, and
whether skipping `commit_fwd` outright preserves output is **not settleable without
the surgery** (KV truncation + hidden-state slicing from the chunked forward).

**Status: specced, NOT built.** Honest prize accounting before anyone starts:
1.396× on a loop `server.py` does not use, for a feature below its own break-even
(τ 1.67–2.33 vs ~2.8). The strongest argument is indirect — removing 33 ms from 70%
of steps lowers that break-even and could make speculation pay at all.

**If it is built, the correctness test is the §7 three-arm quality harness**, not a
token-equality check: this stack already changes text on 33% of prompts from kernel
numerics alone with no measurable quality cost (+0.19pp, CI [−0.77, +1.17]), so
"the text changed" is not by itself a defect here.

---

## 18. REFUTED (built and measured) — skipping `commit_fwd` buys 1.008×, not 1.4×

The surgery from §17 was **built**: recurrent state replayed via
`fused_recurrent_gated_delta_rule`, conv state reconstructed from the `conv1d`
input window, attention KV truncated with `DynamicLayer.crop()`, hidden states
sliced from the chunked forward. It works — and it does not pay.

| arm | tokens | wall s | **tok/s** | ms/step |
| :--- | ---: | ---: | ---: | ---: |
| baseline | 269 | 6.91 | **38.93** | 77.6 |
| surgery | 169 | 4.31 | **39.24** | 71.8 |

**⚠️ THE HEADLINE "1.605× WALL CLOCK" IS AN ARTEFACT AND MY OWN BENCHMARK PRINTED
IT.** The arms do not emit the same number of tokens — the surgery arm hits EOS
earlier and generated **37% fewer tokens**. It finished sooner because it produced
less, not because it produced faster.

Normalised properly:
- **per-token throughput 1.008× — nothing**
- per-step time 1.082× — modest
- τ fell **2.022 → 1.817** (−10%), tripping the script's own guard

**Why the predicted saving never appeared.** Removing 33 ms from 75% of steps
predicts **52.9 ms/step**; measured **71.8**. About **19 ms of the 24.75 ms
expected saving was consumed by the repair itself** — 24 sequential Python-driven
kernel launches with shape-varying Triton autotuning (m ranges 1–5), plus conv
slicing with `.contiguous()` and `crop()` reallocating KV tensors.

**The isolated 1.23 ms micro-benchmark did not survive contact with the loop.**
That is the same lesson as §12, where a synthetic `seq_len=512` Liger benchmark
gave the *opposite sign* to the real workload: **component micro-benchmarks do not
predict in-loop cost.** §17's 27.8× was real and irrelevant.

**Also refuted: the indirect argument.** The hope was that a cheaper step lowers
the acceptance break-even and makes speculation pay. τ moved the *wrong way*
(−10%), because the draft head now conditions on chunked-path hidden states rather
than the re-forward's, and empirically drafts worse from them.

**Not run: the quality harness.** The idea is dead on throughput alone; measuring
the quality of a change that buys 0.8% would be GPU time spent on a settled
decision. The 37% shorter output is recorded as a behavioural regression, not
quantified.

**Kept as evidence, not as a component.** Nothing was wired into the serving path;
`server.py` still has no draft path at all.

---

## 19. RE-MEASURED — FlashNorm is worth ~0.5%, ReplaySSM is unresolved; both headlines were warmup

Two proposals arrived with strong headline numbers. Both re-measured with warmup,
repeats and control arms; both headlines were measurement artefacts. **Both
implementations are correct** — the mechanisms work, they are just small.

### FlashNorm weight folding: claimed +0.5–3.1% (gate 7.75%) → **+0.46%**

| | astral | postgresql | financial | mean |
| :--- | ---: | ---: | ---: | ---: |
| flash vs arm1 *(original comparison)* | +3.4% | +3.1% | −1.1% | **+1.79%** |
| **stock vs arm1 — the SAME arm re-run, no fold** | +2.0% | +2.5% | −0.6% | **+1.32%** |
| flash vs arm3 *(matched warmth)* | +1.4% | +0.6% | −0.6% | **+0.46%** |

**Re-running the stock arm reproduces 74% of the reported "speedup".** Position in
the run is worth ~1.3% on this box — a number worth remembering for every future
A/B here. The original baseline was also cold: warmed stock is 35.3–36.1 tok/s vs
31.3–32.9 reported, ~11% slow, and warmed stock is FLAT across domains (1.1%
spread vs 5%) as the mechanism requires.

Gate 1.3's **7.75%** used a ~11.2 ms/token denominator (≈89 tok/s) against a model
that runs ~28 ms/token — **~17× overstated**. Component arithmetic also fails to
close: **64 norms are folded, not 73**, so Gate 1.1 offers 6.60 µs × 64 = 422 µs
while Gate 1.3 claims 869 µs saved per token, a **2.1× overshoot**.

**⚠️ DRIFT CONFIRMED.** `unfold_rmsnorm` restores by dividing out `(1+γ)` instead
of copying a pristine buffer. After **one** fold/unfold round trip, astral's
generated text **changed**. This is exactly the "reconstruct by inverse
arithmetic" pattern `NOVELTY.md` retired for adapter folding, and it reproduces on
the first cycle. Not live (server folds once at boot and never unfolds), but it
must never be used in a loop, and the reversibility unit test passes only because
it checks a tolerance rather than generated output.

**Genuinely correct and load-bearing:** adapter `V` factors must be scaled by
`(1+γ)` — unscaled diverges 28.1%. That finding would have silently corrupted
every adapter and it was caught.

### ReplaySSM ring buffer: claimed +17.45% → **+9.58%, and unresolved**

No warmup: Arm A of prompt 1 read **18.9 tok/s** against prompt 2's **42.1 tok/s**
on an identical config. With warmup and 3 interleaved repeats: +15.28%, +7.45%,
+6.00% (mean +9.58%) — but **every prompt's arm spreads overlap**, so nothing is
resolved. Arm B's slowest prompt-1 run is slower than Arm A's slowest.

The mechanism bounds it anyway: 498 µs saved × ~13 rollbacks ≈ 6.5 ms of a
~1300 ms run = **~0.5%**. The residual is 20× that, i.e. variance. Arm A also
always ran first — the FlashNorm control above measures that bias at ~1.3%.

Two factual errors: **8 full-attention layers, not 12**
(`{'linear_attention': 24, 'full_attention': 8}`), and the recurrent ring buffer
is **fp32, not bf16** (51.0 MB only computes in fp32; bf16 would be 25.2 MB).

**Correct and worth keeping:** 100% exact text match, accept rates identical
across arms (47.1/67.3/50.0), bit-exact restore, 51 MB, tests green.

### The transferable lesson

Every A/B on this box now needs: **a warmup pass, repeats with spreads, and where
possible a control arm that re-runs the SAME configuration in the treatment's
position.** The third one is what converted "FlashNorm gains 1.79%" into "position
gains 1.32%, FlashNorm gains 0.46%", and no amount of repeats alone would have
found it.

---

## 20. MEASURED — what the state ring buffer unlocks, and the asymmetry that decides it

ReplaySSM's direct speedup is unresolved and bounded near ~0.5% (§19). Judging it
on that alone would be near-sighted — **a primitive's value is not only its direct
speedup**. Three capability claims were made for it; all three now have a
benchmark, **including the one predicted to fail**
([`benchmark_branching_primitives.py`](../benchmarks/runtime/speculative/state_replay/benchmark_branching_primitives.py)).

| claim | verdict | measurement |
| :--- | :--- | :--- |
| **② Time-travel steering** | **SUPPORTED** | rewind is exact and **120× cheaper than re-prefilling** (0.39 ms vs 35.7–47 ms). Resumed tokens match a fresh-cache reference. Bounded to `max_depth=8` checkpoints. |
| **③ Local beam search** | **PARTIAL** | 2-branch search costs 220 ms when the second branch wins, **340 ms when the first wins (+54.6%)** — its KV was cropped and must be regenerated. |
| **① Speculative trees** | **NOT SUPPORTED** | the ring retains **zero KV bytes** (stores `list`s of ints per attention layer), so branch A dies when branch B is explored. |

### ⚠️ The state geometry — and it is the OPPOSITE of what was assumed

| | measured |
| :--- | ---: |
| SSM state per checkpoint | **51.90 MB** (fixed, length-independent) |
| Attention KV | **32.77 KB per token** (grows) |
| Ring buffer, 8 slots | **415.2 MB** |

**The crossover is ~1,584 tokens.** One SSM checkpoint costs as much as 1,584
tokens of KV. Speculative trees span 8–32 tokens, so **SSM state is 50–200× more
expensive than attention KV in this regime.**

Both the ReplaySSM write-up and my own review asserted that attention KV was the
expensive half blocking trees. **Both were wrong.** KV for a full width-4 × depth-8
tree is **1.05 MB**; the SSM slots for it are **1,661 MB**.

| tree | live nodes | KV | **SSM** |
| :--- | ---: | ---: | ---: |
| width 2 × depth 4 | 8 | 0.26 MB | **415 MB** |
| width 4 × depth 4 | 16 | 0.52 MB | **830 MB** |
| width 4 × depth 8 | 32 | 1.05 MB | **1,661 MB** |

**Encouraging consequence:** width-2 × depth-4 needs exactly **8 live nodes, which
`max_depth=8` already holds**. The only missing piece is KV forking at **0.26 MB**.
Tree speculation at modest width is a small piece of plumbing, not a
memory-infrastructure project — a materially better position than "needs paged KV
blocks."

**Third factual error in the write-up:** the ring buffer is **415.2 MB (1.7% of
24 GB)**, not the claimed 51.0 MB / 0.208% — 8× understated. Still inside the 5%
kill-switch, so that verdict stands; the number does not. (Errors 1 and 2: 8
full-attention layers not 12; fp32 recurrent states not bf16.)

### The lesson worth keeping

Small direct gains can unlock disproportionate capability, and a repo that only
asks "how much faster?" will discard its own foundations. **The right response is
not to lower the evidence bar for capability claims — it is to write the benchmark
that tests them.** Doing so here beat both available narratives: two of three
claims hold, the third is half-built, the missing half is 200× cheaper than
anyone (including this reviewer) assumed, and the real constraint turned out to be
somewhere nobody was looking.

---

## 21. MEASURED — mid-stream expert hot-swapping works; the carried state is not a liability

**Claim:** swap experts mid-generation within one response stream — Python tooling
first, fold to the postgres expert when the answer reaches an SQL block — without
resetting the KV cache or re-forwarding the prefix.
[`benchmarks/runtime/folding/midstream_swap/`](../benchmarks/runtime/folding/midstream_swap/)

### The attribution in the proposal is wrong, even though the capability holds

The claim credits "In-Place Folding + ReplaySSM". **ReplaySSM contributes nothing
here.** The KV cache is a *separate object* from the model weights;
`WeightFoldingEngine.activate()` mutates weights in place and never touches the
cache passed to `forward()`. A mid-stream swap therefore already works with
folding alone — there is nothing for a rollback primitive to "hold stable"
because nothing is rolled back. ReplaySSM is a REWIND mechanism; hot-swapping
CONTINUES. It would only matter for *undoing* a swap.

### Measured: 4 arms, post-swap text scored on the repo's postgres term lists

| arm | score | |
| :--- | ---: | :--- |
| A — astral throughout | **0.00** | floor: never reaches postgres vocabulary |
| B — postgres throughout | **100.00** | ceiling |
| **C — swap, state CARRIED** | **100.00** | **the claim — hits the ceiling** |
| D — swap, prefix RE-PREFILLED under postgres | **66.67** | carried-state control |

**Swap costs 17.63 ms** (independently consistent with the 17.97 ms fold from
`calibrate_transition_costs.py`) against **55.03 ms** to re-prefill — **3.1×
cheaper**, and it reaches the same quality as never having used the other expert.

### The counter-intuitive part: re-prefilling is WORSE

D pays full price to re-encode the prefix under postgres and scores *lower* than
simply carrying the astral-encoded state. The likely reason is that the prefix
content genuinely *is* astral-domain (uv/ruff/Python tooling), so encoding it with
astral is the better representation. Carrying the cache leaves **each region
encoded by the expert appropriate to it** — a hybrid context, not a stale one.

That reframes "without resetting the KV cache" from a risk to be tolerated into
the mechanism's actual advantage.

### ⚠️ Limits — do not over-read this

- **n=3 prompts**, and the good/bad ratio metric **saturates at 0/100**, so these
  are three coarse samples, not a distribution. The direction is clear; the
  magnitude is not.
- Prompt 1's D=0.00 was inspected directly and is **not degenerate output** — it
  is fluent, similar length to C (297 vs 308 chars), and simply reaches for
  `api.postgres`/`openai_embed` imports instead of naming `pgvector`. The metric
  is doing something reasonable, but it is a keyword ratio, not a quality judge.
- C and D texts diverge on 3/3 prompts, which is expected: different cache
  encodings, not an error.
- No test of *repeated* swapping, or of swapping back. Cross-task subspace overlap
  is ~chance on the input side but 6–7× chance on the output side (§13), so
  repeated swaps are not obviously safe by extension from one.

---

## 22. MEASURED — branching lifts τ, W=2 is the optimum, and the "2.8 break-even" is STALE

The only question that decides tree speculation is whether acceptance improves
enough to pay. Measured directly, W independent K-token drafts, verifier keeps the
best ([`benchmarks/runtime/speculative/tree_search/`](../benchmarks/runtime/speculative/tree_search/)):

| W | τ_eff | vs W=1 | branch wins |
| ---: | ---: | ---: | :--- |
| 1 | 1.506 | — | 100% |
| **2** | **1.795** | **+0.288** | 86%, 14% |
| 4 | 1.873 | +0.367 | 83%, 13%, 4%, **0%** |

**Branching works, and it saturates immediately.** W=1→2 buys +0.288; W=2→4 buys
only +0.078 more. At W=4 the top branch already wins **83%** of steps and the
fourth branch wins **zero**. Combined with the measured batch cost (28.68 / 28.93 /
32.31 ms), **W=2 is the optimum at 1.391× vs autoregressive; W=4 is WORSE at
1.351%** — cost outgrows acceptance.

### ⚠️ THE "~2.8 BREAK-EVEN" IS STALE, AND I REPEATED IT ALL SESSION

`mtp_draft.py` states break-even is ~2.8 accepted tokens because "verifying K
tokens costs 2.84× one token". That was measured **before `fla` was available** —
the same stale docstring that claimed fla was "not buildable on this AMD rig"
(§15). Measured now:

| | measured |
| :--- | ---: |
| chunked verify of a 5-token chunk | **36.36 ms** |
| one autoregressive token | 28.68 ms |
| **ratio** | **1.27×** — not 2.84× |

**Speculation therefore already pays on this rig and always did.** Recomputing
from measured kernel costs gives 1.18× (W=1), 1.32× (W=2), 1.36× (W=4) — and the
independently-run speculation matrix measured **1.138×–1.294×**, which is
consistent. Every statement in this session's earlier notes that speculation "sits
below its own break-even" is **withdrawn**; it inherited a number that predates the
current kernel stack.

### What this means for tree search

- **Worth doing, at W=2 only.** ~1.11× on top of single-path speculation, which
  itself already pays. Not the transformative win the proposal implied.
- **None of the proposed mechanism is what delivers it.** Not ReplaySSM rollback,
  not the "SSMs avoid KV explosion" premise (backwards — §20), not tree attention
  masks (incompatible with recurrent layers). What delivers it is **batching W
  branches**, priced from the existing batch-scaling table.
- **The branch-win distribution is the real limit.** 83% top-branch dominance at
  W=4 means the draft head's second choice is rarely right. Improving τ further
  needs a *better drafter*, not a wider tree — and §5 already established that
  adapting the draft head makes acceptance worse.

### Limits of this measurement

- Branches are verified **sequentially**; acceptance is layout-independent so the
  τ numbers are exact, but the batched engine was **not built**.
- The net-speedup column applies **batch-scaling costs measured at seq=1** to a
  seq=5 chunked verify. That is an approximation, and the W=2 vs W=4 ordering
  could shift under a direct measurement.
- 6 prompts × 48 tokens, one domain (astral), K=4, no repeats.

---

## 23. CONFIRMED — the matched draft adapter works; §5's "never adapt the head" was scoped too widely

**Result:** folding the SAME expert into the MTP draft head that the backbone is
already wearing raises acceptance in both domains tested:

| domain | τ head pristine | τ head matched | Δ |
| :--- | ---: | ---: | ---: |
| astral | 1.938 | **2.013** | **+0.074** |
| postgresql | 1.613 | **1.690** | **+0.076** |

Two independent domains landing within **0.002** of each other. Worth **+2.6% to
+2.9%** throughput, at **zero VRAM cost** — the adapter is already resident, and
folding it into the head touches 7 modules.

### This SHARPENS §5 rather than contradicting it

`benchmark_mtp_head_adapter_acceptance.py` measured τ 2.456 → 1.531–1.988 and
concluded "you cannot adapt a draft head". Its method says: *"The backbone is
identical across conditions — only the head changes."* It tested an adapted head
against a **PRISTINE** backbone — the mismatched case. Its own stated mechanism
("the head forms its own opinions and disagrees with the backbone") **predicts**
that matching them restores agreement, and it does.

**The corrected rule: a draft head must AGREE WITH ITS BACKBONE. When the backbone
is domain-folded, the head should be too.** The engine always serves a folded
backbone, so the matched case is the one that was operationally relevant all along.

Mechanically clean: the MTP head is architecturally a Qwen decoder layer with
shape-identical modules (`mlp.gate_proj` (9216,2560) in both), so backbone LoRA
factors fold in with no reshaping. Source is the last shape-compatible backbone
layer (31), 7 modules, adapter's own scaling.

### The compounded stack, all measured

| | factor |
| :--- | ---: |
| speculative decoding (τ 1.506) | **1.181×** |
| × W=2 branching (τ → 1.795, §22) | **1.116×** |
| × matched draft head (τ → ~1.87 equivalent) | **1.026×** |
| **= vs autoregressive** | **1.352×** |

**28.7 ms/token → 21.2 ms.**

### ⚠️ THE BLOCKER IS NOT ANY OF THIS — `server.py` HAS NO SPECULATION AT ALL

Confirmed by grep: zero matches for `speculat|draft|mtp` in `server.py`. Every
number above is benchmark-only and **users currently receive none of it**. The
ordering that actually moves user latency:

1. **Speculative decode in the serving loop** — worth 1.18× on its own, and it is
   the only step that changes what a user experiences today.
2. W=2 branching — a further 1.12×.
3. Matched draft adapter — a further 1.03×, free.

Step 1 is worth more than 2 and 3 combined and none of the rest ships without it.

### Correction carried from §22

Earlier notes in this session repeatedly said speculation "sits below its own
break-even (~2.8)". That bar came from a `mtp_draft.py` docstring measured before
`fla` was available. Chunked verify is **1.27×** a single token, not 2.84×.
Speculation pays and always did — the stale number caused every speculative result
here to be undersold.

---

## 24. ⚠️ SERVING GATE — every speculative number here used a baseline the server does not run

Before wiring speculation into `server.py`, the comparison that actually decides it
([`benchmarks/runtime/speculative/serving_gate/`](../benchmarks/runtime/speculative/serving_gate/)):

| arm | tok/s | vs eager | **vs graph** |
| :--- | ---: | ---: | ---: |
| A graph autoregressive — **what server.py runs today** | **36.67** | 1.135× | — |
| B eager autoregressive — the baseline every spec benchmark used | 32.31 | 1.000× | 0.881× |
| C eager speculative | 37.55 | 1.162× | **1.024×** |

τ = 1.994.

**The CUDA graph is worth 1.135× by itself.** Speculation is worth 1.162×. They are
**mutually exclusive**: the graph is captured for a fixed single-token shape, while
speculative verification is a K+1 chunk and the commit re-forward is n_acc+1 tokens.
Switching to speculation *gives up* the graph to buy speculation — netting **+2.4%**.

**⚠️ This invalidates the framing of §22–23.** The "1.352× compounded stack"
(speculation × W=2 branching × matched draft head) was computed against **eager**
decode. The server does not decode eagerly. Against the real serving path the
shippable, measured gain is **+2.4%**, and the branching and matched-head factors
have **not** been re-measured against the graph baseline — they may or may not
survive it. Do not quote 1.352× as a serving number.

### The resolution is bucketed graph capture, and it is not optional

The shape set is small and bounded: verification is always **K+1**, the commit
re-forward is **1…K** — **5 graphs at K=4**. Capturing those makes graph replay and
speculation compose instead of compete, which is the difference between a 2.4% win
and a large one.

**Two stack-specific obstacles, both real:**

1. **fla's Triton kernels autotune.** `chunk_gated_delta_rule` picks a config by
   launching variants and timing them — host synchronisation, illegal during
   capture. Every bucket must be warmed to populate the autotune cache *before*
   capture, or capture fails silently or records a timing path.
2. **Rollback invalidates captured pointers.** Graph replay needs pointer-stable
   memory; `FoldedCudaGraphDecoder` gets that from a pre-allocated `StaticCache`.
   The speculative path grows KV dynamically and rolls back via
   `layer.crop(saved_len)`, which re-views or reallocates the KV tensor. A captured
   graph holding the old address would then read wrong memory. **The speculative
   path must move to a pre-allocated static KV with rollback by length-pointer**,
   not tensor slicing — the same discipline the ring buffer already applies to SSM
   state, extended to KV.

**Recommendation: do not ship the eager speculative path for +2.4%.** Build the
bucketed capture first; it is what makes the measured 1.135× and 1.162× additive
rather than exclusive.

---

## 25. SHIPPED-READY — bucketed graph capture makes speculation and graph replay ADDITIVE (+8.5%)

§24 measured the fork: graph replay and speculation were mutually exclusive, and
wiring in eager speculation netted only **+2.0%** because it gave back the graph's
own advantage. Bucketed capture resolves it.

| arm | tok/s | vs eager | **vs graph (server today)** |
| :--- | ---: | ---: | ---: |
| A graph autoregressive — server today | 36.81 | 1.086× | — |
| B eager autoregressive | 33.88 | 1.000× | 0.920× |
| C eager speculative | 37.55 | 1.108× | 1.020× |
| **D bucketed graph speculative** | **39.94** | **1.179×** | **1.085×** |

**+8.5% over the current server**, and **+6.3% over eager speculation** — that
second number is exactly the graph benefit that the eager path was discarding.
τ = 1.994.

### Why pointer stability turned out to be free

The obstacle §24 flagged — rollback invalidating captured pointers — dissolves
under `StaticCache`. Probed on this hybrid config it contains:

| layer type | count | rollback |
| :--- | ---: | :--- |
| `StaticLayer` (attention) | 8 | **rewind `cache_position`. Nothing else.** KV beyond it is stale, never read, overwritten on the next write. No crop, no re-view, no moved pointers. |
| `LinearAttentionLayer` (SSM) | 24 | fixed-size `recurrent_states`/`conv_states`, restored by in-place `copy_` from buffers allocated once at capture |

The eager speculative path used `layer.crop()`, which re-views the KV tensor and
**would** have broken a captured graph. On a static cache the same rollback is a
counter decrement. Recurrent state and attention KV now roll back under one
pointer-stable discipline, with different mechanisms (copy vs counter).

### The capture hazard that had to be handled explicitly

fla's Triton kernels autotune by launching variants and timing them — host
synchronisation, illegal during capture. **Every bucket is warmed on its own shape
before its capture** so the autotune config is already chosen. Capturing cold
either fails or records a timing path, silently.

### Shape set

Verification is always K+1; the commit re-forward is 1…K. **5 graphs at K=4**, all
sharing one `StaticCache` and one CUDA graph memory pool.

`src/gnn_experiment/bucketed_speculative.py`

### Not yet stacked on top

W=2 branching (§22) and the matched draft head (§23) both raise τ and should
compound with this, but neither has been re-measured on the bucketed path. The
+8.5% is the bucketed path alone.

---

## §26 — The captured graph owned the cache length counter; §25's +8.5% is withdrawn

**Status: RESOLVED (bug), and the feature is REJECTED for serving.**

A long request pinned the GPU at 100% for 4h20m with no progress and ignored
`SIGINT`. Short requests were fine, which is why §25 never saw it.

### Root cause

`transformers`' `StaticLayer` holds its KV write offset in a **device tensor** and
advances it in place — and both lines are captured into the graph:

```python
cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
self.cumulative_length.add_(kv_length)
```

Every replay advances the counter by `width` regardless of the `cache_position`
the caller supplies; the attention layers compute their own offset and never read
ours. Autoregressive decode is immune (one replay, one token, reset per request).
**Speculation replays overlapping ranges** — a K+1 verify then a commit at the
same position — so the counter never rewinds. Committed tokens were written at
drifting offsets from the first partial accept, and past `max_cache_len` the
attention kernel indexed out of range and hung the GPU.

Wedge points match `2048/width` exactly: ~350–400 replays at width 5, ~1950–2000
at width 1, step 300–303 in real decode. Position, the shared graph memory pool,
the SSM snapshot/restore, and decode itself were each ruled out by measurement
before this was found.

**Fix:** pin the counter to the true position before a replay, only when drifted
(after a verify it already sits where the next verify wants it, so only rollback
pays). 5,200 replays clean vs 350 before; 1,400-token decode completes; costs 4%.

### Why the feature is still rejected

512 tokens, chat-template prompt, one process:

| arm | tok/s | vs graph | tau |
| :--- | ---: | ---: | ---: |
| graph autoregressive (server today) | 25.51 | 1.000x | — |
| bucketed speculative, pinned (correct) | 19.91 | **0.781x** | 1.65 |
| bucketed speculative, unpinned (wrong) | 20.78 | 0.815x | 1.68 |

**22% slower than what the server already runs**, correct or broken. tau is 2.5
over the first ~60 tokens of predictable thinking preamble and 1.65 over a
real-length answer, where a 5-token verify costs more than it saves.

### What this invalidates

§25's **+8.5%** and the **+20.6%** live-server figure are **withdrawn**. Both were
measured at 64 `max_tokens` — inside the inflated-tau window — *and* on a decoder
writing KV at wrong offsets. §22 (W=2 branching) and §23 (matched draft head) were
measured on the same corrupted path and are now **unverified**.

`SPECULATIVE_DECODE` stays `0`. The decoder and its repro stay in-tree as evidence.

### The methodology failure

A 64-token cap manufactured the entire result. This repo's own benchmarks default
to 192–1024 tokens; the serving benchmarks used 64 and the gap was never checked.
Short generations sit entirely inside the preamble, where tau is inflated and the
counter never overruns — so the benchmark measured the one regime where a broken,
unprofitable feature looks like a win.

`benchmarks/runtime/speculative/cache_length_counter/`

---

## §27 — RETIRED: Qwen3.5-0.8B as a draft model. It is depth-bound, and it isolates the real bottleneck

**Status: CLOSED. Do not fine-tune, distil, or re-tune a 0.8B drafter on this hardware.**

`CURRENT.md` recorded this as FAILED, but only via a library refusal
(`assisted generation is not supported with stateful models`), which our own
snapshot/restore loop no longer hits. Reopened and measured properly.

**Compatibility is fine.** Byte-identical tokenizer, `vocab_size=248320` both
sides, 14.4 GB together under the 22 GB cap.

**Acceptance is good.** tau = **2.323** with the astral expert folded (2.089
base), alpha chain 0.776 / 0.630 / 0.505 / 0.411 — far better than the shipped
MTP head's 1.65. The folded target is *easier* to draft for, not harder, which
runs opposite to §23's matched-adapter reasoning.

**Cost kills it.** Break-even tau is 2.64; we have 2.323. **Projected 0.913x.**
K=3 -> 0.917x, K=2 -> 0.888x, K=6 -> 0.857x. Nothing rescues it.

The reason is physics, not tuning: the 0.8B has **5.3x fewer parameters but only
2.14x lower latency** (16.56 vs 35.51 ms/token, both graph-captured), because
batch-1 decode here is **depth-bound** and the 0.8B is 24 layers against the 4B's
32 — only 25% fewer sequential steps. Too deep to be a drafter at any weight size.

### What it isolated, which is the point

| component | ms | units |
| :--- | ---: | ---: |
| baseline width-1 graph replay | 35.51 | 1.00 |
| verify, width-5 graph replay | 31.27 | **0.88** |
| **commit re-forward, width-3** | **31.78** | **0.89** |

**The chunked verify is nearly free** — 0.88 units for 5 tokens, because graph
replay makes width almost costless. The hybrid tax was never the verify. It is
the **commit re-forward**: a full forward, ~25–40% of every speculative step,
producing zero new tokens, existing only to recover the recurrent state after
`n_acc+1` tokens.

Everything else it appears to provide is already in the verify output: attention
KV for the accepted prefix is causally correct and needs no rewrite (§25 rewinds
a counter, §26 pins it), and the hidden state at `n_acc` is already returned.
**Only the SSM state is missing.**

### Modelled prize, and the caveat that matters

With the commit re-forward at zero, the arithmetic inverts — the *cheap* MTP head
gains most, because its drafting is nearly free (~15 ms/step vs the 0.8B's 66):

| drafter | with commit | commit at 0 (upper bound) |
| :--- | ---: | ---: |
| 0.8B graphed | 0.913x | ~1.21x |
| MTP head | 0.78x measured | ~2.0x |

⚠️ **Commit-at-zero is an upper bound, not a plan.** Removing it is not free and
not untried: **§18 built recurrent state replay and measured 1.008x.** §18
predates bucketed graph capture (§25), so its replacement cost was 24 eager
per-layer launches — the same depth/launch physics that killed the 0.8B above.
Graph-capturing the state-replay path (one bucket per `n_acc`, as §25 does for
widths) is the untried variable. That is the reason to re-measure, not a
prediction that it will work.

`benchmarks/runtime/speculative/draft_model_0_8b/`

---

## §28 — This rig is kernel-LAUNCH bound, not work bound. Optimise launches, never arithmetic.

**Status: a design law, measured four independent ways.**

| change | work removed | speed gained |
| :--- | :--- | ---: |
| 0.8B drafter instead of the 4B (§27) | 5.3x fewer parameters | **2.14x** |
| §18 recurrent state replay instead of a forward | a whole forward | **1.008x** |
| incremental head_prefill instead of full rebuild | **180x** less work | **1.001x** |
| graph-capturing head_draft | none — identical math | **-34% on that phase** |

Only the last removes *launches* rather than *work*, and it is the only one that
moves. The sharpest statement of it: `head_draft` costs **23.91 ms for four
SINGLE-layer forwards** against **34 ms for a full 32-layer graph-replayed
forward**. One layer costing 18% of a 32-layer model is essentially all overhead.

This retro-explains three earlier results that were each written up as their own
puzzle: §18's repair being eaten by "24 sequential Python-driven kernel launches",
§27's 0.8B being depth-bound rather than bandwidth-bound, and §25's bucketed
capture paying at all.

### The speculative step budget, 99.8% accounted

512 tokens, tau=1.803, step 118.00 ms, 23.71 tok/s: verify 34.34 | commit 27.16 |
head_draft 23.91 | head_prefill 11.69 | accept_sync 8.24 | bookkeep 7.04 |
snapshot 4.71 | cat_hids 0.66.

**The commit re-forward is 23% of the step, not the bottleneck it looked like.**
Removing it entirely yields ~1.10x, not the 1.36-1.64x a component-level model
predicts, because 56.5 ms/step is not model compute. Any plan that targets it
alone is sized wrong.

### Attempted, not achieved

Graph-capturing the draft head delivered its predicted speed (head_draft
24.02 -> 15.74 ms, step 114.48 -> 101.77) but **not equivalence**: tau fell
1.803 -> 1.311 without an attention mask and 1.803 -> **0.910** with a
device-built one, so "attends to unwritten static-cache slots" is refuted.
Net 0.927x and 0.706x. The tau loss costs several times the ~5 ms saved.

**Do not retry until the mechanism is understood.** Two hypotheses were tested
and both were wrong; a third guess is not a plan.

`benchmarks/runtime/speculative/step_budget/`

---

## §29 — Graphed draft head: bit-exact in isolation, diverges in the loop. PARKED with a known-good starting point.

**Status: NOT ACHIEVED. Four hypotheses tested and refuted; the single-step
formulation is verified correct and preserved so a fifth attempt starts warm.**

§28 says the only thing that pays here is removing kernel launches, and
`head_draft` is the most compressible target: ~20-25 ms/step for FOUR
single-layer forwards against 34 ms for a full 32-layer graph replay.

### The one number worth keeping

Per-graph-replay cost is a hard floor, stable across three attempts:
**3.94 / 3.86 / 3.83 ms**. Four replays cost ~15.4 ms, so an **unrolled K-step
graph** (one replay for all K drafts) should land near **4 ms** — a ~4x cut on
that phase. That is measured, not projected, and it is the case for unrolling.

### Verified correct

`_run_layer` hardcodes `attention_mask=None`. Correct over a tight DynamicCache,
**wrong** over a 2048-slot StaticCache — it attends to unwritten slots
(max|dlogit| **0.40625**). With an explicit additive mask the StaticCache path is
**bit-exact (0.00000, argmax 4/4)**, and so is the captured graph, using a
shape-(1,) position buffer, a materialised mask, and §26 counter pinning. The
probe carries a numeric gate that must pass before generation runs, and it passes.

### Not solved

Across a generation tau collapses **1.803 -> 0.910** (0.694x) despite every step
being exact in isolation. Lockstep localises it precisely:

```
step   pos  hfilled  prev_nacc  stale_tail  max|dlogit|
   0    53       52         -1           0      0.00000
   1    58       57          4          -1     14.78906
```

Immediate, large, on the **first full accept** — the one path where `new_h` comes
from the verify replay rather than the commit re-forward. `stale_tail=-1` rules
out live rejected speculative entries.

### Refuted — do not repeat

1. shared CUDA graph memory pool (private pools changed nothing)
2. attending to unwritten StaticCache slots (true in isolation; masking made the
   generation **worse**)
3. graph wiring — 0-dim position scalar / in-graph mask (rewired to the bit-exact
   form, gate passes, generation unchanged)
4. stale speculative cache entries (`stale_tail=-1` at the divergence)

### Judgement

Five hypotheses in one session, four measured wrong, is the signal to stop
guessing. The next attempt should **diff the head's K/V position-by-position at
lockstep step 1**, not reason about it. Note also that tau is worth several times
more than the milliseconds at stake: a 4 ms/step saving against a tau of 1.803 is
worth ~4%, while getting tau wrong costs 30%.

`benchmarks/runtime/speculative/graphed_draft_head/`

---

## §30 — The streaming/serving path costs ~0. Production runs at 93% of the decoder ceiling.

**Status: REFUTED as a target, by measurement.**

§28's "host-bound" finding made the streaming path an obvious suspect: per token
the server does a GPU sync, a Python BPE decode, three nested pydantic
constructions, a `model_dump()`, a `json.dumps()` and an event-loop tick.

Measured, it is nothing. Serialisation tail **13.8 us/token = 0.04%** of a
35.51 ms token (`tokenizer.decode` 2.70 us, construct+dump+json 11.15 us).

| arm | tok/s | vs decoder |
| :--- | ---: | ---: |
| decoder, batch | 28.98 | 1.000x |
| decoder, streaming | 29.39 | 1.014x |
| decoder, stream + pydantic + json | 30.01 | 1.036x |
| live server, non-streaming | 26.90 | **0.928x** |
| live server, streaming | 25.76 | 0.889x |

TTFT 46 ms cold (45.98 = expert swap), **0 ms** warm.

### The distinction §28 did not make explicit

**Host-bound here means GPU kernel-DISPATCH bound, not Python-application bound.**
Graph replay costs 3.85 ms (1 layer) to 34 ms (32 layers); tokenizer + JSON work
costs ~14 us. Three orders of magnitude apart. `.item()` per token is free for the
same reason — the CPU is blocked on the GPU anyway.

**Do not optimise the tokenizer, pydantic models, or SSE framing.** The serving
layer gives up 7%, and the decoder is at the hardware's graph-replay ceiling.

`benchmarks/runtime/performance/serving_path/`

---

## §31 — Expert routing: the corpus was the defect, not the routing. Gate flipped, but the benchmark now saturates.

**Status: postgresql corpus REBUILT and the regression FIXED. Net win over base is
positive but NOT significant — the gate has a ceiling problem.**

### The gate

Three arms on identical 3-step cross-domain agent conversations, scored
objectively (SQL parses as Postgres via `sqlglot`, Python via `ast.parse`) with a
pre-registered rubric audited for giveaways. Decision rule fixed in advance.

**Routing was never the problem: 15/15 correct, and self-routing == oracle in
both runs.** Pillar-1 plumbing works.

### The defect was the training data

`data/postgresql/training_data.jsonl`, 411 records:

| shape | share |
| :--- | ---: |
| about-style ("What is...", "How does...") | **85.2%** |
| write-style ("Write...", "Generate...") | 14.8% |
| book/author biography | **10.9%** |

Not a coverage gap — it mentions hnsw/ivfflat/pgvector **205 times**. It is
book-derived recitation that teaches talking about Postgres, never writing it,
exactly as §9 found for financial. Records like *"What is Marc Linster's
background?"* are author biography in a database expert's corpus.

### The rebuild and the result

`scripts/corpus/build_postgresql_applied_examples.py` -> 741 records, **54.1% applied**,
**342/342 generated SQL answers verified by sqlglot** (a gate the financial
rebuild could not have). Diversity asserted: 30 phrasings, 22 schemas, no family
above 11.1%. Retrained 4:18, loss 1.527 -> 0.804.

| arm | v1 | v2 |
| :--- | ---: | ---: |
| A base | 0.867 | 0.867 |
| B oracle | 0.790 (−0.077) | **0.907 (+0.040)** |
| C self | 0.790 (−0.077) | **0.907 (+0.040)** |

`ann_method` 0.400 -> **1.000**, `cosine_opclass` 0.200 -> **1.000**. That part is
categorical and consistent across all 15 steps.

⚠️ **The net advantage is not significant**: 95% CI **[−0.0133, +0.1067]**, with
**11 of 15 steps tied** because base already scores 1.000 on 8 of 10 checks.

**Next step is the benchmark, not the model.** The tasks must be hard enough that
base does not saturate them, or the gate cannot answer the blueprint's question
however good the experts get.

### Also settled: thinking mode is unusable on this stack

Qwen3.5 never closes `</think>` here — measured at 512, 1024, 2048 and **6144**
tokens. It is a degenerate loop ("Wait, I should check X... Yes." repeated), not
test-time compute: **12-gram repetition 0.636 greedy**, and **0.410 under Qwen's
own recommended temp 0.6 / top_p 0.95**. No token runway fixes it and sampling
only softens it. `enable_thinking=False` is required, and our decoder is
greedy-only anyway.

`benchmarks/factory/agentic/handoff_gate/`, `scripts/corpus/build_postgresql_applied_examples.py`

---

## §32 — The trainer was computing loss on the prompt. 48.4% of every batch was question text.

**Status: FIXED and verified live.**

`train_expert_CURRENT_m2.py` used `SFTConfig(dataset_text_field="text")` with no
collator, no `completion_only_loss`, no `assistant_only_loss` — so the loss ran
over the WHOLE string. The adapter was trained to **generate the corpus's
questions**, not merely answer them.

Measured on a real batch, not estimated:

```
prompt/completion split: 741/741 (completion-only loss ON)
MASK VERIFIED: 183/378 label positions are -100
               (48.4% of the batch is prompt, excluded from loss)
```

**Nearly half the gradient signal was teaching question text.** That is a direct
mechanism for the narrowing §31's held-out benchmark measured (experts −0.233 vs
base, CI [−0.356, −0.111], on constructs absent from their corpora): training on
question text teaches the corpus's *distribution*.

TRL 1.9.2 exposed `completion_only_loss`, `assistant_only_loss`, `padding_free`
and `packing` the whole time. None were used.

### The fix, and why it asserts

`load_dataset_records` now emits prompt/completion pairs split on the
`\n\n### Answer:\n` marker, and `completion_only_loss=True` masks the prompt.

A **hard assertion** verifies the mask on a real batch before training starts:
some labels must be −100 and some must not. §13 records an entire run whose
orthogonality penalty silently evaluated to `0.000000` because a name did not
match — a masking flag that quietly does nothing fails identically, and only
shows up as behaviour months later. `regime.json` now records
`prompt_mask_verified` with the measured fraction.

### Also wired: the geometry probes

The pre-flight SVD / times-above-chance probes ran only as standalone scripts.
They are now computed after every train and recorded in `regime.json`.

Doing so exposed a **broken import**: `build_orthogonality_map.py` imported from
`scripts.factory.geometry...`, a path that ceased to exist when the probes moved
under `benchmarks/`. The module raised ModuleNotFoundError. Repaired.

New adapter vs its siblings (k=32):

| pair | times above chance |
| :--- | ---: |
| pg v3 vs astral v2 | 1.26x |
| pg v3 vs pg v2 (same domain) | **26.94x** |
| pg v3 vs financial | 1.29x |

Cross-domain sits at chance, same-domain at 27x — the probe discriminates.

### Factory audit, all eight requested items

| item | status |
| :--- | :--- |
| Dynamic sequence masking / prompt-target separation | **ADDED** |
| Response-only completion loss | **ADDED**, mask asserted |
| Agentic corpus-to-instruction pipeline | present (`build_*_applied_examples.py`) |
| Liger fused kernels (FLCE + rms_norm + swiglu, rope off) | already present |
| Stock LoRA r=8 alpha=128, 7 projections | already present |
| Pre-flight SVD subspace probe | **WIRED** into regime.json |
| Times-above-chance SVD probe | **WIRED** into regime.json |
| Hyperparameter scale calibration (alpha sweep) | ⚠️ **NOT wired, deliberately** |

⚠️ The alpha sweep measures **bf16 merge-absorption error**
(`merge_rel_err ≈ 0.167 / (‖dW‖/‖W‖)`), not capability retention. Calibrating
alpha from it would optimise the wrong objective — its own finding is that
*larger* alpha merges more faithfully, which is the opposite of what narrowing
needs. The right instrument for choosing alpha is §31's held-out benchmark.

**Unproven:** completion-only loss is a well-motivated fix for narrowing, not a
demonstrated one. It may only improve answer quality. §31's gate decides.

`results/adapters/m2_postgresql_r8a128_v3`

---

## §33 — Completion-only loss recovers HALF the held-out regression. Narrowing is real but was ~50% self-inflicted.

**Status: MEASURED. The training defect was a major cause, not the only cause.**

Both experts retrained under §32's completion-only loss (postgresql v3: 48.4% of
batch masked; astral v3: 40.8%), then re-run through §31's held-out benchmark
unchanged.

| arm | v2 (prompt loss) | v3 (completion-only) |
| :--- | ---: | ---: |
| A base | 0.4667 | 0.4667 (identical — deterministic control) |
| B oracle | 0.2333 | **0.3444** |
| C self | 0.2333 | 0.3444 |
| edge vs base | **−0.2333** | **−0.1222** |
| 95% CI | [−0.3556, −0.1111] | [−0.2556, −0.0000] |

**Half the regression was the trainer, not the method.** The ~48% recovery
matches the ~48% of gradient that had been spent on prompt text — suggestive, not
proof, but the magnitudes line up.

### Per held-out construct

| construct | base | v2 | v3 |
| :--- | ---: | ---: | ---: |
| `LATERAL` | 0.000 | 0.000 | **0.333** (beats base) |
| `LEAD` | 0.500 | 0.000 | **0.500** (tied) |
| `FILTER (WHERE)` | 0.250 | 0.000 | **0.250** (tied) |
| `__slots__` | 1.000 | 0.000 | 0.667 |
| `functools.partial` | 1.000 | 0.000 | 0.333 |
| `DISTINCT ON` | 0.750 | 0.000 | 0.250 |
| `Protocol` | 1.000 | 1.000 | 0.667 (regressed) |
| `percentile_cont` | 0.500 | 1.000 | 0.500 (regressed) |

### What still stands, and what it implies

Experts remain BELOW base on held-out machinery, with the CI upper bound at
exactly −0.0000 — marginal, not robust. So narrowing is real and only about half
of it was the loss defect. Remaining suspects, in order:

1. **alpha = 128 at r=8 (scaling 16)** — untested against capability retention
2. **no retention/replay mixing** — the Factory has no general-data stage at all

### ⚠️ Consequence for previously REJECTED work

A defect worth half the held-out regression could have flipped any marginal
adapter-training result. Rejections that trained adapters under the defective
loss are now **suspect and cheap to re-test (~4-5 min each)**:

| rejection | why suspect |
| :--- | :--- |
| **§5 domain-adapted MTP drafting** | verdict was "adapters LOWER tau, a draft head must AGREE with its backbone". An adapter spending ~half its gradient learning question text is exactly an adapter that disagrees with its backbone. **The measurement and the defect are the same phenomenon.** Highest priority. |
| **§2 PiSSA / OLoRA / SVD init** | measured inert at matched alpha; an INITIALISATION advantage is what halved, diluted gradient would most easily mask |
| **§4 adapter stacking** | ast+fin −10.96pp interference, measured on adapters each wasting ~half their capacity on question format |
| **§13 orthogonalisation penalty** | penalised a subspace that was partly question-generation directions |
| 2-stage identity-Kronecker (+DoRA) | both arms shared the defect, so the relative call is more robust — mild |

**NOT affected** (timing/architecture, adapter-independent): AITER operators,
zero-copy boundary, hybrid radix caching (§11), ROCm kernel router, fused
non-linear ops, in-place swap kernel, APSP router and continuous batching (§6),
multi-token verify loop and native MTP engine (§26), RoPE/GDN sizing (§14), and
the 0.8B drafter (§27 — its verdict was depth-bound cost arithmetic).

§3 (shared-basis/VeRA) weakly SURVIVES: if half of every adapter encoded
question-format, which is shared across corpora, overlap should have been
elevated; it measured near chance (1.10–1.28x).

`results/adapters/m2_postgresql_r8a128_v3`, `results/adapters/m2_astral_r8a128_v3`

---

## §34 — §5 re-test: my hypothesis FAILED, but §5's framing is incomplete

**Status: hypothesis refuted at n=36; §5 neither reproduced nor refuted. In-domain
acceptance RISES, which §5's blanket claim does not accommodate.**

§33 flagged §5 as the highest-priority re-test: its verdict was *"all five adapters
lower tau (2.456 -> 1.531-1.988), a draft head must AGREE with its backbone, not be
domain-fluent"*, and §32 showed those adapters were trained with ~48% of the
gradient on question text — so the measurement and the defect might be the same
phenomenon.

Re-measured with `benchmark_mtp_acceptance_vs_adapter.py` (same instrument,
`ADAPTER_SET` override), K=6, offsets [0,8,16,24]:

| condition | overall (of 6) | vs base |
| :--- | ---: | ---: |
| un-adapted (base) | **2.44** | — |
| astral v2 (prompt loss) | 2.44 | −0.00 |
| astral v3 (completion-only) | 2.11 | −0.33 |
| postgres v2 | 2.33 | −0.11 |
| postgres v3 | 2.19 | −0.25 |

**The prediction was wrong.** Completion-only loss did not restore acceptance; it
trended slightly worse. The proposed mechanism did not appear.

**And the deltas are not resolvable.** n = 12 drafts/domain, **36/condition**,
against §5's **n=160**. Per-domain means span 1.42–3.92, so SE ~ 0.33 — the −0.33
"effect" is about one standard error. Every condition here sits within noise of
every other.

So §5 is **not refuted either**: base reproduces exactly (2.44 vs its 2.456), but
no adapted condition reaches its 1.531–1.988 range, and at n=36 that cannot be
distinguished from sampling.

### The unpredicted signal, which matters more

| adapter folded | financial | astral | postgres |
| :--- | ---: | ---: | ---: |
| none (base) | 2.08 | 2.75 | 2.50 |
| postgres v2 | 1.42 | 1.67 | **3.92** |
| astral v2 | 1.83 | 2.17 | 3.33 |

**In-domain acceptance RISES** — postgres adapter on postgres prompts scores 3.92
against base's 2.50, the largest effect in the table — while out-of-domain falls.
§5's framing ("a draft head must agree with its backbone, not be domain-fluent")
does not accommodate this: the adapted backbone makes the head BETTER inside its
domain. The averaging that produced §5's single number hides a sign flip.

That also matches §16/CURRENT.md's unresolved in-domain speculation result
(financial 1.011x, repeats 0.996–1.012), which straddles 1.0 rather than losing.

### What would settle it

n=160/condition as §5 used — roughly 45 minutes at this instrument's rate, versus
the ~10 minutes spent here. Worth doing only if speculation is reopened; §26
rejected the serving path on throughput grounds independent of tau.

`results/mtp_acceptance_v2_vs_v3.json`

---

## §35 — Admissible alpha window is real, measured, and per-adapter (alpha_opt = 96, not 128)

**Status: CONFIRMED. An earlier note in this session dismissed the alpha sweep as "the wrong objective". That was a framing error, corrected here.**

A lower bound is not the wrong objective — it is **half of a two-sided one**. Bounding alpha from below by floating-point physics and from above by held-out narrowing defines an admissible window with an interior optimum.

### 1. The physical axis is |dW|/|W|, not alpha
`results/alpha_absorption_sweep.json` ran at **rank_total = 64**, so its scaling = alpha/64. Our adapters are r=8, alpha=128 → scaling **16**. Its alpha column therefore does not transfer across ranks.

⚠️ **A claim made mid-session — that we sat "4x beyond the worst point measured" — was wrong for exactly that reason.** It compared alpha across different ranks.

Measured on our r=8 adapters:

| adapter | \|dW\|/\|W\| at alpha=128 | predicted merge_err |
| :--- | ---: | ---: |
| `m2_postgresql_r8a128_v3` | **0.0750** | 2.23% |
| `m2_astral_r8a128_v3` | 0.0741 | 2.25% |

Against the sweep's own perturbation axis:

| sweep point | \|dW\|/\|W\| | its quality |
| :--- | ---: | ---: |
| scaling 0.5 | 0.0451 | 85.8 (peak) |
| **ours, r=8 alpha=128** | **0.0750** | — |
| scaling 1.0 | 0.0901 | 75.0 |
| scaling 4.0 | 0.3578 | 58.3 (collapse) |

So alpha=128 was **suboptimal, not catastrophic** — 4.8x below the collapse point.
The fitted law reproduces all five sweep points to <5%:
`merge_rel_err_pct ≈ 0.167 / (|dW|/|W|)`

Because `dW = (alpha/r) · B@A`, **|dW|/|W| is exactly linear in alpha** — one post-train measurement yields the entire lower-bound curve at zero further cost.

---

## §36 — v4 corpora + reserved-construct gate (contamination eliminated)

**Status: CONFIRMED. Reservation by removal and verification eliminates eval contamination.**

### 1. What was broken
The held-out gate picked its test constructs by scanning the **v2** corpora for zero occurrences. The v3 corpora then trained **32 of the gate's 45 steps (71%)**, because `DISTINCT ON` / `FILTER` / `LATERAL` / `percentile_cont` / `singledispatch` / `TaskGroup` / `Protocol` / `__slots__` are standard advanced features.

### 2. The fix — reserve by REMOVAL, then verify
`scripts/corpus/reserve_eval_constructs.py` holds an explicit reserved-family list, strips those families from training, and **verifies zero residual occurrences**.

| corpus | records | dropped | result |
| :--- | ---: | ---: | :--- |
| postgresql v3 → **v4** | 1599 → **1385** | 214 (13.4%) | 11/11 constructs CLEAN |
| astral v3 → **v4** | 1605 → **1433** | 172 (10.7%) | 6/6 constructs CLEAN |

Verification caught a real leak: **`Protocol` survived at 103 hits** after removing `py_protocol_slots`, because `modern_typing` also emits it. That family is valuable general typing content, so `Protocol` was dropped as a *gate* construct and replaced with `cached_property` (verified at zero).

### 3. Controlled results — same gate, base identical at 0.3889
| adapters | base | oracle | edge | 95% CI | significant |
| :--- | ---: | ---: | ---: | :--- | :--- |
| v3, alpha=128 (control) | 0.3889 | 0.2889 | **−0.1000** | [−0.2111, +0.0000] | no |
| **v4, alpha=128** | 0.3889 | 0.3556 | **−0.0333** | [−0.2000, +0.1333] | **no** |
| v4, alpha=96 | 0.3889 | 0.3556 | −0.0333 | [−0.1889, +0.1111] | no |

The corpus work cut the regression by 67% (−0.1000 → −0.0333), and narrowing is no longer statistically detectable.

---

## §37 — REFUTED: Stacking scaling alpha/sqrt(K) at long horizon (2048 tokens)

**Status: REFUTED. alpha/sqrt(K) scaling collapses on long-horizon generation.**

On 192-token prompt/eval snippets, alpha/sqrt(K) appeared to mitigate interference. When evaluated at the true 2048 token generation context:
- Solo adapter adherence: 90.3%
- 2-Way Stacked (alpha/sqrt(2)): 57.7%
- 3-Way Stacked (alpha/sqrt(3)): 27.5%

Scaling down adapter alpha linearly with 1/sqrt(K) or 1/K dilutes domain signal below the activation activation threshold required to sustain multi-turn generation without falling back to generic base behavior or corrupting downstream formatting.

---

## §38 — REFUTED: Four geometric predictors of multi-adapter stacking damage

**Status: REFUTED across 4 separate geometric instruments.**

Four separate geometric heuristics proposed to predict multi-adapter stacking interference failed empirical verification:
1. **Weight-Space Subspace Overlap (SVD)**: Measured at 1.10–1.28x chance for *all* cross-task adapter pairs. Uniform across both high-interfering pairs (fin+pg) and low-interfering pairs (ast+pg).
2. **Activation Cosine Similarity (Own Domain)**: Measured at +0.0046 to +0.0176 across all pairs, failing to discriminate between constructive and destructive combinations.
3. **Weight Frobenius Norm**: Total ||dW||_F failed to correlate with degradation.
4. **Prompt Cross-Entropy Loss (Norm Meter)**: Unmasked evaluation on prompt tokens acts as an uncalibrated norm meter that penalizes any adapter weight perturbation.

The true predictive metric is **Merit vs. Conflict Decomposition** (`benchmarks/factory/geometry/activation_inertia_probe/probe_stacking_merit.py`):
`MERIT_X = (solo gain over base on X's domain) / (mean ||delta_X||/||h|| on peer domains)`
`CONFLICT_XY = cosine(delta_X, delta_Y) on shared domain prompts`

---

## §39 — L_inert Post-Mortem: Selective Silence Mechanics (v5 vs v5b)

**Status: DIAGNOSED. Loss formulation details decide whether selective silence works.**

1. **v5 Failure Mode**: `L_inert` was initially applied indiscriminately across all batch tokens, including in-domain prompt tokens. This resulted in a uniform 0.843x suppression across both in-domain and out-of-domain activations, leaving the Activation Selectivity Ratio (ASR) unchanged at 0.96x.
2. **v5b Replay Masking**: Applying `L_inert` strictly to out-of-domain replay batches while masking padding tokens restored selective penalization, lifting ASR toward target selective silence.
3. **Gradient Checkpointing Interaction**: Forward hooks computing activation norms must remain pure functions of their tensor inputs without mutable internal state branching across passes to prevent PyTorch `CheckpointError`.

---

## §40 — Architecture Canon: canon.py as the Single Source of Truth

**Status: SHIPPED. `src/gnn_experiment/canon.py` eliminates directory arithmetic failures.**

Using `Path(__file__).parent.parent` arithmetic breaks the moment scripts are reorganized into subdirectories (as occurred in the 32-file scripts/ reorg). `gnn_experiment.canon` establishes:
- Single canonical `REPO_ROOT`
- Centralized adapter directory paths (`REPO_ROOT / "results" / "adapters"`)
- Standardized metadata regimes (`regime.json`) and audit hooks.
Verified by `scripts/audit/check_canon.py`.
