# ❓ SVD-Guided Subspace Initialization — assessment before building

**Status: MEASURED 2026-08-17. The GPU time was spent. Answer: no win, do not adopt.**

> ## Result — the experiment this document proposed
>
> 10 matched runs (`scripts/old/compare_init_schemes.py`): 3 domains × {stock, pissa,
> olora} at α=128, plus a PiSSA alpha sweep and a **stock control at matched
> alpha** on astral. Only `init_lora_weights` and `alpha` varied. Step-1 losses
> agreed to ≤ 0.011, confirming the arms start from the same model.
>
> **(a) iteration win — NONE.** Neither scheme reaches the stock run's final loss
> within 150 steps, in any of the three domains. Nothing to bank.
>
> **(b) quality win — NONE.** At matched α=8 on astral (noise floor 0.0171):
> PiSSA **−0.009** (0.5× noise, indistinguishable), OLoRA **+0.019** (1.1× noise).
> Best config in the entire sweep is unchanged: **stock LoRA r=8 α=128**.
>
> **⚠️ The scary-looking numbers were a scaling artefact.** At the repo's α=128
> these schemes look catastrophic (PiSSA +0.127, OLoRA +0.557); at their
> paper-native α=r they are inert. **96.6% of OLoRA's apparent disaster evaporates
> at matched alpha.** PiSSA's deficit is monotonic in alpha (α=8 +0.024, α=32
> +0.054, α=128 +0.127). Never report these schemes without holding alpha fixed.
>
> **This also kills a plausible wrong story:** that both schemes init on
> `W0`-derived directions which the task doesn't use (1.37× chance), so they waste
> early steps unlearning them. It fits the α=128 numbers perfectly and it is not
> what is happening — hold alpha fixed and the init scheme does nothing at all.
>
> See `docs/DECISIONS.md` §2 and the ⚠️ retraction in §2 below — PiSSA's
> serving-compatible form is **rank 2r**, so it is not free downstream either.

---

## 1. What it actually is

"SVD-guided subspace initialization" is **PiSSA** (Principal Singular values and
Singular vectors Adaptation, Meng et al. 2024). [`docs/RESEARCH_ROADMAP.md`](../../../../docs/RESEARCH_ROADMAP.md) already names it.
Standard LoRA starts at `B=0, A~gaussian` so `dW=0`; PiSSA instead initialises
`A, B` from the top-r SVD of the base weight `W0`, and subtracts that component
from the frozen residual.

**It is already implemented in this repo's dependency tree.** peft 0.20.0:

```python
init_lora_weights: bool | Literal[
    "gaussian", "eva", "olora", "pissa", "pissa_niter_[N]",
    "corda", "loftq", "orthogonal", "mica"
]
```

Enabling it in `train/train_expert.py` is a one-line change:

```python
LoraConfig(..., init_lora_weights="pissa")     # or "pissa_niter_16" for the fast variant
```

So this was never an implementation project. Per this repo's own tiering rule —
*"Re-using a library exactly as its authors intended for the purpose they
intended → ⭐, not 🔥"* — adopting PiSSA is **⭐ Industry Standard**. The only
thing that could earn a tier here is the *measurement* on this stack.

## 2. The premise stated correctly

Your framing is right and worth keeping verbatim: **it helps convergence during
training and has exactly zero impact on runtime serving speed.**

> ### ⚠️ RETRACTED (2026-08-17) — the second half of this claim is wrong
>
> This section used to continue: *"A PiSSA-trained adapter is the same shape, same
> rank, same `dW`, folded by the same `WeightFoldingEngine` in the same 18 ms.
> Nothing downstream of training changes."* **Measured, it is not.**
>
> PiSSA trains against a **mutated base** (`W_res = W0 − scaling·B₀A₀`). This
> engine folds onto a **pristine `W0`** buffer, so a naively-saved PiSSA adapter
> adds the principal component twice. peft converts it back by emitting
> `dW = scaling·(B_trained A_trained − B₀A₀)` — and refactorising a *difference of
> two rank-r products* yields **rank 2r**:
>
> | | rank | alpha | on disk |
> | :--- | ---: | ---: | ---: |
> | production `m2_financial_r8a128` | 8 | 128 | **41 MB** |
> | PiSSA r=8, serving-compatible | **16** | **256** | **82 MB** |
>
> So the real trade is: **double the adapter and the fold FLOPs, or abandon the
> pristine-`W0` design the swap architecture rests on.** Not free. peft warns about
> this itself; `train/train_expert.py --init-lora-weights` now does the
> conversion automatically.

Faster adapter iteration is a stated goal of this project, so the training-side
benefit is a real benefit — it just has to be sized correctly.

## 3. Sizing the benefit — per run it is small, per sweep it is not

> `CURRENT.md`: 150 steps, r=8, alpha=128 — **~130 s for a 150-step run**

| unit of work | today | if PiSSA halves steps-to-target |
| :--- | ---: | ---: |
| one adapter | 130 s | ~65 s |
| one 5-alpha × 3-domain sweep (15 runs) | 32.5 min | **~16 min** |
| the 63 adapters already on disk | 136 min | ~68 min |

**A single run is not worth optimising; a sweep is.** The value lives in sweeps
and re-trains, where 130 s gets multiplied by 15–60. That is the regime this
project actually operates in — 63 adapters exist, 17 of them alpha-sweep members.

Two caveats decide whether that table is real or imaginary:

1. **"Halves steps-to-target" is an assumption, not a measurement.** It is exactly
   what the experiment in §5 exists to determine. It could equally be 1.1×, which
   would make the sweep saving ~3 minutes and not worth the config change.
2. **The saving is only bankable if `max_steps` actually gets cut.** Running PiSSA
   for the same 150 steps costs the same 130 s and buys no time at all — it would
   have to buy *quality* instead. Those are two different wins, and the experiment
   should report which one is on offer:

   > **(a) iteration win** — reaches the current final loss in fewer steps → cut
   > `max_steps`, bank the wall-clock.
   > **(b) quality win** — reaches a better loss/accuracy at the same 150 steps →
   > keep the budget, bank the quality.

Both are worth knowing, and they need different instruments: **(a) is
well-powered by the loss curve; (b) is what §4 shows this eval cannot resolve.**

Note that (a) is the *stated* PiSSA claim while (b) is not — so (b) must not be
justified by citing the paper's convergence plots.

## 4. The power problem — read this before designing the A/B

The obvious experiment is "train both, compare domain accuracy." On this eval
that experiment **cannot resolve the effect it is looking for.**

Observed 95% CI half-widths on the n=40 astral eval, taken from the stacking runs:

| comparison | 95% CI | half-width |
| :--- | :--- | ---: |
| `ast+fin` vs solo | [−21.50, −1.62] | **±9.9pp** |
| `ast+fin+pg` vs solo | [−6.63, +18.26] | **±12.4pp** |

`CURRENT.md` explains the shape: only 15–18 of 40 questions change at all, and
those that do swing the full [−100, +100] (sd 32–40pp). So this eval resolves
roughly **±10pp**.

A PiSSA-vs-LoRA difference at a fixed 150-step budget is plausibly **0–5pp**.
The measurement is underpowered by at least 2×. Running it and reporting "no
significant difference" would be uninformative — a null that was guaranteed by
the instrument, not by the method. This repo has already paid for that mistake
once (the retracted single-shot speculation disable).

## 5. If it gets measured, measure the right endpoint

The convergence claim should be tested on **training loss**, not downstream
accuracy. Loss is sampled every step over 150 steps, has low variance, and is
the quantity PiSSA actually claims to move.

**Primary endpoint** (well-powered):
- training loss at each step, PiSSA vs default init, ≥3 seeds each
- steps-to-reach a fixed loss threshold (e.g. the loss default-init hits at 150)

`CURRENT.md` already records reference trajectories: `1.941→0.859` and
`1.943→0.868` for two matched runs — so seed-to-seed spread on final loss is
~0.009, which makes loss a genuinely sharp instrument here.

**Secondary endpoint** (report with CI, never headline):
- final domain accuracy, paired bootstrap, acknowledged as ±10pp-resolution

**Cost:** 2 inits × 3 domains × 3 seeds × 130 s ≈ **40 GPU-minutes** for the
primary endpoint. That is cheap enough to be worth doing *if* the pre-flight gate
below passes.

## 6. The pre-flight gate (CPU, free)

`TODO.md` prescribes exactly this discipline:

> **Measure subspace overlap before building any further shared-basis scheme.**

PiSSA's premise is that `W0`'s **principal** singular directions are where the
important adaptation happens. This repo has already measured a lot about the
geometry of *updates* — that they are ~one independent direction per layer and
near-orthogonal across tasks (1.10–1.28× chance) — but **nobody has measured
whether updates align with `W0`'s top-r subspace**, which is the specific claim
PiSSA rests on.

[`probe_pissa_premise.py`](probe_pissa_premise.py) measures it on CPU, with no
training and no GPU: for each layer it takes the already-trained m2 `dW` and
computes how much of its energy lies in `W0`'s top-r subspace, against the
random-chance floor, using the `x_above_floor` convention already established by
`probe_subspace_overlap.py`.

| result | reading | action |
| :--- | :--- | :--- |
| `x_above_floor ≈ 1` | the learned update is no more aligned with `W0`'s principal subspace than a random matrix | **do not** spend GPU time on the training A/B |
| `x_above_floor` 2–10× | ambiguous | judgement call |
| `x_above_floor >> 10` | update concentrates exactly where PiSSA initialises | run the A/B |

**Known limit of the gate:** it is correlational. It measures where a
*LoRA-initialised* run ended up, which is not necessarily where a
*PiSSA-initialised* run would go — different inits can reach different optima. A
null is evidence against the premise on this model, not proof PiSSA cannot help.
It is a cheap filter on an expensive experiment, not a substitute for it.

**Prior probability is not favourable.** The repo's unifying geometric result is
that adaptation here occupies ~one direction per layer, mutually near-orthogonal
across depth *and* across tasks, with no shared low-dimensional manifold. A
model whose updates are that idiosyncratic is not obviously one whose updates
live in the base weight's dominant directions. The probe settles it either way.

## 6b. RESULT — the gate was run, and it fails

**Measured 2026-08-17** on `m2_astral_r8a128`, all 32 layers, 128 modules,
exact deterministic SVD (`results/pissa_premise_probe.json`):

> **Median alignment of the trained update with `W0`'s top-8 subspace:
> 1.37x the random-chance floor.**

Per module type (median x above floor):

| module | median | range |
| :--- | ---: | :--- |
| `down_proj` | 1.85x | see JSON |
| `gate_proj` | 1.29x | see JSON |
| `up_proj` | 1.31x | see JSON |
| `k_proj` | 1.06x | see JSON |
| `o_proj` | 1.43x | see JSON |
| `q_proj` | 1.72x | see JSON |
| `v_proj` | 1.10x | see JSON |

Spectral band profile — where in `W0`'s spectrum the update actually sits:

| band of `W0` | x above floor |
| :--- | ---: |
| top-8 | 1.38x |
| top-32 | 1.31x |
| top-128 | 1.16x |
| top-512 | 1.06x |

**There is no concentration anywhere in the spectrum.** Not in the top-8 that
PiSSA would initialise into, and not deeper either.

**Calibration against this repo's own scale:** `probe_subspace_overlap.py`
measured cross-task adapter subspaces at **1.10–1.28x chance** and this project
calls that *"statistically orthogonal"*. The update's alignment with `W0`'s
principal subspace is **1.37x** — essentially the same band. By the standard
already in use here, the trained update is near-orthogonal to the subspace PiSSA
starts from.

### Verdict

**Do not spend GPU time on a PiSSA training A/B on the strength of the published
claim.** The premise it rests on does not hold for this model. PiSSA also
*subtracts* the top-r component from the frozen residual, so initialising into a
subspace the update does not use is not a free bet.

This does not prove PiSSA cannot help (see the correlational caveat in §6). It
does mean the expected value no longer justifies the 40 GPU-minutes, and that the
iteration-speed win in §3 should be treated as unlikely rather than merely
unmeasured.

### ⚠️ Methodological warning that came out of this — reusable

The first version of this probe used `torch.svd_lowrank` and reported **1.28x**,
reaching the same verdict **by luck**. Randomised SVD is unreliable on these
weights because their spectrum is nearly flat (measured `s[0]/s[7] = 2.05` on
layer 0 `down_proj`), so no top-r subspace is well separated:

- five different seeds returned retentions spanning **2.56x** (1.01e-05 .. 2.60e-05)
- every one of them **underestimated** the exact value
- on that module the randomised estimate said ~1.3x above floor; exact says **6.59x**

That error is large enough to invert a verdict. The probe now computes the top-r
subspace **exactly** via the Gram matrix on the small side (validated to 0.001%
against full `linalg.svd`, and bit-identical across seeds).

**Checked for spillover: none.** `scripts/old/extract_svd_basis.py` and
`probe_subspace_overlap.py` both use exact `torch.linalg.svd` on a QR-reduced
matrix, so the existing SVD findings in this repo are unaffected.

**Rule: do not use randomised SVD on these weight matrices.** Their spectra are
too flat for it.

## 7. Recommendation

1. **Do not implement anything.** peft already has it; it is a config flag.
2. ~~Run the gate~~ **DONE — it fails at 1.37x chance (§6b).**
3. **The gate did not pass, so do not run the A/B** on PiSSA's published rationale.
   If it is ever run anyway (e.g. to test a different init like `olora` or `eva`),
   use training loss as the primary endpoint and treat accuracy as the
   ±10pp-resolution secondary it is.
4. **Tier it ⭐** regardless of outcome. A favourable measurement on this stack is a useful data point, not a novel contribution — PiSSA is published, peft ships it, and the serving path is untouched either way. The win, if it exists, is **iteration speed on sweeps**, which is a legitimate project goal — just not a novelty claim.
   a useful data point, not a novel contribution — PiSSA is published, peft ships
   it, and the serving path is untouched either way.

**Scheduling note:** the probe is CPU-heavy (randomised SVD on 2560×9216
matrices). `TODO.md` records that batch-1 decode on this box is CPU-bound on
kernel launches, so running it next to a decode benchmark can skew that
benchmark's tok/s. Run it on an idle machine.
