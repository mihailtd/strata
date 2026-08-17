# ❓ SVD-Guided Subspace Initialization — assessment before building

**Status: nothing to build. The decision is whether to spend GPU time measuring it.**

---

## 1. What it actually is

"SVD-guided subspace initialization" is **PiSSA** (Principal Singular values and
Singular vectors Adaptation, Meng et al. 2024). `IDEAS.md:57` already names it.
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

Enabling it in `train_expert_CURRENT_m2.py` is a one-line change:

```python
LoraConfig(..., init_lora_weights="pissa")     # or "pissa_niter_16" for the fast variant
```

So this was never an implementation project. Per this repo's own tiering rule —
*"Re-using a library exactly as its authors intended for the purpose they
intended → ⭐, not 🔥"* — adopting PiSSA is **⭐ Industry Standard**. The only
thing that could earn a tier here is the *measurement* on this stack.

## 2. The premise stated correctly

Your framing is right and worth keeping verbatim: **it helps convergence during
training and has exactly zero impact on runtime serving speed.** A PiSSA-trained
adapter is the same shape, same rank, same `dW`, folded by the same
`WeightFoldingEngine` in the same 18 ms. Nothing downstream of training changes.

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

## 7. Recommendation

1. **Do not implement anything.** peft already has it; it is a config flag.
2. **Run the CPU gate** ([`probe_pissa_premise.py`](probe_pissa_premise.py)) when
   the machine is idle. Cost: minutes, no GPU.
3. **Only if the gate passes**, run the loss-curve A/B (~40 GPU-min), with
   training loss as the primary endpoint and accuracy as a clearly-underpowered
   secondary.
4. **Tier it ⭐** regardless of outcome. A favourable measurement on this stack is a useful data point, not a novel contribution — PiSSA is published, peft ships it, and the serving path is untouched either way. The win, if it exists, is **iteration speed on sweeps**, which is a legitimate project goal — just not a novelty claim.
   a useful data point, not a novel contribution — PiSSA is published, peft ships
   it, and the serving path is untouched either way.

**Scheduling note:** the probe is CPU-heavy (randomised SVD on 2560×9216
matrices). `TODO.md` records that batch-1 decode on this box is CPU-bound on
kernel launches, so running it next to a decode benchmark can skew that
benchmark's tok/s. Run it on an idle machine.
