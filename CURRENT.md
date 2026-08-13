# CURRENT

What is live right now: which trainer, which adapters, which findings hold, and
what has been retracted. Companion to `SYSTEM.md` (hardware/software snapshot).

**Read this before citing any number or running any benchmark.** Several results
in this repo were invalidated by using an adapter that was not what its script
claimed. The registry below exists to prevent that.

Last updated: 2026-08-13

---

## Training methodology

| id | precision | fused kernels | status |
| :--- | :--- | :--- | :--- |
| **m2** | bf16 | Liger (`fused_linear_cross_entropy`, `rms_norm`, `swiglu`; rope off) | **CURRENT** |
| m1 | 4-bit NF4 | none | legacy — do not use |

**Use `scripts/train_expert_CURRENT_m2.py` for all new expert training.**
Exactly one trainer is CURRENT at a time and its filename says so.

```bash
uv run --env-file .env scripts/train_expert_CURRENT_m2.py --domain astral
```

Fixed hyperparameters across every domain, so experts stay comparable:
r=8, alpha=128 (scaling 16), 7 projections, 150 steps, batch 2, grad-accum 2,
lr 2e-4, cosine, max_length 512.

**Why m1 is dead.** It trains against a 4-bit NF4 base while every folding,
speculation and stacking benchmark loads bf16 — so adapters learned a correction
to quantized weights and were then folded into unquantized ones. The seam is
measurable: the financial expert moved from **−5.00pp to +4.17pp** on its own
domain when retrained bf16, with a data change that altered zero rubric terms.

**To iterate the methodology:** rename this file to `train_expert_m2.py`, move it
to `scripts/superseded/`, create `train_expert_CURRENT_m3.py` with
`METHODOLOGY = "m3"`. Adapters then get named `m3_<domain>_r8a128`.

**Liger notes (verified, not assumed).** The patch must be applied *before*
`from_pretrained` — with `model=None` it rebinds at class level, so reversing
those two lines makes it silently do nothing. `rope` stays False because liger
raises `NotImplementedError` for Qwen3.5; this is a blanket opt-out, **not**
per-layer detection of the hybrid attention mix. Numerically equivalent: liger's
RMSNorm uses `offset=1.0` + `casting_mode="gemma"`, matching stock
`output * (1.0 + weight.float())` then cast; loss trajectories match a non-Liger
run (1.941→0.859 vs 1.943→0.868). Speedup: ~130 s for a 150-step run.

---

## Current adapters

`m2_<domain>_r8a128` is the live expert set — **verified methodology-matched**
across methodology, liger, precision, rank, alpha, max_steps and lr:

| adapter | methodology | liger | scaling | steps | records |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `m2_astral_r8a128` | m2 | yes | 16.0 | 150 | 815 |
| `m2_postgresql_r8a128` | m2 | yes | 16.0 | 150 | 411 |
| `m2_financial_r8a128` | m2 | yes | 16.0 | 150 | 304 |

Verify before use:

```bash
uv run python scripts/audit_adapters.py          # drift check + current roles
uv run python scripts/audit_adapters.py --write  # re-record after training
```

`ADAPTER_MANIFEST.json` (repo root, tracked) records a content hash per adapter.
If an adapter changed under a published result, the drift check says so — this is
the mechanism that was missing when `ctl_lora_fin_a128` was overwritten three
times in one day while the README kept describing the second version.

Legacy adapters are kept for reproducibility of committed findings:
`ctl_lora_r8_a*` / `ctl_idk_*` back the controlled head-to-head; `*_sweep_a*`
back the absorption law. **All are m1 (4-bit).** Do not mix them with m2 experts
in one comparison.

---

## Findings that hold

| finding | number | where |
| :--- | :--- | :--- |
| Batching is the largest lever | **~6–7×** aggregate at B=8 (6.83 and 6.18 on two runs) | `benchmark_batch_scaling.py` |
| Weight folding beats wrapped execution | **1.83×** vs peft wrapper; 1.21× vs `NovelLoraLinear` | `benchmark_batch_scaling.py` §3 |
| Speculative decoding works via fla | **~1.14×**, uniform across domains | `benchmark_mtp_indomain_speculation_matrix.py` |
| bf16 merge-absorption law | `merge_rel_err ≈ 0.167 / (‖dW‖/‖W‖)` | `benchmark_alpha_absorption_sweep.py` |
| Cross-task adapter subspaces are near-orthogonal | 1.10–1.28× chance | `probe_subspace_overlap.py` |
| id_kron ≈ stock LoRA at matched scaling | parity; LoRA more stable (id_kron diverges past scaling ~2) | `eval_controlled_headtohead.py` |
| Prefill is a small share of turn time | 1.5% @2k, 3.0% @4k, **5.9% @8k** | `benchmark_prefill_share.py` |

**Two folding numbers, two questions.** 1.83× is escaping peft's wrapper; 1.21×
is escaping `NovelLoraLinear`. Folded decode is ~29.2 ms against a 28.7 ms bare
model — folding itself is free. Never quote them as one number.

---

## Retracted / closed — do not retry

| claim | what actually happened |
| :--- | :--- |
| "In-domain speculation penalty; disable astral" | Adapters were id_kron mislabelled as Stock LoRA; single measurement; contradicted its own break-even rule. Corrected: diagonal 1.138× vs off-diagonal 1.140×, **all 9 cells win**, router all-True. |
| "Speculative loop diverges from its verifier (r=+0.44)" | The gate was broken, not the loop. `chunked_reference()` teacher-forces a full forward while the verifier steps incrementally. Controls: spec-vs-greedy **80.0%** against a *non-speculative* control of **83.3%**, determinism 100%. 100% exactness is unreachable here. |
| "Hybrid Radix caching is MANDATORY, prefill = 54%" | ~33 s of Triton JIT landed inside the timed region. 2× context gave 1.06× time — impossible as compute. Corrected to 5.9% @8k. **Off the critical path.** |
| "Stacking retention 144–247%" | Degenerate statistic: financial's solo gain is +4.17pp = **0.83 questions** at n=20. Values >100% mean the denominator broke. Superseded by absolute pp deltas with paired bootstrap CIs. |
| "MTP head adaptation improves drafting" | All five adapters **lower** acceptance (τ 2.456 → 1.531–1.988), every CI excludes zero. A draft head must *agree with its backbone*, not be domain-fluent. Retrying needs a new objective (distil backbone outputs), not tuning. |
| "Fold the MTP head to cut draft latency" | Nothing is wrapped — the head loads bare checkpoint tensors, no peft layer. Drafting is ~27% of a round, so the Amdahl ceiling is ~1.14× anyway. |
| Multi-expert chain "zero-copy" results | Fabricated; no `runs.jsonl`, impossible key structure. Deleted. |

---

## Open questions

1. **Does stacking degrade experts?** PARTIALLY ANSWERED (m2 matched set,
   astral, n=40, paired bootstrap; base 6.04% -> solo 56.91%, a +50.87pp gain =
   20.3 question-equivalents, the best-powered stacking run made here).

   | stack | astral | vs solo | 95% CI | verdict |
   | :--- | :---: | :---: | :---: | :--- |
   | `ast` | 56.91% | — | — | baseline |
   | `ast+fin` | 45.95% | **-10.96pp** | [-21.50, -1.62] | **RESOLVED loss** |
   | `ast+fin+pg` | 62.66% | +5.74pp | [-6.63, +18.26] | not resolvable |

   A specific pair interferes. It does **not** decay monotonically — adding a
   third expert moves the score back *above* solo. Any "stack at most N" rule is
   unsupported; generalising from one resolved cell would repeat the error that
   produced the retracted 144.4% retention claim.

   Wide CIs are the metric's shape, not sample size: only 15-18 of 40 questions
   change at all, but those that do swing the full [-100, +100] (sd 32-40pp),
   because the good/bad term ratio is bimodal per question.
2. **Is the m1→m2 seam large elsewhere?** Financial moved +9.17pp. Astral and
   postgres now retrained under m2; comparing them against their m1 versions
   measures it.
3. **Liger speedup, unverified.** ~130 s/run is plausible but has no measured
   non-Liger baseline. `--no-liger` exists for the A/B.

---

## Rules learned the hard way

- **Scaling behaviour catches broken timings.** 2× context giving 1.06× time
  exposed the prefill artifact; absolute values looked fine.
- **Never hardcode `alpha/rank`.** Derive `rank_total` from tensors — id_kron uses
  `rank_in × rank_out`. Hardcoding produced the retracted matrix.
- **Verify a flag is accepted before a long run.** A silent no-op patch let a run
  report a configuration it did not execute.
- **Re-measure baselines in-process**, never against a remembered number.
- **Inferring "what's in use" from source is unreliable** — four consecutive
  detection bugs (`{}` templates, f-strings, bare templates, filter regex), twice
  producing lists containing adapters that back live findings. Use the explicit
  registry.
- **`git add -A` sweeps others' work into your commit.** Stage explicit paths.
