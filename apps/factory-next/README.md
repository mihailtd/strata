# factory-next — notes, not a project

**This is a parking lot for reasoning, not a scaffolded project.** Unlike
`apps/runtime-next` (a real Cargo stub with committed intent to actually
build it), there is no decision to rewrite `apps/factory` (the LoRA/PEFT
training pipeline) in Rust. This folder exists only because a hypothetical
came up worth recording, and it earns a promotion to real scaffolding only
if a concrete decision to build it is made later — the same way
`runtime-next` earned its scaffolding.

Everything below was held to the same bar as `apps/runtime-next/ECOSYSTEM_NOTES.md`:
a claimed benefit needs to trace to something actually verified in this
project's own code or to real, cited third-party evidence — not to generic
"Rust is better" reasoning. Two benefits cleared that bar; a third is real
evidence but explicitly unproven *for this project's own workload*; several
plausible-sounding arguments were checked and discarded.

## Where this came from

Evaluating `flodl` (a Rust training framework on libtorch) for `runtime-next`
surfaced that its actual domain — training, not inference — is a closer
match for `apps/factory` than for the inference server. flodl itself isn't
a fit even there (no support for Qwen3.5's hybrid GatedDeltaNet architecture,
and its own authors state ROCm has never run a real training step on any
AMD GPU as of v0.8.0) — see `apps/runtime-next/ECOSYSTEM_NOTES.md` for that
evaluation in full. What's recorded here is the separate question it raised:
independent of flodl specifically, would a Rust rewrite of the training side
itself be worth it?

## Real benefit #1: deterministic GPU memory release would remove an actual, currently-manual reliability risk

This isn't hypothetical — it's the current code, checked directly:

- `apps/factory/train_all_27b_experts.py:150-153` trains multiple domain
  experts sequentially in one process and calls `gc.collect()` +
  `torch.cuda.empty_cache()` between each domain, because Python/PyTorch
  does not reliably release the previous domain's model and optimizer-state
  VRAM on its own.
- `apps/factory/calibrate_expert_alpha.py:92` does the same with an explicit
  `del model` inside a calibration sweep loop.
- `apps/factory/train_expert.py` imports `set_hard_vram_cap` from the
  runtime's own VRAM-policing module — the same manual-cap machinery
  `apps/runtime-next/ECOSYSTEM_NOTES.md` already flagged as scar tissue on
  the inference side.

None of this is a "hope it works" concern in the sense of having actually
failed — but it *is* a best-effort pattern (garbage-collect, then hope the
allocator actually gives the memory back) standing in for a guarantee. In
Rust, tying a model/optimizer's GPU buffers to ownership and `Drop` makes
release deterministic and immediate at scope exit, not a hint to a garbage
collector. This would delete this category of code, not reimplement it —
same argument already made for `runtime-next`, independently verified here
against `apps/factory`'s own actual scripts rather than assumed to transfer.

## Real benefit #2 (conditional): a correct `runtime-next` forward pass would cover most of the hard part of a training port too

If `apps/runtime-next` is ever actually built out to a working, validated
Rust implementation of Qwen3.5's forward pass (GatedDeltaNet chunked scan,
the gated RMSNorm variant, the custom conv1d layer — the architecture-
specific work that's hard regardless of language), that implementation
would cover most of what a training rewrite also needs on the forward side.
Backward-pass/autodiff would still be new, substantial work — gradient
correctness is a materially higher bar than forward-only inference — but it
would build on an already-debugged forward pass instead of starting from
zero on both at once.

**This is conditional, not current**: `runtime-next` is still a stub
(`src/main.rs` prints a startup message). This benefit exists only in the
future where that work actually gets done — it is a reason to sequence
`runtime-next` before `factory-next`, not a reason to start `factory-next`
now.

## Real evidence, but unproven for this project specifically: framework-overhead reduction

flodl measured real, third-party numbers: 8 wins / 2 ties / 0 regressions
against PyTorch 2.10 across 10 models, up to 31% faster on the models tested,
using the *same* libtorch kernels underneath (their own framing: "the
convnet tie proves both frameworks dispatch identical CUDA kernels — the
speed gap is pure framework overhead"). That's real evidence that a Rust
training loop can meaningfully cut Python-interpreter/dispatch overhead
without touching kernels at all.

**What's NOT established**: whether that overhead is a meaningful fraction
of *this project's* actual training step time. flodl's benchmark models
(transformer/mlp/residual_tower/feedback_fixed/convnet) are small enough
that per-op dispatch overhead is a visible fraction of total time. LoRA
fine-tuning on a multi-billion-parameter model is plausibly far more
GPU-kernel-time-dominated per step, where shaving interpreter overhead off
the host side matters proportionally less. This project has never measured
its own factory training steps to find out. Per this project's own standing
rule (performance claims arrive with numbers or they are not made — see
`apps/runtime-next/ECOSYSTEM_NOTES.md`'s corroboration notes from both
renew-engine and flodl on exactly this point), this is logged as "worth
measuring before assuming," not as a benefit banked as real.

## Checked and discarded

Argued out loud and rejected as not actually well-supported here, so they
don't get silently re-argued the same way later:

- **"Rust's type system would prevent bugs like the ones found this
  session."** Checked against what those bugs actually were: wrong dict
  keys (`training_db.update_airm_metrics` reading `scale_comp` instead of
  `final_scale`), a wrong hardcoded filename (`evaluation_data.jsonl` vs.
  the real `evaluation_data_disposition.jsonl`), a stale hardcoded ground-
  truth table that didn't match any on-disk artifact. None of these are
  type errors — they're wrong string literals and stale data, which Rust's
  type system does not prevent any better than Python's. Not a real
  argument for a rewrite.
- **"Simpler deployment / dependency management."** `apps/factory` already
  uses `uv` + `moon`, a genuinely disciplined toolchain (this project's own
  benchmarking conventions memory notes this explicitly). No evidenced
  current pain point here to fix.

## Blockers, stated honestly

- Qwen3.5's hybrid architecture (GatedDeltaNet + attention, custom conv1d,
  gated RMSNorm) has no off-the-shelf Rust training-framework support found
  so far (flodl included). Building it means implementing both forward
  *and* correct backward passes for a nonstandard architecture from
  scratch — a large, high-risk undertaking, and harder than the inference-
  only version `runtime-next` needs.
- Every Rust ML framework checked so far treats AMD/ROCm as, at best,
  compiles-and-links-tested — not proven on real training workloads. This
  is a harder bar to clear for training (needs correct gradient kernels,
  not just correct forward kernels) than for inference.
- No current measurement establishes that Python/framework overhead is
  actually a meaningful cost in `apps/factory`'s real training runs (see
  above). Absent that, the strongest *established* benefit here is
  reliability (benefit #1), not speed.

## Status

Not started. Nothing here is scoped, sized, or scheduled. Revisit if/when
`runtime-next` produces a real, validated Qwen3.5 forward pass (benefit #2
stops being conditional), or if a factory training run is ever profiled and
framework overhead turns out to be a real fraction of its wall-clock time.
