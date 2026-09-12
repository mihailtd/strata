# Experiments — Stage 1: does this idea work at all?

Pre-integration work: synthetic microbenchmarks (isolated kernel/operation timing, no real model in the loop) and the quick empirical follow-ups that prove an idea can survive a real run before anyone wires it into `apps/runtime*` or `apps/factory`. See the repo's [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) for the full 3-stage lifecycle this is stage 1 of, and the classification rule that decides whether something belongs here or has graduated to [`benchmarks/`](../benchmarks/).

**Nothing here is a production claim.** A script in this tree measuring 3x is a reason to consider building the real thing, not a result to cite.

## Layout

```
experiments/
  factory/           geometry & training-methodology R&D with no live apps/factory module yet
    geometry/          activation-inertia, orthogonality, POET decomposition, etc. probes
  runtime/
    speculative/       candidate gating/hedging strategies not yet wired into the shipped speculative path
  frontier/            pure simulations of 4 "frontier" performance ideas -- no model, no live module
  agentic/             mock-engine prototypes for multi-agent tool-pipeline speculation
  <loose files>        synthetic single-file microbenchmarks (kv-cache, batched prefill, HIP graph capture, ...)
```

## Graduating out of here

When a live `apps/runtime*` or `apps/factory` module actually implements what a script here explores, that script (and its README, and any test pinning it) moves to [`benchmarks/`](../benchmarks/) — it's no longer measuring a hypothesis, it's measuring a shipped feature. If an idea instead turns out not to work, it moves to [`benchmarks/superseded/`](../benchmarks/superseded/) with a retirement rationale, same as anything else in this repo — see `docs/METHODOLOGY.md`.

Tests pinning a probe's correctness live right next to it (e.g. `factory/geometry/latent_variable_glasso/test_latent_variable_glasso.py`), not in a separate tree — see the root `tests/README.md`.
