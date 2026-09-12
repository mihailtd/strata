# Renko Brick Smoothing for Dynamic Team Routing

Loads a real `Qwen3.5-9B` model, folds a `WeightFoldingEngine` around it, captures a real CUDA graph decoder, and benchmarks `RenkoBrickSmoother` (`apps/runtime/dynamic_team_router.py`) — the epsilon-box hysteresis mechanism that prevents `dynamic_team_router`'s expert routing from flapping between adjacent experts on small latent perturbations.

See [`docs/RENKO_SMOOTHING.md`](../../docs/RENKO_SMOOTHING.md) for the full write-up of what Renko brick smoothing is and why it's needed.

## Run

```bash
uv run python benchmarks/renko_smoothing/renko_smoothing_benchmark.py --epsilon 5.0 --max-tokens 400
```
