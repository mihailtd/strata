# Thinking supervisor: synthetic reasoning-control benchmark

Simulates autoregressive generation with scripted logit/latent-state dynamics (an "attractor loop" scenario, a "cognitive convergence" scenario, and an unconstrained baseline) to exercise the **real** `ThinkingRuntimeSupervisor` (`apps/runtime/thinking_supervisor.py`) — its actual `process_step()` and `notify_token_emitted()` decision logic, budget enforcement, and forced-transition injection.

This is a stage-1 experiment: token logits are random (`torch.randn`), not from a real model forward pass, and the scenario dynamics (when the "loop" or "convergence" pattern kicks in) are scripted, not observed. What's real is the supervisor's control logic being exercised under synthetic stress scenarios — does it enforce the token budget, break the attractor loop, and exit immediately on convergence.

Moved here from `benchmarks/` (misclassified — synthetic control-logic check, not a live measurement).

## Run

```bash
uv run python experiments/runtime/thinking_supervisor/benchmark_thinking_supervisor.py
```
