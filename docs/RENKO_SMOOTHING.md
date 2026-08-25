# Renko Brick Smoothing for Continuous Latent LoRA Routing

## Background

In streaming generation architectures that use dynamic LoRA expert routing, the engine continuously evaluates the active context to determine which expert (or blend of experts) should be active. 

However, naively evaluating every single generated token causes severe high-frequency jitter. Common words (`"the"`, `"and"`) or punctuation (`","`, `"."`) can temporarily shift the semantic focus of the prompt, triggering false-positive adapter swap requests. Swapping adapters is mathematically expensive and destroys KV cache continuity, severely degrading throughput.

## The Riemannian Solution (Chapter 4)

To solve this, we adapt the concept of **Renko Bricks** from financial charting to the Riemannian latent space of the LLM. 

Instead of evaluating routing on every token or using an arbitrary token-count timer, the router accumulates the mathematical displacement of the hidden states in continuous space:

$$D_t = D_{t-1} + \Vert h_t - h_{\text{last\_brick}} \Vert_2$$

A routing re-classification event is **only emitted** when the accumulated displacement breaks a discrete boundary ($\Delta D \ge \epsilon_{\text{box}}$). 

### Benefits
1. **Zero High-Frequency Jitter**: Commas and articles do not possess enough mathematical magnitude in the latent space to break the brick boundary, naturally filtering out noise.
2. **Instant Semantic Reaction**: If the prompt sharply shifts topics (e.g., transitioning abruptly from writing Python code to explaining PostgreSQL schemas), the distance between hidden states explodes, instantly breaking the boundary and triggering an expert swap precisely when it is needed.
3. **No Arbitrary Timers**: The system routes based on true semantic drift, not an arbitrary "every 10 tokens" rule.

## Implementation Details

- **`RenkoBrickSmoother`**: Located in `src/runtime/dynamic_team_router.py`. Tracks the `last_brick` tensor and evaluates incoming hidden states via `smoother.step(h_t)`.
- **Latent Exposure**: The CUDA graph decoder (`src/runtime/cuda_graph.py`) has been modified to capture and yield `h_t` out of the generation stream alongside the decoded text.
- **Server Hook**: `src/runtime/server.py` evaluates every token against the Smoother. When a boundary breaks, it triggers a latent evaluation event.

*Note: The current server implementation logs the boundary breaks for telemetry. The final step is to hook these break events into a trained latent clustering model to execute the physical LoRA swaps.*
