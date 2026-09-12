# Frontier ideas: live validation

Unlike `experiments/frontier/` (pure simulation, no model, no live module — see its own README), this script hits a **live, running** engine server over real HTTP and manages real server processes, to check whether "frontier" ideas actually hold up in a real request/response loop before anyone considers building them for real. Measures real GPU tok/s, TTFT, and Pass@1 accuracy across 3 real engineering challenge prompts, head-to-head against raw Ollama.

Kept as its own subfolder rather than merged into `experiments/frontier/` because it's categorically different — that folder is explicitly "no model in the loop," this one requires a live server.

## Run

Requires the target runtime server already running.

```bash
uv run python experiments/frontier_e2e_validation/run_real_empirical_frontier_benchmark.py
```
