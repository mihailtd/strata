# Fabricated root-level "benchmarks"

Six scripts that used to sit loose in `benchmarks/` root, each presenting at least one side of a head-to-head comparison as a live measurement when it was actually a hardcoded constant or a made-up formula. Found while reorganizing `benchmarks/`'s loose root files into proper subfolders (see `benchmarks/README.md`) — moving them into a real subfolder would have implied they belonged there as legitimate results, so they were checked first.

See `benchmarks/superseded/README.md` for this repo's general retirement policy. Full detail is in each file's own retirement docstring; summary:

| Script | What was fabricated |
| :--- | :--- |
| `benchmark_qwen27b_gguf_inference.py` | Real GGUF header parse + real kernel timing, but on dummy zero-filled weights — then falsely reports the on-disk GGUF file size as "Allocated in GPU VRAM". |
| `benchmark_raw_unsupervised_comparison.py` | Real Ollama baseline loaded from disk; "our" side is a hardcoded `tok_s` constant and made-up TTFT formulas. |
| `benchmark_w4a16_vs_ollama_head_to_head.py` | Real Ollama baseline; "native" side literally commented "Simulate Native Engine Telemetry" — hardcoded token counts, hardcoded decode speed, unused `ThinkingRuntimeSupervisor` instance. |
| `run_10turn_cross_domain_benchmark.py` | Real Ollama HTTP calls; native side computed from a hardcoded `native_tok_s = 66.40`. |
| `run_4turn_lora_vs_ollama_eval.py` | Same pattern as the 10-turn script, at 4 turns. |
| `run_supercharged_2x_ollama_benchmark.py` | Real Ollama baseline (read from the 10-turn script's output); "supercharged" side is a hardcoded constant under a comment titled "Specifications", combining features that include a kernel (`Fused SwiGLU`) that never existed anywhere in this repo. |

None of these were cited by any live doc (checked before retiring), so no documentation needed correcting as a result — the fabricated numbers never made it further than these scripts and their own output JSONs (deleted).
