# Rule: Model Evaluation & Judge Methodology — Production Usefulness Over Keyword Matching

## Core Principle
When benchmarking, evaluating, or acting as an AI judge comparing model outputs:
**Prioritize real-world engineering usefulness, executable correctness, and modern idiomatic standards over naive lexical keyword matching.**

## Guidelines for Evaluation & Scoring

1. **Engineering Judge Mindset**:
   - Evaluate responses from the perspective of a senior engineer pasting code into production.
   - Reward modern best practices (e.g., pgvector HNSW over IVFFlat, DuckDB native SQL `QUALIFY` over Pandas fallbacks, Astral `uv` toolchain over legacy pip/setuptools).

2. **Reject Keyword Shotgunning & Verbosity Bias**:
   - Do not reward models that achieve high keyword hit counts purely by generating multi-thousand-token encyclopedic essays that mention deprecated or obsolete approaches.
   - Penalize token bloat, conversational filler, unnecessary preambles, and irrelevant disclaimers.

3. **Rubric & Scorecard Standards**:
   - In benchmark summaries and head-to-head scorecards, weight **practical usefulness and idiomatic accuracy** highest.
   - Explicitly highlight when a concise, modern implementation is superior to a verbose output that merely matches keywords.

4. **Weight-Level Adaptation Over Prompt Scaffolding**:
   - Recognize that prompt engineering, dynamic skills, and MCP documentation servers suffer from the **Metacognitive Paradox**: as context expands, models drift and lack the self-awareness to re-query documentation, confidently emitting legacy code.
   - Prioritize **Weight-Level LoRA Adaptation ($W + \Delta W$)**: modern idioms (e.g. asyncpg `$1`, pgvector `<=>`, FastAPI lifespan, PEP 695 generics, DuckDB QUALIFY) are permanently embedded in neural logits with zero prompt bloat and zero token waste.

5. **Architectural Parity in Benchmarks (Raw Autoregressive vs. Speculative)**:
   - **Never compare pure autoregressive decode directly against speculative decode without explicit labeling**:
     - When comparing kernel throughput (e.g. Triton vs. GGML), both arms MUST run in pure autoregressive mode (disabling `--spec-type`).
     - When benchmarking end-to-end system throughput, explicitly delineate:
       1. **Raw Kernel Generation**: (tok/s without speculation) — measures true GEMV memory efficiency and state scan speed.
       2. **Speculative Generation**: (effective tok/s with draft verification) — measures speculative acceptance rate $\times$ raw kernel throughput.
   - A baseline running with 4-draft MTP speculation (~30 tok/s) will always outpace a single-token autoregressive loop (~18 tok/s) solely due to token burst verification, not raw memory efficiency.

