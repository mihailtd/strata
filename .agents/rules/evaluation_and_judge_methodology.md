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
