# Agentic Capability & Multi-Turn Benchmark Suite

This directory contains the benchmarks, telemetry probes, and evaluation ladders measuring multi-turn agentic behaviors, tool selection disposition, adherence, handoff gating, and causal workflow structures.

---

## Submodule Overview

| Submodule | Tier | Key Contribution |
| :--- | :---: | :--- |
| **[`disposition/`](disposition/)** | **🔥 Applied Practice** | **Tool Selection Disposition Benchmark**: 4-arm token efficiency and native tool preference evaluation (`base`, `sysprompt`, `expert`, `both`). |
| **[`adherence/`](adherence/)** | **⭐ Standard** | **Format & Constraint Adherence**: Measures strict JSON, TOML, and tool signature compliance across domain adapters. |
| **[`chained_holdout/`](chained_holdout/)** | **⭐ Standard** | **Multi-Step Chained Evaluation**: Sequential tool chain execution on held-out tasks. |
| **[`handoff_gate/`](handoff_gate/)** | **🔥 Applied Practice** | **Domain Handoff Routing Gate**: Evaluates sub-millisecond expert adapter switching during dynamic agent workflows. |
| **[`indomain_reserved/`](indomain_reserved/)** | **⭐ Standard** | **In-Domain Capability Retention**: Strict evaluation preventing catastrophic forgetting on core domain tasks. |
| **[`poet_tool_causal_graph/`](poet_tool_causal_graph/)** | **🚀 Genuine Discovery** | **POET + NOTEARS Causal Graph Discovery**: Proved that static factor subtraction destroys causal cascade propagation in Structural Equation Models; NOTEARS directly on event sequences achieves **$90\%$ TPR** and minimal SHD error. |
