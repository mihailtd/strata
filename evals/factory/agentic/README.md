# Agentic capability & held-out evaluation

Task/quality evaluation of trained adapters' agentic behavior — tool selection disposition, format adherence, multi-step chained tool use, domain-handoff gating, and held-out capability retention. Moved here from `benchmarks/factory/agentic/` (the split point: does this measure a shipped feature's performance, or does it score a trained adapter's correctness on a task? These are the latter). See [`evals/README.md`](../../README.md) and the repo's [`docs/METHODOLOGY.md`](../../../docs/METHODOLOGY.md).

`benchmarks/factory/agentic/` kept the entries that measure an integrated router feature (`dynamic_morphing`, `poet_tool_causal_graph`) rather than scoring task correctness.

| Submodule | Tier | Key Contribution |
| :--- | :---: | :--- |
| **[`disposition/`](disposition/)** | **🔥 Applied Practice** | **Tool Selection Disposition Benchmark**: 4-arm token efficiency and native tool preference evaluation (`base`, `sysprompt`, `expert`, `both`). |
| **[`adherence/`](adherence/)** | **⭐ Standard** | **Format & Constraint Adherence**: Measures strict JSON, TOML, and tool signature compliance across domain adapters. |
| **[`chained_holdout/`](chained_holdout/)** | **⭐ Standard** | **Multi-Step Chained Evaluation**: Sequential tool chain execution on held-out tasks. |
| **[`handoff_gate/`](handoff_gate/)** | **🔥 Applied Practice** | **Domain Handoff Routing Gate**: Evaluates sub-millisecond expert adapter switching during dynamic agent workflows. |
| **[`indomain_reserved/`](indomain_reserved/)** | **⭐ Standard** | **In-Domain Capability Retention**: Strict evaluation preventing catastrophic forgetting on core domain tasks. |
| **[`attribution/`](attribution/)** | — | Crediting which trained adapter caused which behavior/output. |
