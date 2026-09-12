# Agentic router feature benchmarks

Performance measurement of integrated agentic-routing features — dynamic expert-team morphing and the POET+NOTEARS causal-graph scheduler (both live in `apps/runtime/dynamic_team_router.py` / `cut_set_router.py`). Task/quality evaluation of trained adapters' agentic *behavior* (disposition, adherence, chained-holdout, handoff gating, attribution) moved to [`evals/factory/agentic/`](../../../evals/factory/agentic/) — see the repo's [`docs/METHODOLOGY.md`](../../../docs/METHODOLOGY.md) for the split rationale.

---

## Submodule Overview

| Submodule | Tier | Key Contribution |
| :--- | :---: | :--- |
| **[`dynamic_morphing/`](dynamic_morphing/)** | — | Dynamic expert-team co-activation via the Riemannian Team Router (`apps/runtime/dynamic_team_router.py`). |
| **[`poet_tool_causal_graph/`](poet_tool_causal_graph/)** | **🚀 Genuine Discovery** | **POET + NOTEARS Causal Graph Discovery**: Proved that static factor subtraction destroys causal cascade propagation in Structural Equation Models; NOTEARS directly on event sequences achieves **$90\%$ TPR** and minimal SHD error. |
