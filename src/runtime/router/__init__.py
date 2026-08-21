"""VRAM expert state and the measured transition-cost model.

Deliberately contains no scheduler and no shortest-path solver:

- shortest-path routing is provably vacuous under this cost model —
  see `benchmarks/superseded/apsp_floyd_warshall/`
- request scheduling has nothing to schedule in a single-tenant engine —
  see `benchmarks/superseded/sla_bounded_cluster_scheduler.py` and
  `docs/DECISIONS.md` §6
"""

from runtime.router.vram_state_router import (
    TransitionCosts,
    VRAMState,
    VRAMStateGraph,
)

__all__ = [
    "TransitionCosts",
    "VRAMState",
    "VRAMStateGraph",
]
