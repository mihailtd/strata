"""VRAM expert state + the measured transition-cost model.

This module holds two things and deliberately no scheduler.

WHAT THE HARDWARE ACTUALLY DOES
-------------------------------
`WeightFoldingEngine.activate(e)` writes `W_live = W0 + s*(U@V)` as ONE fused
addmm per slot, reading from the pristine buffer. It does not restore first and
it does not read the currently-live weights. Measured on an RX 7900 XTX (see
`results/vram_transition_costs.json`, reproduce with
`calibrate_transition_costs.py`):

    from \\ to        astral   postgresql   financial
    pristine          17.97       17.92       17.99
    astral            18.27       18.10       18.17
    postgresql        18.13       18.19       18.18
    financial         18.12       18.16       18.22

Every row is identical. Spread across source states is 0.23-0.31 ms against
0.66-2.12 ms of measurement noise. So transition cost is DESTINATION-ONLY:

    C(u, v) = 0     if u == v          (cache hit)
    C(u, v) = f(v)  otherwise          (independent of u)

WHY THIS FILE STILL EXISTS WITH NOTHING TO SCHEDULE
---------------------------------------------------
The cost model is the physics that retired two components, so it stays here,
tested, rather than being deleted along with them:

1. **Shortest-path routing is vacuous.** For any detour u -> k -> v with k != v,
   cost is f(k) + f(v) > f(v) = C(u, v), because f > 0. The direct edge is always
   the shortest path, so an all-pairs solve returns its own input -- for ANY
   expert set, at any size. That killed the Floyd-Warshall router; see
   `benchmarks/superseded/apsp_floyd_warshall/`.
   `test_direct_edge_is_always_optimal` pins the property, so a future
   source-dependent cost model (e.g. an engine that subtracts a delta in place)
   fails loudly instead of silently reviving the trap.

2. **Ordering is cost-free; only CLUSTERING pays.** A batch touching a set S of
   distinct states costs sum_{v in S} f(v) regardless of visit order. That made
   an SLA-bounded cluster scheduler the right shape *if* there were concurrent
   requests to cluster. There are not: this engine is single-tenant, one agent
   walking a deterministic tool DAG, one node active at a time. The scheduler is
   retired to `benchmarks/superseded/sla_bounded_cluster_scheduler.py` with the
   numbers it earned, in case the engine ever becomes multi-tenant.
   See docs/DECISIONS.md §6.

`VRAMState` is still used by the server to track which expert is resident.
`stacking_saving_ms` reports the ~10.4 ms co-residency saving as cost analysis
only -- stacking stays off in serving because `ast+fin` costs astral -10.96pp
(CI [-21.50, -1.62]).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --- Measured cost model -----------------------------------------------------


@dataclass(frozen=True)
class TransitionCosts:
    """Measured VRAM state-transition costs in milliseconds.

    Defaults are the medians measured on an RX 7900 XTX with Qwen3.5-4B in
    bfloat16. Override via `from_calibration()` to use a fresh measurement.

    Note these are NOT decomposable into "restore + fold". A cross-expert swap
    is a single fused addmm (18.0 ms); `restore()` is a separate pure-copy path
    (14.0 ms). The old model assumed C(u,v) = t_restore + t_fold = 8 + 10, which
    happened to total the right number for cross-expert swaps and the wrong one
    for every transition involving pristine.
    """

    restore_ms: float = 13.95  # engine.restore() -- pure pristine copy
    fold_single_ms: float = 17.97  # engine.activate(e) -- one fused addmm
    fold_extra_expert_ms: float = 7.89  # marginal cost of each extra stacked expert

    @classmethod
    def from_calibration(cls, path: str | Path) -> TransitionCosts:
        """Loads costs from a `calibrate_transition_costs.py` report."""
        data = json.loads(Path(path).read_text())
        single = data["t_fold_single_ms"]
        stacked2 = data.get("t_fold_stacked2_ms")
        extra = (stacked2 - single) if stacked2 else cls.fold_extra_expert_ms
        return cls(
            restore_ms=data["t_restore_ms"],
            fold_single_ms=single,
            fold_extra_expert_ms=extra,
        )

    def cost_of_entering(self, state: VRAMState) -> float:
        """f(v): the destination-only cost of entering `state` from anywhere else."""
        n = len(state.expert_names)
        if n == 0:
            return self.restore_ms
        return self.fold_single_ms + (n - 1) * self.fold_extra_expert_ms


@dataclass(frozen=True)
class VRAMState:
    """A discrete GPU weight configuration."""

    name: str
    expert_names: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def pristine(cls) -> VRAMState:
        return cls(name="pristine", expert_names=frozenset())

    @classmethod
    def single(cls, expert_name: str) -> VRAMState:
        return cls(name=expert_name, expert_names=frozenset([expert_name]))

    @classmethod
    def stacked(cls, expert_names: Iterable[str]) -> VRAMState:
        names = frozenset(expert_names)
        if not names:
            return cls.pristine()
        if len(names) == 1:
            return cls.single(next(iter(names)))
        return cls(name="+".join(sorted(names)), expert_names=names)

    def is_pristine(self) -> bool:
        return len(self.expert_names) == 0

    def is_superset_of(self, other: VRAMState) -> bool:
        return self.expert_names.issuperset(other.expert_names)

    def is_subset_of(self, other: VRAMState) -> bool:
        return self.expert_names.issubset(other.expert_names)

    @classmethod
    def from_expert(cls, expert: Any | None) -> VRAMState:
        """Maps a FoldableExpert (or None, meaning pristine) to its VRAMState."""
        if expert is None:
            return cls.pristine()
        return cls.single(expert.name)


class VRAMStateGraph:
    """VRAM states and their measured transition costs.

    Edges follow the measured destination-only model: entering state v costs
    f(v) from every source except v itself. `t_restore_ms` / `t_fold_per_expert_ms`
    are accepted for backwards compatibility with older call sites but the
    authoritative source is `costs`.
    """

    def __init__(
        self,
        experts: list[str],
        allow_stacking: bool = False,
        costs: TransitionCosts | None = None,
        t_restore_ms: float | None = None,
        t_fold_per_expert_ms: float | None = None,
    ):
        self.experts = sorted(experts)
        self.allow_stacking = allow_stacking

        if costs is None:
            costs = TransitionCosts()
            # Honour legacy positional overrides if a caller supplied them.
            if t_restore_ms is not None or t_fold_per_expert_ms is not None:
                costs = TransitionCosts(
                    restore_ms=t_restore_ms if t_restore_ms is not None else costs.restore_ms,
                    fold_single_ms=(t_fold_per_expert_ms if t_fold_per_expert_ms is not None else costs.fold_single_ms),
                    fold_extra_expert_ms=(
                        t_fold_per_expert_ms if t_fold_per_expert_ms is not None else costs.fold_extra_expert_ms
                    ),
                )
        self.costs = costs

        self.states: list[VRAMState] = [VRAMState.pristine()]
        self.states.extend(VRAMState.single(e) for e in self.experts)

        if allow_stacking and len(self.experts) >= 2:
            for i in range(len(self.experts)):
                for j in range(i + 1, len(self.experts)):
                    self.states.append(VRAMState.stacked([self.experts[i], self.experts[j]]))

        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        self.num_nodes = len(self.states)

        self.adj_matrix: list[list[float]] = [
            [0.0 if i == j else self.costs.cost_of_entering(v) for j, v in enumerate(self.states)]
            for i in range(self.num_nodes)
        ]

    def cost(self, src: VRAMState, dst: VRAMState) -> float:
        """C(src, dst) under the measured destination-only model."""
        if src == dst:
            return 0.0
        return self.costs.cost_of_entering(dst)

    def stacking_saving_ms(self, a: VRAMState, b: VRAMState) -> float:
        """Milliseconds saved by co-residing `a` and `b` instead of folding each.

        Positive means stacking is cheaper. Measured at ~10.4 ms for two single
        experts. Cost-only: says nothing about whether the stacked state
        preserves either domain's accuracy.
        """
        if not (a.expert_names and b.expert_names):
            return 0.0
        merged = VRAMState.stacked(a.expert_names | b.expert_names)
        separate = self.costs.cost_of_entering(a) + self.costs.cost_of_entering(b)
        return separate - self.costs.cost_of_entering(merged)
