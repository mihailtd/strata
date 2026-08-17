"""VRAM State Router — SLA-bounded cluster scheduling over measured transition costs.

WHAT THE HARDWARE ACTUALLY DOES
-------------------------------
`WeightFoldingEngine.activate(e)` writes `W_live = W0 + s*(U@V)` as ONE fused
addmm per slot, reading from the pristine buffer. It does not restore first and
it does not read the currently-live weights. The consequence, measured on an
RX 7900 XTX (see `results/vram_transition_costs.json`, reproduce with
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

TWO CONSEQUENCES, BOTH PROVABLE
-------------------------------
1. Shortest-path routing is vacuous. For any detour u -> k -> v with k != v,
   the cost is f(k) + f(v) > f(v) = C(u, v), because f > 0. The direct edge is
   always the shortest path, so an all-pairs shortest-path solve returns its own
   input. This holds for ANY expert set and ANY number of experts -- it is a
   property of the cost model, not of any particular graph.

   This module used to carry a Floyd-Warshall solver on that basis. It has been
   REMOVED, not merely disabled: it never influenced a scheduling decision and
   could not have. The implementation, the proof, and the measurements are kept
   in `benchmarks/superseded/apsp_floyd_warshall/` -- read that before adding any
   graph pathfinding here. `test_direct_edge_is_always_optimal` pins the property
   so that a future source-dependent cost model (e.g. an engine that subtracts a
   delta in place) fails the test loudly instead of silently reviving the trap.

2. Ordering is cost-free; only CLUSTERING pays. Serving a batch that touches a
   set S of distinct states costs sum_{v in S} f(v) no matter what order the
   clusters are visited in. Total cost depends only on how many distinct states
   you enter, not the sequence. Therefore:
       - the cost-optimal schedule is any schedule that enters each needed state
         exactly once (i.e. perfect clustering), and
       - the entire ordering freedom is free to spend on SLA/fairness.

   This is why the scheduler below is a cluster-drain scheduler with an SLA
   bound, not a travelling-salesman solver. Under destination-only costs, the
   TSP framing has no objective to optimise.

WHAT THIS BUYS, MEASURED
------------------------
Not latency. Swap overhead is 0.86% of wall clock end-to-end, and a live A/B
against FIFO moved mean latency by -62 ms, 95% CI [-1481, +1417] -- not
significant. What it does buy is deadline protection: across the offered-load
sweep it beats both FIFO and greedy affinity on SLA violations (98% -> 79.5% at
rho=1.5, 62.5% -> 47% at rho=0.95), by paying more swaps than greedy to avoid
starving minority domains.

WHERE REAL ROUTING DECISIONS DO EXIST
-------------------------------------
Stacking. `activate_many([a, b])` measured 25.86 ms versus 2 x 18.14 = 36.28 ms
for two separate folds, so co-residency saves ~10.4 ms when a batch needs both
domains. That IS a genuine cost decision. It is implemented (`stacking_saving_ms`)
but DISABLED by default, because whether a stacked state preserves each domain's
task accuracy is an open question in this repo (see `activate_many`'s docstring)
and this module must not trade correctness for 10 ms on an unverified premise.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
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

    restore_ms: float = 13.95        # engine.restore() -- pure pristine copy
    fold_single_ms: float = 17.97    # engine.activate(e) -- one fused addmm
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
                    fold_single_ms=(
                        t_fold_per_expert_ms if t_fold_per_expert_ms is not None else costs.fold_single_ms
                    ),
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


@dataclass
class PendingRequest:
    """An incoming request waiting for execution."""

    req_id: str
    target_state: VRAMState
    arrival_time: float
    sla_deadline_s: float
    metadata: dict[str, Any] = field(default_factory=dict)
    service_time_s: float | None = None  # est. generation time; falls back to scheduler default

    def get_age_s(self, current_time_s: float | None = None) -> float:
        now = current_time_s if current_time_s is not None else time.perf_counter()
        return max(0.0, now - self.arrival_time)

    def deadline_at(self) -> float:
        """Absolute wall-clock time by which this request should have completed."""
        return self.arrival_time + self.sla_deadline_s

    def is_urgent(self, current_time_s: float | None = None) -> bool:
        return self.get_age_s(current_time_s) >= self.sla_deadline_s


class VRAMStateScheduler:
    """SLA-bounded cluster-drain scheduler.

    Under destination-only transition costs, total transition cost equals
    sum(f(v)) over the DISTINCT states a schedule enters, independent of order.
    So the scheduler:

      1. Groups pending requests by target state (each state entered once =>
         cost-optimal by construction, not by heuristic search).
      2. Drains the current cluster for free, because staying put costs 0.
      3. Breaks the drain early -- and only then -- when continuing would push
         another cluster's earliest deadline past the point of no return. That
         is the SLA bound, and it is what stops perfect clustering from starving
         a minority domain indefinitely.
      4. Picks the next cluster by earliest deadline, tie-broken by entry cost
         (pristine at 13.95 ms is genuinely cheaper to enter than a fold).

    The previous implementation scored clusters with
        oldest_age + urgency_bonus + affinity_bonus - gamma * transition_cost_s
    where gamma * cost was 0.0009 against an affinity bonus of 3.0 -- a factor of
    3333. Sweeping gamma over [0, 0.05, 1, 1000] produced byte-identical
    schedules, i.e. the cost term never influenced a single decision; the
    clustering came entirely from the hardcoded affinity constant. It also
    drained whole clusters unconditionally, so its "SLA bound" could not
    actually preempt a drain in progress. See
    `benchmarks/superseded/apsp_floyd_warshall/` for the full retirement note.
    """

    def __init__(
        self,
        graph: VRAMStateGraph,
        default_sla_deadline_s: float = 2.0,
        default_service_time_s: float = 0.576,
    ):
        self.graph = graph
        self.default_sla_deadline_s = default_sla_deadline_s
        self.default_service_time_s = default_service_time_s
        self.current_state: VRAMState = VRAMState.pristine()

    def _service_time(self, req: PendingRequest) -> float:
        return req.service_time_s if req.service_time_s is not None else self.default_service_time_s

    def schedule_batch(
        self,
        queue: list[PendingRequest],
        current_state: VRAMState | None = None,
        current_time_s: float | None = None,
        gamma_cost_weight: float | None = None,  # accepted, unused; see class docstring
    ) -> list[PendingRequest]:
        """Returns the requests in execution order. Never drops or duplicates a request."""
        if not queue:
            return []

        now = current_time_s if current_time_s is not None else time.perf_counter()
        state = current_state or self.current_state

        clusters: dict[VRAMState, list[PendingRequest]] = defaultdict(list)
        for req in queue:
            clusters[req.target_state].append(req)
        for reqs in clusters.values():
            reqs.sort(key=lambda r: r.arrival_time)

        scheduled: list[PendingRequest] = []
        clock = now

        while any(clusters.values()):
            live = {s: r for s, r in clusters.items() if r}

            if state in live:
                nxt = live[state][0]
                # Would serving one more here doom another cluster that is
                # currently still saveable? If so, break the drain now.
                finish_if_continue = clock + self._service_time(nxt)
                preempt_for = self._find_preempt_target(live, state, clock, finish_if_continue)
                if preempt_for is None:
                    scheduled.append(clusters[state].pop(0))
                    clock += self._service_time(nxt)
                    continue
                target = preempt_for
            else:
                target = self._pick_next_cluster(live, state, clock)

            # Transition, then serve one request from the new cluster.
            clock += self.graph.cost(state, target) / 1000.0
            state = target
            nxt = clusters[state].pop(0)
            scheduled.append(nxt)
            clock += self._service_time(nxt)

        return scheduled

    def _find_preempt_target(
        self,
        live: dict[VRAMState, list[PendingRequest]],
        state: VRAMState,
        clock: float,
        finish_if_continue: float,
    ) -> VRAMState | None:
        """The cluster we must switch to NOW, or None if draining is still safe.

        A cluster is 'still saveable' if switching immediately would meet its
        earliest deadline, and 'doomed by continuing' if serving one more request
        from the current cluster first would not. Only those force a preemption --
        clusters that are already past saving do not, since breaking the drain
        for them would cost a transition and fix nothing.
        """
        best: VRAMState | None = None
        best_deadline = float("inf")

        for cand, reqs in live.items():
            if cand == state:
                continue
            head = reqs[0]
            deadline = head.deadline_at()
            switch_cost_s = self.graph.cost(state, cand) / 1000.0
            done_if_switch_now = clock + switch_cost_s + self._service_time(head)
            done_if_continue = finish_if_continue + switch_cost_s + self._service_time(head)

            if done_if_switch_now <= deadline < done_if_continue and deadline < best_deadline:
                best, best_deadline = cand, deadline

        return best

    def _pick_next_cluster(
        self,
        live: dict[VRAMState, list[PendingRequest]],
        state: VRAMState,
        clock: float,
    ) -> VRAMState:
        """Earliest deadline first; ties broken by cheaper entry cost, then by name."""
        return min(
            live,
            key=lambda s: (
                min(r.deadline_at() for r in live[s]),
                self.graph.cost(state, s),
                s.name,
            ),
        )


SLABoundedClusterScheduler = VRAMStateScheduler  # descriptive alias
