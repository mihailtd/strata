"""RETIRED — SLA-bounded cluster scheduler for VRAM expert states.

Kept for provenance. Do not import this from the live engine.

WHY IT WAS RETIRED (2026-08-17)
-------------------------------
Not because it was wrong — it worked, and it measurably beat both baselines on
deadline misses:

    offered load rho    FIFO     Greedy    Router
    0.8                 43.5%    38.5%     33.0%
    0.95                62.5%    51.5%     47.0%
    1.5                 98.0%    96.0%     79.5%

It was retired because **the engine has no concurrency for it to schedule.**
This is a single-tenant system: one agent walking a deterministic tool DAG, one
node active at a time. With one caller there is nothing to reorder, so the
scheduler is inert by construction, and the minority-domain starvation it
protects against cannot occur.

The end-to-end A/B said the same thing before the architecture did: swap overhead
is 0.86% of wall clock, mean latency moved -62 ms with a 95% CI of
[-1481, +1417] (not significant), and batch-order commitment cost +3.6 s of P95.

REVIVE THIS ONLY IF the engine becomes multi-tenant — concurrent callers sharing
one GPU with different target experts. Then the numbers above are what you get
back. Until then it is machinery guarding against a condition that cannot arise.

What survives in `router/vram_state_router.py` is the measured cost model
(`TransitionCosts`, `VRAMStateGraph`) — that is the physics that retired the APSP
router, and it stays tested — plus `VRAMState`, which the server still uses to
track which expert is resident.

See docs/DECISIONS.md §6.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# VRAMState / VRAMStateGraph still live in gnn_experiment.router.vram_state_router;
# this file is provenance only and is not imported by anything.
from gnn_experiment.router.vram_state_router import VRAMState, VRAMStateGraph  # noqa: F401


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
