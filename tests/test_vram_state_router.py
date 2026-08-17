"""Tests for the VRAM State Router.

These pin two things: the measured cost model, and the consequences that follow
from it — including the fact that all-pairs shortest-path routing is degenerate
here. That degeneracy is a finding, so it is asserted rather than assumed.
"""

import time

from gnn_experiment.router.vram_state_router import (
    PendingRequest,
    TransitionCosts,
    VRAMState,
    VRAMStateGraph,
    VRAMStateScheduler,
)

EXPERTS = ["astral", "postgresql", "financial"]


def _graph(allow_stacking: bool = False) -> VRAMStateGraph:
    return VRAMStateGraph(EXPERTS, allow_stacking=allow_stacking)


def count_transitions(reqs, init_state):
    transitions, curr = 0, init_state
    for r in reqs:
        if r.target_state != curr:
            transitions += 1
            curr = r.target_state
    return transitions


# --- Cost model --------------------------------------------------------------

def test_transition_cost_is_destination_only():
    """C(u, v) must not depend on u. This is the measured behaviour of activate()."""
    graph = VRAMStateGraph(EXPERTS, allow_stacking=True)
    for dst in graph.states:
        costs = {graph.cost(src, dst) for src in graph.states if src != dst}
        assert len(costs) == 1, f"cost of entering {dst.name} varies by source: {costs}"


def test_identity_transition_is_free():
    graph = VRAMStateGraph(EXPERTS)
    for s in graph.states:
        assert graph.cost(s, s) == 0.0


def test_measured_costs_match_calibration():
    """Defaults are the medians from results/vram_transition_costs.json."""
    c = TransitionCosts()
    assert c.restore_ms == 13.95
    assert c.fold_single_ms == 17.97
    # A stacked pair costs less than two separate folds — the one real routing win.
    graph = VRAMStateGraph(EXPERTS, allow_stacking=True)
    saving = graph.stacking_saving_ms(
        VRAMState.single("astral"), VRAMState.single("postgresql")
    )
    assert saving > 0
    assert abs(saving - (2 * 17.97 - (17.97 + 7.89))) < 1e-6


# --- Why there is no shortest-path solver ------------------------------------
#
# These pin the property that made Floyd-Warshall dead code. If a future cost
# model becomes source-dependent (e.g. an engine that subtracts a delta in place,
# making unstacking cheaper than a full re-fold), these fail LOUDLY — which is
# the signal that graph routing might finally be worth having. Until then, see
# benchmarks/superseded/apsp_floyd_warshall/ and do not re-add a solver.

def test_direct_edge_is_always_optimal():
    """No detour u -> k -> v can beat the direct edge, so shortest paths are trivial.

    cost(u->k->v) = f(k) + f(v) > f(v) = C(u,v), because f > 0.
    """
    for allow_stacking in (False, True):
        graph = _graph(allow_stacking)
        for u in graph.states:
            for v in graph.states:
                if u == v:
                    continue
                direct = graph.cost(u, v)
                for k in graph.states:
                    if k in (u, v):
                        continue
                    detour = graph.cost(u, k) + graph.cost(k, v)
                    assert detour > direct - 1e-9, (
                        f"detour {u.name}->{k.name}->{v.name} ({detour:.3f}ms) beats the "
                        f"direct edge ({direct:.3f}ms) — the cost model is no longer "
                        "destination-only; revisit the superseded APSP note."
                    )


def test_property_holds_as_expert_count_grows():
    """This is a property of the cost model, not of a 3-expert graph."""
    for n in (1, 2, 5, 8):
        graph = VRAMStateGraph([f"e{i}" for i in range(n)], allow_stacking=True)
        for u in graph.states:
            for v in graph.states:
                if u == v:
                    continue
                assert graph.cost(u, v) == graph.costs.cost_of_entering(v)


def test_batch_cost_is_order_independent():
    """Total cost depends only on WHICH states are entered, never the sequence.

    This is what makes cluster-drain scheduling cost-optimal by construction, and
    what leaves the entire ordering freedom available to spend on deadlines.
    """
    import itertools

    graph = _graph()
    targets = [VRAMState.single(e) for e in EXPERTS]

    costs = set()
    for order in itertools.permutations(targets):
        total, state = 0.0, VRAMState.pristine()
        for t in order:
            total += graph.cost(state, t)
            state = t
        costs.add(round(total, 9))

    assert len(costs) == 1, f"visit order changed total cost: {costs}"


# --- Scheduling --------------------------------------------------------------

def test_clustering_is_cost_optimal_without_sla_pressure():
    """With slack deadlines, each needed state is entered exactly once."""
    scheduler = VRAMStateScheduler(_graph(), default_service_time_s=0.05)
    now = time.perf_counter()

    state_pg = VRAMState.single("postgresql")
    state_astral = VRAMState.single("astral")
    state_fin = VRAMState.single("financial")

    requests = [
        PendingRequest("r1", state_pg, now, 1e6),
        PendingRequest("r2", state_astral, now, 1e6),
        PendingRequest("r3", state_pg, now, 1e6),
        PendingRequest("r4", state_fin, now, 1e6),
        PendingRequest("r5", state_astral, now, 1e6),
        PendingRequest("r6", state_pg, now, 1e6),
        PendingRequest("r7", state_fin, now, 1e6),
    ]

    scheduled = scheduler.schedule_batch(requests, current_state=state_pg, current_time_s=now)

    assert count_transitions(requests, state_pg) == 6
    # 3 distinct states, already sitting on one of them => 2 entries is the floor.
    assert count_transitions(scheduled, state_pg) == 2
    # Current state drains first, at zero transition cost.
    assert [r.target_state.name for r in scheduled[:3]] == ["postgresql"] * 3


def test_sla_bound_preempts_a_drain_to_prevent_starvation():
    """A long drain must break for a request that is still saveable but about to expire."""
    scheduler = VRAMStateScheduler(_graph(), default_service_time_s=0.5)
    now = time.perf_counter()

    state_pg = VRAMState.single("postgresql")
    state_astral = VRAMState.single("astral")

    pg_reqs = [PendingRequest(f"pg{i}", state_pg, now, 1e6) for i in range(10)]
    astral = PendingRequest("astral_urgent", state_astral, now, 1.6)

    scheduled = scheduler.schedule_batch(
        pg_reqs + [astral], current_state=state_pg, current_time_s=now
    )
    order = [r.req_id for r in scheduled]

    # It drains pg while safe, breaks for astral before the deadline is lost,
    # then resumes pg. Two transitions total — the drain is not abandoned.
    idx = order.index("astral_urgent")
    assert 0 < idx < len(order) - 1, f"astral not preempted mid-drain: {order}"
    assert count_transitions(scheduled, state_pg) == 2

    # And the deadline is actually met under the scheduler's own timing model.
    completion = now
    state = state_pg
    for r in scheduled:
        completion += scheduler.graph.cost(state, r.target_state) / 1000.0
        state = r.target_state
        completion += 0.5
        if r.req_id == "astral_urgent":
            assert completion <= astral.deadline_at()
            break


def test_already_doomed_request_does_not_thrash_the_drain():
    """A request past saving must not trigger a preemption that fixes nothing."""
    scheduler = VRAMStateScheduler(_graph(), default_service_time_s=0.5)
    now = time.perf_counter()

    state_pg = VRAMState.single("postgresql")
    state_astral = VRAMState.single("astral")

    pg_reqs = [PendingRequest(f"pg{i}", state_pg, now, 1e6) for i in range(5)]
    # Arrived 100s ago with a 1s SLA — unsaveable no matter what we do.
    doomed = PendingRequest("doomed", state_astral, now - 100.0, 1.0)

    scheduled = scheduler.schedule_batch(
        pg_reqs + [doomed], current_state=state_pg, current_time_s=now
    )
    assert count_transitions(scheduled, state_pg) == 1
    assert scheduled[-1].req_id == "doomed"


def test_schedule_is_a_permutation_of_the_queue():
    """No request may be dropped or duplicated."""
    scheduler = VRAMStateScheduler(_graph(), default_service_time_s=0.05)
    now = time.perf_counter()
    states = [VRAMState.single(e) for e in EXPERTS] + [VRAMState.pristine()]

    queue = [
        PendingRequest(f"r{i}", states[i % len(states)], now - (i * 0.01), 0.5 + (i % 3))
        for i in range(40)
    ]
    scheduled = scheduler.schedule_batch(queue, current_state=states[0], current_time_s=now)

    assert len(scheduled) == len(queue)
    assert {r.req_id for r in scheduled} == {r.req_id for r in queue}


def test_empty_queue_returns_empty():
    scheduler = VRAMStateScheduler(_graph())
    assert scheduler.schedule_batch([]) == []


def test_gamma_parameter_is_inert():
    """Kept for call-site compatibility; it must not change any decision.

    The old scorer multiplied gamma by a cost in seconds (0.0009) against a
    hardcoded affinity bonus of 3.0 — a factor of 3333 — so it never mattered.
    """
    scheduler = VRAMStateScheduler(_graph(), default_service_time_s=0.05)
    now = time.perf_counter()
    states = [VRAMState.single(e) for e in EXPERTS]
    queue = [
        PendingRequest(f"r{i}", states[i % 3], now - (i * 0.01), 2.0) for i in range(15)
    ]

    orders = [
        [r.req_id for r in scheduler.schedule_batch(
            list(queue), current_state=states[0], current_time_s=now, gamma_cost_weight=g
        )]
        for g in (0.0, 0.05, 1.0, 1000.0)
    ]
    assert all(o == orders[0] for o in orders)
