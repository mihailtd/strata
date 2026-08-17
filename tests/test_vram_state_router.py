"""Tests for the VRAM expert-state cost model.

These pin the measured physics and the two consequences that retired components
on the strength of it: shortest-path routing is degenerate (APSP), and ordering
is cost-free so only clustering pays (which is why the cluster scheduler was the
right shape — before the engine turned out to be single-tenant and have nothing
to schedule).

Both are findings, so they are asserted rather than assumed. If a future engine
makes the cost model source-dependent, these fail loudly.

The scheduler's own tests retired with it; see
`benchmarks/superseded/sla_bounded_cluster_scheduler.py`.
"""

from gnn_experiment.router.vram_state_router import (
    TransitionCosts,
    VRAMState,
    VRAMStateGraph,
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
