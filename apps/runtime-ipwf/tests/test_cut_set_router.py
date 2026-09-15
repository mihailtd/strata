"""Unit tests for Chapter 6: k-out-of-n Reliability & Minimal Cut Sets for Agent Tool DAGs."""

from cut_set_router import ReliabilityDAGExecutor, ReliabilityGraph, ReliabilityNode


def test_series_dag_all_nodes_are_order_1_cut_sets():
    """In a pure series DAG (S -> A -> B -> T), every node is a critical single point of failure."""
    g = ReliabilityGraph(source="S", sink="T")
    g.add_node(ReliabilityNode("S", "Source", reliability=1.0))
    g.add_node(ReliabilityNode("A", "SQL Query", reliability=0.80))
    g.add_node(ReliabilityNode("B", "Python Clean", reliability=0.95))
    g.add_node(ReliabilityNode("T", "Summary", reliability=1.0))

    g.add_edge("S", "A")
    g.add_edge("A", "B")
    g.add_edge("B", "T")

    paths = g.find_all_paths()
    assert len(paths) == 1
    assert paths[0] == ["S", "A", "B", "T"]

    order_1 = g.get_order_1_cut_sets()
    assert set(order_1) == {"S", "A", "B", "T"}

    # Bottleneck check (R < 0.90)
    bottlenecks = g.get_critical_bottlenecks(target_reliability=0.90)
    assert bottlenecks == ["A"]


def test_diamond_dag_minimal_cut_sets():
    """In a diamond DAG (S -> A -> T, S -> B -> T), S and T are Order-1, while {A, B} is Order-2."""
    g = ReliabilityGraph(source="S", sink="T")
    g.add_node(ReliabilityNode("S", "Source", reliability=1.0))
    g.add_node(ReliabilityNode("A", "Worker 1", reliability=0.85))
    g.add_node(ReliabilityNode("B", "Worker 2", reliability=0.85))
    g.add_node(ReliabilityNode("T", "Sink", reliability=1.0))

    g.add_edge("S", "A")
    g.add_edge("S", "B")
    g.add_edge("A", "T")
    g.add_edge("B", "T")

    paths = g.find_all_paths()
    assert len(paths) == 2
    assert ["S", "A", "T"] in paths
    assert ["S", "B", "T"] in paths

    cut_sets = g.find_minimal_cut_sets()
    # Order-1 cuts: {S}, {T}
    # Order-2 cuts: {A, B}
    assert {"S"} in cut_sets
    assert {"T"} in cut_sets
    assert {"A", "B"} in cut_sets

    # Neither A nor B is an Order-1 Cut Set individually
    order_1 = g.get_order_1_cut_sets()
    assert set(order_1) == {"S", "T"}


def test_analytical_system_reliability():
    """Verify that hedging an order-1 node elevates total system reliability."""
    g = ReliabilityGraph(source="S", sink="T")
    g.add_node(ReliabilityNode("S", "Source", reliability=1.0))
    g.add_node(ReliabilityNode("A", "Fragile Tool", reliability=0.80))
    g.add_node(ReliabilityNode("B", "Robust Tool", reliability=0.95))
    g.add_node(ReliabilityNode("T", "Sink", reliability=1.0))

    g.add_edge("S", "A")
    g.add_edge("A", "B")
    g.add_edge("B", "T")

    # Baseline reliability without hedging: 1.0 * 0.80 * 0.95 * 1.0 = 0.760
    r_unhedged = g.compute_system_reliability()
    assert abs(r_unhedged - 0.760) < 1e-4

    # With k=1 of n=2 hedging on node A: R(A) becomes 1 - 0.2^2 = 0.96
    # Total reliability: 1.0 * 0.96 * 0.95 * 1.0 = 0.912
    r_hedged = g.compute_system_reliability(active_hedges={"A"})
    assert abs(r_hedged - 0.912) < 1e-4
    assert r_hedged > r_unhedged + 0.15


def test_dag_executor_averts_transient_tool_failure():
    """Verify that ReliabilityDAGExecutor intercepts primary failure and succeeds via fallback."""
    g = ReliabilityGraph(source="S", sink="T")

    def primary_sql_flaky(ctx):
        # Simulates a runtime SQL syntax exception
        raise ValueError("Postgres SyntaxError: relation 'customer_tab' does not exist")

    def fallback_sql_safe(ctx):
        # Safe schema introspection fallback
        return [{"id": 1, "revenue": 500.0}]

    def python_clean(ctx):
        rows = ctx["A"]
        return sum(r["revenue"] for r in rows)

    g.add_node(ReliabilityNode("S", "Start", reliability=1.0, execute_fn=lambda ctx: "started"))
    g.add_node(
        ReliabilityNode("A", "SQL", reliability=0.75, execute_fn=primary_sql_flaky, fallback_fn=fallback_sql_safe)
    )
    g.add_node(ReliabilityNode("B", "Python", reliability=0.99, execute_fn=python_clean))
    g.add_node(ReliabilityNode("T", "Finish", reliability=1.0, execute_fn=lambda ctx: ctx["B"]))

    g.add_edge("S", "A")
    g.add_edge("A", "B")
    g.add_edge("B", "T")

    executor = ReliabilityDAGExecutor(target_reliability=0.90)
    context, telemetry = executor.execute_dag(g)

    # Execution must complete successfully with the correct data
    assert context["T"] == 500.0
    assert telemetry["exceptions_averted"] == 1
    assert "A" in telemetry["hedged_nodes"]
    # Total calls: S (1) + A (2 hedged) + B (1) + T (1) = 5
    assert telemetry["total_tool_calls"] == 5


def test_robust_nodes_not_duplicated():
    """Verify that nodes with reliability >= target_reliability are executed as n=1 without overhead."""
    g = ReliabilityGraph(source="S", sink="T")
    g.add_node(ReliabilityNode("S", "Start", reliability=1.0, execute_fn=lambda ctx: "s"))
    g.add_node(
        ReliabilityNode(
            "A", "Solid Tool", reliability=0.98, execute_fn=lambda ctx: "solid", fallback_fn=lambda ctx: "fb"
        )
    )
    g.add_node(ReliabilityNode("T", "Finish", reliability=1.0, execute_fn=lambda ctx: "done"))

    g.add_edge("S", "A")
    g.add_edge("A", "T")

    executor = ReliabilityDAGExecutor(target_reliability=0.90)
    context, telemetry = executor.execute_dag(g)

    assert telemetry["hedged_nodes"] == []
    assert telemetry["total_tool_calls"] == 3  # S(1) + A(1) + T(1)
    assert telemetry["exceptions_averted"] == 0
