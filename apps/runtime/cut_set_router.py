"""Chapter 6: k-out-of-n Reliability & Minimal Cut Sets for Graph-of-Thoughts (GoT) Tool Routing.

Theoretical Grounding:
Chapter 6 (System Failure Modeling – k-out-of-n System Model, Minimal Paths & Cuts,
Jaejin Hwang, Reliability Analysis Using MINITAB and Python).

Mathematical Formulation:
1. Reliability Block Diagram (RBD):
   Given an agent execution DAG G = (V, E) with source s and sink t:
   - A Minimal Path Set P_j is a minimal set of nodes whose functioning ensures s -> t connectivity.
   - A Minimal Cut Set C_i is a minimal set of nodes whose failure completely disconnects s -> t.

2. System Reliability:
   R_sys = P(At least one minimal path works) = 1 - P(At least one minimal cut set fails)

3. k-out-of-n Redundancy for Order-1 Cut Sets:
   Any node v appearing in a cut set of order 1 (|C| = 1) is a Single Point of Failure (SPOF).
   If empirical node reliability R(v) < R_target (e.g. 0.95), the runtime dynamically promotes
   the node to a k=1 out of n=2 speculative parallel execution:
     R_{1/2}(v) = 1 - (1 - R(v))^2 = 2R(v) - R(v)^2
   For non-critical nodes (|C| > 1 or R >= R_target), execution remains n=1 (0% compute overhead).
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any


@dataclass
class ReliabilityNode:
    """Represents a tool execution step in the agent reasoning DAG."""

    node_id: str
    name: str
    reliability: float = 0.90  # Historical / empirical success probability
    execute_fn: Callable[..., Any] | None = None
    fallback_fn: Callable[..., Any] | None = None


class ReliabilityGraph:
    """Decomposes Graph-of-Thoughts (GoT) tool chains into Minimal Path & Cut Sets."""

    def __init__(self, source: str, sink: str) -> None:
        self.source = source
        self.sink = sink
        self.nodes: dict[str, ReliabilityNode] = {}
        self.adj: dict[str, list[str]] = {}
        self.in_degree: dict[str, int] = {}

    def add_node(self, node: ReliabilityNode) -> None:
        self.nodes[node.node_id] = node
        if node.node_id not in self.adj:
            self.adj[node.node_id] = []
        if node.node_id not in self.in_degree:
            self.in_degree[node.node_id] = 0

    def add_edge(self, u: str, v: str) -> None:
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Both nodes must exist before adding edge: {u} -> {v}")
        self.adj[u].append(v)
        self.in_degree[v] = self.in_degree.get(v, 0) + 1

    def find_all_paths(self) -> list[list[str]]:
        """Finds all simple directed paths from source to sink using DFS."""
        paths: list[list[str]] = []

        def dfs(current: str, current_path: list[str], visited: set[str]):
            if current == self.sink:
                paths.append(list(current_path))
                return

            for neighbor in self.adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    current_path.append(neighbor)
                    dfs(neighbor, current_path, visited)
                    current_path.pop()
                    visited.remove(neighbor)

        if self.source in self.nodes:
            dfs(self.source, [self.source], {self.source})

        return paths

    def find_minimal_cut_sets(self) -> list[set[str]]:
        """Computes all Minimal Cut Sets of the execution DAG.

        A cut set is a set of nodes whose removal destroys all source-to-sink paths.
        A minimal cut set contains no proper subset that is also a cut set.
        """
        paths = self.find_all_paths()
        if not paths:
            return []

        # Candidate nodes (excluding source and sink or including intermediate nodes)
        all_nodes = set(self.nodes.keys())
        cut_sets: list[set[str]] = []

        # Check candidate subsets from size 1 upwards
        for r in range(1, len(all_nodes) + 1):
            found_at_this_r = False
            for subset in itertools.combinations(all_nodes, r):
                subset_set = set(subset)
                # Check if this subset intersects every source-to-sink path
                if all(any(node in subset_set for node in path) for path in paths):
                    # Check minimality: no existing minimal cut set is a subset of this
                    if not any(existing.issubset(subset_set) for existing in cut_sets):
                        cut_sets.append(subset_set)
                        found_at_this_r = True

        return cut_sets

    def get_order_1_cut_sets(self) -> list[str]:
        """Returns all Single Points of Failure (Order-1 Minimal Cut Sets)."""
        minimal_cuts = self.find_minimal_cut_sets()
        order_1 = [list(c)[0] for c in minimal_cuts if len(c) == 1]
        return order_1

    def compute_system_reliability(self, active_hedges: set[str] | None = None) -> float:
        """Computes analytical system reliability under optional k=1-of-n=2 hedging."""
        paths = self.find_all_paths()
        if not paths:
            return 0.0

        hedges = active_hedges or set()

        # Effective node reliabilities
        effective_r = {}
        for nid, node in self.nodes.items():
            base_r = node.reliability
            if nid in hedges:
                # k=1 out of n=2 redundancy
                effective_r[nid] = 1.0 - (1.0 - base_r) ** 2
            else:
                effective_r[nid] = base_r

        # For series-parallel / DAG reliability, compute via inclusion-exclusion over minimal paths
        n_paths = len(paths)
        if n_paths == 1:
            # Simple series product
            r_sys = 1.0
            for node_id in paths[0]:
                r_sys *= effective_r[node_id]
            return float(r_sys)

        # Inclusion-Exclusion over paths for general DAGs
        r_sys = 0.0
        for r in range(1, n_paths + 1):
            sign = 1.0 if (r % 2 == 1) else -1.0
            for path_comb in itertools.combinations(paths, r):
                # Union of nodes in these paths
                union_nodes = set().union(*path_comb)
                p_union = 1.0
                for node_id in union_nodes:
                    p_union *= effective_r[node_id]
                r_sys += sign * p_union

        return float(max(0.0, min(1.0, r_sys)))

    def get_critical_bottlenecks(self, target_reliability: float = 0.95) -> list[str]:
        """Identifies vulnerable Order-1 Cut Sets requiring speculative hedging."""
        order_1_cuts = self.get_order_1_cut_sets()
        bottlenecks = []
        for nid in order_1_cuts:
            node = self.nodes[nid]
            if node.reliability < target_reliability:
                bottlenecks.append(nid)
        return bottlenecks


class ReliabilityDAGExecutor:
    """Universal execution middleware with targeted k-out-of-n speculative hedging."""

    def __init__(self, target_reliability: float = 0.95, max_workers: int = 4) -> None:
        self.target_reliability = target_reliability
        self.max_workers = max_workers

    def execute_dag(
        self,
        graph: ReliabilityGraph,
        initial_context: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Executes the agent DAG with automatic minimal cut-set speculative hedging.

        Returns:
            Tuple of:
            - context: Final execution dictionary containing results from all steps
            - telemetry: Metadata (hedged_nodes, exceptions_averted, total_tool_calls)
        """
        context = dict(initial_context or {})
        bottlenecks = set(graph.get_critical_bottlenecks(self.target_reliability))

        # Topological Sort
        in_degrees = {k: graph.in_degree.get(k, 0) for k in graph.nodes}
        queue = [k for k, deg in in_degrees.items() if deg == 0]
        topo_order = []
        while queue:
            curr = queue.pop(0)
            topo_order.append(curr)
            for nxt in graph.adj.get(curr, []):
                in_degrees[nxt] -= 1
                if in_degrees[nxt] == 0:
                    queue.append(nxt)

        telemetry = {
            "total_nodes": len(graph.nodes),
            "order_1_cut_sets": graph.get_order_1_cut_sets(),
            "hedged_nodes": list(bottlenecks),
            "total_tool_calls": 0,
            "exceptions_averted": 0,
            "failed_nodes": [],
            "execution_time_ms": 0.0,
        }

        t_start = time.perf_counter()

        for node_id in topo_order:
            node = graph.nodes[node_id]
            is_hedged = (node_id in bottlenecks) and (node.fallback_fn is not None)

            if is_hedged:
                # k=1 out of n=2 Speculative Parallel Race
                telemetry["total_tool_calls"] += 2
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {
                        pool.submit(node.execute_fn, dict(context)): "primary",
                        pool.submit(node.fallback_fn, dict(context)): "fallback",
                    }
                    success = False
                    first_res = None
                    primary_failed = False

                    for fut in as_completed(futures):
                        arm = futures[fut]
                        try:
                            res = fut.result()
                            if not success:
                                first_res = res
                                success = True
                        except Exception:
                            if arm == "primary":
                                primary_failed = True

                    if success:
                        context[node_id] = first_res
                        if primary_failed:
                            telemetry["exceptions_averted"] += 1
                    else:
                        telemetry["failed_nodes"].append(node_id)
                        raise RuntimeError(f"All speculative replicas failed on critical node {node_id}")

            else:
                # Standard n=1 execution
                telemetry["total_tool_calls"] += 1
                if node.execute_fn is not None:
                    try:
                        res = node.execute_fn(dict(context))
                        context[node_id] = res
                    except Exception as exc:
                        telemetry["failed_nodes"].append(node_id)
                        raise RuntimeError(f"Step {node_id} failed: {exc}")

        telemetry["execution_time_ms"] = (time.perf_counter() - t_start) * 1000.0
        return context, telemetry
