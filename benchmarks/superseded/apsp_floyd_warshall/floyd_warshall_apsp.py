"""RETIRED — Floyd-Warshall all-pairs shortest path over VRAM states.

Kept for provenance and as a runnable proof of why it was removed. Do not import
this from the live engine; the router uses a direct cost lookup instead.

WHY IT WAS RETIRED
------------------
It never did any work, and could never have done any work, because the cost model
it ran on is destination-only.

`WeightFoldingEngine.activate(e)` writes `W_live = W0 + s*(U@V)` as one fused
addmm per slot, reading from the pristine buffer. It does not restore first and
it does not read the currently-live weights. So the cost of entering state v is
the same from every source. Measured on RX 7900 XTX / Qwen3.5-4B bf16
(`results/vram_transition_costs.json`):

    from \\ to        astral   postgresql   financial
    pristine          17.97       17.92       17.99
    astral            18.27       18.10       18.17
    postgresql        18.13       18.19       18.18
    financial         18.12       18.16       18.22

Spread across sources: 0.23-0.31 ms. Measurement noise: 0.66-2.12 ms. Every row
is the same row.

THE PROOF
---------
Let C(u,v) = f(v) for u != v, with f(v) > 0 for all v.
For any intermediate k not in {u, v}:

    cost(u -> k -> v) = f(k) + f(v) > f(v) = C(u, v)

since f(k) > 0. So no detour is ever cheaper than the direct edge, the shortest
path between any two states IS the direct edge, and Floyd-Warshall provably
returns its own input matrix. This holds for ANY number of experts and ANY
positive f — it is a property of the cost model, not of the 3-expert graph that
happened to be configured.

Two further consequences, which is what the live router is built on instead:

  - Serving a batch that touches a set S of distinct states costs sum_{v in S} f(v)
    REGARDLESS of visit order. So there is no travelling-salesman objective here:
    ordering is free, and only clustering (entering each state once) pays.
  - Because ordering is cost-free, the entire ordering freedom can be spent on
    SLA/fairness rather than on cost.

RUN IT
------
    uv run python benchmarks/superseded/apsp_floyd_warshall/floyd_warshall_apsp.py

Prints the number of pairs the APSP solve improved. It is 0, at every expert
count tested.
"""

from __future__ import annotations


class RetiredFloydWarshallAPSP:
    """The retired solver. `dist` always ends up equal to the input adjacency matrix."""

    def __init__(self, adj_matrix: list[list[float]]):
        self.adj = [row[:] for row in adj_matrix]
        self.N = len(adj_matrix)
        self.dist = [row[:] for row in adj_matrix]
        self.next_hop: list[list[int | None]] = [
            [j if i != j else None for j in range(self.N)] for i in range(self.N)
        ]
        self._solve()

    def _solve(self) -> None:
        for k in range(self.N):
            for i in range(self.N):
                for j in range(self.N):
                    nd = self.dist[i][k] + self.dist[k][j]
                    if nd < self.dist[i][j]:
                        self.dist[i][j] = nd
                        self.next_hop[i][j] = self.next_hop[i][k]

    def improved_pairs(self, tol: float = 1e-9) -> int:
        return sum(
            1
            for i in range(self.N)
            for j in range(self.N)
            if abs(self.dist[i][j] - self.adj[i][j]) > tol
        )


def build_destination_only_adjacency(
    num_experts: int,
    restore_ms: float = 13.95,
    fold_single_ms: float = 17.97,
    fold_extra_ms: float = 7.89,
    allow_stacking: bool = True,
) -> tuple[list[list[float]], list[str]]:
    """Builds the VRAM-state adjacency matrix under the MEASURED cost model."""
    names = [f"e{i}" for i in range(num_experts)]
    states: list[frozenset[str]] = [frozenset()]
    states.extend(frozenset([n]) for n in names)
    if allow_stacking and num_experts >= 2:
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                states.append(frozenset([names[i], names[j]]))

    def f(state: frozenset[str]) -> float:
        n = len(state)
        if n == 0:
            return restore_ms
        return fold_single_ms + (n - 1) * fold_extra_ms

    adj = [
        [0.0 if i == j else f(v) for j, v in enumerate(states)]
        for i in range(len(states))
    ]
    labels = ["pristine" if not s else "+".join(sorted(s)) for s in states]
    return adj, labels


def main() -> None:
    print("Floyd-Warshall over the MEASURED (destination-only) VRAM cost model")
    print("=" * 66)
    print(f"{'experts':>8} {'states':>8} {'pairs':>8} {'improved by APSP':>18}")
    print("-" * 66)

    total_improved = 0
    for n in (1, 2, 3, 4, 6, 8, 10):
        adj, _ = build_destination_only_adjacency(n)
        solver = RetiredFloydWarshallAPSP(adj)
        improved = solver.improved_pairs()
        total_improved += improved
        print(f"{n:>8} {len(adj):>8} {len(adj)**2:>8} {improved:>18}")

    print("-" * 66)
    print(f"Total pairs improved across every configuration: {total_improved}")
    print()
    if total_improved == 0:
        print("The APSP solve returned its own input at every size, as proved in the")
        print("module docstring. This is why the algorithm was removed from the engine.")
    else:  # pragma: no cover - would mean the cost model stopped being destination-only
        print("APSP improved something — the cost model is no longer destination-only.")
        print("Revisit the live router: shortest-path routing may now be worth having.")


if __name__ == "__main__":
    main()
