"""Unit tests for Entropy-Adaptive Dynamic Tree Speculative Decoder."""

import pytest
import torch
import torch.nn as nn

from runtime.adaptive_tree_speculator import (
    EntropyAdaptiveTreeSpeculator,
    SpeculationRegime,
    TreeTopology,
)


class MockDraftHead(nn.Module):
    def __init__(self, vocab_size: int = 1000):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(128, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_entropy_computation():
    # 1. Delta distribution (Zero Entropy)
    logits_zero = torch.tensor([[100.0, -100.0, -100.0, -100.0]])
    ent_zero = EntropyAdaptiveTreeSpeculator.compute_shannon_entropy(logits_zero)
    assert ent_zero < 0.01, f"Expected near-zero entropy, got {ent_zero}"

    # 2. Uniform distribution (Max Entropy for 4 elements = log2(4) = 2.0)
    logits_uniform = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    ent_uniform = EntropyAdaptiveTreeSpeculator.compute_shannon_entropy(logits_uniform)
    assert abs(ent_uniform - 2.0) < 0.05, f"Expected ~2.0 bits entropy, got {ent_uniform}"


def test_adaptive_topology_selection():
    draft_head = MockDraftHead(vocab_size=500)
    speculator = EntropyAdaptiveTreeSpeculator(
        draft_head=draft_head,
        entropy_low_threshold=0.25,
        entropy_high_threshold=1.00,
    )

    # Low entropy -> Deep burst
    topo_low = speculator.select_topology(0.15)
    assert topo_low.regime == SpeculationRegime.DEEP_BURST
    assert topo_low.depth == 4

    # Mid entropy -> Balanced 2x2
    topo_mid = speculator.select_topology(0.65)
    assert topo_mid.regime == SpeculationRegime.BALANCED_TREE
    assert topo_mid.depth == 2

    # High entropy -> Shallow guard
    topo_high = speculator.select_topology(1.50)
    assert topo_high.regime == SpeculationRegime.SHALLOW_GUARD
    assert topo_high.depth == 1


def test_speculation_step_execution():
    draft_head = MockDraftHead(vocab_size=500)
    speculator = EntropyAdaptiveTreeSpeculator(
        draft_head=draft_head,
        entropy_low_threshold=0.25,
        entropy_high_threshold=1.00,
    )

    hidden = torch.randn((1, 128))
    tree = speculator.build_candidate_tree(hidden, current_token=42)
    assert tree.tokens.dim() == 2
    assert tree.tokens.size(0) >= 1

    # Verify oracle
    def oracle_fn(paths: torch.Tensor) -> torch.Tensor:
        return paths

    res = speculator.verify_and_accept(tree, oracle_fn)
    assert res.num_accepted >= 1
    assert res.cycle_time_ms > 0
    assert res.effective_throughput_tok_s > 0

    telemetry = speculator.get_telemetry_summary()
    assert telemetry["total_cycles"] == 1
    assert telemetry["total_accepted_tokens"] >= 1
