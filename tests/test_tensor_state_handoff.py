"""Unit and integration tests for Tensor-Level Recurrent State Handoff ($S_t$) & Dual Protocol."""

import sys
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runtime.canon import configure_deterministic_attention
from src.runtime.novel_peft import FoldableExpert, WeightFoldingEngine
from src.runtime.state_handoff import (
    AgentHandoffSession,
    RecurrentStateSnapshot,
    capture_recurrent_state,
)


@pytest.fixture(scope="module")
def shared_model_and_engine():
    configure_deterministic_attention()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen3.5-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    exp_pg = FoldableExpert.from_dir(REPO_ROOT / "results/adapters/m2_postgresql_r8a128_v4", "postgresql")
    exp_astral = FoldableExpert.from_dir(REPO_ROOT / "results/adapters/m2_astral_r8a128_v4", "astral")
    engine = WeightFoldingEngine(model, [exp_pg, exp_astral], keep_pristine=True)
    experts = {"postgresql": exp_pg, "astral": exp_astral}

    return model, tokenizer, engine, experts


def test_snapshot_and_clone_immutability(shared_model_and_engine):
    """Verify that snapshotting and cloning S_t preserves bit-exact values without aliasing."""
    model, tokenizer, engine, experts = shared_model_and_engine
    engine.activate(experts["postgresql"])

    inputs = tokenizer("SELECT * FROM users WHERE id = 1;", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        cache = out.past_key_values

    snapshot_1 = capture_recurrent_state(cache)
    assert snapshot_1.total_mb > 0.05, f"Expected >0.05 MB state, got {snapshot_1.total_mb:.2f} MB"

    # Clone snapshot
    snapshot_2 = snapshot_1.clone()
    assert snapshot_2.total_bytes == snapshot_1.total_bytes

    # Verify tensor equality and verify they are distinct memory buffers
    for layer1, layer2 in zip(snapshot_1.cache_instance.layers, snapshot_2.cache_instance.layers, strict=True):
        for k in layer1.__dict__:
            v1 = getattr(layer1, k)
            v2 = getattr(layer2, k)
            if isinstance(v1, torch.Tensor):
                assert torch.equal(v1, v2), f"Tensor mismatch for key {k}"
                assert v1.data_ptr() != v2.data_ptr(), f"Tensors share memory buffer for key {k}!"
            elif isinstance(v1, list) and all(isinstance(x, torch.Tensor) for x in v1):
                for t1, t2 in zip(v1, v2):
                    assert torch.equal(t1, t2), f"Tensor mismatch in list {k}"
                    assert t1.data_ptr() != t2.data_ptr(), f"List tensors share memory for {k}!"


def test_multi_agent_recurrent_state_handoff(shared_model_and_engine):
    """Verify end-to-end multi-agent execution with S_t state handoff."""
    model, tokenizer, engine, experts = shared_model_and_engine

    session = AgentHandoffSession(model, tokenizer, engine, experts)

    # Turn 1: PostgreSQL Expert creates a schema
    res_1 = session.execute_turn(
        expert_name="postgresql",
        instruction="Create a table 'metrics' with id integer and latency_ms float.",
        max_new_tokens=48,
    )
    assert res_1.expert_name == "postgresql"
    assert res_1.generated_tokens > 0
    assert "metrics" in res_1.full_output_text.lower() or "table" in res_1.full_output_text.lower()
    assert len(res_1.human_summary) > 0

    # Turn 2: Astral / Python Expert receives S_t state and writes code referencing the table
    res_2 = session.execute_turn(
        expert_name="astral",
        instruction="Write a Python function to insert a metric row.",
        state_handoff=res_1.state_snapshot,
        max_new_tokens=48,
    )
    assert res_2.expert_name == "astral"
    assert res_2.generated_tokens > 0
    assert len(res_2.human_summary) > 0

    # Ensure prefill latency for the 2nd turn was minimal (short prompt only)
    assert res_2.prompt_tokens < 40
