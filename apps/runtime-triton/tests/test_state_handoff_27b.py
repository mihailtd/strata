"""Unit tests for True O(1) Tensor State Handoff on Native 27B Triton Engine."""

from pathlib import Path

from state_handoff_27b import AgentTurn, StateHandoffSession, TurnResult


def test_agent_turn_dataclass():
    turn = AgentTurn(
        agent_id="agent_1",
        role="postgresql",
        instruction="SELECT * FROM metrics",
        expert_lora="postgresql",
        max_new_tokens=32,
        temperature=0.0,
    )
    assert turn.agent_id == "agent_1"
    assert turn.role == "postgresql"
    assert turn.instruction == "SELECT * FROM metrics"
    assert turn.expert_lora == "postgresql"
    assert turn.max_new_tokens == 32
    assert turn.temperature == 0.0


def test_turn_result_dataclass():
    res = TurnResult(
        agent_id="agent_1",
        role="postgresql",
        expert_lora="postgresql",
        output_text="Result text",
        tokens_generated=10,
        prefill_tokens=15,
        tokens_avoided=0,
        prefill_ms=12.5,
        decode_ms=80.0,
        handoff_ms=0.5,
        lora_swap_ms=1.2,
        total_ms=94.2,
        tok_per_sec=106.16,
        vram_used_gb=15.4,
        state_tensor_mb=154.0,
    )
    assert res.tokens_generated == 10
    assert res.prefill_tokens == 15
    assert res.tokens_avoided == 0
    assert res.state_tensor_mb == 154.0
    assert res.handoff_ms == 0.5


def test_state_handoff_session_initialization():
    session = StateHandoffSession(engine=None, session_id="test_sess_001", clone_on_handoff=True)
    assert session.session_id == "test_sess_001"
    assert session.clone_on_handoff is True
    assert session.active_state is None
    assert session.turn_history == []
    assert session.cumulative_history_tokens == 0


def test_server_has_state_handoff_routes():
    server_path = Path(__file__).parent.parent / "server.py"
    source = server_path.read_text()
    assert "/api/multi_agent/run_pipeline" in source
    assert "/api/engine/multi_agent/run_pipeline" in source
    assert "RunPipelineRequest" in source
    assert "StateHandoffSession" in source
