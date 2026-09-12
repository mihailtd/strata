"""Tests for SQLite training ledger and telemetry database (training_db.py)."""

from pathlib import Path

import pytest
from runtime import training_db


def test_init_db_and_start_run(tmp_path: Path):
    db_file = tmp_path / "factory.db"
    training_db.init_db(db_file)
    assert db_file.exists()

    run_id = training_db.start_run(
        domain="python_modern",
        dataset_path="data/python_modern/training_data_v6.jsonl",
        n_records=1418,
        rank=8,
        alpha=128,
        target_dw_w=0.071,
        db_path=db_file,
    )
    assert run_id.startswith("run_")
    assert "python_modern" in run_id

    runs = training_db.list_runs(db_path=db_file)
    assert len(runs) == 1
    assert runs[0]["domain"] == "python_modern"
    assert runs[0]["status"] == "running"
    assert runs[0]["n_records"] == 1418
    assert runs[0]["scaling"] == 16.0


def test_record_step_metrics_and_finish_run(tmp_path: Path):
    db_file = tmp_path / "factory.db"
    run_id = training_db.start_run(
        domain="postgresql",
        dataset_path="data/postgresql/training_data_v6.jsonl",
        n_records=1000,
        db_path=db_file,
    )

    # Record 3 steps
    training_db.record_step(run_id, step=10, loss=1.45, token_accuracy=0.65, dw_over_w=0.02, db_path=db_file)
    training_db.record_step(run_id, step=50, loss=0.45, token_accuracy=0.91, dw_over_w=0.065, db_path=db_file)
    training_db.record_step(run_id, step=70, loss=0.14, token_accuracy=0.96, dw_over_w=0.072, db_path=db_file)

    # Finish run
    training_db.finish_run(
        run_id,
        status="completed",
        stopped_at_step=70,
        stop_reason="target_reached",
        final_loss=0.14,
        final_token_acc=0.96,
        final_dw_w=0.072,
        predicted_merge_err=2.31,
        runtime_seconds=55.0,
        db_path=db_file,
    )

    # Retrieve full run details
    run = training_db.get_run(run_id, db_path=db_file)
    assert run is not None
    assert run["status"] == "completed"
    assert run["stopped_at_step"] == 70
    assert run["stop_reason"] == "target_reached"
    assert len(run["steps"]) == 3
    assert run["steps"][2]["step"] == 70
    assert run["steps"][2]["dw_over_w"] == 0.072


def test_update_airm_metrics(tmp_path: Path):
    db_file = tmp_path / "factory.db"
    run_id = training_db.start_run(
        domain="astral",
        dataset_path="data/astral/training_data_v6.jsonl",
        n_records=800,
        db_path=db_file,
    )
    training_db.finish_run(run_id, status="completed", db_path=db_file)

    # Update AIRM metrics
    training_db.update_airm_metrics(
        domain="astral",
        airm_scale=3.33,
        airm_shape=103.11,
        gate_passed=True,
        run_id=run_id,
        db_path=db_file,
    )

    run = training_db.get_run(run_id, db_path=db_file)
    assert run["airm_scale"] == pytest.approx(3.33, abs=1e-2)
    assert run["airm_shape"] == pytest.approx(103.11, abs=1e-2)
    assert run["gate_passed"] == 1
