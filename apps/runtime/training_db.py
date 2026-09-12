"""SQLite Database & Telemetry Ledger for Adapter Training Runs (results/factory.db).

Maintains a queryable history of all expert training runs, per-step loss/geometry curves,
and automated post-training AIRM Acceptance Gate audit scores.
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any

from runtime.canon import REPO_ROOT

DEFAULT_DB_PATH = REPO_ROOT / "results" / "factory.db"


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Connect to SQLite database with foreign keys and row factory enabled."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    """Initialize the schema for runs and step_metrics tables."""
    with get_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                adapter_version TEXT NOT NULL DEFAULT 'v6',
                status TEXT NOT NULL DEFAULT 'running', -- running, completed, failed, cancelled
                dataset_path TEXT NOT NULL,
                n_records INTEGER NOT NULL DEFAULT 0,
                rank INTEGER NOT NULL DEFAULT 8,
                alpha INTEGER NOT NULL DEFAULT 128,
                scaling REAL NOT NULL DEFAULT 16.0,
                lr REAL NOT NULL DEFAULT 0.0002,
                max_steps INTEGER NOT NULL DEFAULT 150,
                target_dw_w REAL NOT NULL DEFAULT 0.071,
                adapter_size_mb REAL NOT NULL DEFAULT 40.53,
                stopped_at_step INTEGER,
                stop_reason TEXT, -- target_reached, ceiling, plateau, max_steps, error
                final_loss REAL,
                final_token_acc REAL,
                final_dw_w REAL,
                predicted_merge_err REAL,
                airm_scale REAL,
                airm_shape REAL,
                gate_passed INTEGER DEFAULT NULL, -- 1 = passed (scale <= 8.0), 0 = failed, NULL = not evaluated
                runtime_seconds REAL,
                config_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME
            );

            CREATE TABLE IF NOT EXISTS step_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                loss REAL,
                grad_norm REAL,
                learning_rate REAL,
                token_accuracy REAL,
                dw_over_w REAL,
                rel_growth REAL,
                merge_err_pct REAL,
                entropy REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_runs_domain ON training_runs(domain);
            CREATE INDEX IF NOT EXISTS idx_runs_created ON training_runs(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_step_metrics_run ON step_metrics(run_id, step);
        """)
        try:
            conn.execute("ALTER TABLE training_runs ADD COLUMN adapter_size_mb REAL NOT NULL DEFAULT 40.53")
        except Exception:
            pass


def start_run(
    domain: str,
    dataset_path: str,
    n_records: int,
    rank: int = 8,
    alpha: int = 128,
    lr: float = 0.0002,
    max_steps: int = 150,
    target_dw_w: float = 0.071,
    adapter_version: str = "v6",
    config: dict[str, Any] | None = None,
    db_path: Path | str | None = None,
) -> str:
    """Register the start of a training run in SQLite."""
    init_db(db_path)
    now_str = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{now_str}_{domain}"
    scaling = float(alpha) / float(rank) if rank > 0 else 1.0

    created_iso = datetime.datetime.now(datetime.UTC).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO training_runs (
                run_id, domain, adapter_version, status, dataset_path, n_records,
                rank, alpha, scaling, lr, max_steps, target_dw_w, config_json, created_at
            ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                domain,
                adapter_version,
                str(dataset_path),
                n_records,
                rank,
                alpha,
                scaling,
                lr,
                max_steps,
                target_dw_w,
                json.dumps(config or {}),
                created_iso,
            ),
        )
    return run_id


def record_step(
    run_id: str,
    step: int,
    loss: float | None = None,
    grad_norm: float | None = None,
    learning_rate: float | None = None,
    token_accuracy: float | None = None,
    dw_over_w: float | None = None,
    rel_growth: float | None = None,
    merge_err_pct: float | None = None,
    entropy: float | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Record a single step's metrics into step_metrics."""
    try:
        with get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO step_metrics (
                    run_id, step, loss, grad_norm, learning_rate,
                    token_accuracy, dw_over_w, rel_growth, merge_err_pct, entropy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step,
                    loss,
                    grad_norm,
                    learning_rate,
                    token_accuracy,
                    dw_over_w,
                    rel_growth,
                    merge_err_pct,
                    entropy,
                ),
            )
    except Exception:
        pass


def finish_run(
    run_id: str,
    status: str = "completed",
    stopped_at_step: int | None = None,
    stop_reason: str | None = None,
    final_loss: float | None = None,
    final_token_acc: float | None = None,
    final_dw_w: float | None = None,
    predicted_merge_err: float | None = None,
    runtime_seconds: float | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Mark a training run as finished (completed, failed, or cancelled)."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE training_runs
            SET status = ?,
                stopped_at_step = ?,
                stop_reason = ?,
                final_loss = ?,
                final_token_acc = ?,
                final_dw_w = ?,
                predicted_merge_err = ?,
                runtime_seconds = ?,
                completed_at = ?
            WHERE run_id = ?
            """,
            (
                status,
                stopped_at_step,
                stop_reason,
                final_loss,
                final_token_acc,
                final_dw_w,
                predicted_merge_err,
                runtime_seconds,
                now_iso,
                run_id,
            ),
        )


def update_airm_metrics(
    domain: str,
    airm_scale: float,
    airm_shape: float,
    gate_passed: bool,
    run_id: str | None = None,
    db_path: Path | str | None = None,
) -> None:
    """Update AIRM acceptance gate metrics for a run (or latest run for domain)."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if run_id:
            conn.execute(
                """
                UPDATE training_runs
                SET airm_scale = ?, airm_shape = ?, gate_passed = ?
                WHERE run_id = ?
                """,
                (airm_scale, airm_shape, 1 if gate_passed else 0, run_id),
            )
        else:
            conn.execute(
                """
                UPDATE training_runs
                SET airm_scale = ?, airm_shape = ?, gate_passed = ?
                WHERE rowid = (
                    SELECT rowid FROM training_runs
                    WHERE domain = ? ORDER BY created_at DESC LIMIT 1
                )
                """,
                (airm_scale, airm_shape, 1 if gate_passed else 0, domain),
            )


def list_runs(domain: str | None = None, limit: int = 50, db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """List recent training runs ordered with the newest runs first."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        if domain:
            cur = conn.execute(
                """
                SELECT * FROM training_runs
                WHERE domain = ?
                ORDER BY created_at DESC, rowid DESC LIMIT ?
                """,
                (domain, limit),
            )
        else:
            cur = conn.execute(
                """
                SELECT * FROM training_runs
                ORDER BY created_at DESC, rowid DESC LIMIT ?
                """,
                (limit,),
            )
        return [dict(row) for row in cur.fetchall()]


def get_run(run_id: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Get full details of a run including its step metrics series."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT * FROM training_runs WHERE run_id = ?", (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        run_data = dict(row)

        cur_steps = conn.execute(
            """
            SELECT step, loss, grad_norm, learning_rate, token_accuracy,
                   dw_over_w, rel_growth, merge_err_pct, entropy
            FROM step_metrics
            WHERE run_id = ?
            ORDER BY step ASC
            """,
            (run_id,),
        )
        run_data["steps"] = [dict(s) for s in cur_steps.fetchall()]
        return run_data
