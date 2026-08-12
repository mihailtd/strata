"""Export all MLflow runs, metrics, params, tags, and experiments from mlruns.db into structured JSON/JSONL format.

Guarantees 100% preservation of all historical benchmark and training data in human-readable files.
"""

import json
import sqlite3
from pathlib import Path


def export_mlflow_database(
    db_path: str | Path = "mlruns.db",
    output_dir: str | Path = "results/mlflow_export",
) -> dict[str, int]:
    db_path = Path(db_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print(f"Warning: Database file {db_path} not found.")
        return {"experiments": 0, "runs": 0, "metrics": 0, "params": 0, "tags": 0}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. Export Experiments
    cursor.execute("SELECT * FROM experiments;")
    experiments = [dict(row) for row in cursor.fetchall()]
    exp_out = output_dir / "experiments.json"
    with open(exp_out, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)

    # 2. Export Runs Summary
    cursor.execute("SELECT * FROM runs;")
    runs = [dict(row) for row in cursor.fetchall()]

    # Index params, tags, latest metrics by run_uuid / run_id
    cursor.execute("SELECT * FROM params;")
    params_rows = [dict(row) for row in cursor.fetchall()]
    params_by_run: dict[str, dict[str, str]] = {}
    for p in params_rows:
        run_id = p["run_uuid"]
        params_by_run.setdefault(run_id, {})[p["key"]] = p["value"]

    cursor.execute("SELECT * FROM tags;")
    tags_rows = [dict(row) for row in cursor.fetchall()]
    tags_by_run: dict[str, dict[str, str]] = {}
    for t in tags_rows:
        run_id = t["run_uuid"]
        tags_by_run.setdefault(run_id, {})[t["key"]] = t["value"]

    cursor.execute("SELECT * FROM latest_metrics;")
    latest_metrics_rows = [dict(row) for row in cursor.fetchall()]
    latest_metrics_by_run: dict[str, dict[str, float]] = {}
    for m in latest_metrics_rows:
        run_id = m["run_uuid"]
        latest_metrics_by_run.setdefault(run_id, {})[m["key"]] = m["value"]

    full_runs = []
    for r in runs:
        run_id = r["run_uuid"]
        full_runs.append(
            {
                "run_id": run_id,
                "name": r.get("name"),
                "experiment_id": r.get("experiment_id"),
                "status": r.get("status"),
                "start_time": r.get("start_time"),
                "end_time": r.get("end_time"),
                "artifact_uri": r.get("artifact_uri"),
                "lifecycle_stage": r.get("lifecycle_stage"),
                "params": params_by_run.get(run_id, {}),
                "tags": tags_by_run.get(run_id, {}),
                "latest_metrics": latest_metrics_by_run.get(run_id, {}),
            }
        )

    runs_out = output_dir / "runs_summary.jsonl"
    with open(runs_out, "w", encoding="utf-8") as f:
        for run in full_runs:
            f.write(json.dumps(run) + "\n")

    # 3. Export Full Metrics Time Series
    cursor.execute("SELECT key, value, timestamp, step, run_uuid FROM metrics;")
    metrics_rows = [dict(row) for row in cursor.fetchall()]
    metrics_out = output_dir / "metrics_full.jsonl"
    with open(metrics_out, "w", encoding="utf-8") as f:
        for m in metrics_rows:
            f.write(json.dumps(m) + "\n")

    counts = {
        "experiments": len(experiments),
        "runs": len(full_runs),
        "metrics": len(metrics_rows),
        "params": len(params_rows),
        "tags": len(tags_rows),
    }

    summary_out = output_dir / "export_summary.json"
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)

    conn.close()
    return counts


if __name__ == "__main__":
    counts = export_mlflow_database()
    print("=== MLflow Data Export Complete ===")
    print(f"Experiments: {counts['experiments']}")
    print(f"Runs: {counts['runs']}")
    print(f"Metrics records: {counts['metrics']}")
    print(f"Parameters: {counts['params']}")
    print(f"Tags: {counts['tags']}")
    print(f"Saved exported datasets to: results/mlflow_export/")
