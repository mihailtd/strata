"""Master Real LoRA Training & Empirical Evaluation Suite.

Trains real rank-8 LoRA adapters for all 6 engineering domains on GPU:
  1. postgresql
  2. astral
  3. python_web
  4. python_modern
  5. duckdb
  6. financial_planning

Uses memory-safe gradient checkpointing and batch parameters.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

DOMAINS = [
    ("postgresql", "PostgreSQL & pgvector HNSW"),
    ("astral", "Astral UV & Ruff Tooling"),
    ("python_web", "FastAPI & Pydantic v2"),
    ("python_modern", "Python 3.12 Generics"),
    ("duckdb", "DuckDB SQL & QUALIFY"),
    ("financial_planning", "Financial Risk & VaR"),
]


def train_domain(domain: str, max_steps: int = 30) -> dict:
    out_dir = Path(f"results/adapters/m2_{domain}_r8a128_v7_real")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "uv", "run", "python", "scripts/train/train_expert.py",
        "--domain", domain,
        "--v7",
        "--qlora",
        "--batch-size", "1",
        "--grad-accum", "4",
        "--gradient-checkpointing",
        "--max-length", "384",
        "--max-steps", str(max_steps),
        "--logging-steps", "5",
        "--out", str(out_dir)
    ]
    
    print(f"\n▶ [TRAINING] Domain: {domain} | Steps: {max_steps} | Output: {out_dir}", flush=True)
    t0 = time.perf_counter()
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    
    success = (res.returncode == 0) and (out_dir / "adapter_model.safetensors").exists()
    
    # Extract training loss from logs
    loss = None
    for line in res.stdout.splitlines():
        if "train_loss" in line:
            print(f"  {line.strip()}", flush=True)
            try:
                import ast
                d = ast.literal_eval(line.strip()[line.find("{"):])
                loss = d.get("train_loss")
            except Exception:
                pass
    
    print(f"  Result: {'✅ SUCCESS' if success else '❌ FAILED'} in {elapsed:.1f}s (Loss: {loss})", flush=True)
    
    return {
        "domain": domain,
        "success": success,
        "elapsed_s": round(elapsed, 2),
        "train_loss": loss,
        "adapter_path": str(out_dir),
    }


def main():
    print("=" * 80)
    print("🚀 LAUNCHING FULL 6-DOMAIN REAL LORA TRAINING PIPELINE")
    print("   Target: AMD Radeon RX 7900 XTX (hip:0 / cuda:0)")
    print("=" * 80, flush=True)

    summary = []
    t_start = time.perf_counter()

    for domain, label in DOMAINS:
        print(f"\n--- {label} ({domain}) ---")
        info = train_domain(domain, max_steps=30)
        summary.append(info)

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"🏆 ALL 6 DOMAIN LORAS TRAINED IN {total_time/60:.2f} MINUTES")
    print("=" * 80)
    for s in summary:
        print(f"  * {s['domain']:<20} | Success: {'✅' if s['success'] else '❌'} | Loss: {s['train_loss']} | Time: {s['elapsed_s']}s")

    out_file = Path("results/benchmarks/real_lora_training_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\n💾 Training log saved to: {out_file}")


if __name__ == "__main__":
    main()
