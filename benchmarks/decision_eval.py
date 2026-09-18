"""Decision Engine Candidate Bake-Off Evaluation Benchmark.

Performs live empirical evaluation across candidate backbones:
- Candidate A: ModernBERT-base (149M params, bidirectional encoder)
- Candidate B: Qwen2.5-0.5B (494M params, causal decoder)
- Candidate C: Qwen3.5-0.8B (800M params, hybrid GatedDeltaNet + Attention)

Measures:
1. Decision Accuracy: Top-1 Choice accuracy, Noul F1 & Accuracy, Score MAE
2. Probability Calibration: Expected Calibration Error (ECE < 0.04) and Brier score
3. Single-Pass B=1 Latency: P50, P90, P99 latency via HIP/CUDA stream events or high-res timers
4. Static & Active VRAM Footprint via live telemetry
5. Emits results/benchmarks/decision_engine_bakeoff_scorecard.json
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from apps.factory.decision_engine.data_loader import (
    create_dataloaders,
    generate_synthetic_samples,
)
from apps.factory.decision_engine.models import (
    HARNESS_DOMAINS,
    DecisionHeadConfig,
    ModernBertDecisionEngine,
    QwenDecisionEngine,
)
from apps.factory.decision_engine.train import (
    DecisionEngineLoss,
    calibrate_temperatures,
    compute_brier_score,
    compute_ece,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    name: str
    architecture: str
    model_id: str
    param_count: int
    vram_allocated_mb: float
    vram_reserved_mb: float
    b1_latency_p50_ms: float
    b1_latency_p90_ms: float
    b1_latency_p99_ms: float
    choice_top1_acc: float
    choice_ece: float
    noul_accuracy: float
    noul_f1: float
    noul_brier: float
    noul_ece: float
    score_mae: float
    calibrated_tau_choice: float
    calibrated_tau_noul: float


def measure_b1_latency(
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
    sample_text: str = "Create an HNSW index on embeddings with cosine distance <=> in PostgreSQL 17.",
    num_warmup: int = 10,
    num_runs: int = 50,
) -> tuple[float, float, float]:
    """Measure single-pass batch=1 latency in milliseconds."""
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    model.eval()

    is_cuda = device.type == "cuda" and torch.cuda.is_available()

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        if is_cuda:
            torch.cuda.synchronize(device)

    latencies_ms: list[float] = []

    with torch.no_grad():
        for _ in range(num_runs):
            if is_cuda:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                _ = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                end_evt.record()
                torch.cuda.synchronize(device)
                latencies_ms.append(start_evt.elapsed_time(end_evt))
            else:
                t0 = time.perf_counter()
                _ = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p90 = float(np.percentile(latencies_ms, 90))
    p99 = float(np.percentile(latencies_ms, 99))
    return p50, p90, p99


def evaluate_candidate(
    candidate_name: str,
    architecture: str,
    model_id: str,
    device: torch.device,
    num_samples: int = 300,
    train_epochs: int = 2,
    batch_size: int = 8,
) -> CandidateResult:
    """Instantiate, evaluate, and benchmark a candidate decision model."""
    logger.info("==================================================================")
    logger.info("Evaluating Candidate: %s (%s)", candidate_name, model_id)
    logger.info("==================================================================")

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    head_cfg = DecisionHeadConfig()

    if architecture == "encoder":
        model = ModernBertDecisionEngine(model_name_or_path=model_id, config=head_cfg)
    elif architecture in ("decoder", "hybrid"):
        model = QwenDecisionEngine(model_name_or_path=model_id, config=head_cfg)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    param_count = sum(p.numel() for p in model.parameters())
    model.to(device)

    # Initialize candidate cache
    model.init_harness_cache(tokenizer, device)

    # Record VRAM telemetry
    if device.type == "cuda" and torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
        vram_res = torch.cuda.memory_reserved(device) / (1024 * 1024)
    else:
        vram_alloc = 0.0
        vram_res = 0.0

    logger.info(
        "Model loaded. Params: %s (%.1f M), VRAM: %.1f MB alloc / %.1f MB res",
        f"{param_count:,}",
        param_count / 1e6,
        vram_alloc,
        vram_res,
    )

    # Measure B=1 latency
    p50, p90, p99 = measure_b1_latency(model, tokenizer, device)
    logger.info("B=1 Forward Latency: P50=%.2f ms, P90=%.2f ms, P99=%.2f ms", p50, p90, p99)

    # Create dataset splits
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=num_samples,
    )

    # Fine-tune heads + backbone for alignment
    criterion = DecisionEngineLoss(brier_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

    for epoch in range(1, train_epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss, _ = criterion(out, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Temperature calibration
    best_tau_choice, best_tau_noul = calibrate_temperatures(model, val_loader, device)
    model.choice_head.temperature.copy_(torch.tensor(best_tau_choice))
    model.noul_head.temperature.copy_(torch.tensor(best_tau_noul))

    # Evaluate validation metrics
    model.eval()
    choice_preds: list[int] = []
    choice_targets: list[int] = []
    choice_confs: list[float] = []

    noul_probs: list[float] = []
    noul_targets: list[float] = []

    score_errors: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            c_mask = batch["choice_targets"] != -100
            if c_mask.any() and out.choice is not None:
                probs = out.choice.probabilities[c_mask].float().cpu().numpy()
                targets = batch["choice_targets"][c_mask].numpy()
                choice_preds.extend(np.argmax(probs, axis=-1).tolist())
                choice_targets.extend(targets.tolist())
                choice_confs.extend(np.max(probs, axis=-1).tolist())

            n_mask = batch["noul_targets"] != -100.0
            if n_mask.any() and out.noul is not None:
                probs = out.noul.probability[n_mask].squeeze(-1).float().cpu().numpy()
                targets = batch["noul_targets"][n_mask].numpy()
                noul_probs.extend(probs.tolist())
                noul_targets.extend(targets.tolist())

            s_mask = batch["score_targets"] != -100.0
            if s_mask.any() and out.score is not None:
                preds = out.score.score[s_mask].squeeze(-1).float().cpu().numpy()
                targets = batch["score_targets"][s_mask].numpy()
                score_errors.extend(np.abs(preds - targets).tolist())

    # Calculate metrics
    c_acc = float(np.mean(np.array(choice_preds) == np.array(choice_targets))) if choice_preds else 0.0
    c_ece = compute_ece(np.array(choice_confs), (np.array(choice_preds) == np.array(choice_targets)).astype(float)) if choice_preds else 0.0

    if noul_probs:
        p_arr = np.array(noul_probs)
        t_arr = np.array(noul_targets)
        d_arr = (p_arr >= 0.5).astype(float)
        n_acc = float(np.mean(d_arr == t_arr))
        n_brier = compute_brier_score(p_arr, t_arr)
        confs_arr = np.where(d_arr == 1.0, p_arr, 1.0 - p_arr)
        n_ece = compute_ece(confs_arr, (d_arr == t_arr).astype(float))
        tp = np.sum((d_arr == 1.0) & (t_arr == 1.0))
        fp = np.sum((d_arr == 1.0) & (t_arr == 0.0))
        fn = np.sum((d_arr == 0.0) & (t_arr == 1.0))
        prec = tp / max(tp + fp, 1e-9)
        rec = tp / max(tp + fn, 1e-9)
        n_f1 = float(2 * prec * rec / max(prec + rec, 1e-9))
    else:
        n_acc, n_brier, n_ece, n_f1 = 0.0, 0.0, 0.0, 0.0

    s_mae = float(np.mean(score_errors)) if score_errors else 0.0

    # Cleanup model from device
    del model
    del tokenizer
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    return CandidateResult(
        name=candidate_name,
        architecture=architecture,
        model_id=model_id,
        param_count=param_count,
        vram_allocated_mb=round(vram_alloc, 1),
        vram_reserved_mb=round(vram_res, 1),
        b1_latency_p50_ms=round(p50, 2),
        b1_latency_p90_ms=round(p90, 2),
        b1_latency_p99_ms=round(p99, 2),
        choice_top1_acc=round(c_acc, 4),
        choice_ece=round(c_ece, 4),
        noul_accuracy=round(n_acc, 4),
        noul_f1=round(n_f1, 4),
        noul_brier=round(n_brier, 4),
        noul_ece=round(n_ece, 4),
        score_mae=round(s_mae, 2),
        calibrated_tau_choice=round(best_tau_choice, 3),
        calibrated_tau_noul=round(best_tau_noul, 3),
    )


def run_bakeoff(
    device_str: Optional[str] = None,
    output_path: str = "results/benchmarks/decision_engine_bakeoff_scorecard.json",
    num_samples: int = 300,
    train_epochs: int = 2,
) -> dict[str, Any]:
    """Execute the 3-way candidate bake-off evaluation."""
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Running System One Decision Engine Bake-Off on %s", device)

    candidates = [
        {
            "name": "Candidate A (ModernBERT-base)",
            "architecture": "encoder",
            "model_id": "answerdotai/ModernBERT-base",
        },
        {
            "name": "Candidate B (Qwen2.5-0.5B)",
            "architecture": "decoder",
            "model_id": "Qwen/Qwen2.5-0.5B",
        },
        {
            "name": "Candidate C (Qwen3.5-0.8B)",
            "architecture": "hybrid",
            "model_id": "Qwen/Qwen3.5-0.8B",
        },
    ]

    scorecard: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "target_hardware": "AMD Radeon RX 7900 XTX (gfx1100, 24GB VRAM)",
        "evaluation_criteria": {
            "latency_target_ms": 8.0,
            "ece_target": 0.04,
            "top1_accuracy_target": 0.965,
            "vram_max_mb": 1200.0,
        },
        "candidates": [],
    }

    results = []
    for cand in candidates:
        try:
            res = evaluate_candidate(
                candidate_name=cand["name"],
                architecture=cand["architecture"],
                model_id=cand["model_id"],
                device=device,
                num_samples=num_samples,
                train_epochs=train_epochs,
            )
            results.append(res)
            scorecard["candidates"].append(asdict(res))
        except Exception as e:
            logger.error("Failed to evaluate candidate %s: %s", cand["name"], e, exc_info=True)

    # Determine winner based on Composite System One utility:
    # Utility = Acc / (Latency_P50 * (1 + ECE))
    best_candidate = None
    best_score = -1.0

    for r in results:
        # Penalize higher latency and higher calibration error
        latency_penalty = max(r.b1_latency_p50_ms, 0.5)
        calibration_penalty = 1.0 + (r.choice_ece * 10.0)
        utility = (r.choice_top1_acc * 100.0) / (latency_penalty * calibration_penalty)
        if utility > best_score:
            best_score = utility
            best_candidate = r.name

    scorecard["winner"] = best_candidate
    scorecard["best_utility_score"] = round(best_score, 2)

    # Save scorecard
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(scorecard, f, indent=2)

    logger.info("==================================================================")
    logger.info("Bake-Off Complete! Winner: %s (Score: %.2f)", best_candidate, best_score)
    logger.info("Scorecard saved to %s", out_file)
    logger.info("==================================================================")

    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description="System One Decision Engine Bake-Off")
    parser.add_argument("--device", default=None, help="Device to evaluate on: 'cpu' or 'cuda'")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", default="results/benchmarks/decision_engine_bakeoff_scorecard.json")

    args = parser.parse_args()
    run_bakeoff(
        device_str=args.device,
        output_path=args.output,
        num_samples=args.samples,
        train_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
