"""Training and Calibration script for System One Decision Engines.

Implements multi-task composite loss with Brier score regularization,
post-hoc temperature calibration scaling for ECE < 0.04, and validation metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

from .data_loader import create_dataloaders
from .models import (
    DecisionHeadConfig,
    DecisionOutputs,
    ModernBertDecisionEngine,
    QwenDecisionEngine,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Calibration Metrics: Expected Calibration Error (ECE) & Brier Score
# -----------------------------------------------------------------------------

def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error across n_bins equal probability intervals."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Compute mean squared error between probabilities and binary targets."""
    if len(probabilities) == 0:
        return 0.0
    return float(np.mean((probabilities - targets) ** 2))


# -----------------------------------------------------------------------------
# Multi-Task Loss with Brier Regularization
# -----------------------------------------------------------------------------

class DecisionEngineLoss(nn.Module):
    """Composite loss function with task loss, Brier score penalty, and Huber loss."""

    def __init__(
        self,
        brier_weight: float = 0.5,
        score_max: float = 100.0,
    ) -> None:
        super().__init__()
        self.brier_weight = brier_weight
        self.score_max = score_max
        self.huber_loss = nn.SmoothL1Loss(reduction="mean")

    def forward(
        self,
        outputs: DecisionOutputs,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss = torch.tensor(0.0, device=outputs.pooled_hidden.device)
        metrics: dict[str, float] = {}

        # 1. Choice Loss (Cross-Entropy over active choice samples)
        choice_targets = batch["choice_targets"].to(outputs.pooled_hidden.device)
        choice_mask = choice_targets != -100
        if choice_mask.any() and outputs.choice is not None:
            c_logits = outputs.choice.logits[choice_mask]
            c_targets = choice_targets[choice_mask]
            choice_loss = F.cross_entropy(c_logits, c_targets)
            loss = loss + choice_loss
            metrics["loss_choice"] = float(choice_loss.item())

        # 2. Noul Loss (BCE + Brier Score Penalty)
        noul_targets = batch["noul_targets"].to(outputs.pooled_hidden.device)
        noul_mask = noul_targets != -100.0
        if noul_mask.any() and outputs.noul is not None:
            n_logits = outputs.noul.logit[noul_mask].squeeze(-1)
            n_probs = outputs.noul.probability[noul_mask].squeeze(-1)
            n_targets = noul_targets[noul_mask]

            bce = F.binary_cross_entropy_with_logits(n_logits, n_targets)
            # Brier penalty: MSE between probabilities and binary targets
            brier = torch.mean((n_probs - n_targets) ** 2)
            noul_loss = bce + self.brier_weight * brier

            loss = loss + noul_loss
            metrics["loss_noul"] = float(noul_loss.item())
            metrics["brier_noul"] = float(brier.item())

        # 3. Score Loss (Huber / SmoothL1 on normalized [0, 1] range)
        score_targets = batch["score_targets"].to(outputs.pooled_hidden.device)
        score_mask = score_targets != -100.0
        if score_mask.any() and outputs.score is not None:
            norm_predicted = outputs.score.score[score_mask].squeeze(-1) / self.score_max
            norm_targets = score_targets[score_mask] / self.score_max

            score_loss = self.huber_loss(norm_predicted, norm_targets)
            loss = loss + score_loss
            metrics["loss_score"] = float(score_loss.item())

        metrics["loss_total"] = float(loss.item())
        return loss, metrics


# -----------------------------------------------------------------------------
# Post-Hoc Temperature Optimization
# -----------------------------------------------------------------------------

def calibrate_temperatures(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Optimize validation temperatures tau_choice and tau_noul using grid/line search."""
    model.eval()
    all_choice_logits: list[torch.Tensor] = []
    all_choice_targets: list[torch.Tensor] = []

    all_noul_logits: list[torch.Tensor] = []
    all_noul_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Choice
            c_mask = batch["choice_targets"] != -100
            if c_mask.any() and outputs.choice is not None:
                # Get unscaled logits (before temperature)
                cur_temp = float(model.choice_head.temperature)
                unscaled_c = outputs.choice.logits[c_mask] * cur_temp
                all_choice_logits.append(unscaled_c.float().cpu())
                all_choice_targets.append(batch["choice_targets"][c_mask])

            # Noul
            n_mask = batch["noul_targets"] != -100.0
            if n_mask.any() and outputs.noul is not None:
                cur_temp_noul = float(model.noul_head.temperature)
                unscaled_n = outputs.noul.logit[n_mask] * cur_temp_noul
                all_noul_logits.append(unscaled_n.float().cpu().squeeze(-1))
                all_noul_targets.append(batch["noul_targets"][n_mask])

    best_tau_choice = 1.0
    if all_choice_logits:
        c_logits = torch.cat(all_choice_logits, dim=0).numpy()
        c_targets = torch.cat(all_choice_targets, dim=0).numpy()

        best_ece = float("inf")
        # Grid search over temperatures [0.2, 5.0]
        for tau in np.linspace(0.2, 4.0, 39):
            probs = np.exp(c_logits / tau) / np.sum(np.exp(c_logits / tau), axis=-1, keepdims=True)
            confs = np.max(probs, axis=-1)
            preds = np.argmax(probs, axis=-1)
            accs = (preds == c_targets).astype(float)
            ece = compute_ece(confs, accs)
            if ece < best_ece:
                best_ece = ece
                best_tau_choice = float(tau)

    best_tau_noul = 1.0
    if all_noul_logits:
        n_logits = torch.cat(all_noul_logits, dim=0).numpy()
        n_targets = torch.cat(all_noul_targets, dim=0).numpy()

        best_ece = float("inf")
        for tau in np.linspace(0.2, 4.0, 39):
            probs = 1.0 / (1.0 + np.exp(-n_logits / tau))
            preds = (probs >= 0.5).astype(float)
            confs = np.where(preds == 1.0, probs, 1.0 - probs)
            accs = (preds == n_targets).astype(float)
            ece = compute_ece(confs, accs)
            if ece < best_ece:
                best_ece = ece
                best_tau_noul = float(tau)

    return best_tau_choice, best_tau_noul


# -----------------------------------------------------------------------------
# Validation Evaluation Routine
# -----------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: DecisionEngineLoss,
    device: torch.device,
) -> dict[str, float]:
    """Run full evaluation on validation split."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    choice_preds: list[int] = []
    choice_targets: list[int] = []
    choice_confs: list[float] = []

    noul_probs: list[float] = []
    noul_targets: list[float] = []

    score_errors: list[float] = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, _ = criterion(outputs, batch)
            total_loss += loss.item()
            num_batches += 1

            # Choice metrics
            c_mask = batch["choice_targets"] != -100
            if c_mask.any() and outputs.choice is not None:
                probs = outputs.choice.probabilities[c_mask].float().cpu().numpy()
                targets = batch["choice_targets"][c_mask].numpy()
                preds = np.argmax(probs, axis=-1)
                confs = np.max(probs, axis=-1)

                choice_preds.extend(preds.tolist())
                choice_targets.extend(targets.tolist())
                choice_confs.extend(confs.tolist())

            # Noul metrics
            n_mask = batch["noul_targets"] != -100.0
            if n_mask.any() and outputs.noul is not None:
                probs = outputs.noul.probability[n_mask].squeeze(-1).float().cpu().numpy()
                targets = batch["noul_targets"][n_mask].numpy()

                noul_probs.extend(probs.tolist())
                noul_targets.extend(targets.tolist())

            # Score metrics
            s_mask = batch["score_targets"] != -100.0
            if s_mask.any() and outputs.score is not None:
                preds = outputs.score.score[s_mask].squeeze(-1).float().cpu().numpy()
                targets = batch["score_targets"][s_mask].numpy()
                errors = np.abs(preds - targets)
                score_errors.extend(errors.tolist())

    results: dict[str, float] = {
        "val_loss": total_loss / max(num_batches, 1),
    }

    if choice_preds:
        preds_arr = np.array(choice_preds)
        targets_arr = np.array(choice_targets)
        confs_arr = np.array(choice_confs)

        acc = float(np.mean(preds_arr == targets_arr))
        ece = compute_ece(confs_arr, (preds_arr == targets_arr).astype(float))
        results["choice_top1_accuracy"] = acc
        results["choice_ece"] = ece

    if noul_probs:
        p_arr = np.array(noul_probs)
        t_arr = np.array(noul_targets)
        d_arr = (p_arr >= 0.5).astype(float)

        acc = float(np.mean(d_arr == t_arr))
        brier = compute_brier_score(p_arr, t_arr)
        confs_arr = np.where(d_arr == 1.0, p_arr, 1.0 - p_arr)
        ece = compute_ece(confs_arr, (d_arr == t_arr).astype(float))

        tp = np.sum((d_arr == 1.0) & (t_arr == 1.0))
        fp = np.sum((d_arr == 1.0) & (t_arr == 0.0))
        fn = np.sum((d_arr == 0.0) & (t_arr == 1.0))
        precision = tp / max(tp + fp, 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        results["noul_accuracy"] = acc
        results["noul_f1"] = float(f1)
        results["noul_brier"] = brier
        results["noul_ece"] = ece

    if score_errors:
        results["score_mae"] = float(np.mean(score_errors))

    return results


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------

def train_decision_engine(
    model_type: str = "modernbert",
    model_name_or_path: str = "answerdotai/ModernBERT-base",
    output_dir: str = "results/models/decision_engine_v1",
    num_epochs: int = 5,
    batch_size: int = 16,
    lr: float = 3e-4,
    device_str: str | None = None,
    use_lora: bool = False,
    num_samples: int = 1200,
) -> dict[str, Any]:
    """Train, calibrate, and save a System One Decision Engine."""
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Initializing training on device: %s", device)
    logger.info("Model type: %s, backbone: %s", model_type, model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    head_config = DecisionHeadConfig(use_lora=use_lora)

    if model_type == "modernbert":
        model = ModernBertDecisionEngine(
            model_name_or_path=model_name_or_path,
            config=head_config,
        )
    elif model_type in ("qwen", "causal"):
        model = QwenDecisionEngine(
            model_name_or_path=model_name_or_path,
            config=head_config,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(device)

    # Initialize candidate cache on device
    model.init_harness_cache(tokenizer, device)

    # Prepare datasets
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        num_samples=num_samples,
    )

    criterion = DecisionEngineLoss(brier_weight=0.5, score_max=100.0)

    # Separate head parameters from backbone parameters
    head_params = (
        list(model.choice_head.parameters())
        + list(model.noul_head.parameters())
        + list(model.score_head.parameters())
    )

    optimizer = AdamW(
        [
            {"params": head_params, "lr": lr},
            {"params": model.backbone.parameters(), "lr": lr * 0.1},
        ],
        weight_decay=0.01,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

    logger.info("Starting training loop: %d epochs, %d batches/epoch", num_epochs, len(train_loader))

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            loss, metrics = criterion(outputs, batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f - Val Loss: %.4f - Choice Acc: %.2f%% - Noul F1: %.3f - Score MAE: %.2f",
            epoch,
            num_epochs,
            avg_loss,
            val_metrics.get("val_loss", 0.0),
            val_metrics.get("choice_top1_accuracy", 0.0) * 100.0,
            val_metrics.get("noul_f1", 0.0),
            val_metrics.get("score_mae", 0.0),
        )

    # Post-Hoc Temperature Calibration
    logger.info("Calibrating temperatures on validation set...")
    best_tau_choice, best_tau_noul = calibrate_temperatures(model, val_loader, device)
    model.choice_head.temperature.copy_(torch.tensor(best_tau_choice))
    model.noul_head.temperature.copy_(torch.tensor(best_tau_noul))
    logger.info(
        "Calibrated temperatures: tau_choice=%.3f, tau_noul=%.3f",
        best_tau_choice,
        best_tau_noul,
    )

    # Final post-calibration validation metrics
    final_metrics = evaluate(model, val_loader, criterion, device)
    final_metrics["temperature_choice"] = best_tau_choice
    final_metrics["temperature_noul"] = best_tau_noul
    logger.info("Final Calibrated Metrics: %s", final_metrics)

    # Save model checkpoint and metadata
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "model_name_or_path": model_name_or_path,
            "config": head_config,
            "final_metrics": final_metrics,
        },
        out_path / "decision_engine.pt",
    )

    with open(out_path / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("Model and metrics successfully saved to %s", out_path)
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train System One Decision Engine")
    parser.add_argument("--model-type", default="modernbert", choices=["modernbert", "qwen", "causal"])
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--output-dir", default="results/models/decision_engine_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--num-samples", type=int, default=1200)

    args = parser.parse_args()
    train_decision_engine(
        model_type=args.model_type,
        model_name_or_path=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        use_lora=args.use_lora,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
