"""Neural System One Decision Engine Models.

Non-autoregressive decision models producing calibrated Choice, Score, and Noul outputs
in a single parallel forward pass.

Supports:
- ModernBertDecisionEngine: Bidirectional encoder (e.g. ModernBERT-base/large)
- QwenDecisionEngine / CausalDecisionEngine: Causal decoder (e.g. Qwen2.5-0.5B, Qwen3.5-0.8B)
- Dynamic candidate matching (logits = (W_c * h)^T (W_e * e_i) / sqrt(d))
- Pre-computed candidate caching for fixed specialist adapter domains
- Platt / temperature calibration scaling for Expected Calibration Error (ECE < 0.04)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# -----------------------------------------------------------------------------
# Dynamic Self-Describing Adapter Registry (Domain-Agnostic Invariant)
# -----------------------------------------------------------------------------

from apps.factory.decision_engine.registry import (
    BASELINE_DOMAIN_DESCRIPTIONS,
    AdapterRegistry,
)

# Backwards compatibility aliases
HARNESS_DOMAIN_DESCRIPTIONS: dict[str, str] = BASELINE_DOMAIN_DESCRIPTIONS
HARNESS_DOMAINS: list[str] = list(BASELINE_DOMAIN_DESCRIPTIONS.keys())



# -----------------------------------------------------------------------------
# Configuration and Output Dataclasses
# -----------------------------------------------------------------------------

@dataclass
class DecisionHeadConfig:
    """Hyperparameters for decision engine projection heads."""

    hidden_size: int = 768
    proj_dim: int = 256
    noul_hidden_dim: int = 256
    score_hidden_dim: int = 256
    score_max: float = 100.0
    temperature_choice: float = 1.0
    temperature_noul: float = 1.0
    dropout: float = 0.1
    pooling_mode: str = "mean"  # "mean" or "cls" for encoder, "last" for decoder
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class ChoiceOutput:
    """Output from the Dynamic Candidate Choice head."""

    logits: torch.Tensor  # [B, K]
    probabilities: torch.Tensor  # [B, K]
    chosen_index: list[int]
    confidence: list[float]
    chosen_label: list[str] | None = None


@dataclass
class NoulOutput:
    """Output from the Calibrated Boolean Noul gate."""

    logit: torch.Tensor  # [B, 1]
    probability: torch.Tensor  # [B, 1]
    decision: list[bool]  # thresholded at 0.5
    confidence: list[float]


@dataclass
class ScoreOutput:
    """Output from the Continuous Rubric Score head."""

    raw_logit: torch.Tensor  # [B, 1]
    score: torch.Tensor  # [B, 1] scaled to [0, score_max]
    scores_list: list[float]


@dataclass
class DecisionOutputs:
    """Consolidated container for decision engine outputs."""

    choice: ChoiceOutput | None = None
    noul: NoulOutput | None = None
    score: ScoreOutput | None = None
    pooled_hidden: torch.Tensor | None = None


# -----------------------------------------------------------------------------
# Head Modules
# -----------------------------------------------------------------------------

class DynamicChoiceHead(nn.Module):
    """Dynamic Candidate Matching Head.

    Computes categorical selection over K dynamic or pre-cached candidates:
        logits_i = (LayerNorm(W_c * h))^T (LayerNorm(W_e * e_i)) / sqrt(D_proj) / tau
    """

    def __init__(self, config: DecisionHeadConfig, cand_dim: int | None = None) -> None:
        super().__init__()
        self.config = config
        cand_dim = cand_dim or config.hidden_size
        self.proj_dim = config.proj_dim

        self.context_proj = nn.Linear(config.hidden_size, self.proj_dim, bias=False)
        self.context_ln = nn.LayerNorm(self.proj_dim)

        self.cand_proj = nn.Linear(cand_dim, self.proj_dim, bias=False)
        self.cand_ln = nn.LayerNorm(self.proj_dim)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0

        # Learned or post-hoc calibrated temperature
        self.temperature = nn.Parameter(
            torch.tensor(float(config.temperature_choice)), requires_grad=False
        )

        # Optional cached candidate projections: [K, proj_dim]
        self.register_buffer("cached_cand_projs", None, persistent=False)
        self.cached_labels: list[str] | None = None

    def set_cached_candidates(
        self,
        candidate_embeddings: torch.Tensor,
        labels: list[str] | None = None,
    ) -> None:
        """Cache pre-projected candidate embeddings for instant O(1) fixed routing.

        Args:
            candidate_embeddings: [K, cand_dim] tensor.
            labels: list of K label strings (e.g. HARNESS_DOMAINS).
        """
        with torch.no_grad():
            candidate_embeddings = candidate_embeddings.to(self.cand_proj.weight.dtype)
            cand_proj = self.cand_proj(candidate_embeddings)
            cand_proj = self.cand_ln(cand_proj)
            # Normalize for cosine-like stability
            cand_proj = F.normalize(cand_proj, p=2, dim=-1)
            self.cached_cand_projs = cand_proj
            self.cached_labels = list(labels) if labels is not None else None

    def clear_cached_candidates(self) -> None:
        """Clear cached candidate projections."""
        self.cached_cand_projs = None
        self.cached_labels = None

    def forward(
        self,
        pooled_context: torch.Tensor,
        candidate_embeddings: torch.Tensor | None = None,
        candidate_labels: list[str] | None = None,
        temperature: float | None = None,
    ) -> ChoiceOutput:
        """Forward pass for dynamic choice matching.

        Args:
            pooled_context: [B, hidden_size]
            candidate_embeddings: [B, K, cand_dim] or [K, cand_dim].
                                  If None, uses cached_cand_projs.
            candidate_labels: Optional string labels for candidates.
            temperature: Optional override for temperature scaling.

        Returns:
            ChoiceOutput with logits, probabilities, chosen indices, and confidence.
        """
        B = pooled_context.shape[0]
        temp = temperature if temperature is not None else float(self.temperature)
        temp = max(temp, 1e-4)

        # Cast context to projection dtype
        pooled_context = pooled_context.to(self.context_proj.weight.dtype)

        # Project context: [B, proj_dim]
        ctx_proj = self.context_proj(self.dropout(pooled_context))
        ctx_proj = self.context_ln(ctx_proj)
        ctx_proj = F.normalize(ctx_proj, p=2, dim=-1)

        labels = candidate_labels

        if candidate_embeddings is not None:
            # Dynamic candidates
            candidate_embeddings = candidate_embeddings.to(self.cand_proj.weight.dtype)
            if candidate_embeddings.dim() == 2:
                # [K, cand_dim] shared across batch
                c_proj = self.cand_proj(candidate_embeddings)
                c_proj = self.cand_ln(c_proj)
                c_proj = F.normalize(c_proj, p=2, dim=-1)  # [K, proj_dim]
                # Matmul: [B, proj_dim] x [proj_dim, K] -> [B, K]
                logits = torch.matmul(ctx_proj, c_proj.t()) * self.scale
            elif candidate_embeddings.dim() == 3:
                # [B, K, cand_dim] per-sample candidates
                c_proj = self.cand_proj(candidate_embeddings)
                c_proj = self.cand_ln(c_proj)
                c_proj = F.normalize(c_proj, p=2, dim=-1)  # [B, K, proj_dim]
                # Batch matmul: [B, 1, proj_dim] x [B, proj_dim, K] -> [B, 1, K] -> [B, K]
                logits = torch.bmm(ctx_proj.unsqueeze(1), c_proj.transpose(1, 2)).squeeze(1) * self.scale
            else:
                raise ValueError(
                    f"candidate_embeddings must be 2D or 3D, got {candidate_embeddings.shape}"
                )
        elif self.cached_cand_projs is not None:
            # Fast cached path: single gemm [B, proj_dim] x [proj_dim, K]
            logits = torch.matmul(ctx_proj, self.cached_cand_projs.t()) * self.scale
            if labels is None:
                labels = self.cached_labels
        else:
            raise ValueError(
                "Either candidate_embeddings must be provided or set_cached_candidates() called."
            )

        # Scale by calibration temperature
        scaled_logits = logits / temp
        probs = F.softmax(scaled_logits, dim=-1)

        chosen_indices = torch.argmax(probs, dim=-1).tolist()
        confidences = torch.max(probs, dim=-1).values.tolist()

        chosen_names = None
        if labels is not None:
            chosen_names = [labels[idx] for idx in chosen_indices]

        return ChoiceOutput(
            logits=scaled_logits,
            probabilities=probs,
            chosen_index=chosen_indices,
            confidence=confidences,
            chosen_label=chosen_names,
        )


class NoulHead(nn.Module):
    """Calibrated Boolean Gate Head (Noul).

    Computes calibrated probability for binary readiness gates, verification checks,
    and if/else conditions via a 2-layer MLP with GELU and temperature scaling:
        logit = W_2 * GELU(W_1 * h + b_1) + b_2
        P(True) = sigmoid(logit / tau_noul)
    """

    def __init__(self, config: DecisionHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.noul_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.noul_hidden_dim, 1),
        )
        self.temperature = nn.Parameter(
            torch.tensor(float(config.temperature_noul)), requires_grad=False
        )

    def forward(
        self,
        pooled_context: torch.Tensor,
        threshold: float = 0.5,
        temperature: float | None = None,
    ) -> NoulOutput:
        """Forward pass for boolean readiness prediction.

        Args:
            pooled_context: [B, hidden_size]
            threshold: Probability decision threshold (default 0.5).
            temperature: Optional override for temperature scaling.

        Returns:
            NoulOutput with logit, probability, boolean decision, and confidence.
        """
        temp = temperature if temperature is not None else float(self.temperature)
        temp = max(temp, 1e-4)

        pooled_context = pooled_context.to(self.mlp[0].weight.dtype)
        raw_logit = self.mlp(pooled_context)  # [B, 1]
        scaled_logit = raw_logit / temp
        prob = torch.sigmoid(scaled_logit)  # [B, 1]

        prob_flat = prob.squeeze(-1).tolist()
        if not isinstance(prob_flat, list):
            prob_flat = [prob_flat]

        decisions = [p >= threshold for p in prob_flat]
        confidences = [p if d else (1.0 - p) for p, d in zip(prob_flat, decisions)]

        return NoulOutput(
            logit=scaled_logit,
            probability=prob,
            decision=decisions,
            confidence=confidences,
        )


class ScoreHead(nn.Module):
    """Continuous Rubric Score Head.

    Computes bounded continuous scalar evaluation over code diff quality, test pass
    likelihood, or progress metrics:
        Score = S_max * sigmoid(W_2 * GELU(W_1 * h + b_1) + b_2)
    """

    def __init__(self, config: DecisionHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.score_max = float(config.score_max)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.score_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.score_hidden_dim, 1),
        )

    def forward(self, pooled_context: torch.Tensor) -> ScoreOutput:
        """Forward pass for continuous score regression.

        Args:
            pooled_context: [B, hidden_size]

        Returns:
            ScoreOutput with raw_logit, score in [0, S_max], and python float list.
        """
        pooled_context = pooled_context.to(self.mlp[0].weight.dtype)
        raw_logit = self.mlp(pooled_context)  # [B, 1]
        norm_score = torch.sigmoid(raw_logit)  # [B, 1] in [0, 1]
        score = norm_score * self.score_max  # [B, 1] in [0, score_max]

        scores_list = score.squeeze(-1).tolist()
        if not isinstance(scores_list, list):
            scores_list = [scores_list]

        return ScoreOutput(
            raw_logit=raw_logit,
            score=score,
            scores_list=scores_list,
        )


# -----------------------------------------------------------------------------
# Complete Decision Engine Architectures
# -----------------------------------------------------------------------------

class ModernBertDecisionEngine(nn.Module):
    """ModernBERT Bidirectional Encoder Decision Engine.

    Passes text through ModernBertModel and pools representations (mean or CLS)
    into parallel DynamicChoiceHead, NoulHead, and ScoreHead.
    """

    def __init__(
        self,
        model_name_or_path: str = "answerdotai/ModernBERT-base",
        config: DecisionHeadConfig | None = None,
        backbone: PreTrainedModel | None = None,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path

        if backbone is not None:
            self.backbone = backbone
        else:
            token = os.environ.get("HF_TOKEN")
            try:
                self.backbone = AutoModel.from_pretrained(
                    model_name_or_path,
                    local_files_only=True,
                    token=token,
                )
            except Exception:
                self.backbone = AutoModel.from_pretrained(
                    model_name_or_path,
                    token=token,
                )

        hidden_size = getattr(self.backbone.config, "hidden_size", 768)

        self.config = config or DecisionHeadConfig(
            hidden_size=hidden_size,
            pooling_mode="mean",
        )
        self.config.hidden_size = hidden_size

        self.choice_head = DynamicChoiceHead(self.config)
        self.noul_head = NoulHead(self.config)
        self.score_head = ScoreHead(self.config)

        dtype = getattr(self.backbone, "dtype", torch.float32)
        if dtype is not None and dtype != torch.float32:
            self.choice_head.to(dtype)
            self.noul_head.to(dtype)
            self.score_head.to(dtype)

    def pool_representation(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool token representations into a single sentence vector."""
        if self.config.pooling_mode == "cls":
            return last_hidden_state[:, 0]

        # Default: mean pooling masked by attention_mask
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)

        mask_expanded = attention_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def get_context_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract pooled context embedding for input text."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.pool_representation(outputs.last_hidden_state, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        candidate_embeddings: torch.Tensor | None = None,
        candidate_labels: list[str] | None = None,
        tasks: Sequence[str] = ("choice", "noul", "score"),
    ) -> DecisionOutputs:
        """Unified parallel forward pass across requested heads."""
        pooled = self.get_context_embedding(input_ids, attention_mask)

        choice_out = None
        noul_out = None
        score_out = None

        if "choice" in tasks:
            choice_out = self.choice_head(
                pooled,
                candidate_embeddings=candidate_embeddings,
                candidate_labels=candidate_labels,
            )
        if "noul" in tasks:
            noul_out = self.noul_head(pooled)
        if "score" in tasks:
            score_out = self.score_head(pooled)

        return DecisionOutputs(
            choice=choice_out,
            noul=noul_out,
            score=score_out,
            pooled_hidden=pooled,
        )

    def init_from_registry(self, registry: AdapterRegistry, tokenizer: Any, device: torch.device) -> None:
        """Dynamically pre-compute and cache candidate projections for all discovered adapters."""
        domains = registry.domains
        descriptions = registry.descriptions
        tokens = tokenizer(descriptions, padding=True, return_tensors="pt").to(device)

        self.eval()
        with torch.no_grad():
            cand_embs = self.get_context_embedding(
                tokens["input_ids"], tokens["attention_mask"]
            )
            self.choice_head.set_cached_candidates(cand_embs, labels=domains)

    def init_harness_cache(self, tokenizer: Any, device: torch.device) -> None:
        """Initialize candidate cache from the dynamic adapter registry."""
        registry = AdapterRegistry()
        self.init_from_registry(registry, tokenizer, device)


class QwenDecisionEngine(nn.Module):
    """Causal Decoder Decision Engine (e.g. Qwen 2.5 0.5B / Qwen 3.5 0.8B).

    Extracts representations from the last non-padded prompt token (prefix pooling)
    and passes them to parallel DynamicChoiceHead, NoulHead, and ScoreHead.
    Supports PEFT LoRA adapters for memory-efficient fine-tuning.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B",
        config: DecisionHeadConfig | None = None,
        backbone: PreTrainedModel | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path

        if backbone is not None:
            self.backbone = backbone
        else:
            self.backbone = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )

        # Detect hidden size across architectures (Qwen 2.5 vs Qwen 3.5 text_config)
        cfg = self.backbone.config
        if hasattr(cfg, "hidden_size") and cfg.hidden_size is not None:
            hidden_size = cfg.hidden_size
        elif hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            hidden_size = cfg.text_config.hidden_size
        else:
            hidden_size = 896  # fallback default

        self.config = config or DecisionHeadConfig(
            hidden_size=hidden_size,
            pooling_mode="last",
        )
        self.config.hidden_size = hidden_size

        # Optional LoRA configuration
        if self.config.use_lora:
            try:
                from peft import LoraConfig, get_peft_model

                peft_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=self.config.target_modules,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type="FEATURE_EXTRACTION",
                )
                self.backbone = get_peft_model(self.backbone, peft_config)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning("Could not initialize LoRA: %s", e)

        self.choice_head = DynamicChoiceHead(self.config)
        self.noul_head = NoulHead(self.config)
        self.score_head = ScoreHead(self.config)

        dtype = getattr(self.backbone, "dtype", torch.float32)
        if dtype is not None and dtype != torch.float32:
            self.choice_head.to(dtype)
            self.noul_head.to(dtype)
            self.score_head.to(dtype)

    def pool_representation(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract representation from the last non-padded token."""
        if attention_mask is None:
            # Assume no padding, take last position
            return last_hidden_state[:, -1, :]

        # Find index of last 1 in attention_mask for each sample
        seq_lengths = attention_mask.sum(dim=1).long() - 1  # [B]
        seq_lengths = torch.clamp(seq_lengths, min=0)
        B = last_hidden_state.shape[0]
        batch_idx = torch.arange(B, device=last_hidden_state.device)
        return last_hidden_state[batch_idx, seq_lengths, :]

    def get_context_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract pooled context embedding for input text."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.pool_representation(outputs.last_hidden_state, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        candidate_embeddings: torch.Tensor | None = None,
        candidate_labels: list[str] | None = None,
        tasks: Sequence[str] = ("choice", "noul", "score"),
    ) -> DecisionOutputs:
        """Unified parallel forward pass across requested heads."""
        pooled = self.get_context_embedding(input_ids, attention_mask)

        choice_out = None
        noul_out = None
        score_out = None

        if "choice" in tasks:
            choice_out = self.choice_head(
                pooled,
                candidate_embeddings=candidate_embeddings,
                candidate_labels=candidate_labels,
            )
        if "noul" in tasks:
            noul_out = self.noul_head(pooled)
        if "score" in tasks:
            score_out = self.score_head(pooled)

        return DecisionOutputs(
            choice=choice_out,
            noul=noul_out,
            score=score_out,
            pooled_hidden=pooled,
        )

    def init_from_registry(self, registry: AdapterRegistry, tokenizer: Any, device: torch.device) -> None:
        """Dynamically pre-compute and cache candidate projections for all discovered adapters."""
        domains = registry.domains
        descriptions = registry.descriptions
        tokens = tokenizer(descriptions, padding=True, return_tensors="pt").to(device)

        self.eval()
        with torch.no_grad():
            cand_embs = self.get_context_embedding(
                tokens["input_ids"], tokens["attention_mask"]
            )
            self.choice_head.set_cached_candidates(cand_embs, labels=domains)

    def init_harness_cache(self, tokenizer: Any, device: torch.device) -> None:
        """Initialize candidate cache from the dynamic adapter registry."""
        registry = AdapterRegistry()
        self.init_from_registry(registry, tokenizer, device)


# Alias for backward compatibility
CausalDecisionEngine = QwenDecisionEngine
