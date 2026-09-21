"""Dynamic Mixture-of-Adapters (Multi-Expert LoRA Stacker).

Performs exact weight-space block concatenation and linear task arithmetic merging
across multiple specialist LoRA adapters:

  B_stacked = [sqrt(γ_1)*B_1, sqrt(γ_2)*B_2, ..., sqrt(γ_K)*B_K]  (dim=1)
  A_stacked = [sqrt(γ_1)*A_1; sqrt(γ_2)*A_2; ...; sqrt(γ_K)*A_K]  (dim=0)

Mathematically guarantees:
  B_stacked @ A_stacked ≡ sum_{k=1}^K γ_k * (B_k @ A_k)

Enables simultaneous multi-domain mastery (e.g. PostgreSQL + FastAPI + DuckDB)
in a single unified low-rank adapter without mathematical distortion.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class DynamicAdapterStacker:
    """Combines multiple LoRA safetensors into a unified multi-expert adapter."""

    def __init__(self, adapters_dir: str | Path = REPO_ROOT / "results" / "adapters"):
        self.adapters_dir = Path(adapters_dir)

    def stack_adapters(
        self,
        expert_weights: dict[str, float],
        output_path: str | Path | None = None,
        device: str = "cpu",
        normalize_weights: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Stacks multiple domain adapters with given mixture weights.

        Args:
            expert_weights: Mapping of adapter_name/path -> weight gamma_k.
            output_path: Optional path to save fused adapter_model.safetensors.
            device: Compute device ('cpu' or 'cuda'/'hip').
            normalize_weights: If True, weights are normalized to sum to 1.0. If False,
                weights represent direct unattenuated multipliers (e.g. 1.0 for each expert).

        Returns:
            Tuple of (fused_state_dict, fused_config)
        """
        t0 = time.perf_counter()
        # 1. Process weights
        if normalize_weights:
            total_w = sum(expert_weights.values())
            if total_w <= 0:
                raise ValueError(f"Sum of expert weights must be > 0, got {total_w}")
            norm_weights = {k: v / total_w for k, v in expert_weights.items()}
        else:
            norm_weights = {k: float(v) for k, v in expert_weights.items()}

        # 2. Resolve adapter paths
        loaded_tensors: list[tuple[str, float, dict[str, torch.Tensor]]] = []
        base_config: dict[str, Any] | None = None

        for name, gamma in norm_weights.items():
            adapter_file = self._resolve_adapter_path(name)
            tensors = load_file(str(adapter_file), device=device)
            loaded_tensors.append((name, gamma, tensors))

            # Load adapter_config.json if available
            config_file = adapter_file.parent / "adapter_config.json"
            if config_file.exists() and base_config is None:
                with open(config_file) as f:
                    base_config = json.load(f)

        if not loaded_tensors:
            raise ValueError("No adapters loaded for stacking")

        # 3. Discover all module keys
        all_keys = set()
        for _, _, tensors in loaded_tensors:
            all_keys.update(tensors.keys())

        # Separate lora_A and lora_B keys
        a_keys = {k for k in all_keys if "lora_A" in k}
        b_keys = {k for k in all_keys if "lora_B" in k}

        fused_state_dict: dict[str, torch.Tensor] = {}
        total_rank = 0

        # 4. Fuse matching A and B matrices
        for a_key in sorted(a_keys):
            b_key = a_key.replace("lora_A", "lora_B")

            a_parts = []
            b_parts = []

            for _name, gamma, tensors in loaded_tensors:
                if a_key in tensors and b_key in tensors:
                    a_t = tensors[a_key].to(device)
                    b_t = tensors[b_key].to(device)

                    sqrt_gamma = math.sqrt(gamma)
                    a_parts.append(a_t * sqrt_gamma)
                    b_parts.append(b_t * sqrt_gamma)

            if a_parts and b_parts:
                first_a = a_parts[0]
                first_b = b_parts[0]

                if first_a.shape[0] == first_b.shape[1] and (
                    first_a.shape[1] != first_b.shape[0] or first_a.shape[0] < first_a.shape[1]
                ):
                    # Standard HuggingFace PEFT format: A is (r, in_features), B is (out_features, r)
                    # Concat A along dim=0 (rank rows), Concat B along dim=1 (rank cols)
                    a_fused = torch.cat(a_parts, dim=0)
                    b_fused = torch.cat(b_parts, dim=1)
                    total_rank = a_fused.shape[0]
                elif first_a.shape[1] == first_b.shape[0]:
                    # Native 27B GGUF format: A is (in_features, r), B is (r, out_features)
                    # Concat A along dim=1 (rank cols), Concat B along dim=0 (rank rows)
                    a_fused = torch.cat(a_parts, dim=1)
                    b_fused = torch.cat(b_parts, dim=0)
                    total_rank = a_fused.shape[1]
                elif first_a.shape[0] == first_b.shape[1]:
                    # Fallback HuggingFace PEFT format
                    a_fused = torch.cat(a_parts, dim=0)
                    b_fused = torch.cat(b_parts, dim=1)
                    total_rank = a_fused.shape[0]
                else:
                    raise ValueError(f"Incompatible LoRA shapes for key {a_key}: A={first_a.shape}, B={first_b.shape}")

                fused_state_dict[a_key] = a_fused
                fused_state_dict[b_key] = b_fused

        # Copy any auxiliary non-A/B tensors if present
        for key in all_keys - a_keys - b_keys:
            # Weighted average for biases/scalings if any
            accum = None
            for _, gamma, tensors in loaded_tensors:
                if key in tensors:
                    t = tensors[key].to(device) * gamma
                    accum = t if accum is None else accum + t
            if accum is not None:
                fused_state_dict[key] = accum

        # 5. Build fused config
        fused_config = base_config.copy() if base_config else {}
        fused_config["r"] = total_rank
        fused_config["expert_mixture"] = norm_weights
        fused_config["stacked_at_timestamp"] = time.time()

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"✅ Fused {len(loaded_tensors)} LoRA experts (Rank-{total_rank}) in {elapsed_ms:.2f}ms")

        # 6. Save if output path provided
        if output_path is not None:
            out_path = Path(output_path)
            if out_path.is_dir() or not out_path.suffix:
                out_path.mkdir(parents=True, exist_ok=True)
                config_file = out_path / "adapter_config.json"
                out_file = out_path / "adapter_model.safetensors"
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                config_file = out_path.parent / "adapter_config.json"
                out_file = out_path
            save_file(fused_state_dict, str(out_file))
            with open(config_file, "w") as f:
                json.dump(fused_config, f, indent=2)
            print(f"💾 Saved stacked adapter to: {out_file}")

        return fused_state_dict, fused_config

    def _resolve_adapter_path(self, name: str) -> Path:
        """Finds the safetensors file from name or path."""
        p = Path(name)
        if p.exists() and p.is_file():
            return p
        if p.exists() and p.is_dir() and (p / "adapter_model.safetensors").exists():
            return p / "adapter_model.safetensors"

        # Search in adapters_dir
        candidates = [
            self.adapters_dir / f"m2_{name}_r8a128_v7_27b" / "adapter_model.safetensors",
            self.adapters_dir / f"m2_{name}_r8a128_v7_real" / "adapter_model.safetensors",
            self.adapters_dir / f"m2_{name}_r8a128_v7_ornith35b" / "adapter_model.safetensors",
            self.adapters_dir / f"m2_{name}_r8a128_v7" / "adapter_model.safetensors",
            self.adapters_dir / f"{name}" / "adapter_model.safetensors",
        ]
        for c in candidates:
            if c.exists():
                return c

        # Fallback wildcard search
        matches = list(self.adapters_dir.rglob(f"*{name}*/*.safetensors"))
        if matches:
            return matches[0]

        raise FileNotFoundError(f"Could not resolve adapter path for: '{name}' in {self.adapters_dir}")
