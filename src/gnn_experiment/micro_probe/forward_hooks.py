"""PyTorch Forward Hook diagnostics and hidden-state velocity measurement.

Registers forward hooks on model decoder layers to compute:
1. Hidden-state velocity: \\Delta h_l^{(t)} = ||h_l^{(t)} - h_l^{(t-1)}||_2 / ||h_l^{(t-1)}||_2
2. Quiet layer ratio under dynamic velocity threshold \\epsilon
3. Backward pass gradient norms and tensor stability
"""

import math
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn


class MicroProbeForwardHooks:
    """Manages forward hooks for layer velocity diagnostics."""

    def __init__(self, model: nn.Module, epsilon: float = 0.05):
        self.model = model
        self.epsilon = epsilon
        self.hooks = []
        self.layer_activations: Dict[int, torch.Tensor] = {}
        self.layer_prev_activations: Dict[int, torch.Tensor] = {}
        self.layer_velocities: Dict[int, List[float]] = {}
        self.quiet_counts: Dict[int, int] = {}
        self.total_tokens_evaluated: int = 0
        self._register_hooks()

    def _get_decoder_layers(self) -> List[nn.Module]:
        """Extract decoder layers from Qwen / Transformer architecture."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        elif hasattr(self.model, "layers"):
            return list(self.model.layers)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        else:
            # Fallback: inspect module children
            return [m for name, m in self.model.named_modules() if "layer" in name.lower() or "block" in name.lower()]

    def _register_hooks(self):
        layers = self._get_decoder_layers()
        for idx, layer in enumerate(layers):
            self.layer_velocities[idx] = []
            self.quiet_counts[idx] = 0

            def make_hook(layer_idx: int):
                def hook(module: nn.Module, input_args, output_tensor):
                    # Output can be tensor or tuple (hidden_states, ...)
                    if isinstance(output_tensor, tuple):
                        h = output_tensor[0]
                    else:
                        h = output_tensor

                    if isinstance(h, torch.Tensor):
                        # Detach to save memory
                        h_flat = h.detach().float()
                        if layer_idx in self.layer_prev_activations:
                            prev_h = self.layer_prev_activations[layer_idx]
                            # Compute norm delta across hidden dimension
                            if prev_h.shape == h_flat.shape:
                                diff = torch.norm(h_flat - prev_h, p=2, dim=-1)
                                base = torch.norm(prev_h, p=2, dim=-1) + 1e-8
                                vel = (diff / base).mean().item()
                                self.layer_velocities[layer_idx].append(vel)
                                if vel < self.epsilon:
                                    self.quiet_counts[layer_idx] += 1
                        self.layer_prev_activations[layer_idx] = h_flat

                return hook

            h_handle = layer.register_forward_hook(make_hook(idx))
            self.hooks.append(h_handle)

    def remove_hooks(self):
        """Remove all active PyTorch forward hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Compute aggregated statistics for layer state velocity and quietness."""
        summary = {}
        total_evals = 0
        total_quiet = 0

        avg_velocities = {}
        for idx, v_list in self.layer_velocities.items():
            if v_list:
                avg_v = sum(v_list) / len(v_list)
                avg_velocities[f"layer_{idx}_velocity"] = avg_v
                q_count = self.quiet_counts.get(idx, 0)
                total_evals += len(v_list)
                total_quiet += q_count

        overall_avg_velocity = (
            sum(avg_velocities.values()) / len(avg_velocities) if avg_velocities else 0.0
        )
        quiet_ratio = (total_quiet / total_evals) if total_evals > 0 else 0.0

        summary["overall_avg_velocity"] = overall_avg_velocity
        summary["quiet_layer_ratio"] = quiet_ratio
        summary["total_velocity_evaluations"] = total_evals
        summary["layer_velocities"] = avg_velocities
        return summary


def compute_gradient_stability(model: nn.Module) -> Dict[str, float]:
    """Compute gradient norm statistics across all trainable parameters."""
    total_norm_sq = 0.0
    param_count = 0
    max_grad = 0.0

    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.detach().data.norm(2).item()
            total_norm_sq += param_norm**2
            param_count += 1
            if param_norm > max_grad:
                max_grad = param_norm

    total_grad_norm = math.sqrt(total_norm_sq)
    avg_grad_norm = total_grad_norm / max(1, param_count)

    return {
        "total_grad_norm": total_grad_norm,
        "avg_grad_norm": avg_grad_norm,
        "max_grad_norm": max_grad,
        "trainable_param_groups": param_count,
    }
