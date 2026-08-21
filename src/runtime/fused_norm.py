"""Drop-in RMSNorm replacements for Qwen3.5, in plain PyTorch.

WHY THIS IS NOT CALLED "AITER"
------------------------------
This file replaces `aiter_ops.py`, which claimed to dispatch to AMD's AITER
fused HIP kernels. It never did, and on this hardware it could not:

  * `aiter` was not installed, so `HAS_AITER` was False and every op fell
    through to the PyTorch path. Worse, the PyPI package named `aiter` is an
    unrelated async-iterator library from 2019 -- installing *that* would have
    flipped `HAS_AITER` to True while `aiter.rmsnorm` raised AttributeError
    into a bare `except Exception: pass`.
  * The real AITER (github.com/ROCm/aiter) was installed and measured here on
    gfx1100 / ROCm 7.2. Findings, all reproducible:
      - `aiter.rope` and `aiter.swiglu` do not exist; the API the old module
        called was invented.
      - `aiter.rmsnorm2d_fwd` silently returns ALL ZEROS at hidden=2560 and
        8192. It is only correct at 32768 -- which is why AMD's own
        op_tests/test_rmsnorm2d.py passes (it tests 32768/65536). Qwen3.5-4B
        is hidden=2560, so the headline kernel is silently wrong for us.
      - `aiter.rmsnorm2d_fwd_opus` IS numerically correct at 2560
        (rel err 7.4e-3, within bf16 rounding) but is 2-6x SLOWER than
        torch.nn.functional.rms_norm at every shape we use:
            rows=1   6.62us torch vs 14.98us aiter  (0.44x)
            rows=128 4.95us torch vs 29.81us aiter  (0.17x)
    AITER targets CDNA/MI300 at large shapes; on RDNA3 at hidden=2560 ATen
    wins. There is no AITER speedup available on this box.

So this module does what it actually does: reimplement Qwen3.5's two RMSNorm
variants in PyTorch, exactly. It exists to be CUDA-graph friendly and to be a
correct drop-in -- not to be faster than ATen, which it is not.

NO SILENT FALLBACKS. The old module wrapped every kernel call in
`except Exception: pass`, which is how a completely inert "fused" path
survived unnoticed. Nothing here swallows exceptions.
"""

from __future__ import annotations

import torch
from torch import nn


class ExactRMSNorm(nn.Module):
    """Numerically exact stand-in for Qwen3_5RMSNorm / Qwen3_5RMSNormGated.

    The two upstream variants differ in more than a flag, and the previous
    implementation got the gated one wrong (measured max abs err 6.25e-2 per
    layer, compounding over 24 GatedDeltaNet layers into a 5.5e-3 logit MSE):

      Qwen3_5RMSNorm (unit_offset=True), weight init zeros:
          out = norm(x) * (1 + w)          -- all in fp32, cast at the end
      Qwen3_5RMSNormGated (unit_offset=False), weight init ones:
          out = (norm(x).to(dtype) * w) * silu(gate.float())
          note the cast to input dtype BEFORE the weight multiply, and silu
          computed in fp32 -- both of which the old version got backwards.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, unit_offset: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size) if unit_offset else torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.unit_offset = unit_offset

    def _norm_f32(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        normed = self._norm_f32(hidden_states)

        if self.unit_offset:
            # matches Qwen3_5RMSNorm: weight applied in fp32, single cast at end
            return (normed * (1.0 + self.weight.float())).type_as(hidden_states)

        # matches Qwen3_5RMSNormGated: cast first, then weight, then fp32 gate
        out = self.weight * normed.to(input_dtype)
        if gate is not None:
            out = out * torch.nn.functional.silu(gate.to(torch.float32))
        return out.to(input_dtype)


class ScaleFreeRMSNorm(nn.Module):
    """Scale-free RMSNorm stand-in (post-folding).

    Eliminates scale parameter memory access and elementwise multiply.
    Stores the original weight in a buffer `_original_weight` for exact reversibility.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        unit_offset: bool = True,
        original_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.unit_offset = unit_offset
        if original_weight is not None:
            self.register_buffer("_original_weight", original_weight.detach().clone())
        else:
            self.register_buffer("_original_weight", None)

    def _norm_f32(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        normed = self._norm_f32(hidden_states)
        if gate is not None:
            normed = normed * torch.nn.functional.silu(gate.to(torch.float32))
        return normed.type_as(hidden_states)


def fold_rmsnorm_into_linear(model: nn.Module, fold_weights: bool = True) -> int:
    """Absorb RMSNorm layer scales into downstream Linear projection weights.

    Eliminates RMSNorm kernel launch and scale loading overhead at decode-time.
    Supports Qwen3.5 decoder layers:
      - input_layernorm -> self_attn.{q,k,v}_proj OR linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a}
      - post_attention_layernorm -> mlp.{gate_proj, up_proj}
      - final norm -> lm_head (if untied)

    Multi-fan-out is handled by broadcasting the same (1 + γ) scale across all
    downstream linear projection columns (dimension 1).
    """
    if not fold_weights:
        return 0

    folded_count = 0

    # Look for text model layers
    text_model = getattr(model, "model", model)
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    if hasattr(text_model, "model"):
        text_model = text_model.model

    layers = getattr(text_model, "layers", [])

    for i, layer in enumerate(layers):
        # 1. input_layernorm -> Attention projections
        input_norm = getattr(layer, "input_layernorm", None)
        if input_norm is not None and hasattr(input_norm, "weight") and not isinstance(input_norm, ScaleFreeRMSNorm):
            unit_offset = getattr(input_norm, "unit_offset", True)
            w_norm = input_norm.weight.data.float()
            gamma = (1.0 + w_norm) if unit_offset else w_norm
            gamma_col = gamma.unsqueeze(0)  # (1, in_features)

            downstream: list[nn.Linear] = []
            if hasattr(layer, "self_attn"):
                for proj_name in ("q_proj", "k_proj", "v_proj"):
                    proj = getattr(layer.self_attn, proj_name, None)
                    if isinstance(proj, nn.Linear):
                        downstream.append(proj)
            elif hasattr(layer, "linear_attn"):
                for proj_name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
                    proj = getattr(layer.linear_attn, proj_name, None)
                    if isinstance(proj, nn.Linear):
                        downstream.append(proj)

            if downstream:
                for proj in downstream:
                    with torch.no_grad():
                        w_dtype = proj.weight.dtype
                        proj.weight.data.copy_((proj.weight.data.float() * gamma_col.to(proj.weight.device)).to(w_dtype))

                # Replace with ScaleFreeRMSNorm
                eps = getattr(input_norm, "variance_epsilon", getattr(input_norm, "eps", 1e-6))
                sf_norm = ScaleFreeRMSNorm(
                    input_norm.weight.shape[0],
                    eps=eps,
                    unit_offset=unit_offset,
                    original_weight=input_norm.weight.data,
                ).to(device=input_norm.weight.device)
                layer.input_layernorm = sf_norm
                folded_count += 1

        # 2. post_attention_layernorm -> MLP projections
        post_norm = getattr(layer, "post_attention_layernorm", None)
        if post_norm is not None and hasattr(post_norm, "weight") and not isinstance(post_norm, ScaleFreeRMSNorm):
            unit_offset = getattr(post_norm, "unit_offset", True)
            w_norm = post_norm.weight.data.float()
            gamma = (1.0 + w_norm) if unit_offset else w_norm
            gamma_col = gamma.unsqueeze(0)  # (1, in_features)

            downstream = []
            if hasattr(layer, "mlp"):
                for proj_name in ("gate_proj", "up_proj"):
                    proj = getattr(layer.mlp, proj_name, None)
                    if isinstance(proj, nn.Linear):
                        downstream.append(proj)

            if downstream:
                for proj in downstream:
                    with torch.no_grad():
                        w_dtype = proj.weight.dtype
                        proj.weight.data.copy_((proj.weight.data.float() * gamma_col.to(proj.weight.device)).to(w_dtype))

                # Replace with ScaleFreeRMSNorm
                eps = getattr(post_norm, "variance_epsilon", getattr(post_norm, "eps", 1e-6))
                sf_norm = ScaleFreeRMSNorm(
                    post_norm.weight.shape[0],
                    eps=eps,
                    unit_offset=unit_offset,
                    original_weight=post_norm.weight.data,
                ).to(device=post_norm.weight.device)
                layer.post_attention_layernorm = sf_norm
                folded_count += 1

    # 3. Final model norm -> lm_head
    final_norm = getattr(text_model, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    embed_tokens = getattr(text_model, "embed_tokens", None)

    if (
        final_norm is not None
        and hasattr(final_norm, "weight")
        and not isinstance(final_norm, ScaleFreeRMSNorm)
        and isinstance(lm_head, nn.Linear)
    ):
        # Check if untied
        is_untied = embed_tokens is None or (lm_head.weight.data_ptr() != embed_tokens.weight.data_ptr())
        if is_untied:
            unit_offset = getattr(final_norm, "unit_offset", True)
            w_norm = final_norm.weight.data.float()
            gamma = (1.0 + w_norm) if unit_offset else w_norm
            gamma_col = gamma.unsqueeze(0)

            with torch.no_grad():
                w_dtype = lm_head.weight.dtype
                lm_head.weight.data.copy_((lm_head.weight.data.float() * gamma_col.to(lm_head.weight.device)).to(w_dtype))

            eps = getattr(final_norm, "variance_epsilon", getattr(final_norm, "eps", 1e-6))
            sf_norm = ScaleFreeRMSNorm(
                final_norm.weight.shape[0],
                eps=eps,
                unit_offset=unit_offset,
                original_weight=final_norm.weight.data,
            ).to(device=final_norm.weight.device)
            text_model.norm = sf_norm
            folded_count += 1

    return folded_count


def unfold_rmsnorm(model: nn.Module) -> int:
    """Reverse FlashNorm weight folding, restoring ExactRMSNorm modules."""
    unfolded_count = 0

    text_model = getattr(model, "model", model)
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    if hasattr(text_model, "model"):
        text_model = text_model.model

    layers = getattr(text_model, "layers", [])

    for layer in layers:
        # 1. input_layernorm
        if isinstance(getattr(layer, "input_layernorm", None), ScaleFreeRMSNorm):
            sf_norm = layer.input_layernorm
            orig_w = sf_norm._original_weight
            if orig_w is not None:
                unit_offset = sf_norm.unit_offset
                gamma = (1.0 + orig_w.float()) if unit_offset else orig_w.float()
                inv_gamma = (1.0 / gamma).unsqueeze(0)

                downstream = []
                if hasattr(layer, "self_attn"):
                    for proj_name in ("q_proj", "k_proj", "v_proj"):
                        proj = getattr(layer.self_attn, proj_name, None)
                        if isinstance(proj, nn.Linear):
                            downstream.append(proj)
                elif hasattr(layer, "linear_attn"):
                    for proj_name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
                        proj = getattr(layer.linear_attn, proj_name, None)
                        if isinstance(proj, nn.Linear):
                            downstream.append(proj)

                for proj in downstream:
                    with torch.no_grad():
                        w_dtype = proj.weight.dtype
                        proj.weight.data.copy_((proj.weight.data.float() * inv_gamma.to(proj.weight.device)).to(w_dtype))

                exact_norm = ExactRMSNorm(sf_norm.hidden_size, eps=sf_norm.variance_epsilon, unit_offset=unit_offset)
                exact_norm.weight.data.copy_(orig_w)
                exact_norm.to(device=orig_w.device, dtype=orig_w.dtype)
                layer.input_layernorm = exact_norm
                unfolded_count += 1

        # 2. post_attention_layernorm
        if isinstance(getattr(layer, "post_attention_layernorm", None), ScaleFreeRMSNorm):
            sf_norm = layer.post_attention_layernorm
            orig_w = sf_norm._original_weight
            if orig_w is not None:
                unit_offset = sf_norm.unit_offset
                gamma = (1.0 + orig_w.float()) if unit_offset else orig_w.float()
                inv_gamma = (1.0 / gamma).unsqueeze(0)

                downstream = []
                if hasattr(layer, "mlp"):
                    for proj_name in ("gate_proj", "up_proj"):
                        proj = getattr(layer.mlp, proj_name, None)
                        if isinstance(proj, nn.Linear):
                            downstream.append(proj)

                for proj in downstream:
                    with torch.no_grad():
                        w_dtype = proj.weight.dtype
                        proj.weight.data.copy_((proj.weight.data.float() * inv_gamma.to(proj.weight.device)).to(w_dtype))

                exact_norm = ExactRMSNorm(sf_norm.hidden_size, eps=sf_norm.variance_epsilon, unit_offset=unit_offset)
                exact_norm.weight.data.copy_(orig_w)
                exact_norm.to(device=orig_w.device, dtype=orig_w.dtype)
                layer.post_attention_layernorm = exact_norm
                unfolded_count += 1

    # 3. Final norm
    final_norm = getattr(text_model, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if isinstance(final_norm, ScaleFreeRMSNorm) and isinstance(lm_head, nn.Linear):
        orig_w = final_norm._original_weight
        if orig_w is not None:
            unit_offset = final_norm.unit_offset
            gamma = (1.0 + orig_w.float()) if unit_offset else orig_w.float()
            inv_gamma = (1.0 / gamma).unsqueeze(0)

            with torch.no_grad():
                w_dtype = lm_head.weight.dtype
                lm_head.weight.data.copy_((lm_head.weight.data.float() * inv_gamma.to(lm_head.weight.device)).to(w_dtype))

            exact_norm = ExactRMSNorm(final_norm.hidden_size, eps=final_norm.variance_epsilon, unit_offset=unit_offset)
            exact_norm.weight.data.copy_(orig_w)
            exact_norm.to(device=orig_w.device, dtype=orig_w.dtype)
            text_model.norm = exact_norm
            unfolded_count += 1

    return unfolded_count


def inject_exact_rmsnorm(model: nn.Module) -> int:
    """Swap Qwen3.5 RMSNorm modules for ExactRMSNorm. Returns how many.

    `unit_offset` is derived from the class name: Qwen3_5RMSNorm stores its
    weight as an offset from 1 (init zeros), Qwen3_5RMSNormGated stores it
    directly (init ones). Getting this backwards silently rescales the layer.
    """
    injected = 0
    for name, module in list(model.named_modules()):
        cls = module.__class__.__name__
        if "RMSNorm" not in cls:
            continue
        if not hasattr(module, "weight"):
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = model if not parent_name else model.get_submodule(parent_name)

        eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
        unit_offset = "gated" not in cls.lower()

        fused = ExactRMSNorm(module.weight.shape[0], eps=eps, unit_offset=unit_offset)
        fused.weight.data.copy_(module.weight.data)
        fused.to(device=module.weight.device, dtype=module.weight.dtype)

        setattr(parent, child_name, fused)
        injected += 1

    return injected


def scale_expert_factors_for_folded_norms(model: nn.Module, experts: list | tuple) -> int:
    """Scale adapter V factors by (1 + γ) for weights whose input RMSNorm was folded.

    Ensures that when WeightFoldingEngine activates an adapter (W_live = W0 + s*U@V),
    the delta dW receives the exact same normalization scale (1 + γ) that was folded
    into the base weight W0.
    """
    scaled_count = 0

    text_model = getattr(model, "model", model)
    if hasattr(text_model, "language_model"):
        text_model = text_model.language_model
    if hasattr(text_model, "model"):
        text_model = text_model.model

    layers = getattr(text_model, "layers", [])

    for i, layer in enumerate(layers):
        # 1. input_layernorm
        input_norm = getattr(layer, "input_layernorm", None)
        if isinstance(input_norm, ScaleFreeRMSNorm) and input_norm._original_weight is not None:
            gamma = (
                (1.0 + input_norm._original_weight.float())
                if input_norm.unit_offset
                else input_norm._original_weight.float()
            )
            gamma_col = gamma.unsqueeze(0)  # (1, in_features)

            target_keys = [
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"model.layers.{i}.linear_attn.in_proj_qkv.weight",
                f"model.layers.{i}.linear_attn.in_proj_z.weight",
                f"model.layers.{i}.linear_attn.in_proj_b.weight",
                f"model.layers.{i}.linear_attn.in_proj_a.weight",
            ]

            for expert in experts:
                for k in target_keys:
                    if k in expert.factors:
                        u, v = expert.factors[k]
                        v_scaled = (v.float() * gamma_col.to(v.device)).to(v.dtype)
                        expert.factors[k] = (u, v_scaled)
                        scaled_count += 1

        # 2. post_attention_layernorm
        post_norm = getattr(layer, "post_attention_layernorm", None)
        if isinstance(post_norm, ScaleFreeRMSNorm) and post_norm._original_weight is not None:
            gamma = (
                (1.0 + post_norm._original_weight.float())
                if post_norm.unit_offset
                else post_norm._original_weight.float()
            )
            gamma_col = gamma.unsqueeze(0)

            target_keys = [
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"model.layers.{i}.mlp.up_proj.weight",
            ]

            for expert in experts:
                for k in target_keys:
                    if k in expert.factors:
                        u, v = expert.factors[k]
                        v_scaled = (v.float() * gamma_col.to(v.device)).to(v.dtype)
                        expert.factors[k] = (u, v_scaled)
                        scaled_count += 1

    return scaled_count


