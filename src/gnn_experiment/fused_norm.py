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
