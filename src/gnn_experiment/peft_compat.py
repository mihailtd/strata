"""Runtime compatibility patches for third-party libraries.

Currently one patch: **peft's LoKr does not support 4-bit quantized base
layers.** Every adapter in this project trains on a bitsandbytes-quantized
base (`load_in_4bit=True`), so without this, any LoKr run dies on the first
forward pass with e.g.::

    RuntimeError: shape '[11796480, 1]' is invalid for input of size 23592960

Cause: `LoKrLayer.get_delta_weight` reshapes the Kronecker product to
`base_layer.weight.shape`. For a `bitsandbytes.nn.Linear4bit`, `.weight` is
the *packed* uint8 tensor -- two 4-bit values per byte -- so its shape is
`[out_features * in_features / 2, 1]`, exactly half the elements the delta
actually has. (11796480 = 9216*2560/2, the packed byte count for gate_proj.)
LoRA handles this via dedicated dispatch modules (`peft/tuners/lora/bnb.py`,
`gptq.py`, `awq.py`); LoKr ships no equivalent -- grep for `bitsandbytes` under
`peft/tuners/lokr/` returns nothing.

Fix: derive the target shape from `out_features`/`in_features`, which
`Linear4bit` exposes correctly, and only fall back to `.weight.shape` for
layer types that lack them. This matches what `update_layer` already uses to
size the factors, so the reshape becomes a no-op for Linear layers rather than
a reinterpretation.

WHY THIS LIVES HERE instead of in site-packages: it was originally applied by
editing `.venv/.../peft/tuners/lokr/layer.py` in place. uv *hardlinks* its
global cache into the venv (verified: same inode, 4 links), so that edit also
mutated `~/.cache/uv/archive-v0/<hash>/peft/...` -- i.e. it silently
propagated to every other project on this machine installing peft 0.20.0, and
had already reached this repo's `training/.venv`. It was also invisible to git
and uv.lock, and `uv cache clean` or a reinstall would revert it with no
warning and no obvious symptom beyond a confusing shape error. A repo-local,
version-guarded, idempotent patch is durable across all of those.
"""

from __future__ import annotations

import warnings

# The peft version this patch was written against and verified on. If peft is
# upgraded, `get_delta_weight`'s body may have changed upstream (including
# possibly gaining native 4-bit support), so re-check rather than silently
# overwriting a newer implementation with this older copy.
VERIFIED_PEFT_VERSION = "0.20.0"

_PATCH_SENTINEL = "_gnn_experiment_4bit_patch"


def patch_lokr_4bit_support(force: bool = False) -> bool:
    """Teach peft's LoKr layers to compute delta shapes from in/out_features.

    Idempotent and safe to call repeatedly. Returns True if this call
    installed the patch, False if it was already present or was skipped.
    """
    try:
        import peft
        from peft.tuners.lokr.layer import (
            LoKrLayer,
            make_weight_cp,  # noqa: F401  (used by the copied body)
        )
    except ImportError:  # peft not installed / layout changed -- nothing to patch
        return False

    if getattr(LoKrLayer.get_delta_weight, _PATCH_SENTINEL, False):
        return False  # already patched

    installed = getattr(peft, "__version__", "unknown")
    if installed != VERIFIED_PEFT_VERSION and not force:
        warnings.warn(
            f"peft_compat: peft {installed} is installed but the LoKr 4-bit patch was written and "
            f"verified against {VERIFIED_PEFT_VERSION}. NOT patching, because overwriting a newer "
            "get_delta_weight with an older copy could silently reintroduce fixed bugs. Verify "
            "whether upstream now supports 4-bit base layers (grep for 'bitsandbytes' under "
            "peft/tuners/lokr/); if it still does not, re-verify this patch and bump "
            "VERIFIED_PEFT_VERSION. LoKr training on a 4-bit base will fail until then.",
            stacklevel=2,
        )
        return False

    import torch
    from peft.tuners.lokr.layer import make_kron

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        # Body copied from peft 0.20.0 LoKrLayer.get_delta_weight, with ONLY the
        # target-shape computation changed (marked below). Kept as a full copy
        # because the upstream bug is mid-method -- there is no seam to wrap.
        if adapter_name in self.lokr_w1:
            w1 = self.lokr_w1[adapter_name]
        else:
            w1 = self.lokr_w1_a[adapter_name] @ self.lokr_w1_b[adapter_name]

        if adapter_name in self.lokr_w2:
            w2 = self.lokr_w2[adapter_name]
        elif adapter_name in self.lokr_t2:
            w2 = make_weight_cp(self.lokr_t2[adapter_name], self.lokr_w2_a[adapter_name], self.lokr_w2_b[adapter_name])
        else:
            w2 = self.lokr_w2_a[adapter_name] @ self.lokr_w2_b[adapter_name]

        weight = make_kron(w1, w2, self.scaling[adapter_name])

        base_layer = self.get_base_layer()
        # ---- THE PATCH ----------------------------------------------------
        # upstream: weight = weight.reshape(base_layer.weight.shape)
        # `.weight` is packed uint8 on a 4-bit quantized layer; in/out_features
        # are correct for both quantized and dense Linear layers.
        if hasattr(base_layer, "out_features") and hasattr(base_layer, "in_features"):
            target_shape = (base_layer.out_features, base_layer.in_features)
        else:
            target_shape = base_layer.weight.shape
        weight = weight.reshape(target_shape)
        # ---- END PATCH ----------------------------------------------------

        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (torch.rand(weight.size(0)) > rank_dropout).float()
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            if self.rank_dropout_scale[adapter_name]:
                drop /= drop.mean()
            weight *= drop

        return weight

    setattr(get_delta_weight, _PATCH_SENTINEL, True)
    LoKrLayer.get_delta_weight = get_delta_weight
    return True
