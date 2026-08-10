"""Two novel training-time techniques for LoRA fine-tuning, benchmarked against
the existing QLoRA baseline in `scripts/export_adapter.py` (see GOAL_1.md):

1. Cross-Layer Tucker Factorization ("tucker" mode): instead of each target
   Linear owning an independent (A, B) pair, every layer sharing a given
   (module_suffix, in_features, out_features) shape draws from ONE global
   factor pair (U_in, U_out). Each layer keeps only a tiny private core
   matrix. Delta is computed as a chained matmul (x @ U_in) @ core @ U_out —
   the full d_in x d_out matrix is never materialized, same as ordinary LoRA.

2. Velocity-Masked SFT ("velocity gate"): a forward hook on each decoder
   layer measures hidden-state velocity (relative L2 change of the residual
   stream across that layer). An EMA of that velocity (updated once per
   optimizer step, after a warmup) decides which layers are "quiet" for the
   *next* step; wrapped Linears skip their adapter delta entirely for quiet
   layers, so no backward graph nodes are created for them that step.

Both are orthogonal and can be combined by wrapping with mode="tucker" and
passing a VelocityGate.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from torch import nn


def set_hard_vram_cap(cap_gb: float, device: int = 0) -> None:
    """Force PyTorch's caching allocator to refuse allocations beyond `cap_gb`
    on `device`, raising a normal catchable torch.OutOfMemoryError instead of
    issuing the underlying driver allocation.

    This matters specifically on this rig: ROCm-over-WSL (the `/dev/dxg`
    paravirtualized path) does not always hard-fail an over-allocation the
    way native Linux ROCm does -- it can silently satisfy it from *shared
    host memory* instead, which balloons system RAM rather than cleanly
    erroring (see EXPERIMENTS.md T-11 notes; reproduced firsthand when an
    early, misconfigured smoke test here spiked system RAM badly enough to
    nearly crash the host). Call this once, early, before loading any model,
    on every script that touches the GPU in this project.
    """
    if not torch.cuda.is_available():
        return
    total_mem = torch.cuda.get_device_properties(device).total_memory
    fraction = min(0.97, (cap_gb * 1024**3) / total_mem)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    total_gb = total_mem / 1024**3
    print(f"[safety] Hard VRAM cap: {cap_gb:.1f} GB ({fraction:.1%} of {total_gb:.1f} GB total) on device {device}")


# ---------------------------------------------------------------------------
# Cross-Layer Tucker Factorization
# ---------------------------------------------------------------------------


class TuckerFactorBank(nn.Module):
    """Owns the global (U_in, U_out) factor pairs, keyed by shape signature.

    Registered once at the top level of the model so its parameters show up
    in `model.parameters()` exactly once (individual NovelLoraLinear wrappers
    only hold a plain-function lookup into this bank, never a direct
    nn.Module/Parameter reference, to avoid re-registering the same shared
    parameter under dozens of different submodule paths).
    """

    def __init__(self):
        super().__init__()
        self.u_in: nn.ParameterDict = nn.ParameterDict()
        self.u_out: nn.ParameterDict = nn.ParameterDict()

    def get_or_create(
        self,
        key: str,
        in_features: int,
        out_features: int,
        rank_in: int,
        rank_out: int,
        device,
        dtype,
    ) -> tuple[nn.Parameter, nn.Parameter]:
        if key not in self.u_in:
            u_in = nn.Parameter(
                torch.randn(in_features, rank_in, device=device, dtype=dtype) * (1.0 / math.sqrt(in_features))
            )
            # zero-init U_out, same convention as LoRA-B: adapter starts as a no-op.
            u_out = nn.Parameter(torch.zeros(rank_out, out_features, device=device, dtype=dtype))
            self.u_in[key] = u_in
            self.u_out[key] = u_out
        return self.u_in[key], self.u_out[key]


def _make_bank_lookup(
    bank: TuckerFactorBank,
) -> Callable[[str], tuple[torch.Tensor, torch.Tensor]]:
    def lookup(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        return bank.u_in[key], bank.u_out[key]

    return lookup


# ---------------------------------------------------------------------------
# Velocity-Masked SFT (Foveated Training)
# ---------------------------------------------------------------------------


class VelocityGate(nn.Module):
    """Tracks per-decoder-layer hidden-state velocity and decides, once per
    optimizer step, which layers are "quiet" for the following step.

    Decision is EMA-based and lagged by one step on purpose: at the moment a
    given layer actually runs inside a single teacher-forced forward pass, we
    cannot yet know its own output without running it, so "is this layer
    quiet" is answered using the *previous* step's measured velocity (an EMA
    across steps), not the current step's. This mirrors a closed-loop
    controller reacting to the last observation rather than an oracle.
    """

    def __init__(
        self,
        num_layers: int,
        threshold: float = 0.05,
        ema_decay: float = 0.9,
        warmup_steps: int = 20,
        max_quiet_fraction: float = 0.9,
    ):
        super().__init__()
        self.ema_velocity: torch.Tensor
        self.register_buffer("ema_velocity", torch.full((num_layers,), float("nan")))
        self.num_layers = num_layers
        self.threshold = threshold
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        # Safety valve: never mask every layer in the same step, or the loss tensor
        # loses requires_grad entirely (nothing trainable touches the graph) and
        # Trainer's loss.backward() raises. Keeps at least a floor of active layers.
        self.max_quiet_fraction = max_quiet_fraction
        self.step_count = 0
        self.quiet_layers: set[int] = set()
        self.quiet_fraction_history: list[float] = []

    @torch.no_grad()
    def record(self, layer_idx: int, velocity: torch.Tensor) -> None:
        v = float(velocity.detach().float().cpu())
        cur = self.ema_velocity[layer_idx]
        if torch.isnan(cur):
            self.ema_velocity[layer_idx] = v
        else:
            self.ema_velocity[layer_idx] = self.ema_decay * cur + (1 - self.ema_decay) * v

    def end_step(self) -> None:
        self.step_count += 1
        if self.step_count > self.warmup_steps:
            vals = self.ema_velocity.tolist()
            candidates = [(v, i) for i, v in enumerate(vals) if not math.isnan(v) and v < self.threshold]
            cap = int(self.num_layers * self.max_quiet_fraction)
            if len(candidates) > cap:
                # keep the `cap` quietest layers quiet; the rest stay active this step.
                candidates.sort()
                candidates = candidates[:cap]
            self.quiet_layers = {i for _, i in candidates}
        else:
            self.quiet_layers = set()
        self.quiet_fraction_history.append(len(self.quiet_layers) / self.num_layers)

    def is_quiet(self, layer_idx: int) -> bool:
        return layer_idx in self.quiet_layers


def install_velocity_hooks(layers: Iterable[nn.Module], gate: VelocityGate) -> list:
    """Forward hooks measuring per-layer relative hidden-state velocity."""

    handles = []

    def make_hook(idx: int):
        def hook(module, inputs, output):
            hs_in = inputs[0]
            hs_out = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                diff = (hs_out.float() - hs_in.float()).norm(dim=-1)
                norm = hs_in.float().norm(dim=-1).clamp_min(1e-6)
                rel = (diff / norm).mean()
            gate.record(idx, rel)

        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))
    return handles


class VelocityGateStepCallback:
    """Plain (non-transformers) callback object; wired via a transformers
    `TrainerCallback` in the training script to avoid importing transformers
    here."""

    def __init__(self, gate: VelocityGate):
        self.gate = gate

    def on_step_end(self):
        self.gate.end_step()


# ---------------------------------------------------------------------------
# The wrapped Linear
# ---------------------------------------------------------------------------


class NovelLoraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        layer_idx: int,
        mode: str,
        alpha: float,
        dropout: float = 0.0,
        rank: int | None = None,
        rank_in: int | None = None,
        rank_out: int | None = None,
        factor_lookup: Callable[[str], tuple[torch.Tensor, torch.Tensor]] | None = None,
        factor_key: str | None = None,
        is_quiet_fn: Callable[[int], bool] | None = None,
    ):
        super().__init__()
        assert mode in ("standard", "tucker")
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.layer_idx = layer_idx
        self.mode = mode
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._is_quiet_fn = is_quiet_fn or (lambda idx: False)

        device = next(base_layer.parameters()).device
        compute_dtype = torch.bfloat16

        if mode == "tucker":
            assert factor_lookup is not None and factor_key is not None and rank_in and rank_out
            self.rank = rank_in  # scaling uses the input-side rank, matching LoRA convention
            self.scaling = alpha / rank_in
            self._factor_lookup = factor_lookup
            self.factor_key = factor_key
            core = torch.empty(rank_in, rank_out, device=device, dtype=compute_dtype)
            nn.init.normal_(core, std=0.01)
            self.core = nn.Parameter(core)
        else:
            assert rank is not None
            self.rank = rank
            self.scaling = alpha / rank
            in_f, out_f = base_layer.in_features, base_layer.out_features
            lora_a = torch.empty(in_f, rank, device=device, dtype=compute_dtype)
            nn.init.kaiming_uniform_(lora_a, a=math.sqrt(5))
            self.lora_a = nn.Parameter(lora_a)
            self.lora_b = nn.Parameter(torch.zeros(rank, out_f, device=device, dtype=compute_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self._is_quiet_fn(self.layer_idx):
            return result

        h = self.dropout(x).to(self.core.dtype if self.mode == "tucker" else self.lora_a.dtype)
        if self.mode == "tucker":
            u_in, u_out = self._factor_lookup(self.factor_key)
            delta = (h @ u_in) @ self.core
            delta = delta @ u_out
        else:
            delta = (h @ self.lora_a) @ self.lora_b
        delta = delta * self.scaling
        return result + delta.to(result.dtype)


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_decoder_layers(model) -> list[nn.Module]:
    for path in (("model",), ("model", "language_model"), ("language_model",), ()):
        m = model
        for attr in path:
            m = getattr(m, attr, None)
            if m is None:
                break
        if m is not None and hasattr(m, "layers"):
            return list(m.layers)
    raise ValueError("Could not locate decoder layer list on model")


def apply_novel_lora(
    model,
    mode: str,
    target_modules: list[str] = TARGET_MODULES,
    rank: int = 8,
    rank_in: int = 8,
    rank_out: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
    velocity_gate: VelocityGate | None = None,
) -> dict:
    """Freezes the whole model, then replaces every target Linear inside every
    decoder layer with a NovelLoraLinear. Returns a summary dict (wrapped
    count, trainable param count, and the bank if mode == "tucker")."""

    for p in model.parameters():
        p.requires_grad_(False)

    layers = get_decoder_layers(model)
    is_quiet_fn = velocity_gate.is_quiet if velocity_gate is not None else None

    bank = TuckerFactorBank() if mode == "tucker" else None
    factor_lookup = _make_bank_lookup(bank) if bank is not None else None

    wrapped = 0
    for layer_idx, layer in enumerate(layers):
        # snapshot (name, module) pairs first: we mutate the module tree while
        # iterating, and isinstance narrows module to nn.Linear (bnb's Linear4bit
        # subclasses it) so in_features/out_features type-check as int, not the
        # Tensor | Module that nn.Module.__getattr__ would otherwise infer.
        targets: list[tuple[str, nn.Linear]] = [
            (name, module)
            for name, module in layer.named_modules()
            if name.split(".")[-1] in target_modules and isinstance(module, nn.Linear)
        ]
        for name, module in targets:
            parts = name.split(".")
            parent = layer.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else layer
            child_attr = parts[-1]

            if mode == "tucker":
                assert bank is not None and factor_lookup is not None  # implied by mode == "tucker"
                key = f"{child_attr}_{module.in_features}x{module.out_features}"
                bank.get_or_create(
                    key,
                    module.in_features,
                    module.out_features,
                    rank_in,
                    rank_out,
                    device=next(module.parameters()).device,
                    dtype=torch.bfloat16,
                )
                wrapper = NovelLoraLinear(
                    module,
                    layer_idx,
                    mode="tucker",
                    alpha=alpha,
                    dropout=dropout,
                    rank_in=rank_in,
                    rank_out=rank_out,
                    factor_lookup=factor_lookup,
                    factor_key=key,
                    is_quiet_fn=is_quiet_fn,
                )
            else:
                wrapper = NovelLoraLinear(
                    module,
                    layer_idx,
                    mode="standard",
                    alpha=alpha,
                    dropout=dropout,
                    rank=rank,
                    is_quiet_fn=is_quiet_fn,
                )
            setattr(parent, child_attr, wrapper)
            wrapped += 1

    if bank is not None:
        model.add_module("novel_lora_bank", bank)

    if velocity_gate is not None:
        install_velocity_hooks(layers, velocity_gate)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    return {
        "bank": bank,
        "wrapped_count": wrapped,
        "trainable_params": trainable_params,
        "total_params": all_params,
        "num_layers": len(layers),
    }


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_novel_adapter(model, out_dir: str | Path, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = {}
    for name, module in model.named_modules():
        if isinstance(module, NovelLoraLinear):
            local = module.state_dict()
            for k, v in local.items():
                if k.startswith("base_layer"):
                    continue
                state[f"{name}.{k}"] = v.detach().cpu().clone()
    if isinstance(getattr(model, "novel_lora_bank", None), TuckerFactorBank):
        for k, v in model.novel_lora_bank.state_dict().items():
            state[f"novel_lora_bank.{k}"] = v.detach().cpu().clone()

    torch.save(state, out_dir / "novel_adapter.pt")
    with open(out_dir / "novel_adapter_config.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_novel_adapter(model, adapter_dir: str | Path, velocity_gate: VelocityGate | None = None) -> dict:
    adapter_dir = Path(adapter_dir)
    meta = json.loads((adapter_dir / "novel_adapter_config.json").read_text())

    summary = apply_novel_lora(
        model,
        mode=meta["mode"],
        target_modules=meta.get("target_modules", TARGET_MODULES),
        rank=meta.get("rank", 8),
        rank_in=meta.get("rank_in", 8),
        rank_out=meta.get("rank_out", 8),
        alpha=meta.get("alpha", 16),
        dropout=0.0,
        velocity_gate=velocity_gate,
    )

    state = torch.load(adapter_dir / "novel_adapter.pt", map_location="cpu")
    msd = model.state_dict()
    missing = []
    for k, v in state.items():
        if k in msd:
            msd[k].data.copy_(v.to(msd[k].dtype).to(msd[k].device))
        else:
            missing.append(k)
    if missing:
        raise RuntimeError(f"Adapter state keys not found in freshly-wrapped model: {missing}")

    summary["meta"] = meta
    return summary
