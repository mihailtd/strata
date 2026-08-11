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

3. Inference-time velocity skipping: the same VelocityGate/is_quiet_fn
   machinery, but driven per generated *token* instead of per training step
   (see VelocityGateTickLogitsProcessor) -- a distinct experiment from
   training-time masking, since it changes decode-time FLOPs/memory reads
   rather than backward-pass cost, and needs its own threshold calibration
   (generation-time per-layer velocity is a different distribution than
   training-batch velocity).
"""

from __future__ import annotations

import json
import math
import random
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from torch import nn
from transformers import LogitsProcessor


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
# EXPERIMENT STATUS: FAILED — DO NOT USE FOR NEW WORK
#
# Tucker factorization was evaluated in two configurations:
#   Tucker v1 — r=8, alpha=16   → 487K trainable params, 1.86 MB optimizer state
#   Tucker v2 — r=32, alpha=64, + per-layer diagonal scale → 2.05M params, 7.83 MB
#
# Results vs. custom_standard reference (10.6M params):
#   Tucker v1  training loss 1.613,  adherence 17.27%
#   Tucker v2  training loss 1.399,  adherence 15.66%  ← capacity fix worked on loss…
#   custom_std training loss 1.366,  adherence 59.95%  ← …but adherence didn't move
#
# Root cause: every layer in a shape-group is forced through the SAME shared
# basis directions (U_in, U_out).  A per-layer diagonal scale vector lets each
# layer independently rescale that shared subspace, but CANNOT rotate it into a
# genuinely different subspace.  The pip→uv / black→ruff swap behaviour we
# measure requires each layer to push in a different gradient direction — not
# just a differently-scaled version of one shared direction.
#
# This is NOT a parameter-count problem (4× more capacity in v2 didn't help).
# Remaining candidate fixes (per-depth-block factor groups, HOSVD init) were
# not implemented because the shared-subspace limitation is likely fundamental
# for this class of tasks.
#
# Code is preserved here for reference.  Do not add new Tucker experiments
# without first addressing the shared-basis subspace problem.
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
        quiet_percentile: float = 25.0,
        ema_decay: float = 0.9,
        warmup_steps: int = 20,
        max_quiet_fraction: float = 0.9,
        selection: str = "velocity",
        seed: int = 0,
    ):
        """`quiet_percentile`: each step, the bottom `quiet_percentile`% of
        layers by EMA velocity go quiet -- relative to the *current* velocity
        distribution, not a fixed absolute number. A fixed threshold (this
        gate's original design) has to be recalibrated by hand any time the
        training data changes, and silently produces zero quiet layers if the
        real distribution shifts away from it without warning (measured: the
        original 0.45 threshold, tuned on one dataset, produced 0% quiet
        layers when the underlying data composition changed -- see
        goal1-novel-adapter-results memory / this session's postgres run).
        Percentile gating self-calibrates to whatever the real distribution
        is, so it can't silently go inert this way.

        `selection`:
          "velocity" -- mask the bottom `quiet_percentile`% of layers by EMA
                        velocity (the actual technique).
          "random"   -- mask a uniformly random subset *of the same size*,
                        redrawn every step, ignoring the measured velocities.
                        This is the control arm: it isolates "does ranking by
                        velocity carry signal" from "does masking N layers at
                        all change anything". If random matches or beats
                        velocity, the velocity ranking contributes nothing and
                        the technique reduces to a capacity/regularization
                        knob. Hooks stay installed in both modes so the
                        measurement overhead and code path are identical.
        """
        if selection not in ("velocity", "random"):
            raise ValueError(f"selection must be 'velocity' or 'random', got {selection!r}")
        super().__init__()
        self.ema_velocity: torch.Tensor
        self.register_buffer("ema_velocity", torch.full((num_layers,), float("nan")))
        # Per-step scratch space the hooks write into. Kept as a GPU buffer
        # (not a Python dict/list) specifically so `record()` never needs a
        # host sync -- see its docstring.
        self.pending_velocity: torch.Tensor
        self.register_buffer("pending_velocity", torch.full((num_layers,), float("nan")))
        self.num_layers = num_layers
        self.quiet_percentile = quiet_percentile
        self.ema_decay = ema_decay
        self.warmup_steps = warmup_steps
        # Safety valve: never mask every layer in the same step, or the loss tensor
        # loses requires_grad entirely (nothing trainable touches the graph) and
        # Trainer's loss.backward() raises. Keeps at least a floor of active layers.
        self.max_quiet_fraction = max_quiet_fraction
        self.selection = selection
        # Own Random instance, not the global `random` module, so drawing the
        # control arm's masks can't perturb any other seeded stream.
        self._rng = random.Random(seed)
        self.step_count = 0
        self.quiet_layers: set[int] = set()
        self.quiet_fraction_history: list[float] = []
        # Layer-level churn: how much the quiet set changes step to step. The
        # velocity arm is expected to be near-static (measured: per-layer
        # velocity is nearly constant batch-to-batch, so the ranking barely
        # moves); the random arm re-draws every step. Recorded so the two are
        # distinguishable in the results rather than assumed.
        self.quiet_churn_history: list[float] = []
        self._warned_no_observations = False

    @torch.no_grad()
    def record(self, layer_idx: int, velocity: torch.Tensor) -> None:
        """Write this layer's velocity for the in-progress step.

        Deliberately does nothing but a same-device indexed write -- no
        `.item()`/`.cpu()`/Python `if` on a tensor anywhere in this method.
        Each of those forces a host<->device sync, and this runs once per
        decoder layer per forward (doubled again under gradient checkpointing,
        which re-invokes every layer's forward during backward). The first,
        naive version of this method called `.cpu()` here and made training
        29% *slower* despite skipping real compute -- ~64 forced GPU stalls
        every step ate more wall-clock than the skipped matmuls saved. All
        the actual math (EMA blend, NaN handling, host sync for the quiet-set
        decision) now happens exactly once per step in `end_step()` instead
        of once per layer here. Requires `self` to already be on the same
        device as the model (call `.to(model_device)` right after
        construction) or this write itself becomes a sync point again.
        """
        self.pending_velocity[layer_idx] = velocity.detach()

    def end_step(self) -> None:
        self.step_count += 1
        with torch.no_grad():
            fired = ~torch.isnan(self.pending_velocity)
            first_obs = torch.isnan(self.ema_velocity) & fired
            blend = fired & ~first_obs
            self.ema_velocity = torch.where(first_obs, self.pending_velocity, self.ema_velocity)
            blended = self.ema_decay * self.ema_velocity + (1 - self.ema_decay) * self.pending_velocity
            self.ema_velocity = torch.where(blend, blended, self.ema_velocity)
            self.pending_velocity.fill_(float("nan"))

        if self.step_count > self.warmup_steps:
            vals = self.ema_velocity.tolist()  # the one intentional sync/step, not one per layer
            observed = sorted((v, i) for i, v in enumerate(vals) if not math.isnan(v))
            if not observed and not self._warned_no_observations:
                # Warmup has passed and not a single layer has ever recorded a
                # velocity -- the gate is structurally inert (permanently 0%
                # quiet, no regularization effect) with no other symptom: no
                # crash, no exception, training proceeds normally. Measured
                # real cause once already (skip_recompute=True's
                # torch.is_grad_enabled() heuristic misfiring under
                # non-reentrant gradient checkpointing, see
                # install_velocity_hooks docstring) -- but *anything* that
                # stops the hooks from firing looks identical from here, so
                # this checks the symptom, not that one specific cause.
                warnings.warn(
                    "VelocityGate: warmup complete but zero layers have ever recorded a "
                    "velocity -- quiet_layers will stay permanently empty. The gate is having "
                    "no effect. Check that install_velocity_hooks' hooks are actually firing "
                    "(a common cause: skip_recompute=True combined with non-reentrant gradient "
                    "checkpointing, where torch.is_grad_enabled() never signals 'recompute' the "
                    "way that flag assumes).",
                    stacklevel=2,
                )
                self._warned_no_observations = True
            # Bottom quiet_percentile% of currently-observed layers, capped by
            # max_quiet_fraction (safety valve, see __init__). Relative to
            # `len(observed)`, not `self.num_layers`, so an early step with
            # only some layers having fired yet doesn't undercount the target.
            target_n = min(
                round(len(observed) * self.quiet_percentile / 100),
                int(self.num_layers * self.max_quiet_fraction),
            )
            prev_quiet = self.quiet_layers
            if target_n <= 0:
                self.quiet_layers = set()
            elif self.selection == "random":
                # Control arm: same count, same pool of observed layers, but
                # redrawn uniformly at random each step -- see __init__.
                self.quiet_layers = set(self._rng.sample([i for _, i in observed], target_n))
            else:
                self.quiet_layers = {i for _, i in observed[:target_n]}
            if prev_quiet or self.quiet_layers:
                changed = len(prev_quiet.symmetric_difference(self.quiet_layers))
                denom = max(len(prev_quiet), len(self.quiet_layers), 1)
                self.quiet_churn_history.append(changed / (2 * denom))
        else:
            self.quiet_layers = set()
        self.quiet_fraction_history.append(len(self.quiet_layers) / self.num_layers)

    def is_quiet(self, layer_idx: int) -> bool:
        return layer_idx in self.quiet_layers


def install_velocity_hooks(layers: Iterable[nn.Module], gate: VelocityGate, skip_recompute: bool = False) -> list:
    """Forward hooks measuring per-layer relative hidden-state velocity.

    `skip_recompute=True` skips the (real, non-trivial: two fp32 upcasts plus
    two norm reductions, ~7 kernel launches) measurement during gradient
    checkpointing's recompute pass, where it would otherwise fire a second,
    redundant time for the same step. Detected via `torch.is_grad_enabled()`:
    for *reentrant* `torch.utils.checkpoint` (the old default), the original
    forward runs under `no_grad()` and only the recompute re-enables grad, so
    "grad enabled inside this hook" reliably means "this is the recompute."

    DO NOT pass True with *non-reentrant* checkpointing (`use_reentrant=False`,
    the modern default `transformers`' `gradient_checkpointing_enable()` uses)
    -- there, grad stays enabled throughout the original forward too, so
    `torch.is_grad_enabled()` is always True and this unconditionally skips
    *every* hook firing, on every layer, every step. Measured directly: this
    silently produced zero recorded velocities and permanently empty
    quiet_layers across full training runs (no crash, no warning at the time
    -- see the warmup-complete-with-zero-observations check in
    `VelocityGate.end_step`, added after finding this). Only pass True when
    the caller has confirmed which checkpointing mode is actually active, not
    just that checkpointing is "on" -- the two modes need opposite settings
    here despite both being "gradient checkpointing enabled."

    Only pass True at all when the caller *knows* gradient checkpointing is
    active on this model -- on an uncheckpointed forward (grad enabled, no
    recompute at all) it has the same failure mode as the non-reentrant case
    above, for the same reason.
    """

    handles = []

    def make_hook(idx: int):
        def hook(module, inputs, output):
            if skip_recompute and torch.is_grad_enabled():
                return
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


class VelocityGateTickLogitsProcessor(LogitsProcessor):
    """Drives a VelocityGate once per *generated token* instead of once per
    training step, for inference-time layer skipping.

    `generate()`'s loop, per step, is: run the forward pass for the current
    token (this fires the same per-layer forward hooks used in training,
    writing this token's velocities into the gate's pending buffer) -> call
    each registered LogitsProcessor on the resulting logits -> sample/pick
    the next token. A LogitsProcessor is therefore called exactly once per
    token, strictly after that token's hooks have already fired and strictly
    before the next token's forward pass begins -- the same "decide the next
    step using the step that just finished" lag `end_step()` already
    implements for training, just retargeted from steps to tokens. Returns
    `scores` unchanged; it exists purely for this side effect.
    """

    def __init__(self, gate: VelocityGate):
        self.gate = gate

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.gate.end_step()
        return scores


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
            # Per-layer trainable magnitude vector over the shared r_out subspace.
            # U_in/U_out and (with a shared shape group) the core's *rank* are the
            # same for every layer in the group; this is the one piece of this
            # layer's adapter that's entirely its own, letting it scale each shared
            # output direction up or down independently instead of every layer in
            # the group being forced through an identical relative mix. Ones-init:
            # a no-op at start (also moot initially since U_out is zero-init).
            self.diag_scale = nn.Parameter(torch.ones(rank_out, device=device, dtype=compute_dtype))
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
            delta = delta * self.diag_scale
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
    velocity_skip_recompute: bool = False,
) -> dict:
    """Freezes the whole model, then replaces every target Linear inside every
    decoder layer with a NovelLoraLinear. Returns a summary dict (wrapped
    count, trainable param count, and the bank if mode == "tucker").

    `velocity_skip_recompute` is forwarded to `install_velocity_hooks` -- only
    pass True when the caller has itself enabled gradient checkpointing on
    `model` (see that function's docstring for why)."""

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
        install_velocity_hooks(layers, velocity_gate, skip_recompute=velocity_skip_recompute)

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
