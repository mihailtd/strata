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
# Master Basis Set Reduction (MasterBasisBank)
# ---------------------------------------------------------------------------


class MasterBasisBank(nn.Module):
    """Owns shared orthogonal Master Basis matrices (basis_u, basis_v), keyed by shape signature.

    Registered once at top-level model as `novel_master_basis_bank` so basis
    matrices reside in VRAM once per model. Task adapters store ONLY the small
    scalar projection coefficients c (sub-kilobyte payloads).
    """

    def __init__(self):
        super().__init__()
        self.basis_u: nn.ParameterDict = nn.ParameterDict()
        self.basis_v: nn.ParameterDict = nn.ParameterDict()

    def get_or_create(
        self,
        key: str,
        in_features: int,
        out_features: int,
        num_basis: int,
        rank_basis: int,
        device,
        dtype,
        correct_fan_in_init: bool = False,
        train_bank: bool = False,
        preloaded: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> tuple[nn.Parameter, nn.Parameter]:
        """`preloaded` supplies a precomputed bank (see scripts/extract_svd_basis.py)
        keyed the same way, i.e. {key: {"basis_u": (k,in,rb), "basis_v": (k,rb,out)}}.
        A random bank spans nothing task-relevant (measured: 12.4-13.4% adherence,
        flat across a 16x alpha sweep), so a preloaded basis is the only
        configuration in which this mode has a chance."""
        if key not in self.basis_u:
            if preloaded is not None and key in preloaded:
                u = preloaded[key]["basis_u"].to(device=device, dtype=dtype).clone()
                v = preloaded[key]["basis_v"].to(device=device, dtype=dtype).clone()
                if u.shape != (num_basis, in_features, rank_basis):
                    raise ValueError(
                        f"preloaded basis_u for {key!r} has shape {tuple(u.shape)}, expected "
                        f"{(num_basis, in_features, rank_basis)} -- rerun extract_svd_basis.py with "
                        f"--num-basis {num_basis} --rank-basis {rank_basis}"
                    )
            else:
                u = torch.empty(num_basis, in_features, rank_basis, device=device, dtype=dtype)
                v = torch.empty(num_basis, rank_basis, out_features, device=device, dtype=dtype)
                if not correct_fan_in_init:
                    nn.init.kaiming_uniform_(u, a=math.sqrt(5))
                else:
                    gain = math.sqrt(2.0 / (1 + 5.0))
                    bound = gain * math.sqrt(3.0 / in_features)
                    with torch.no_grad():
                        u.uniform_(-bound, bound)
                nn.init.normal_(v, std=0.01)
            self.basis_u[key] = nn.Parameter(u, requires_grad=train_bank)
            self.basis_v[key] = nn.Parameter(v, requires_grad=train_bank)
        return self.basis_u[key], self.basis_v[key]


def _make_master_basis_lookup(
    bank: MasterBasisBank,
) -> Callable[[str], tuple[torch.Tensor, torch.Tensor]]:
    def lookup(key: str) -> tuple[torch.Tensor, torch.Tensor]:
        return bank.basis_u[key], bank.basis_v[key]

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
        correct_fan_in_init: bool = False,
    ):
        """`correct_fan_in_init`: opt-in fix for a real init bug, kept OFF by
        default so every previously-measured variant reproduces exactly.

        The down-projection matrices here (`lora_a`, `w_a`, `tucker_a`) are
        stored TRANSPOSED relative to peft's convention -- `(in_features,
        rank)` instead of peft's `(rank, in_features)`. `nn.init.
        kaiming_uniform_` derives fan_in from `tensor.size(1)`, so on this
        layout it reads fan_in = *rank* (8) rather than in_features (2560),
        and initialises the matrix sqrt(in_features/rank) ~= 17.9x too large.

        Measured consequence: because the up-projection is zero-init the
        starting delta is 0 either way (nothing errors, nothing warns), but
        dL/dB is proportional to A, so B grows ~18x faster and the trained
        delta-W comes out 18.29x larger than peft's for an otherwise
        identical config (same r, alpha, scaling, modules, data, steps).
        That inflated update -- not the architecture -- is the leading
        explanation for this repo's custom variants outscoring peft LoRA.

        Setting this True computes the bound from the true fan_in
        (in_features) so the init matches peft's magnitude.
        """
        super().__init__()
        assert mode in ("standard", "tucker", "krotucker", "id_kron", "master_basis")
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        self.layer_idx = layer_idx
        self.mode = mode
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._is_quiet_fn = is_quiet_fn or (lambda idx: False)

        device = next(base_layer.parameters()).device
        compute_dtype = torch.bfloat16

        def init_down_proj(t: torch.Tensor) -> None:
            """kaiming_uniform on a down-projection stored as (in, rank).

            Default reproduces the historical (buggy) behaviour exactly.
            With correct_fan_in_init, the bound is computed from the real
            fan_in (t.size(0), the input dim) instead of letting PyTorch
            infer t.size(1) == rank. See the class docstring.
            """
            if not correct_fan_in_init:
                nn.init.kaiming_uniform_(t, a=math.sqrt(5))
                return
            fan_in = t.size(0)
            gain = math.sqrt(2.0 / (1 + 5.0))  # a = sqrt(5)
            bound = gain * math.sqrt(3.0 / fan_in)
            with torch.no_grad():
                t.uniform_(-bound, bound)

        if mode == "tucker":
            assert factor_lookup is not None and factor_key is not None and rank_in and rank_out
            self.rank = rank_in  # scaling uses the input-side rank, matching LoRA convention
            self.scaling = alpha / rank_in
            self._factor_lookup = factor_lookup
            self.factor_key = factor_key
            core = torch.empty(rank_in, rank_out, device=device, dtype=compute_dtype)
            nn.init.normal_(core, std=0.01)
            self.core = nn.Parameter(core)
            self.diag_scale = nn.Parameter(torch.ones(rank_out, device=device, dtype=compute_dtype))
        elif mode == "master_basis":
            assert factor_lookup is not None and factor_key is not None and rank_in and rank_out
            num_basis = rank_in
            rank_basis = rank_out
            self.num_basis = num_basis
            self.rank_basis = rank_basis
            self.scaling = alpha / num_basis
            self._factor_lookup = factor_lookup
            self.factor_key = factor_key
            coefficients = torch.zeros(num_basis, device=device, dtype=compute_dtype)
            self.coefficients = nn.Parameter(coefficients)
        elif mode == "krotucker":
            assert rank is not None
            self.rank = rank
            self.scaling = alpha / rank
            in_f, out_f = base_layer.in_features, base_layer.out_features
            tucker_a = torch.empty(in_f, rank, device=device, dtype=compute_dtype)
            init_down_proj(tucker_a)
            self.tucker_a = nn.Parameter(tucker_a)
            self.tucker_b = nn.Parameter(torch.zeros(rank, out_f, device=device, dtype=compute_dtype))

            sqrt_r = int(math.isqrt(rank))
            if sqrt_r * sqrt_r == rank:
                kron_a = torch.empty(sqrt_r, sqrt_r, device=device, dtype=compute_dtype)
                kron_b = torch.empty(sqrt_r, sqrt_r, device=device, dtype=compute_dtype)
            else:
                kron_a = torch.empty(rank, 1, device=device, dtype=compute_dtype)
                kron_b = torch.empty(1, rank, device=device, dtype=compute_dtype)
            nn.init.normal_(kron_a, std=0.02)
            nn.init.normal_(kron_b, std=0.02)
            self.kron_a = nn.Parameter(kron_a)
            self.kron_b = nn.Parameter(kron_b)
        elif mode == "id_kron":
            r1 = rank_in if rank_in else 8
            r2 = rank_out if rank_out else 1
            rank_total = r1 * r2
            self.rank = rank_total
            self.scaling = alpha / rank_total
            self.r1 = r1
            self.r2 = r2
            in_f, out_f = base_layer.in_features, base_layer.out_features
            assert in_f % r1 == 0, f"in_features {in_f} must be divisible by r1 {r1}"
            in_sub = in_f // r1
            w_a = torch.empty(in_sub, r2, device=device, dtype=compute_dtype)
            init_down_proj(w_a)
            self.w_a = nn.Parameter(w_a)
            self.b_lora = nn.Parameter(torch.zeros(rank_total, out_f, device=device, dtype=compute_dtype))
        else:
            assert rank is not None
            self.rank = rank
            self.scaling = alpha / rank
            in_f, out_f = base_layer.in_features, base_layer.out_features
            lora_a = torch.empty(in_f, rank, device=device, dtype=compute_dtype)
            init_down_proj(lora_a)
            self.lora_a = nn.Parameter(lora_a)
            self.lora_b = nn.Parameter(torch.zeros(rank, out_f, device=device, dtype=compute_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self._is_quiet_fn(self.layer_idx):
            return result

        h_dtype = (
            self.coefficients.dtype
            if self.mode == "master_basis"
            else (
                self.w_a.dtype
                if self.mode == "id_kron"
                else (
                    self.tucker_a.dtype
                    if self.mode == "krotucker"
                    else (self.core.dtype if self.mode == "tucker" else self.lora_a.dtype)
                )
            )
        )
        h = self.dropout(x).to(h_dtype)
        if self.mode == "tucker":
            u_in, u_out = self._factor_lookup(self.factor_key)
            delta = (h @ u_in) @ self.core
            delta = delta * self.diag_scale
            delta = delta @ u_out
        elif self.mode == "master_basis":
            basis_u, basis_v = self._factor_lookup(self.factor_key)
            bu = basis_u.to(h_dtype)
            bv = basis_v.to(h_dtype)
            coefs = self.coefficients.to(h_dtype)
            h_proj = torch.einsum("...i,kir->...kr", h, bu)
            o_proj = torch.einsum("...kr,kro->...ko", h_proj, bv)
            delta = torch.einsum("...ko,k->...o", o_proj, coefs)
        elif self.mode == "krotucker":
            h_proj = h @ self.tucker_a
            kron_core = torch.kron(self.kron_a, self.kron_b)
            h_core = h_proj @ kron_core
            delta = h_core @ self.tucker_b
        elif self.mode == "id_kron":
            batch_size, seq_len, in_f = h.shape
            h_reshaped = h.view(batch_size, seq_len, self.r1, in_f // self.r1)
            h_proj = torch.matmul(h_reshaped, self.w_a).view(batch_size, seq_len, self.rank)
            delta = h_proj @ self.b_lora
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
    correct_fan_in_init: bool = False,
    preloaded_basis: dict | None = None,
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

    if mode == "tucker":
        bank = TuckerFactorBank()
        factor_lookup = _make_bank_lookup(bank)
    elif mode == "master_basis":
        bank = MasterBasisBank()
        factor_lookup = _make_master_basis_lookup(bank)
    else:
        bank = None
        factor_lookup = None

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
                    correct_fan_in_init=correct_fan_in_init,
                )
            elif mode == "master_basis":
                # isinstance (not just `is not None`) so the MasterBasisBank-only
                # `preloaded` kwarg below type-checks -- `bank` is a union with
                # TuckerFactorBank, whose get_or_create has a different signature.
                assert isinstance(bank, MasterBasisBank) and factor_lookup is not None
                key = f"{child_attr}_{module.in_features}x{module.out_features}"
                bank.get_or_create(
                    key,
                    module.in_features,
                    module.out_features,
                    num_basis=rank_in,
                    rank_basis=rank_out,
                    device=next(module.parameters()).device,
                    dtype=torch.bfloat16,
                    correct_fan_in_init=correct_fan_in_init,
                    preloaded=preloaded_basis,
                )
                wrapper = NovelLoraLinear(
                    module,
                    layer_idx,
                    mode="master_basis",
                    alpha=alpha,
                    dropout=dropout,
                    rank_in=rank_in,
                    rank_out=rank_out,
                    factor_lookup=factor_lookup,
                    factor_key=key,
                    is_quiet_fn=is_quiet_fn,
                    correct_fan_in_init=correct_fan_in_init,
                )
            elif mode == "krotucker":
                wrapper = NovelLoraLinear(
                    module,
                    layer_idx,
                    mode="krotucker",
                    alpha=alpha,
                    dropout=dropout,
                    rank=rank,
                    is_quiet_fn=is_quiet_fn,
                    correct_fan_in_init=correct_fan_in_init,
                )
            elif mode == "id_kron":
                wrapper = NovelLoraLinear(
                    module,
                    layer_idx,
                    mode="id_kron",
                    alpha=alpha,
                    dropout=dropout,
                    rank=rank,
                    rank_in=rank_in,
                    rank_out=rank_out,
                    is_quiet_fn=is_quiet_fn,
                    correct_fan_in_init=correct_fan_in_init,
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
                    correct_fan_in_init=correct_fan_in_init,
                )
            setattr(parent, child_attr, wrapper)
            wrapped += 1

    if bank is not None:
        bank_attr = "novel_lora_bank" if mode == "tucker" else "novel_master_basis_bank"
        model.add_module(bank_attr, bank)

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
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    curr = model
    while hasattr(curr, "module"):
        curr = curr.module

    state = {}
    for name, module in curr.named_modules():
        if "NovelLoraLinear" in type(module).__name__ or isinstance(module, NovelLoraLinear):
            local = module.state_dict()
            for k, v in local.items():
                if k.startswith("base_layer"):
                    continue
                state[f"{name}.{k}"] = v.detach().cpu().clone()

    if isinstance(getattr(curr, "novel_lora_bank", None), TuckerFactorBank):
        for k, v in curr.novel_lora_bank.state_dict().items():
            state[f"novel_lora_bank.{k}"] = v.detach().cpu().clone()
    if meta.get("save_bank", False) and isinstance(getattr(curr, "novel_master_basis_bank", None), MasterBasisBank):
        for k, v in curr.novel_master_basis_bank.state_dict().items():
            state[f"novel_master_basis_bank.{k}"] = v.detach().cpu().clone()

    print(f"[save_novel_adapter] Extracted {len(state)} adapter weight tensors -> {out_dir / 'novel_adapter.pt'}")
    torch.save(state, out_dir / "novel_adapter.pt")
    with open(out_dir / "novel_adapter_config.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_novel_adapter(model, adapter_dir: str | Path, velocity_gate: VelocityGate | None = None) -> dict:
    adapter_dir = Path(adapter_dir)
    meta = json.loads((adapter_dir / "novel_adapter_config.json").read_text())

    # master_basis adapters store ONLY mixture coefficients; the basis bank they
    # index into lives in a separate file and is meaningless to reload randomly
    # (a random bank scores ~base, measured). Restore the exact bank recorded at
    # save time, and fail loudly rather than silently evaluating against a
    # different basis than the coefficients were fitted to.
    preloaded_basis = None
    bank_ref = meta.get("basis_bank")
    if bank_ref:
        bank_path = Path(bank_ref)
        if not bank_path.is_absolute():
            bank_path = Path(__file__).resolve().parent.parent.parent / bank_ref
        if not bank_path.exists():
            raise FileNotFoundError(
                f"Adapter {adapter_dir} references basis bank {bank_ref!r}, which is missing. "
                "The coefficients are unusable without the exact bank they were fitted to."
            )
        preloaded_basis = torch.load(bank_path, map_location="cpu", weights_only=False)["bank"]

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
        preloaded_basis=preloaded_basis,
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


# ---------------------------------------------------------------------------
# Hot-swap primitive
# ---------------------------------------------------------------------------


class AdapterPayload:
    """An adapter's weights held in host memory, ready for repeated H2D copies.

    Deliberately separate from `swap_adapter_weights` so the disk read and the
    device transfer are never conflated -- benchmarking a "swap" that secretly
    includes a `torch.load` measures the SSD, not the bus.

    `pin_memory=True` page-locks the host buffers, which is what actually makes
    `copy_(non_blocking=True)` asynchronous. From ordinary pageable memory the
    driver must stage through an internal bounce buffer and the copy is
    effectively synchronous regardless of the flag -- a common way to
    accidentally measure nothing.
    """

    def __init__(self, state: dict[str, torch.Tensor], pin_memory: bool = True, name: str = ""):
        self.name = name
        self.pinned = pin_memory and torch.cuda.is_available()
        self.state: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            t = v.detach().contiguous()
            self.state[k] = t.pin_memory() if self.pinned else t
        self.nbytes = sum(t.numel() * t.element_size() for t in self.state.values())

    @classmethod
    def from_dir(cls, adapter_dir: str | Path, pin_memory: bool = True) -> AdapterPayload:
        adapter_dir = Path(adapter_dir)
        state = torch.load(adapter_dir / "novel_adapter.pt", map_location="cpu")
        return cls(state, pin_memory=pin_memory, name=adapter_dir.name)

    def __repr__(self) -> str:
        return (
            f"AdapterPayload({self.name!r}, {len(self.state)} tensors, "
            f"{self.nbytes / 1e6:.3f} MB, pinned={self.pinned})"
        )


def prepare_swap_slots(model, payload: AdapterPayload) -> dict[str, torch.Tensor]:
    """Resolve payload keys to the model's EXISTING parameter tensors once.

    Doing this lookup up front means a swap is a pure sequence of `copy_`
    calls with no dict traversal, no attribute walking, and no allocation.
    Shapes/dtypes are validated here so a mismatched adapter fails loudly at
    setup rather than silently corrupting weights mid-run.
    """
    msd = model.state_dict()
    slots: dict[str, torch.Tensor] = {}
    missing, mismatched = [], []
    for k, v in payload.state.items():
        if k not in msd:
            missing.append(k)
            continue
        dst = msd[k]
        if tuple(dst.shape) != tuple(v.shape):
            mismatched.append(f"{k}: model {tuple(dst.shape)} vs payload {tuple(v.shape)}")
            continue
        slots[k] = dst
    if missing:
        raise KeyError(
            f"{len(missing)} payload keys absent from the wrapped model (first 3: {missing[:3]}). "
            "The model must already be wrapped with a compatible adapter architecture -- "
            "swap_adapter_weights never re-wraps."
        )
    if mismatched:
        raise ValueError(f"shape mismatch in {len(mismatched)} tensors (first 3: {mismatched[:3]})")
    return slots


@torch.no_grad()
def swap_adapter_weights(
    slots: dict[str, torch.Tensor],
    payload: AdapterPayload,
    non_blocking: bool = True,
    batched: bool = True,
) -> None:
    """Overwrite an already-wrapped model's adapter weights in place.

    This is the primitive `load_novel_adapter` is NOT: that function calls
    `apply_novel_lora`, which walks the module tree and replaces every target
    Linear with a fresh wrapper. Calling it twice on one model double-wraps and
    corrupts the structure (confirmed by a real crash when loading a second
    adapter onto an already-wrapped model). It also reallocates every adapter
    tensor, so its cost is dominated by Python-side module surgery rather than
    by the actual bytes moved.

    Here the destination tensors are the ones already living in VRAM; a swap is
    a fixed set of device writes into static addresses. No allocation, no graph
    mutation, no autograd bookkeeping.

    `batched=True` issues the whole swap through `torch._foreach_copy_` instead
    of one `copy_` per tensor. These adapters are many small tensors (256 for a
    rank-8 LoRA, 128 for a coefficient set), and measurement showed the swap was
    latency-bound on per-call overhead rather than bandwidth: 21.2 MB moved in
    31 ms is 0.68 GB/s, ~35x below what the bus can do, i.e. ~122 us of overhead
    per tensor and almost no time actually transferring. Batching collapses that
    into far fewer launches.

    NOTE: with non_blocking=True the copies are queued, not completed, on
    return. Call `torch.cuda.synchronize()` before timing or before relying on
    the new weights from host code -- kernels queued afterwards on the same
    stream will observe them correctly regardless.
    """
    if batched and hasattr(torch, "_foreach_copy_"):
        keys = list(slots)
        torch._foreach_copy_([slots[k] for k in keys], [payload.state[k] for k in keys], non_blocking=non_blocking)
        return
    for k, dst in slots.items():
        dst.copy_(payload.state[k], non_blocking=non_blocking)


# ---------------------------------------------------------------------------
# In-Place Adapter Weight Folding (Zero-Allocation Mutating Hot-Swap)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Weight folding: run an adapted model at unwrapped base-model speed
# ---------------------------------------------------------------------------
#
# WHY: at batch 1 the adapter wrapper costs ~19% of decode throughput on this
# rig -- not FLOPs (the adapter is 0.44% of parameters) but 256 extra small
# kernel launches per token, 2 per wrapped Linear. Folding dW into the base
# weight deletes the wrapper from the execution path entirely.
#
# THE THREE THINGS THAT MAKE THIS CORRECT, each of which a naive version gets
# wrong:
#
#   1. FACTORS STAY FACTORED. A dense dW payload for this model is 10.2 GB in
#      fp32 -- an 812x expansion of the 12.6 MB adapter -- so holding three of
#      them needs 30.7 GB of pinned host RAM (this box has 23 GB) and a swap
#      would push 2 x 10.2 GB over a 12.6 GB/s PCIe link: ~1.6 s. Keeping the
#      (out, r) x (r, in) factors on-device instead makes a swap one fused
#      `addmm` per module, bounded by VRAM bandwidth rather than PCIe.
#
#   2. SCALING COMES FROM THE ADAPTER, NOT A CONSTANT. `NovelLoraLinear` uses
#      alpha / rank_total, and for id_kron rank_total = rank_in * rank_out. A
#      hardcoded 2.0 is right for r1*r2 = 8 and 8x too large for r1*r2 = 64.
#      Here rank_total is read off the tensors themselves and cross-checked
#      against the config.
#
#   3. RESTORE IS A COPY, NOT A SUBTRACTION. bf16 has 8 mantissa bits, so
#      (W + dW) - dW != W: measured 4.9e-4 L_inf drift after 400 add/sub
#      cycles. Keeping a pristine W0 and writing W_live = W0 + dW makes every
#      activation independent of history and restore exact by construction.
#      Costs one extra copy of the wrapped weights in VRAM (5.1 GB bf16 here).


def unwrap_novel_lora(model: nn.Module) -> int:
    """Replace every NovelLoraLinear with its base Linear, in place.

    Lets a single process measure wrapped and folded execution of the *same*
    weights back to back, which is the only way to compare them without a
    reload confounding the timing.
    """
    n = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, NovelLoraLinear):
                setattr(parent, name, child.base_layer)
                n += 1
    return n


class FoldableExpert:
    """One adapter as device-resident factors: dW = scaling * (U @ V).

    U is (out, r), V is (r, in), so dW matches an nn.Linear weight (out, in)
    directly and folding is `addmm(W0, U, V, alpha=scaling)`.

    Every supported layout is normalised into that one pair:
      peft     lora_A (r, in), lora_B (out, r)   -> U = B,            V = A
      standard lora_a (in, r), lora_b (r, out)   -> U = lora_b^T,     V = lora_a^T
      id_kron  w_a (in/r1, r2), b_lora (r1*r2, out)
               the wrapper splits h into r1 contiguous chunks and applies the
               shared w_a to each, i.e. an implicit block-diagonal down-proj
               A = blkdiag(w_a x r1) of shape (in, r1*r2)
                                              -> U = b_lora^T,        V = A^T
    """

    def __init__(self, factors: dict[str, tuple[torch.Tensor, torch.Tensor]], scaling: float, name: str = ""):
        self.factors = factors
        self.scaling = float(scaling)
        self.name = name

    @property
    def nbytes(self) -> int:
        return sum(u.numel() * u.element_size() + v.numel() * v.element_size() for u, v in self.factors.values())

    def to(self, device, dtype) -> FoldableExpert:
        self.factors = {k: (u.to(device, dtype), v.to(device, dtype)) for k, (u, v) in self.factors.items()}
        return self

    def __repr__(self) -> str:
        return (
            f"FoldableExpert({self.name!r}, {len(self.factors)} modules, "
            f"scaling={self.scaling}, {self.nbytes / 1e6:.1f} MB)"
        )

    @staticmethod
    def _weight_key(module_path: str) -> str:
        """'model.layers.0.mlp.gate_proj' -> 'model.layers.0.mlp.gate_proj.weight'."""
        return f"{module_path}.weight"

    @classmethod
    def from_dir(cls, adapter_dir: str | Path, name: str = "") -> FoldableExpert:
        adapter_dir = Path(adapter_dir)
        name = name or adapter_dir.name

        if (adapter_dir / "novel_adapter_config.json").exists():
            cfg = json.loads((adapter_dir / "novel_adapter_config.json").read_text())
            state = torch.load(adapter_dir / "novel_adapter.pt", map_location="cpu")
            return cls._from_novel(state, cfg, name)

        if (adapter_dir / "adapter_config.json").exists():
            from peft.utils import load_peft_weights

            cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
            return cls._from_peft(load_peft_weights(str(adapter_dir)), cfg, name)

        raise FileNotFoundError(f"No novel_adapter_config.json or adapter_config.json in {adapter_dir}")

    @classmethod
    def _from_peft(cls, state: dict, cfg: dict, name: str) -> FoldableExpert:
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        for k, v in state.items():
            for tag, slot in ((".lora_A", "A"), (".lora_B", "B")):
                if tag in k:
                    mod = k.split(tag)[0].replace("base_model.model.", "")
                    pairs.setdefault(mod, {})[slot] = v
        factors = {}
        for mod, d in pairs.items():
            if "A" in d and "B" in d:
                factors[cls._weight_key(mod)] = (d["B"].float(), d["A"].float())
        rank = cfg["r"]
        return cls(factors, scaling=cfg["lora_alpha"] / rank, name=name)

    @classmethod
    def _from_novel(cls, state: dict, cfg: dict, name: str) -> FoldableExpert:
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        for k, v in state.items():
            mod, _, leaf = k.rpartition(".")
            if leaf in ("w_a", "b_lora", "lora_a", "lora_b"):
                pairs.setdefault(mod, {})[leaf] = v
        if not pairs:
            raise ValueError(f"{name}: no foldable (w_a/b_lora or lora_a/lora_b) tensors found")

        factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        rank_totals = set()
        for mod, d in pairs.items():
            if "w_a" in d and "b_lora" in d:  # id_kron
                w_a, b = d["w_a"].float(), d["b_lora"].float()
                rank_total, _ = b.shape
                r2 = w_a.shape[1]
                r1 = rank_total // max(1, r2)
                if r1 * r2 != rank_total:
                    raise ValueError(f"{mod}: b_lora rank {rank_total} is not r1*r2 for r2={r2}")
                # implicit block-diagonal down-projection, (in, r1*r2)
                A = torch.block_diag(*([w_a] * r1))
                factors[cls._weight_key(mod)] = (b.T.contiguous(), A.T.contiguous())
            elif "lora_a" in d and "lora_b" in d:  # standard
                a, b = d["lora_a"].float(), d["lora_b"].float()
                rank_total = a.shape[1]
                factors[cls._weight_key(mod)] = (b.T.contiguous(), a.T.contiguous())
            else:
                continue
            rank_totals.add(rank_total)

        if len(rank_totals) != 1:
            raise ValueError(f"{name}: inconsistent rank across modules: {sorted(rank_totals)}")
        rank_total = rank_totals.pop()

        # NovelLoraLinear sets self.scaling = alpha / rank_total. Derive it from
        # the tensors (authoritative) and warn if the config disagrees.
        alpha = cfg.get("alpha", 16)
        scaling = alpha / rank_total
        if cfg.get("mode") == "id_kron":
            cfg_rank = (cfg.get("rank_in") or 8) * (cfg.get("rank_out") or 1)
        else:
            cfg_rank = cfg.get("rank", 8)
        if cfg_rank != rank_total:
            warnings.warn(
                f"{name}: rank from tensors ({rank_total}) != rank from config ({cfg_rank}); "
                f"trusting tensors, scaling = {alpha}/{rank_total} = {scaling}",
                stacklevel=2,
            )
        return cls(factors, scaling=scaling, name=name)


class WeightFoldingEngine:
    """Folds experts into a plain (unwrapped) model's weights and back out.

    Holds a pristine copy of every weight any expert touches, so `activate`
    always writes W0 + dW rather than mutating whatever is currently loaded.
    That makes activations order-independent and `restore` bit-exact.
    """

    def __init__(self, model: nn.Module, experts: Iterable[FoldableExpert], keep_pristine: bool = True):
        params = dict(model.named_parameters())
        self.experts = list(experts)
        self.keep_pristine = keep_pristine

        touched: set[str] = set()
        for e in self.experts:
            for key, (u, v) in e.factors.items():
                if key not in params:
                    raise KeyError(f"{e.name}: '{key}' is not a parameter of the model")
                w = params[key]
                if (u.shape[0], v.shape[1]) != tuple(w.shape):
                    raise ValueError(
                        f"{e.name}: '{key}' delta {(u.shape[0], v.shape[1])} != weight {tuple(w.shape)}"
                    )
                touched.add(key)

        self.slots = {k: params[k] for k in sorted(touched)}
        # move factors onto the weights they fold into
        ref = next(iter(self.slots.values()))
        for e in self.experts:
            e.to(ref.device, ref.dtype)

        self.pristine = {k: v.detach().clone() for k, v in self.slots.items()} if keep_pristine else {}
        self.active: str | None = None

    @property
    def pristine_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.pristine.values())

    @torch.no_grad()
    def activate(self, expert: FoldableExpert) -> None:
        """W_live = W0 + scaling * (U @ V), one fused addmm per module."""
        if not self.keep_pristine:
            raise RuntimeError("activate() requires keep_pristine=True; use activate_delta() otherwise")
        for key, w in self.slots.items():
            f = expert.factors.get(key)
            w0 = self.pristine[key]
            if f is None:
                w.copy_(w0)
                continue
            u, v = f
            torch.addmm(w0, u, v, beta=1.0, alpha=expert.scaling, out=w)
        self.active = expert.name

    @torch.no_grad()
    def restore(self) -> None:
        """Exact: copies the pristine weights back, no arithmetic involved."""
        for key, w in self.slots.items():
            w.copy_(self.pristine[key])
        self.active = None

    @torch.no_grad()
    def max_drift(self) -> float:
        """L_inf between live weights and pristine -- 0 only if truly restored."""
        return max((self.slots[k] - self.pristine[k]).abs().max().item() for k in self.slots)
