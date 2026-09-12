"""Group-wise quantize / dequantize against an arbitrary grid.

The whole of phase 1 lives here: take an fp16/bf16 weight matrix, map it onto a
discrete grid, map it straight back, and measure what you broke. Nothing in
this module allocates anything smaller than fp16 -- the compression is
simulated. ``ftq.packing`` is what makes it real on disk.

Layout convention: weights are quantized along the *input* dimension, i.e. a
Linear weight of shape (out_features, in_features) is reshaped to
(-1, group_size) so that each group is a contiguous run of inputs sharing one
scale. That matches how a GEMM kernel walks the K dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from . import grids


@dataclass
class QuantConfig:
    """One quantization scheme. This is the thing you sweep."""

    grid: str = "uniform"
    bits: int = 4
    group_size: int = 64  # -1 for per-output-channel
    n_levels: int | None = None  # set for fractional bit widths (overrides bits)
    asymmetric: bool = False  # per-group zero point (costs another scalar/group)
    clip_search: int = 0  # >0: try N shrink factors per group, keep lowest MSE
    clip_min: float = 0.55
    grid_kwargs: dict = field(default_factory=dict)
    scale_dtype: torch.dtype = torch.float16

    @property
    def levels_count(self) -> int:
        """Size of the code space this config asks for.

        An upper bound, not a promise: a grid may return fewer levels than the
        bit width allows (``uniform`` spends one code keeping zero exact, so it
        returns 2**bits - 1). What actually got used is
        ``QuantizedTensor.n_levels``, and that is what packing and bit
        accounting read -- 15 levels still costs 4 dense bits, but only
        log2(15) = 3.91 under mixed-radix packing.
        """
        return self.n_levels if self.n_levels is not None else 2**self.bits

    @classmethod
    def parse(cls, spec: str) -> "QuantConfig":
        """Build a config from ``grid:bits:group[:flags]``.

        Flags are comma-separated: ``clipN`` sets the clip search width, ``asym``
        adds a per-group zero point, ``kN`` overrides the level count for
        fractional bit widths, ``pF`` sets the power-grid exponent.

            nf:3:64:clip8        NF-style grid, 3 bits, group 64, 8-step clip search
            uniform:0:64:k6      6 uniform levels = 2.585 bits per weight
            power:4:128:p2.5     companded grid with exponent 2.5
        """
        parts = spec.split(":")
        if len(parts) < 3:
            raise ValueError(f"expected grid:bits:group[:flags], got {spec!r}")
        cfg = cls(grid=parts[0], bits=int(parts[1]), group_size=int(parts[2]))
        for flag in (parts[3].split(",") if len(parts) > 3 else []):
            if flag.startswith("clip"):
                cfg.clip_search = int(flag[4:])
            elif flag == "asym":
                cfg.asymmetric = True
            elif flag.startswith("k"):
                cfg.n_levels = int(flag[1:])
            elif flag.startswith("p"):
                cfg.grid_kwargs["p"] = float(flag[1:])
            else:
                raise ValueError(f"unknown flag {flag!r} in {spec!r}")
        return cfg

    def describe(self) -> str:
        b = grids.effective_bits(self.levels_count)
        g = "chan" if self.group_size == -1 else self.group_size
        extra = "".join(
            [
                "/asym" if self.asymmetric else "",
                f"/clip{self.clip_search}" if self.clip_search else "",
            ]
        )
        return f"{self.grid}@{b:.3g}b g{g}{extra}"


@dataclass
class QuantizedTensor:
    """Codes plus everything needed to reconstruct, still in fat dtypes."""

    codes: Tensor  # int16, shape (rows, groups, group_size)
    scale: Tensor  # (rows, groups, 1)
    zero: Tensor | None  # (rows, groups, 1) in weight units, or None
    levels: Tensor  # (n_levels,) float32, shared by the whole tensor
    shape: torch.Size
    config: QuantConfig

    @property
    def n_levels(self) -> int:
        """Levels the grid actually produced -- the number packing must respect."""
        return self.levels.numel()

    def dequantize(self, dtype: torch.dtype = torch.float16) -> Tensor:
        # level * scale in fp32 even when both are stored fp16: the product is
        # what the weight *is*, and rounding it twice costs accuracy the grid
        # never spent. It also makes this agree bit-for-bit with the packed
        # path, which is what the parity tests check.
        lv = self.levels.float().to(self.codes.device)
        out = lv[self.codes.long()] * self.scale.float()
        if self.zero is not None:
            out = out + self.zero.float()
        return out.reshape(self.shape).to(dtype)

    def overhead_bits_per_weight(self) -> float:
        """Bits per weight spent on scales/zeros (not on codes)."""
        per_scalar = torch.finfo(self.config.scale_dtype).bits
        n_scalars = self.scale.numel() + (self.zero.numel() if self.zero is not None else 0)
        return per_scalar * n_scalars / self.codes.numel()


def _grouped(w: Tensor, group_size: int) -> tuple[Tensor, int]:
    rows, cols = w.shape
    gs = cols if group_size in (-1, None) else group_size
    if cols % gs:
        raise ValueError(f"in_features {cols} not divisible by group_size {gs}")
    return w.reshape(rows, cols // gs, gs), gs


def _nearest(x: Tensor, levels: Tensor) -> Tensor:
    """Nearest-level index via bucketize on midpoints. O(n log K), no K x n matrix."""
    edges = (levels[1:] + levels[:-1]) / 2
    return torch.bucketize(x.contiguous(), edges.contiguous())


def _fit_groups(
    xg: Tensor, levels: Tensor, cfg: QuantConfig
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Fit scale (and zero) per group, returning codes, scale, zero."""
    if cfg.asymmetric:
        lo = xg.amin(dim=-1, keepdim=True)
        hi = xg.amax(dim=-1, keepdim=True)
        span = levels[-1] - levels[0]
        scale = ((hi - lo) / span).clamp_min(1e-12)
        zero = lo - levels[0] * scale
    else:
        # Level 1.0 is by definition the group's absmax; grids are normalized to
        # that. A fitted codebook whose top level is below 1.0 simply never
        # predicts the extreme value, which is what its fit asked for.
        scale = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        zero = None

    def encode(scale: Tensor, zero: Tensor | None) -> tuple[Tensor, Tensor]:
        normed = (xg - zero) / scale if zero is not None else xg / scale
        codes = _nearest(normed, levels)
        recon = levels[codes] * scale + (zero if zero is not None else 0)
        return codes, recon

    codes, recon = encode(scale, zero)
    if not cfg.clip_search:
        return codes, scale, zero

    # Shrinking the scale trades clipping error in the tails for finer steps in
    # the bulk. Which side wins is tensor-dependent, so search it per group.
    best_err = (recon - xg).pow(2).sum(dim=-1, keepdim=True)
    for i in range(1, cfg.clip_search + 1):
        f = 1.0 - (1.0 - cfg.clip_min) * i / cfg.clip_search
        cand_scale = scale * f
        cand_zero = zero * f if zero is not None else None
        cand_codes, cand_recon = encode(cand_scale, cand_zero)
        err = (cand_recon - xg).pow(2).sum(dim=-1, keepdim=True)
        win = err < best_err
        best_err = torch.where(win, err, best_err)
        codes = torch.where(win, cand_codes, codes)
        scale = torch.where(win, cand_scale, scale)
        if zero is not None:
            zero = torch.where(win, cand_zero, zero)
    return codes, scale, zero


def build_levels(cfg: QuantConfig, weight: Tensor | None = None) -> Tensor:
    """Materialize the grid for a config (data-dependent grids get the weights)."""
    kwargs = dict(cfg.grid_kwargs)
    if cfg.n_levels is not None:
        name = {"uniform": "uniform_k", "power": "power_k", "nf": "nf_k"}.get(cfg.grid)
        if name is None:
            raise ValueError(f"grid {cfg.grid!r} has no fractional-level variant")
        return grids.REGISTRY[name](cfg.n_levels, **kwargs)  # type: ignore[operator]
    if cfg.grid == "lloyd":
        if weight is None:
            raise ValueError("grid 'lloyd' needs the weights it is fitted to")
        # Fit on group-normalized weights, not the raw tensor: quantization
        # sees each group after division by its own absmax, and that
        # distribution is markedly wider-tailed than the global one. Fitting on
        # the wrong distribution is how a data-driven codebook ends up losing to
        # a hand-drawn curve.
        kwargs["weights"] = _group_normalize(weight, cfg.group_size)
    return grids.build(cfg.grid, cfg.bits, **kwargs)


def _group_normalize(w: Tensor, group_size: int) -> Tensor:
    xg, _ = _grouped(w.detach().float(), group_size)
    return xg / xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)


# Working set of _fit_groups, measured, as a multiple of the fp16 weight: the
# float32 copy, bucketize's int64 output and the clip search's candidates all
# stack up. 25.3x means a 1.89 GiB lm_head peaks at 47.9 GiB -- twice this card,
# which HIP silently satisfies out of shared host memory instead of failing.
_FIT_OVERHEAD = 26
_CHUNK_BUDGET_BYTES = 64 << 20  # cap VRAM working set transient to 64 MiB per row block


def _rows_per_chunk(rows: int, in_features: int, budget: int = _CHUNK_BUDGET_BYTES) -> int:
    per_row = in_features * _FIT_OVERHEAD * 4  # fp32 element, overhead multiple
    return max(1, min(rows, budget // max(1, per_row)))


def quantize(w: Tensor, cfg: QuantConfig, *, levels: Tensor | None = None) -> QuantizedTensor:
    """Quantize a 2-D weight matrix (out_features, in_features).

    Rows are fitted in blocks sized to a memory budget. Groups never span rows,
    so blocking is exact -- the codes and scales are identical to fitting the
    whole tensor at once, but the transient stays near 1 GiB instead of scaling
    with the tensor.
    """
    if w.dim() != 2:
        raise ValueError(f"expected a 2-D weight, got {tuple(w.shape)}")
    work = w.detach().float()
    lv = (levels if levels is not None else build_levels(cfg, work)).to(work.device).float()
    if cfg.n_levels is not None and lv.numel() != cfg.n_levels:
        raise ValueError(f"grid produced {lv.numel()} levels, config asked for {cfg.n_levels}")
    if lv.numel() > cfg.levels_count:
        raise ValueError(f"grid produced {lv.numel()} levels, more than {cfg.bits} bits can index")

    xg, _ = _grouped(work, cfg.group_size)
    rows = xg.shape[0]
    block = _rows_per_chunk(rows, w.shape[1])

    if block >= rows:
        codes, scale, zero = _fit_groups(xg, lv, cfg)
    else:
        code_parts, scale_parts, zero_parts = [], [], []
        for start in range(0, rows, block):
            c, s, z = _fit_groups(xg[start : start + block], lv, cfg)
            code_parts.append(c.to(torch.int16))  # narrow before accumulating
            scale_parts.append(s.to(cfg.scale_dtype))
            if z is not None:
                zero_parts.append(z.to(cfg.scale_dtype))
            del c, s, z
        codes = torch.cat(code_parts)
        scale = torch.cat(scale_parts)
        zero = torch.cat(zero_parts) if zero_parts else None
        del code_parts, scale_parts, zero_parts
    return QuantizedTensor(
        codes=codes.to(torch.int16),
        scale=scale.to(cfg.scale_dtype),
        zero=zero.to(cfg.scale_dtype) if zero is not None else None,
        levels=lv.cpu(),
        shape=w.shape,
        config=cfg,
    )


def fake_quantize(w: Tensor, cfg: QuantConfig, *, levels: Tensor | None = None) -> Tensor:
    """Quantize and immediately reconstruct, in the input dtype. Phase 1's workhorse."""
    return quantize(w, cfg, levels=levels).dequantize(dtype=w.dtype).to(w.device)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def error_report(w: Tensor, w_hat: Tensor, *, importance: Tensor | None = None) -> dict[str, float]:
    """Reconstruction error of one tensor.

    ``importance`` is an optional per-input-channel weight (shape (in_features,)),
    e.g. mean squared activation magnitude from a calibration pass. Weighted
    error correlates far better with end-task damage than raw error does,
    because a big error on a channel the model never excites is free.
    """
    a, b = w.detach().float(), w_hat.detach().float()
    diff = a - b
    rep = {
        "rel_fro": (diff.norm() / a.norm().clamp_min(1e-12)).item(),
        "rmse": diff.pow(2).mean().sqrt().item(),
        "max_abs": diff.abs().max().item(),
        "cos": torch.nn.functional.cosine_similarity(
            a.flatten(), b.flatten(), dim=0, eps=1e-12
        ).item(),
    }
    if importance is not None:
        imp = importance.detach().float().to(a.device).clamp_min(0)
        rep["weighted_mse"] = (diff.pow(2) * imp).mean().item()
    return rep
