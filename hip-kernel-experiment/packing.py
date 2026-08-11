"""Phase 2: turn codes into bytes.

Two packers, because bit width comes in two flavours:

``pack_bits``  - dense bit packing for integer widths (1..16). Codes are laid
                 out LSB-first and are allowed to straddle uint32 boundaries,
                 so 3-bit codes really cost 3 bits, not 4.
``pack_base``  - mixed-radix packing for non-power-of-two alphabets. K levels
                 cost log2(K) bits, which is how you get 2.585 or 3.459 bits
                 per weight instead of rounding up to 3 or 4.

Both produce uint32 words, which is the unit a GPU kernel wants to load, and
both round-trip exactly -- the tests assert that, because a packer that is
subtly wrong looks exactly like a quantizer that is subtly bad.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .quantizers import QuantConfig, QuantizedTensor

WORD_BITS = 32


# ---------------------------------------------------------------------------
# Dense bit packing
# ---------------------------------------------------------------------------


def pack_bits(codes: Tensor | np.ndarray, nbits: int) -> np.ndarray:
    """Pack integer codes into a flat uint32 array, LSB-first, no padding waste."""
    if not 1 <= nbits <= 16:
        raise ValueError("pack_bits handles 1..16 bit codes")
    flat = _as_u16(codes)
    if flat.size and flat.max() >= (1 << nbits):
        raise ValueError(f"code {flat.max()} does not fit in {nbits} bits")
    bits = np.unpackbits(flat.view(np.uint8).reshape(-1, 2), axis=1, bitorder="little")
    bits = bits[:, :nbits].reshape(-1)
    bits = _pad_to(bits, WORD_BITS)
    return np.packbits(bits, bitorder="little").view("<u4").astype(np.uint32)


def unpack_bits(words: np.ndarray, nbits: int, count: int) -> np.ndarray:
    """Inverse of :func:`pack_bits`, returning ``count`` codes as uint16."""
    bits = np.unpackbits(words.astype("<u4").view(np.uint8), bitorder="little")
    bits = bits[: count * nbits].reshape(count, nbits)
    bits = np.concatenate([bits, np.zeros((count, 16 - nbits), np.uint8)], axis=1)
    return np.packbits(bits, axis=1, bitorder="little").view(np.uint16).reshape(-1)


def pack_bits_torch(codes: Tensor, nbits: int, chunk_words: int = 1 << 22) -> Tensor:
    """Pack integer codes into int32 words on-device, in bounded chunks.

    The numpy path goes through ``np.unpackbits``, which expands every code into
    16 one-byte flags. That is 16 bytes per code: fine for a 3.7M-parameter
    layer, but a 1.017B-parameter lm_head needs a 16.3 GiB intermediate, and on
    this box the allocation spilled into shared host memory and took the machine
    to 100% RAM rather than raising OOM.

    For widths that tile a word (1/2/4/8/16) no code straddles a boundary, so a
    word is just its codes shifted into place and OR-ed together. Working in
    chunks keeps the transient at a few hundred MiB regardless of tensor size.
    """
    if WORD_BITS % nbits:
        raise ValueError(f"pack_bits_torch handles widths dividing 32, not {nbits}")

    per = WORD_BITS // nbits
    flat = codes.reshape(-1)
    pad = (-flat.numel()) % per
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])

    n_words = flat.numel() // per
    out = torch.empty(n_words, dtype=torch.int32, device=flat.device)
    shifts = torch.arange(0, WORD_BITS, nbits, device=flat.device, dtype=torch.int32)
    mask = (1 << nbits) - 1

    for start in range(0, n_words, chunk_words):
        stop = min(start + chunk_words, n_words)
        block = flat[start * per : stop * per].reshape(-1, per).to(torch.int32)
        # Disjoint bit fields, so summing is the same as OR-ing them together.
        out[start:stop] = ((block & mask) << shifts).sum(dim=1)
        del block
    return out


def unpack_bits_torch(words: Tensor, nbits: int, count: int, start: int = 0) -> Tensor:
    """Device-side unpack of ``count`` codes beginning at code index ``start``.

    Reads at most two words per code and stitches them, which is exactly what a
    real kernel does -- this is the readable reference that a Triton or HIP
    implementation gets validated against. ``start`` lets a caller decode a
    slice of a tensor without touching the rest of it.
    """
    dev = words.device

    if start == 0 and WORD_BITS % nbits == 0:
        # Fast path for widths that tile a word exactly (1, 2, 4, 8, 16 -- which
        # covers every power-of-two scheme). No code straddles a boundary, so
        # every code is a shift-and-mask of the word it lives in, and the whole
        # tensor comes out of one broadcast.
        #
        # The generic path below is correct for these widths too, but it builds
        # six int64 temporaries the size of the weight -- 48 bytes of scratch per
        # 4-bit weight. That is 24x the fp16 tensor it is replacing, and it is
        # what made PackedLinear unusable on a large model rather than merely
        # slow.
        shifts = torch.arange(0, WORD_BITS, nbits, device=dev, dtype=torch.int32)
        codes = (words.to(torch.int32).unsqueeze(-1) >> shifts) & ((1 << nbits) - 1)
        return codes.reshape(-1)[:count].to(torch.int64)

    w = words.to(torch.int64) & 0xFFFFFFFF
    pos = (torch.arange(count, device=dev, dtype=torch.int64) + start) * nbits
    idx, off = pos >> 5, pos & 31
    lo = w[idx] >> off
    nxt = torch.clamp(idx + 1, max=w.numel() - 1)
    hi = torch.where(off + nbits > WORD_BITS, w[nxt] << (WORD_BITS - off), torch.zeros_like(lo))
    return (lo | hi) & ((1 << nbits) - 1)


# ---------------------------------------------------------------------------
# Mixed-radix packing (fractional bits per weight)
# ---------------------------------------------------------------------------


def symbols_per_word(n_levels: int) -> int:
    """How many base-K symbols fit in one uint32 without overflow."""
    if n_levels < 2:
        raise ValueError("need at least 2 levels")
    s = int(math.floor(WORD_BITS / math.log2(n_levels)))
    while n_levels**s > (1 << WORD_BITS):
        s -= 1
    return s


def pack_base(codes: Tensor | np.ndarray, n_levels: int) -> np.ndarray:
    """Pack base-K digits into uint32 words: word = sum d[i] * K**i."""
    flat = _as_u16(codes).astype(np.uint64)
    if flat.size and flat.max() >= n_levels:
        raise ValueError(f"code {flat.max()} outside alphabet of size {n_levels}")
    s = symbols_per_word(n_levels)
    padded = _pad_to(flat, s).reshape(-1, s)
    powers = (np.uint64(n_levels) ** np.arange(s, dtype=np.uint64)).reshape(1, s)
    return (padded * powers).sum(axis=1).astype(np.uint32)


def unpack_base(words: np.ndarray, n_levels: int, count: int) -> np.ndarray:
    s = symbols_per_word(n_levels)
    acc = words.astype(np.uint64).reshape(-1, 1).repeat(s, axis=1)
    powers = (np.uint64(n_levels) ** np.arange(s, dtype=np.uint64)).reshape(1, s)
    digits = (acc // powers) % np.uint64(n_levels)
    return digits.reshape(-1)[:count].astype(np.uint16)


def unpack_base_torch(words: Tensor, n_levels: int, count: int, start: int = 0) -> Tensor:
    """Device-side mixed-radix unpack of ``count`` digits from digit index ``start``."""
    s = symbols_per_word(n_levels)
    dev = words.device
    powers = torch.tensor(
        [n_levels**i for i in range(s)], device=dev, dtype=torch.int64
    )
    idx = torch.arange(count, device=dev, dtype=torch.int64) + start
    w = (words.to(torch.int64) & 0xFFFFFFFF)[idx // s]
    return (w // powers[idx % s]) % n_levels


# ---------------------------------------------------------------------------
# Bit accounting
# ---------------------------------------------------------------------------


def storage_report(qt: QuantizedTensor, *, mode: str = "auto") -> dict[str, float]:
    """Honest bits-per-weight, codes plus scales plus the level table."""
    n = qt.codes.numel()
    k = qt.n_levels
    mode = _resolve_mode(mode, k)
    code_bits = _code_bits(n, k, mode)
    meta_bits = torch.finfo(qt.config.scale_dtype).bits * (
        qt.scale.numel() + (qt.zero.numel() if qt.zero is not None else 0)
    )
    table_bits = 32 * qt.levels.numel()
    total = code_bits + meta_bits + table_bits
    return {
        "mode": mode,
        "n_levels": k,
        "ideal_bpw": math.log2(k),
        "code_bpw": code_bits / n,
        "meta_bpw": meta_bits / n,
        "total_bpw": total / n,
        "bytes": total / 8,
        "fp16_bytes": 2 * n,
        "compression_x": (16 * n) / total,
    }


def _resolve_mode(mode: str, n_levels: int) -> str:
    """Pick dense-bit or mixed-radix packing when the caller said "auto".

    Mixed radix only earns its keep when the alphabet sits far from a power of
    two. 15 levels in 4 dense bits wastes 2.3% -- not worth paying a division
    per code to decode. 6 levels in 3 dense bits wastes 14%, which is worth it.
    Pass mode= explicitly to override.
    """
    if mode != "auto":
        return mode
    waste = math.ceil(math.log2(n_levels)) - math.log2(n_levels)
    return "base" if waste > 0.15 else "bits"


def _code_bits(n: int, n_levels: int, mode: str) -> int:
    if mode == "bits":
        nbits = max(1, math.ceil(math.log2(n_levels)))
        return math.ceil(n * nbits / WORD_BITS) * WORD_BITS
    s = symbols_per_word(n_levels)
    return math.ceil(n / s) * WORD_BITS


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------


def pack_tensor(qt: QuantizedTensor, *, mode: str = "auto") -> dict[str, Tensor]:
    """Serialize one quantized tensor to safetensors-compatible tensors."""
    k = qt.n_levels
    mode = _resolve_mode(mode, k)
    nbits = max(1, math.ceil(math.log2(k)))
    if mode == "bits" and WORD_BITS % nbits == 0:
        # On-device, chunked: avoids numpy's 16-bytes-per-code intermediate,
        # which is 16 GiB for a 1B-parameter head.
        qweight = pack_bits_torch(qt.codes, nbits)
    else:
        codes = qt.codes.reshape(-1).cpu().numpy()
        words = pack_bits(codes, nbits) if mode == "bits" else pack_base(codes, k)
        qweight = torch.from_numpy(words.view(np.int32).copy())

    out = {
        # safetensors has no uint32; int32 is the same 4 bytes, reinterpreted
        "qweight": qweight,
        "scale": qt.scale.reshape(qt.scale.shape[0], -1).cpu(),
        "levels": qt.levels.cpu(),
    }
    if qt.zero is not None:
        out["zero"] = qt.zero.reshape(qt.zero.shape[0], -1).cpu()
    return out


def unpack_tensor(
    blob: dict[str, Tensor], cfg: QuantConfig, shape: tuple[int, int], *, mode: str = "auto"
) -> QuantizedTensor:
    k = blob["levels"].numel()
    mode = _resolve_mode(mode, k)
    rows, cols = shape
    gs = cols if cfg.group_size in (-1, None) else cfg.group_size
    count = rows * cols
    words = blob["qweight"].cpu().numpy().view(np.uint32)
    if mode == "bits":
        codes = unpack_bits(words, max(1, math.ceil(math.log2(k))), count)
    else:
        codes = unpack_base(words, k, count)
    zero = blob.get("zero")
    return QuantizedTensor(
        codes=torch.from_numpy(codes.astype(np.int16)).reshape(rows, cols // gs, gs),
        scale=blob["scale"].reshape(rows, cols // gs, 1),
        zero=zero.reshape(rows, cols // gs, 1) if zero is not None else None,
        levels=blob["levels"],
        shape=torch.Size(shape),
        config=cfg,
    )


def save_checkpoint(
    path: str | Path,
    tensors: dict[str, dict[str, Tensor]],
    meta: dict[str, object],
) -> Path:
    """Write packed tensors + a JSON sidecar describing how to read them back."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"{name}.{key}": t for name, blob in tensors.items() for key, t in blob.items()}
    save_file(flat, str(path), metadata={"ftq": json.dumps(meta, default=_jsonable)})
    return path


def load_checkpoint(path: str | Path) -> tuple[dict[str, dict[str, Tensor]], dict]:
    from safetensors import safe_open

    tensors: dict[str, dict[str, Tensor]] = {}
    with safe_open(str(path), framework="pt") as f:
        meta = json.loads(f.metadata()["ftq"])
        for key in f.keys():
            name, _, leaf = key.rpartition(".")
            tensors.setdefault(name, {})[leaf] = f.get_tensor(key)
    return tensors, meta


def config_to_json(cfg: QuantConfig) -> dict:
    d = asdict(cfg)
    d["scale_dtype"] = str(cfg.scale_dtype).removeprefix("torch.")
    return d


def config_from_json(d: dict) -> QuantConfig:
    d = dict(d)
    d["scale_dtype"] = getattr(torch, d.get("scale_dtype", "float16"))
    return QuantConfig(**d)


def _jsonable(o: object) -> str:
    return str(o)


def _as_u16(codes: Tensor | np.ndarray) -> np.ndarray:
    arr = codes.detach().cpu().numpy() if isinstance(codes, Tensor) else codes
    return np.ascontiguousarray(arr.reshape(-1).astype(np.uint16))


def _pad_to(arr: np.ndarray, multiple: int) -> np.ndarray:
    rem = (-arr.size) % multiple
    return np.concatenate([arr, np.zeros(rem, arr.dtype)]) if rem else arr
