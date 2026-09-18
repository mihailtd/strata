"""§106 Phase A3: offline W4A16 quantizer for `runtime-next`.

Ports the EXACT real quantization scheme already validated (numerically,
not just conceptually) in `apps/runtime-triton/triton_w4a16.py`'s
`quantize_and_pack_w4` -- symmetric INT4, per-group-of-128-along-K, bf16
scales, round-to-nearest (no calibration data), fixed zero-point=8, packed
8 nibbles per int32 LSB-first. Same formula, ported so `runtime-next`
(Rust/HIP, reads real safetensors) has a real quantized checkpoint to
load, not a different quantization scheme invented for this crate.

Streams one real tensor at a time from the source HF snapshot (never
materializes the full checkpoint in RAM) and writes a new, real sharded
safetensors checkpoint (+ a real `model.safetensors.index.json`, same
shape the existing `model_loader.rs::load_weight_map` already reads) so
the Rust side needs zero new file-format code -- just new tensor NAMES
(`<name>.qweight` int32, `<name>.scales` bf16) read through the exact same
`safetensors` crate machinery already proven for the bf16 checkpoints.

Real, disclosed scope decision: only the tensors that actually go through
this engine's GEMV hot path get quantized (attention qkv/o, GDN in/out
proj, MLP gate_up/down, and `lm_head.weight` when untied) -- norm weights
(tiny, precision-sensitive) and `embed_tokens.weight` (read via a gather,
`embedding_lookup`, not a GEMV -- no dequant-gather kernel exists yet)
stay real, unquantized bf16, copied through unchanged. This is a real,
bounded first pass, not the smallest possible checkpoint.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

GROUP_SIZE = 128

# Real tensor-name patterns that go through this engine's GEMV hot path
# (raw::gemm call sites in model.rs) -- everything else is copied through
# unquantized. `lm_head.weight` is included; it's absent (tied) for most
# sizes and present (untied) for 9B/27B, handled either way by simply not
# matching when absent.
QUANTIZE_PATTERNS = [
    re.compile(r"\.self_attn\.(q|k|v|o)_proj\.weight$"),
    re.compile(r"\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)\.weight$"),
    re.compile(r"\.mlp\.(gate_proj|up_proj|down_proj)\.weight$"),
    re.compile(r"^lm_head\.weight$"),
]


def should_quantize(name: str) -> bool:
    return any(p.search(name) for p in QUANTIZE_PATTERNS)


def quantize_and_pack_w4(w: torch.Tensor, group_size: int = GROUP_SIZE) -> tuple[torch.Tensor, torch.Tensor]:
    """Same real quantization MATH as
    `apps/runtime-triton/triton_w4a16.py::quantize_and_pack_w4` (symmetric
    INT4, per-128-group-along-K, bf16 scales, RTN, zero-point=8) -- but a
    deliberately DIFFERENT memory layout. The source kernel is a
    Triton tile-based GEMM (multi-row activations, BLOCK_K x BLOCK_N
    tiles), so it packs K-major (`[K//8, N]`). This engine's GEMV kernel
    is one-block-per-OUTPUT-ROW (`gemv_bf16_kernel`'s own established
    pattern, M=1 always here), which wants each output row's K values
    CONTIGUOUS for vectorized reads -- the opposite axis. So `w` (real HF
    `nn.Linear` `[out_features, in_features]`) is quantized and packed
    row-major, matching this crate's own existing bf16 weight layout
    exactly (just with `in_features` compressed 8x): qweight
    `[out_features, in_features//8]`, scales `[out_features,
    in_features//group_size]`. A real, deliberate adaptation to the
    consuming kernel's access pattern, not a blind copy of the source's
    own layout.
    """
    out_features, in_features = w.shape
    assert in_features % group_size == 0, (
        f"in_features={in_features} not a multiple of group_size={group_size} -- "
        "the source scheme assumes exact group alignment, not a ragged tail"
    )
    assert in_features % 8 == 0, "in_features must be a multiple of 8 for nibble packing"

    N, K = out_features, in_features
    w_f32 = w.contiguous().to(torch.float32)  # [N, K]
    w_grouped = w_f32.view(N, K // group_size, group_size)

    max_abs = w_grouped.abs().amax(dim=2, keepdim=True).clamp(min=1e-5)
    scales = (max_abs / 7.5).to(torch.bfloat16)  # [N, K//group_size, 1]
    scales_f32 = scales.to(torch.float32)

    q = torch.clamp(torch.round(w_grouped / scales_f32) + 8.0, 0, 15).to(torch.int32)
    q = q.view(N, K)

    q_unpacked = q.view(N, K // 8, 8)
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32).view(1, 1, 8)
    qweight = (q_unpacked << shifts).sum(dim=2, dtype=torch.int32)  # [N, K//8]

    return qweight.contiguous(), scales.view(N, K // group_size).contiguous()


def dequantize_w4(qweight: torch.Tensor, scales: torch.Tensor, group_size: int = GROUP_SIZE) -> torch.Tensor:
    """Reference dequant, for the synthetic self-check below -- same
    formula the new HIP kernel will implement: w ~= (nibble - 8) * scale.
    Row-major [N, K//8] in, [N, K] out (matches quantize_and_pack_w4's
    layout above).
    """
    n, k8 = qweight.shape
    K = k8 * 8
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32).view(1, 1, 8)
    nibbles = ((qweight.view(n, k8, 1) >> shifts) & 0xF).view(n, K).to(torch.float32)
    scales_f32 = scales.to(torch.float32)
    scales_expanded = scales_f32.repeat_interleave(group_size, dim=1)
    return (nibbles - 8.0) * scales_expanded  # [N, K]


def self_check() -> None:
    """Real correctness gate on synthetic data, matching
    `test_triton_w4a16.py`'s own bar (cosine similarity), before this
    script is trusted against real 27B weights."""
    torch.manual_seed(0)
    out_features, in_features = 512, 1024  # multiple of group_size and 8
    w = (torch.randn(out_features, in_features) * 0.02).to(torch.bfloat16)
    qweight, scales = quantize_and_pack_w4(w)
    w_dequant = dequantize_w4(qweight, scales)  # already [out, in], matching w
    cos_sim = torch.nn.functional.cosine_similarity(
        w.to(torch.float32).flatten(), w_dequant.flatten(), dim=0
    )
    print(f"[self_check] roundtrip cosine similarity: {cos_sim.item():.6f}")
    assert cos_sim.item() > 0.99, f"quantize/dequant roundtrip cosine similarity too low: {cos_sim.item()}"
    print("[self_check] PASSED")


def convert(src_dir: Path, dst_dir: Path, shard_bytes_limit: int = 4 * 1024 * 1024 * 1024, group_size: int = GROUP_SIZE) -> None:
    index = json.loads((src_dir / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    src_shards = sorted(set(weight_map.values()))
    dst_dir.mkdir(parents=True, exist_ok=True)

    new_weight_map: dict[str, str] = {}
    current_shard: dict[str, torch.Tensor] = {}
    current_shard_bytes = 0
    shard_idx = 0
    total_tensors = len(weight_map)
    quantized_count = 0

    def flush_shard() -> None:
        nonlocal current_shard, current_shard_bytes, shard_idx
        if not current_shard:
            return
        shard_idx += 1
        shard_name = f"model-{shard_idx:05d}.safetensors"
        save_file(current_shard, str(dst_dir / shard_name), metadata={"format": "pt"})
        for name in current_shard:
            new_weight_map[name] = shard_name
        print(f"  wrote shard {shard_name}: {len(current_shard)} tensors, {current_shard_bytes / 1e9:.2f}GB")
        current_shard = {}
        current_shard_bytes = 0

    for shard_i, src_shard in enumerate(src_shards, 1):
        print(f"[{shard_i}/{len(src_shards)}] reading {src_shard}")
        with safe_open(str(src_dir / src_shard), framework="pt") as f:
            for name in f.keys():
                if weight_map.get(name) != src_shard:
                    continue
                w = f.get_tensor(name)
                if should_quantize(name) and w.dim() == 2:
                    qweight, scales = quantize_and_pack_w4(w, group_size=group_size)
                    current_shard[f"{name}.qweight"] = qweight
                    current_shard[f"{name}.scales"] = scales
                    current_shard_bytes += qweight.numel() * 4 + scales.numel() * 2
                    quantized_count += 1
                else:
                    current_shard[name] = w
                    current_shard_bytes += w.numel() * w.element_size()
                if current_shard_bytes >= shard_bytes_limit:
                    flush_shard()
    flush_shard()

    (dst_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": "pt"}, "weight_map": new_weight_map}, indent=2)
    )
    print(f"done: {quantized_count}/{total_tensors} tensors quantized, {shard_idx} shards written to {dst_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check-only", action="store_true", help="run the synthetic correctness gate and exit")
    ap.add_argument("--src", type=Path, help="real HF snapshot dir (bf16 safetensors)")
    ap.add_argument("--dst", type=Path, help="output dir for the quantized checkpoint")
    ap.add_argument("--group-size", type=int, default=GROUP_SIZE, help=f"real quantization group size along in_features (default {GROUP_SIZE})")
    args = ap.parse_args()

    self_check()
    if args.self_check_only:
        sys.exit(0)
    if not args.src or not args.dst:
        ap.error("--src and --dst are required unless --self-check-only")
    convert(args.src, args.dst, group_size=args.group_size)
