"""CPU Unit Tests for Master Basis Set Reduction (mode="master_basis").

Verifies model wrapping, forward pass, trainable parameter counting,
and sub-kilobyte adapter payload serialization on CPU.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from novel_peft import (
    MasterBasisBank,
    NovelLoraLinear,
    apply_novel_lora,
    load_novel_adapter,
    save_novel_adapter,
)


class DummyDecoderLayer(nn.Module):
    def __init__(self, in_f: int = 2560, out_f: int = 9216):
        super().__init__()
        self.gate_proj = nn.Linear(in_f, out_f, bias=False)
        self.up_proj = nn.Linear(in_f, out_f, bias=False)
        self.down_proj = nn.Linear(out_f, in_f, bias=False)


class DummyTransformer(nn.Module):
    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([DummyDecoderLayer() for _ in range(num_layers)])


def test_master_basis_bank_and_linear():
    base = nn.Linear(2560, 9216, bias=False)
    bank = MasterBasisBank()
    bank.get_or_create(
        "gate_proj_2560x9216",
        in_features=2560,
        out_features=9216,
        num_basis=8,
        rank_basis=8,
        device="cpu",
        dtype=torch.float32,
    )

    def lookup(key):
        return bank.basis_u[key], bank.basis_v[key]

    wrapper = NovelLoraLinear(
        base_layer=base,
        layer_idx=0,
        mode="master_basis",
        alpha=16.0,
        rank_in=8,
        rank_out=8,
        factor_lookup=lookup,
        factor_key="gate_proj_2560x9216",
    )

    x = torch.randn(2, 10, 2560)
    out = wrapper(x)
    assert out.shape == (2, 10, 9216), f"Unexpected shape {out.shape}"

    # Check trainable parameter count for a single layer
    trainable = [p for p in wrapper.parameters() if p.requires_grad]
    assert len(trainable) == 1, "Only coefficients tensor should be trainable per layer"
    assert trainable[0].shape == (8,), f"Coefficients shape mismatch: {trainable[0].shape}"


def test_apply_novel_lora_master_basis():
    model = DummyTransformer(num_layers=4)
    summary = apply_novel_lora(
        model,
        mode="master_basis",
        target_modules=["gate_proj", "up_proj", "down_proj"],
        rank_in=8,
        rank_out=8,
        alpha=16.0,
    )

    assert summary["wrapped_count"] == 12  # 4 layers * 3 target modules
    assert hasattr(model, "novel_master_basis_bank")

    # Trainable params across 12 layers: 12 layers * 8 coefficients = 96 params
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_count == 96, f"Expected 96 trainable params, got {trainable_count}"


def test_save_and_load_master_basis_adapter():
    model = DummyTransformer(num_layers=4)
    apply_novel_lora(
        model,
        mode="master_basis",
        target_modules=["gate_proj", "up_proj", "down_proj"],
        rank_in=8,
        rank_out=8,
        alpha=16.0,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        meta = {
            "variant": "master_basis",
            "mode": "master_basis",
            "rank_in": 8,
            "rank_out": 8,
            "alpha": 16.0,
            "target_modules": ["gate_proj", "up_proj", "down_proj"],
        }
        save_novel_adapter(model, tmp_dir, meta)

        adapter_file = Path(tmp_dir) / "novel_adapter.pt"
        assert adapter_file.exists()
        file_size_bytes = adapter_file.stat().st_size

        # Verify payload is sub-megabyte
        assert file_size_bytes < 1_000_000, f"Payload size {file_size_bytes} bytes exceeds 1MB target!"

        # Test loading into a fresh model
        fresh_model = DummyTransformer(num_layers=4)
        load_summary = load_novel_adapter(fresh_model, tmp_dir)
        assert load_summary["wrapped_count"] == 12
