"""Train MTP Draft Head Micro-Adapters Across the Entire Canonical v7 Fleet.

Fine-tunes 1-layer EAGLE MTP micro-adapters (r=64, α=64) per domain while the
corresponding domain expert is folded into the 4B backbone.

This closes the "Draft Head Distribution Gap" (§61), lifting speculative
acceptance from 30-45% to 65%+ across all 6 specialized domains.

Usage:
    uv run --env-file .env scripts/train/train_mtp_adapters_fleet.py [--steps 150]
"""

from __future__ import annotations

import os
import sys

# Ensure ROCm HSA runtime is preloaded for AMD Radeon RX 7900 XTX
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
rocm_hsa_lib = "/opt/rocm-7.2.0/lib/libhsa-runtime64.so"
if __name__ == "__main__" and os.path.exists(rocm_hsa_lib) and rocm_hsa_lib not in os.environ.get("LD_PRELOAD", ""):
    current_preload = os.environ.get("LD_PRELOAD", "")
    os.environ["LD_PRELOAD"] = f"{rocm_hsa_lib}:{current_preload}".strip(":")
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.canon import CANON, REPO_ROOT, adapter_path
from runtime.gpu_preflight import ensure_gpu_exclusive
from runtime.mtp_draft import Qwen35MTPDraftHead, mtp_adapter_path
from runtime.novel_peft import (
    FoldableExpert,
    NovelLoraLinear,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

DOMAINS_DATA = {
    "astral": "data/astral/training_data_v6.jsonl",
    "postgresql": "data/postgresql/training_data_v6.jsonl",
    "duckdb": "data/duckdb/training_data_v6.jsonl",
    "financial": "data/financial_planning/training_data_v6.jsonl",
    "python_modern": "data/python_modern/training_data_v6.jsonl",
    "python_web": "data/python_web/training_data_v6.jsonl",
}


class DomainJsonlDataset(Dataset):
    def __init__(self, jsonl_path: Path):
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    text = item.get("text")
                    if not text and "messages" in item:
                        msgs = item["messages"]
                        text = "".join(f"### {m['role'].capitalize()}:\n{m['content']}\n\n" for m in msgs)
                    if text:
                        self.rows.append(text)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {"text": self.rows[idx]}


def attach_mtp_lora(head: Qwen35MTPDraftHead, rank: int = 64, alpha: float = 64.0) -> dict[str, NovelLoraLinear]:
    """Attaches NovelLoraLinear adapters to mtp.fc and mtp.layer projection layers."""
    wrapped: dict[str, NovelLoraLinear] = {}
    target_names = [
        "fc",
        "layer.self_attn.q_proj",
        "layer.self_attn.k_proj",
        "layer.self_attn.v_proj",
        "layer.self_attn.o_proj",
        "layer.mlp.gate_proj",
        "layer.mlp.up_proj",
        "layer.mlp.down_proj",
    ]

    for name, module in list(head.named_modules()):
        if isinstance(module, nn.Linear) and any(name.endswith(t) or name == t for t in target_names):
            parent_name, _, child_name = name.rpartition(".")
            parent = head if not parent_name else head.get_submodule(parent_name)
            lora_layer = NovelLoraLinear(
                base_layer=module,
                layer_idx=0,
                mode="standard",
                alpha=alpha,
                rank=rank,
                correct_fan_in_init=True,
            )
            lora_layer.to(device=module.weight.device, dtype=module.weight.dtype)
            setattr(parent, child_name, lora_layer)
            wrapped[f"mtp.{name}"] = lora_layer

    return wrapped


def detach_mtp_lora(head: Qwen35MTPDraftHead, wrapped: dict[str, NovelLoraLinear]) -> None:
    """Restores the original base nn.Linear layers in the MTP draft head."""
    for key, lora_layer in wrapped.items():
        name = key.replace("mtp.", "")
        parent_name, _, child_name = name.rpartition(".")
        parent = head if not parent_name else head.get_submodule(parent_name)
        setattr(parent, child_name, lora_layer.base_layer)


def save_mtp_adapter(wrapped: dict[str, NovelLoraLinear], out_dir: Path, rank: int, alpha: float) -> None:
    """Saves MTP micro-adapter factors (U, V) and scaling config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {}
    for key, lora in wrapped.items():
        sd[f"{key}.lora_a"] = lora.lora_a.detach().cpu()
        sd[f"{key}.lora_b"] = lora.lora_b.detach().cpu()

    torch.save(sd, out_dir / "novel_adapter.pt")
    cfg = {
        "adapter_type": "mtp_novel_lora",
        "rank": rank,
        "alpha": alpha,
        "scaling": alpha / rank,
        "modules": list(wrapped.keys()),
    }
    (out_dir / "novel_adapter_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"  ✅ Saved MTP adapter to {out_dir.name} ({len(sd) // 2} modules)")


def train_single_mtp_expert(
    domain: str,
    base_model: nn.Module,
    mtp_head: Qwen35MTPDraftHead,
    tokenizer: AutoTokenizer,
    folding_engine: WeightFoldingEngine,
    rank: int = 64,
    alpha: float = 64.0,
    steps: int = 150,
    lr: float = 3e-4,
    batch_size: int = 1,
) -> Path:
    print("=" * 80)
    print(f"🎯 TRAINING MTP ADAPTER FOR DOMAIN: {domain.upper()} (r={rank}, α={alpha}, steps={steps})")
    print("=" * 80)

    # 1. Activate domain expert in backbone
    expert = next(e for e in folding_engine.experts if e.name == domain)
    folding_engine.activate(expert)
    print(f"  Folded backbone expert: {expert.name}")

    # 2. Attach LoRA to MTP draft head
    wrapped = attach_mtp_lora(mtp_head, rank=rank, alpha=alpha)
    trainable_params = [p for name, p in mtp_head.named_parameters() if "lora_" in name]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    # 3. Load dataset
    jsonl_path = REPO_ROOT / DOMAINS_DATA[domain]
    ds = DomainJsonlDataset(jsonl_path)
    print(f"  Loaded {len(ds)} rows from {jsonl_path.name}")

    device = next(base_model.parameters()).device

    def collate_fn(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(texts, truncation=True, max_length=1024, padding=True, return_tensors="pt")
        return enc.input_ids.to(device)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    loader_iter = iter(loader)

    mtp_head.train()
    start_t = time.perf_counter()

    for step in range(1, steps + 1):
        try:
            input_ids = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            input_ids = next(loader_iter)

        if input_ids.shape[1] < 4:
            continue

        optimizer.zero_grad()

        # Step 1: Base Model Forward Pass (Adapted backbone, no grad)
        with torch.no_grad():
            outputs = base_model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states[-1]

        # Step 2: MTP Draft Head Forward Pass with Feature Representation Alignment
        h_in = hidden_states[:, :-1, :]          # (B, N-1, H)
        next_ids = input_ids[:, 1:]               # (B, N-1)
        target_hidden = hidden_states[:, 1:, :]  # (B, N-1, H) exact next-state target from backbone

        mtp_logits, draft_hidden, _ = mtp_head(h_in, next_ids, return_hidden=True)

        # 1. Geometry Alignment: Cosine Distance on 2560-dim residual stream
        mask = (next_ids != tokenizer.pad_token_id).float()  # (B, N-1)
        cos_sim = torch.nn.functional.cosine_similarity(draft_hidden.float(), target_hidden.float(), dim=-1)
        loss_cos = 1.0 - (cos_sim * mask).sum() / mask.sum().clamp(min=1.0)

        # 2. Representation Magnitude: Smooth L1 Loss
        loss_l1 = torch.nn.functional.smooth_l1_loss(
            draft_hidden.float() * mask.unsqueeze(-1),
            target_hidden.float() * mask.unsqueeze(-1),
            reduction="sum",
        ) / (mask.sum() * draft_hidden.shape[-1]).clamp(min=1.0)

        # 3. Soft Logit Regularization
        shift_logits = mtp_logits[:, :-1, :].reshape(-1, mtp_logits.shape[-1])
        shift_labels = input_ids[:, 2:].reshape(-1)
        loss_ce = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(shift_logits, shift_labels)

        # Total Feature Regression Loss
        loss = 2.0 * loss_cos + 1.0 * loss_l1 + 0.1 * loss_ce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1:
            elapsed = time.perf_counter() - start_t
            rate = step / max(1e-5, elapsed)
            print(f"  Step {step:4d}/{steps} | Total: {loss.item():.4f} (Cos: {loss_cos.item():.4f}, L1: {loss_l1.item():.4f}) | {rate:.2f} steps/s")

    # 4. Save MTP adapter
    out_dir = mtp_adapter_path(domain, version="v7")
    save_mtp_adapter(wrapped, out_dir, rank=rank, alpha=alpha)

    # 5. Clean up LoRA layers and restore pristine MTP head
    detach_mtp_lora(mtp_head, wrapped)
    mtp_head.eval()

    # 6. Unfold backbone expert
    folding_engine.restore_pristine()
    print(f"  Completed {domain} in {time.perf_counter() - start_t:.1f}s\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=CANON.BASE_MODEL)
    parser.add_argument("--domains", nargs="+", default=list(DOMAINS_DATA.keys()))
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    ensure_gpu_exclusive()
    set_hard_vram_cap(CANON.VRAM_CAP_GB)

    print("=" * 80)
    print("🚀 MTP DRAFT HEAD FLEET ADAPTATION (v7 Fleet)")
    print(f"   Domains: {', '.join(args.domains)}")
    print(f"   Config: rank={args.rank}, alpha={args.alpha}, steps={args.steps}, lr={args.lr}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Base Backbone: {args.model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    base_model.eval()
    base_model.requires_grad_(False)

    # Load all backbone domain experts into FoldingEngine
    experts = [FoldableExpert.from_dir(adapter_path(d), d) for d in args.domains]
    engine = WeightFoldingEngine(base_model, experts, keep_pristine=True)

    # Load MTP draft head
    print("Loading MTP Draft Head...")
    mtp_head = Qwen35MTPDraftHead(base_model, args.model_id)
    mtp_head.eval()

    total_start = time.perf_counter()
    trained_paths = {}

    for d in args.domains:
        path = train_single_mtp_expert(
            domain=d,
            base_model=base_model,
            mtp_head=mtp_head,
            tokenizer=tokenizer,
            folding_engine=engine,
            rank=args.rank,
            alpha=args.alpha,
            steps=args.steps,
            lr=args.lr,
        )
        trained_paths[d] = str(path)

    print("=" * 80)
    print(f"🎉 FLEET ADAPTATION COMPLETED in {time.perf_counter() - total_start:.1f}s")
    for d, p in trained_paths.items():
        print(f"   • {d:15s} -> {p}")
    print("=" * 80)


if __name__ == "__main__":
    main()
