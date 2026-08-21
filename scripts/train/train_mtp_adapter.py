"""Fine-tune micro-adapters specifically attached to Qwen3.5's MTP draft head.

Aligns the MTP projection layers (`mtp.fc`, `mtp.layers.0.self_attn`, `mtp.layers.0.mlp`)
with domain expert token distributions (Astral, PostgreSQL, Financial) in single-token space (K=1).

USAGE:
    uv run scripts/train/train_mtp_adapter.py \
        --domain astral \
        --rank 64 \
        --alpha 64 \
        --steps 150 \
        --lr 2e-4 \
        --out results/adapters/mtp_astral_lora_r64_a64
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.micro_probe.dataset import load_astral_micro_dataset
from runtime.mtp_draft import Qwen35MTPDraftHead
from runtime.novel_peft import NovelLoraLinear

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
def attach_mtp_lora(
    head: Qwen35MTPDraftHead, rank: int = 64, alpha: float = 64.0
) -> dict[str, NovelLoraLinear]:
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
        if isinstance(module, nn.Linear) and any(
            name.endswith(t) or name == t for t in target_names
        ):
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


def save_mtp_adapter(
    wrapped: dict[str, NovelLoraLinear], out_dir: Path, rank: int, alpha: float
) -> None:
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
    print(f"✅ Saved MTP adapter to {out_dir} ({len(sd) // 2} modules)")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MTP head micro-adapters for domain alignment"
    )
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3.5-4B", help="Base model checkpoint"
    )
    parser.add_argument("--domain", default="astral", help="Target domain dataset")
    parser.add_argument(
        "--rank", type=int, default=64, help="Adapter rank (default: 64)"
    )
    parser.add_argument(
        "--alpha", type=float, default=64.0, help="Adapter alpha (default: 64.0)"
    )
    parser.add_argument(
        "--steps", type=int, default=150, help="Training steps (default: 150)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--out", required=True, help="Output directory for MTP adapter"
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA/ROCm GPU is not available! Run with 'uv run --env-file .env scripts/train/train_mtp_adapter.py' "
            "to load LD_PRELOAD=/opt/rocm-7.2.0/lib/libhsa-runtime64.so."
        )

    from runtime.novel_peft import set_hard_vram_cap

    set_hard_vram_cap(cap_gb=22.0)
    device = torch.device("cuda:0")
    print(f"🚀 Initializing MTP Head Adaptation on GPU ({torch.cuda.get_device_name(0)})...")

    # Load Base Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="cuda:0"
    )
    base_model.eval()
    base_model.requires_grad_(False)

    # Initialize MTP Head
    print("Loading native Qwen3.5 MTP Draft Head...")
    mtp_head = Qwen35MTPDraftHead(base_model, args.model_id)
    mtp_head.to(device=device, dtype=torch.bfloat16)

    # Attach LoRA to MTP projection layers
    wrapped = attach_mtp_lora(mtp_head, rank=args.rank, alpha=args.alpha)
    print(f"Attached NovelLoraLinear (r={args.rank}, α={args.alpha}) to {len(wrapped)} MTP layers.")

    # Collect trainable parameters
    trainable_params = [
        p for name, p in mtp_head.named_parameters() if "lora_" in name
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Load Dataset
    print(f"Loading {args.domain} SFT dataset...")
    ds = load_astral_micro_dataset(conversational=False)

    def collate_fn(batch):
        texts = [b["text"] for b in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        )
        return enc.input_ids.to(device)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    loader_iter = iter(loader)

    print(f"\n🏋️ Starting MTP Head Fine-Tuning ({args.steps} steps)...")
    mtp_head.train()
    start_t = time.perf_counter()

    for step in range(1, args.steps + 1):
        try:
            input_ids = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            input_ids = next(loader_iter)

        optimizer.zero_grad()

        # Step 1: Base Model Forward Pass (No grad)
        with torch.no_grad():
            outputs = base_model(
                input_ids=input_ids, output_hidden_states=True, use_cache=False
            )
            # Last hidden states from base model backbone: (B, N, H)
            hidden_states = outputs.hidden_states[-1]

        # Step 2: MTP Draft Head Forward Pass (Grad enabled)
        # Shift inputs for next-token prediction:
        # h_t corresponds to input_ids[:, :-1], next token embeddings e_{t+1} = embed(input_ids[:, 1:])
        if input_ids.shape[1] < 4:
            continue

        h_in = hidden_states[:, :-1, :]  # (B, N-1, H)
        next_ids = input_ids[:, 1:]  # (B, N-1)

        # MTP Draft Head outputs prediction logits for input_ids[:, 2:]
        mtp_logits, _ = mtp_head(h_in, next_ids)  # (B, N-1, V)

        # Target alignment: predict input_ids[:, 1:]
        shift_logits = mtp_logits[:, :-1, :].reshape(-1, mtp_logits.shape[-1])
        shift_labels = input_ids[:, 2:].reshape(-1)

        loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(shift_logits, shift_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        if step % 25 == 0 or step == 1:
            elapsed = time.perf_counter() - start_t
            step_rate = step / max(1e-5, elapsed)
            print(
                f"  Step {step:4d}/{args.steps} | Loss: {loss.item():.4f} | {step_rate:.2f} steps/s"
            )

    total_time = time.perf_counter() - start_t
    print(f"\n🎉 Fine-tuning complete in {total_time:.2f}s!")

    # Save MTP Adapter
    save_mtp_adapter(wrapped, out_dir, rank=args.rank, alpha=args.alpha)


if __name__ == "__main__":
    main()
