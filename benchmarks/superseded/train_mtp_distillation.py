"""Train Domain-Distilled Speculative Draft Head via Logit KL-Divergence.

HYPOTHESIS & SCIENTIFIC METHOD:
------------------------------
A speculative draft head's mathematical role is to predict the exact logits of
the domain-adapted target backbone, NOT raw dataset tokens.

1. Fold the target domain expert (m2_*_r8a128) into the backbone (bfloat16).
2. Pass domain tokens through the folded backbone to generate teacher hidden
   states h_t and teacher logits z_backbone.
3. Pass (h_t, embed(x_{t+1})) through Qwen35MTPDraftHead to produce draft logits z_draft.
4. Optimize via temperature-scaled KL Divergence loss:
     L = alpha_kd * T^2 * KL( Softmax(z_teacher / T) || Softmax(z_draft / T) ) + (1 - alpha_kd) * L_ce

USAGE:
    uv run --env-file .env python scripts/train_mtp_distillation.py --domain astral --steps 150 --lr 2e-4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Touch CUDA before importing fla/transformers
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.mtp_draft import Qwen35MTPDraftHead  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

DOMAINS = {
    "astral": (
        "data/astral/training_data.jsonl",
        "results/adapters/m2_astral_r8a128",
        "results/adapters/mtp_distilled_astral",
    ),
    "postgresql": (
        "data/postgresql/training_data.jsonl",
        "results/adapters/m2_postgresql_r8a128",
        "results/adapters/mtp_distilled_postgresql",
    ),
    "financial": (
        "data/financial_planning/training_data.jsonl",
        "results/adapters/m2_financial_r8a128",
        "results/adapters/mtp_distilled_financial",
    ),
}


class DomainTextDataset(Dataset):
    def __init__(self, file_path: Path, tokenizer, max_length: int = 512):
        self.samples = []
        with open(file_path) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                text = data.get("text") or (
                    f"### Question:\n{data.get('prompt', '')}\n\n### Answer:\n{data.get('response', '')}"
                )
                enc = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt",
                )
                self.samples.append(enc.input_ids.squeeze(0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def save_distilled_head(mtp_head: Qwen35MTPDraftHead, out_dir: Path, metadata: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save the 15 MTP tensors
    state_dict = {}
    for name, param in mtp_head.named_parameters():
        if param.requires_grad:
            state_dict[name] = param.detach().cpu()

    weights_path = out_dir / "mtp_distilled_weights.pt"
    torch.save(state_dict, weights_path)

    config_path = out_dir / "mtp_distilled_config.json"
    config_path.write_text(json.dumps(metadata, indent=2))
    print(f"\n✅ Saved distilled MTP head to {out_dir}")
    print(f"   Weights: {weights_path} ({os.path.getsize(weights_path) / 1e6:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", choices=sorted(DOMAINS), required=True, help="Target domain")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B", help="Base model backbone")
    parser.add_argument("--steps", type=int, default=150, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for draft head")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--alpha-kd", type=float, default=0.8, help="Weight for KL distillation loss (vs CE)")
    parser.add_argument("--vram-cap-gb", type=float, default=22.0, help="Hard VRAM limit")
    args = parser.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    data_rel, adapter_rel, out_rel = DOMAINS[args.domain]
    data_path = REPO_ROOT / data_rel
    adapter_path = REPO_ROOT / adapter_rel
    out_dir = REPO_ROOT / out_rel

    print("=" * 80)
    print(f" 🎓 Distilling Speculative Draft Head to Domain Expert [{args.domain.upper()}]")
    print("=" * 80)
    print(f"  Base Model       : {args.model_id}")
    print(f"  Folded Expert    : {adapter_path}")
    print(f"  Training Data    : {data_path}")
    print(f"  Output Directory : {out_dir}")
    print(f"  Hyperparameters  : steps={args.steps}, lr={args.lr}, T={args.temperature}, alpha_kd={args.alpha_kd}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model in bfloat16
    print("\nLoading base model backbone...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    base_model.eval()

    # 3. Fold the Domain Expert into the Backbone
    if not adapter_path.exists():
        raise FileNotFoundError(f"Required domain adapter not found at {adapter_path}")

    print(f"Folding domain expert from {adapter_path} into backbone...")
    expert = FoldableExpert.from_dir(adapter_path, name=args.domain)
    engine = WeightFoldingEngine(base_model, [expert], keep_pristine=True)
    engine.activate(expert)
    print("  Domain expert successfully folded into live backbone!")

    # 4. Load Shipped MTP Draft Head
    print("\nInitializing MTP draft head...")
    mtp_head = Qwen35MTPDraftHead(base_model, args.model_id)
    mtp_head.to(device=base_model.device, dtype=torch.bfloat16)
    mtp_head.train()

    # Freeze base model and shared embeddings/lm_head
    for p in base_model.parameters():
        p.requires_grad = False
    for p in mtp_head.parameters():
        p.requires_grad = False

    # Explicitly unfreeze only the 15 MTP draft head tensors (fc, decoder layer, norms)
    mtp_head.pre_fc_norm_hidden.weight.requires_grad = True
    mtp_head.pre_fc_norm_embedding.weight.requires_grad = True
    mtp_head.fc.weight.requires_grad = True
    for p in mtp_head.layer.parameters():
        p.requires_grad = True
    mtp_head.norm.weight.requires_grad = True

    trainable_params = [p for p in mtp_head.parameters() if p.requires_grad]
    param_count = sum(p.numel() for p in trainable_params)
    print(f"  MTP Trainable Parameters: {param_count / 1e6:.2f} M across {len(trainable_params)} tensors")

    # 5. Dataset & Optimizer Setup
    dataset = DomainTextDataset(data_path, tokenizer, max_length=256)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loader_iter = iter(dataloader)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    # 6. Distillation Training Loop
    print(f"\n🏋️ Starting Distillation Training ({args.steps} steps)...")
    start_t = time.perf_counter()
    running_loss = 0.0
    running_kd = 0.0

    for step in range(1, args.steps + 1):
        try:
            input_ids = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            input_ids = next(loader_iter)

        input_ids = input_ids.to(base_model.device)
        optimizer.zero_grad()

        # A. Teacher Forward Pass (Folded Backbone) - No Grad
        with torch.no_grad():
            teacher_outputs = base_model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            teacher_logits = teacher_outputs.logits  # (B, N, V)
            teacher_hidden = teacher_outputs.hidden_states[-1]  # (B, N, H)

        if input_ids.shape[1] < 4:
            continue

        # B. Student Draft Head Forward Pass (Grad Enabled)
        h_in = teacher_hidden[:, :-1, :]  # (B, N-1, H)
        next_ids = input_ids[:, 1:]  # (B, N-1)

        # Draft head outputs prediction for input_ids[:, 2:]
        draft_logits, _ = mtp_head(h_in, next_ids)  # (B, N-1, V)

        # C. Align Logits
        t_logits = teacher_logits[:, 1:-1, :].contiguous()  # (B, N-2, V)
        s_logits = draft_logits[:, :-1, :].contiguous()  # (B, N-2, V)
        labels = input_ids[:, 2:].contiguous()  # (B, N-2)

        # D. Chunked KL Divergence Loss to keep VRAM constant
        T = args.temperature
        num_tokens = t_logits.shape[1]
        chunk_size = 64
        loss_kd = torch.tensor(0.0, device=base_model.device)

        for c in range(0, num_tokens, chunk_size):
            c_end = min(c + chunk_size, num_tokens)
            t_chunk = t_logits[:, c:c_end, :]
            s_chunk = s_logits[:, c:c_end, :]

            t_probs = F.softmax(t_chunk / T, dim=-1)
            s_log_probs = F.log_softmax(s_chunk / T, dim=-1)
            loss_kd = loss_kd + F.kl_div(s_log_probs, t_probs, reduction="sum") * (T * T)

        loss_kd = loss_kd / max(1, t_logits.shape[0] * num_tokens)

        # Hard Cross Entropy on Ground Truth
        loss_ce = F.cross_entropy(
            s_logits.view(-1, s_logits.shape[-1]), labels.view(-1), ignore_index=tokenizer.pad_token_id
        )

        # Combined Loss
        loss = args.alpha_kd * loss_kd + (1.0 - args.alpha_kd) * loss_ce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        running_kd += loss_kd.item()

        if step % 25 == 0 or step == 1:
            elapsed = time.perf_counter() - start_t
            step_rate = step / max(1e-5, elapsed)
            avg_loss = running_loss / (25 if step > 1 else 1)
            avg_kd = running_kd / (25 if step > 1 else 1)
            running_loss = 0.0
            running_kd = 0.0
            print(
                f"  Step {step:4d}/{args.steps} | Total Loss: {avg_loss:.4f} | "
                f"KD Loss: {avg_kd:.4f} | {step_rate:.2f} steps/s"
            )

    # 7. Save Distilled Head
    metadata = {
        "domain": args.domain,
        "base_model": args.model_id,
        "folded_adapter": str(adapter_path),
        "steps": args.steps,
        "learning_rate": args.lr,
        "temperature": args.temperature,
        "alpha_kd": args.alpha_kd,
        "trainable_parameters": param_count,
    }
    save_distilled_head(mtp_head, out_dir, metadata)


if __name__ == "__main__":
    main()
