"""Train an expert with an explicit subspace-orthogonality penalty against a frozen peer.

THE CLAIM UNDER TEST
--------------------
"Explicit Subspace Orthogonalization Penalty" proposes a training-time loss term
pushing each expert's adapter subspace away from the other experts', so experts
can coexist without interfering.

The static probe (`benchmarks/factory/geometry/orthogonality_headroom/`) already
bounds it, and the bound is two-sided:

    INPUT side (what the expert READS)   1.023-1.036x chance  -> 3.5% headroom
    OUTPUT side (what it WRITES)         5.996-7.164x chance  -> real structure

So on the input side there is almost nothing to remove, and on the output side
there is something — but plausibly something worth KEEPING, since all three
experts writing into the same directions is what you would expect if those are
the directions that matter for this model. Forcing them apart may push experts
into less useful output directions.

This settles which. `--penalty-side` selects input, output or both; the penalty is

    L = CE + lambda * sum_modules || Q_new^T Q_frozen ||_F^2

over orthonormalised bases, i.e. the summed squared cosines of principal angles,
which is exactly the quantity the static probe reports.

WHAT TO MEASURE AFTERWARDS
    1. Did overlap actually drop? (re-run the static probe against the new pair)
    2. Did anything IMPROVE? The penalty's only consumer is adapter stacking, so
       the outcome that matters is stacked accuracy, not solo accuracy.

    NOTE: stacking is retired independently (10.4 ms saved in a single-tenant
    engine that folds one expert at a time), so even a clean win here changes
    nothing operationally. This exists to replace an argued dismissal with a
    measured one.

    uv run --env-file .env python scripts/train_with_orthogonality_penalty.py \
        --domain postgresql --frozen-peer astral --lambda-orth 0.1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import FoldableExpert, set_hard_vram_cap  # noqa: E402

DOMAINS = {
    "astral": "data/astral/training_data.jsonl",
    "postgresql": "data/postgresql/training_data.jsonl",
    "financial_planning": "data/financial_planning/training_data.jsonl",
}
PEERS = {
    "astral": "results/adapters/m2_astral_r8a128",
    "postgresql": "results/adapters/m2_postgresql_r8a128",
    "financial_planning": "results/adapters/m2_financial_r8a128",
}


def orthonormal(m: torch.Tensor) -> torch.Tensor:
    """QR basis for the column space, in fp32 for numerical sanity."""
    q, _ = torch.linalg.qr(m.float())
    return q


class OrthPenaltyTrainer(SFTTrainer):
    """SFTTrainer + a subspace-overlap penalty against frozen peer factors.

    The penalty is recomputed from the LIVE lora weights every step, so it tracks
    the adapter as it moves rather than penalising a stale snapshot.
    """

    def __init__(self, *a, peer_bases=None, lam=0.0, side="input", **kw):
        super().__init__(*a, **kw)
        self.peer_bases = peer_bases or {}
        self.lam = lam
        self.side = side
        self.last_penalty = 0.0

    def _penalty(self, model) -> torch.Tensor:
        dev = next(model.parameters()).device
        total = torch.zeros((), device=dev, dtype=torch.float32)
        n = 0
        for name, mod in model.named_modules():
            if not hasattr(mod, "lora_A") or "default" not in getattr(mod, "lora_A", {}):
                continue
            key = self._match(name)
            if key is None:
                continue
            peer_in, peer_out = self.peer_bases[key]
            # peft: lora_A is (r, in) -> read side is its ROWS; lora_B is (out, r).
            #
            # NOTE ON THE ZERO GUARD. LoRA initialises lora_B to EXACTLY ZERO, and
            # QR of a zero matrix is degenerate -- the first version of this
            # penalty returned `nan` at step 0 for precisely that reason. The
            # output-side subspace does not exist at initialisation and the
            # penalty cannot act on it until B has grown away from zero. That is
            # a property of LoRA, not a numerical nuisance, and it is the same
            # B=0/A=random asymmetry that leaves the input side at chance while
            # the output side reaches 6-7x chance.
            if self.side in ("input", "both"):
                a = mod.lora_A["default"].weight.T  # (in, r)
                if a.norm() > 1e-6:
                    qa = orthonormal(a)
                    total = total + (qa.T @ peer_in.to(qa.device)).pow(2).sum()
                    n += 1
            if self.side in ("output", "both"):
                b = mod.lora_B["default"].weight  # (out, r)
                if b.norm() > 1e-6:
                    qb = orthonormal(b)
                    total = total + (qb.T @ peer_out.to(qb.device)).pow(2).sum()
                    n += 1
        return total / max(1, n)

    def _match(self, name: str) -> str | None:
        """peft module path -> the key used in the frozen expert's factor dict.

        peft yields `base_model.model.model.layers.0.mlp.down_proj` while
        FoldableExpert keys are `model.layers.0.mlp.down_proj.WEIGHT`. Missing the
        suffix made every lookup miss, n stayed 0, and the trainer ran to
        completion reporting `final penalty term: 0.000000` — a silently disabled
        experiment that would have "measured" the penalty doing nothing.
        `_assert_matched` now makes that failure loud instead.
        """
        clean = name.replace("base_model.model.", "").replace(".base_layer", "")
        for cand in (f"{clean}.weight", clean):
            if cand in self.peer_bases:
                return cand
        return None

    def _assert_matched(self, model) -> None:
        n = sum(
            1 for name, mod in model.named_modules()
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {})
            and self._match(name) is not None
        )
        if self.lam > 0 and n == 0:
            raise RuntimeError(
                "orthogonality penalty matched 0 modules — the peft module names do "
                "not line up with the frozen expert's factor keys, so the penalty "
                "would be silently inert. Refusing to run a disabled experiment."
            )
        print(f"  penalty active on {n} modules")

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = super().compute_loss(model, inputs, return_outputs=True, **kw)
        loss, outputs = out if isinstance(out, tuple) else (out, None)
        if self.lam > 0:
            pen = self._penalty(model)
            self.last_penalty = float(pen.detach())
            loss = loss + self.lam * pen
        return (loss, outputs) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=sorted(DOMAINS), default="postgresql")
    ap.add_argument("--frozen-peer", choices=sorted(PEERS), default="astral")
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--lambda-orth", type=float, default=0.1)
    ap.add_argument("--penalty-side", choices=["input", "output", "both"], default="both")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--loss-curve-out", default=None)
    args = ap.parse_args()

    assert args.domain != args.frozen_peer, "peer must be a different expert"
    set_hard_vram_cap(args.vram_cap_gb)

    print("=" * 96)
    print(f"  ORTHOGONALITY PENALTY  domain={args.domain}  peer={args.frozen_peer}")
    print(f"  lambda={args.lambda_orth}  side={args.penalty_side}")
    print("=" * 96)

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

    apply_liger_kernel_to_qwen3_5()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )

    peer = FoldableExpert.from_dir(REPO_ROOT / PEERS[args.frozen_peer], args.frozen_peer)
    peer_bases = {}
    for mod, (u, v) in peer.factors.items():
        peer_bases[mod] = (
            orthonormal(v.float().T).to("cuda"),  # input side  (in, r)
            orthonormal(u.float()).to("cuda"),    # output side (out, r)
        )
    print(f"  frozen peer bases: {len(peer_bases)} modules")

    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    from datasets import Dataset

    rows = [{"text": json.loads(x)["text"]}
            for x in (REPO_ROOT / DOMAINS[args.domain]).read_text().splitlines() if x.strip()]
    print(f"  {len(rows)} training records")

    out_dir = Path(args.out)
    trainer = OrthPenaltyTrainer(
        model=model,
        args=SFTConfig(
            output_dir=str(out_dir / "checkpoints"), dataset_text_field="text",
            max_length=2048, per_device_train_batch_size=2, gradient_accumulation_steps=2,
            learning_rate=args.lr, max_steps=args.max_steps, logging_steps=1,
            lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True,
            save_strategy="no", report_to="none",
        ),
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
        peer_bases=peer_bases, lam=args.lambda_orth, side=args.penalty_side,
    )
    trainer._assert_matched(model)
    trainer.train()
    print(f"  final penalty term: {trainer.last_penalty:.6f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "regime.json").write_text(json.dumps({
        "methodology": "m2", "precision": "bfloat16", "liger_fused_kernels": True,
        "trained_by": "scripts/train_with_orthogonality_penalty.py",
        "domain": args.domain, "rank": args.rank, "alpha": args.alpha,
        "orthogonality_penalty": {
            "lambda": args.lambda_orth, "side": args.penalty_side,
            "frozen_peer": args.frozen_peer,
            "final_penalty_value": trainer.last_penalty,
        },
        "max_steps": args.max_steps, "lr": args.lr,
        "dataset": DOMAINS[args.domain], "n_records": len(rows),
    }, indent=2))
    if args.loss_curve_out:
        p = REPO_ROOT / args.loss_curve_out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "domain": args.domain, "methodology": "m2",
            "lambda_orth": args.lambda_orth, "penalty_side": args.penalty_side,
            "max_steps": args.max_steps, "logging_steps": 1, "lr": args.lr,
            "n_records": len(rows), "log_history": trainer.state.log_history,
        }, indent=2))
    print(f"  saved -> {out_dir}")


if __name__ == "__main__":
    main()
