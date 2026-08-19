"""How much of the adapter survives being merged into bf16 base weights?

The folded-vs-wrapped eval showed the two paths give the same aggregate
adherence but different text on 35-70% of prompts. This measures the mechanism
behind that: folding stores the adapter INSIDE the base weight, so `W0 + dW` is
rounded to bf16's 8 mantissa bits. Wherever an element of dW is small relative
to the W0 element it lands on, it is partially or entirely absorbed -- the
folded model runs a quantised copy of the adapter.

    intended  dW = scaling * (U @ V)                      (fp32)
    realised  dW = (W0 + dW) rounded to bf16, minus W0    (what the model uses)

Reports, per expert:
  * relative error   ||realised - intended||_F / ||intended||_F
  * absorbed         fraction of delta elements that changed nothing at all
  * the same figures for an fp32 master weight, as the control -- if fp32 is
    clean then precision is the cause, and the fix is master weights rather
    than anything about the folding mechanism.

    uv run --env-file .env scripts/measure_fold_precision.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import FoldableExpert, set_hard_vram_cap  # noqa: E402

EXPERTS = {
    "astral": "results/adapters/m2_astral_r8a128",
    "postgresql": "results/adapters/m2_postgresql_r8a128",
    "financial_planning": "results/adapters/m2_financial_r8a128",
}


@torch.no_grad()
def measure(expert: FoldableExpert, params: dict, dtype) -> dict:
    tot_int_sq = tot_err_sq = 0.0
    absorbed = total = 0
    per_module = []

    for key, (u, v) in expert.factors.items():
        w0 = params[key]
        intended = expert.scaling * (u.float() @ v.float())

        # what the model actually ends up holding
        folded = (w0.float() + intended).to(dtype)
        realised = folded.float() - w0.float()

        err = (realised - intended).pow(2).sum().item()
        mag = intended.pow(2).sum().item()
        tot_err_sq += err
        tot_int_sq += mag

        # elements the merge threw away entirely
        dead = ((realised == 0) & (intended != 0)).sum().item()
        absorbed += dead
        total += intended.numel()

        per_module.append(
            {
                "module": key,
                "rel_err": (err / mag) ** 0.5 if mag else 0.0,
                "absorbed_pct": 100.0 * dead / intended.numel(),
                "delta_to_weight_ratio": (intended.abs().mean() / w0.float().abs().mean()).item(),
            }
        )

    per_module.sort(key=lambda r: -r["rel_err"])
    return {
        "rel_err": (tot_err_sq / tot_int_sq) ** 0.5 if tot_int_sq else 0.0,
        "absorbed_pct": 100.0 * absorbed / total,
        "worst_modules": per_module[:5],
        "best_modules": per_module[-3:],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/fold_precision.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    print(f"Loading {args.model_name} in bfloat16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    model.eval()
    params = dict(model.named_parameters())

    results = {}
    for name, rel in EXPERTS.items():
        e = FoldableExpert.from_dir(REPO_ROOT / rel, name=name)
        ref = params[next(iter(e.factors))]
        e.to(ref.device, torch.float32)  # keep factors exact; only the merge should round

        print(f"\n{'=' * 68}\n {name}  (scaling={e.scaling})\n{'=' * 68}")
        bf16 = measure(e, params, torch.bfloat16)
        fp32 = measure(e, params, torch.float32)

        print(
            f"  merged into bf16 : rel err {bf16['rel_err'] * 100:6.2f}%   "
            f"delta elements fully absorbed: {bf16['absorbed_pct']:5.2f}%"
        )
        print(
            f"  merged into fp32 : rel err {fp32['rel_err'] * 100:6.2f}%   "
            f"delta elements fully absorbed: {fp32['absorbed_pct']:5.2f}%   (control)"
        )
        print("  worst modules (bf16):")
        for m in bf16["worst_modules"]:
            print(
                f"    {m['module']:52s} rel err {m['rel_err'] * 100:5.2f}%  |dW|/|W| = {m['delta_to_weight_ratio']:.2e}"
            )
        results[name] = {"scaling": e.scaling, "bf16": bf16, "fp32": fp32}

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
