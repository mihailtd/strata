"""Arm 0: compress an existing adapter onto a frozen basis bank by LEAST-SQUARES
PROJECTION -- no training -- and save the result as a master_basis adapter.

This isolates the compression question from the optimisation question. The
least-squares coefficients are the provable ceiling for that (bank, k): no
amount of training can produce a better fit to the source adapter, so if
adherence collapses here it collapses for any trained variant too.

Why this is worth running rather than extrapolating: retained *energy* is not
retained *behaviour*, and this project has already measured an energy-style
proxy (training loss) being ANTI-correlated with adherence -- the 60.80%
winner had nearly the worst loss, and the best loss scored worst. So
"98.23% of the energy survived k=32" does not license "~59.7% adherence".
That number has to be measured.

    uv run scripts/old/project_adapter_to_basis.py \\
        --source results/adapters/astral_qwen3.5_micro_lora_a256 \\
        --basis  results/basis/astral_a256_k32.pt \\
        --out    results/adapters/astral_qwen3.5_micro_mbproj_k32
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from extract_svd_basis import load_adapter_factors  # noqa: E402


def inner_factored(P1: torch.Tensor, Q1: torch.Tensor, P2: torch.Tensor, Q2: torch.Tensor) -> float:
    """<P1@Q1, P2@Q2>_F without materialising either (in x out) matrix."""
    return torch.trace((P1.T @ P2) @ (Q2 @ Q1.T)).item()


def canonical_module_name(name: str) -> str:
    """Strip peft's wrapper prefix so keys match a NovelLoraLinear-wrapped model.

    peft  : base_model.model.model.layers.0.mlp.down_proj
    custom:                  model.layers.0.mlp.down_proj
    The projected adapter is loaded by load_novel_adapter into a model wrapped
    by apply_novel_lora (no peft wrapper), so the saved keys must use the
    latter form or every key misses on load.
    """
    return name.removeprefix("base_model.model.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="Adapter whose delta gets projected.")
    ap.add_argument("--basis", required=True, help="Basis bank .pt from extract_svd_basis.py")
    ap.add_argument("--out", required=True, help="Output adapter dir (master_basis format).")
    ap.add_argument("--alpha", type=int, default=None, help="Alpha to bake in; default = num_basis (scaling 1.0).")
    args = ap.parse_args()

    src = Path(args.source)
    if not src.is_absolute():
        src = REPO_ROOT / src
    blob = torch.load(
        args.basis if Path(args.basis).is_absolute() else REPO_ROOT / args.basis, map_location="cpu", weights_only=False
    )
    bank = blob["bank"]
    num_basis, rank_basis = blob["num_basis"], blob["rank_basis"]

    # scaling is applied in NovelLoraLinear.forward as alpha/num_basis. Use
    # alpha == num_basis so scaling == 1.0 and the coefficients we solve for are
    # exactly the ones the model will apply -- no hidden rescale.
    alpha = args.alpha if args.alpha is not None else num_basis
    scaling = alpha / num_basis

    fac = load_adapter_factors(src)
    print(f"source: {src.name}  ({len(fac)} modules)")
    print(f"basis : k={num_basis} rank_basis={rank_basis}  from {blob.get('sources')}")
    print(f"alpha={alpha} -> scaling={scaling}")

    state: dict[str, torch.Tensor] = {}
    rets = []
    for name, (P, Q, mtype) in sorted(fac.items()):
        key = f"{mtype}_{P.shape[0]}x{Q.shape[1]}"
        if key not in bank:
            raise KeyError(f"{key} missing from basis bank")
        bu = bank[key]["basis_u"].float()
        bv = bank[key]["basis_v"].float()
        k = bu.shape[0]

        # Solve min_c || M - sum_k c_k E_k ||_F  via normal equations H c = b
        H = torch.zeros(k, k, dtype=torch.float64)
        for i in range(k):
            for j in range(i, k):
                val = inner_factored(bu[i], bv[i], bu[j], bv[j])
                H[i, j] = val
                H[j, i] = val
        H += torch.eye(k, dtype=torch.float64) * 1e-10
        b = torch.tensor([inner_factored(P, Q, bu[i], bv[i]) for i in range(k)], dtype=torch.float64)
        c = torch.linalg.solve(H, b)

        m2 = inner_factored(P, Q, P, Q)
        rets.append(max(0.0, min(1.0, (c @ b).item() / m2 if m2 > 0 else 0.0)))

        # forward applies delta = scaling * sum_k c_k (h @ bu_k) @ bv_k, so divide
        # the solved coefficients by scaling to land on exactly the fitted delta.
        state[f"{canonical_module_name(name)}.coefficients"] = (c.float() / scaling).to(torch.bfloat16)

    r = torch.tensor(rets)
    print(
        f"\nleast-squares retained energy: mean {r.mean() * 100:.2f}%  "
        f"median {r.median() * 100:.2f}%  min {r.min() * 100:.2f}%"
    )

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    torch.save(state, out / "novel_adapter.pt")
    meta = {
        "variant": "master_basis_projected",
        "mode": "master_basis",
        "gated": False,
        "target_modules": sorted({n.split(".")[-1] for n in fac}),
        "rank": num_basis,
        "rank_in": num_basis,
        "rank_out": rank_basis,
        "alpha": alpha,
        "model_name": "Qwen/Qwen3.5-4B",
        "basis_bank": str(args.basis),
        "projected_from": str(src),
        "retained_energy_mean": r.mean().item(),
    }
    (out / "novel_adapter_config.json").write_text(json.dumps(meta, indent=2))

    n_coef = sum(v.numel() for v in state.values())
    nbytes = sum(v.numel() * v.element_size() for v in state.values())
    print(f"saved {out}")
    print(f"  coefficients: {n_coef:,} scalars = {nbytes / 1024:.2f} KB (bf16)")
    print("  NOTE: unusable without the basis bank -- see TODO.md for honest storage accounting.")


if __name__ == "__main__":
    main()
