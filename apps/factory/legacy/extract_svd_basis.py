"""Build a MasterBasisBank from the principal directions of already-trained
adapters (CPU only), and optionally project a source adapter onto that basis.

WHY THIS EXISTS
---------------
`master_basis` trains only k scalar coefficients per wrapped Linear (1,024
total here) over a *frozen* basis bank. With a cold random bank that failed
outright (12.4-12.6% adherence vs 11.94% base, flat across a 16x alpha sweep --
the bottleneck is that 8 random directions in a 2560x9216 space span nothing
task-relevant). This script replaces the random bank with directions extracted
from adapters that actually work.

THE KEY STRUCTURAL CONSTRAINT
-----------------------------
`MasterBasisBank` is keyed by `f"{module_type}_{in}x{out}"`, i.e. ONE basis is
shared by all 32 layers of a given shape -- not one per layer. A rank-8 LoRA
delta is exactly rank 8 (verified: 8 non-zero singular values, then exactly
0.0), so 32 layers span up to 256 dimensions which must be compressed into k=8
shared directions. That compression is genuinely lossy, which is what makes
even an "in-task" basis a real test rather than a tautology.

MEMORY
------
Deltas are never materialised. Each layer's delta is kept factored as
M_l = P_l @ Q_l with P_l (in, r), Q_l (r, out). The Gram matrix over layers,
G[i,j] = <M_i, M_j>, is computed as trace((P_i^T P_j)(Q_j Q_i^T)) -- only
(r x r) products. Principal components are then formed as low-rank products
and truncated with a QR + small-SVD, so the largest intermediate is
(L*r, out) rather than 32 x (in x out) (~3 GB in fp32).

USAGE
    # basis from the astral winner
    uv run scripts/old/extract_svd_basis.py \\
        --adapters results/adapters/astral_qwen3.5_micro_lora_a256 \\
        --out results/basis/astral_a256_k8.pt

    # cross-task basis (the transfer test) + projected coefficients
    uv run scripts/old/extract_svd_basis.py \\
        --adapters results/adapters/postgres_qwen3.5_micro_custom_standard \\
        --out results/basis/postgres_k8.pt
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
def load_adapter_factors(adapter_dir: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor, str]]:
    """Return {module_path: (P, Q, module_type)} with delta M = P @ Q, M is (in, out).

    Handles both formats in this repo:
      peft   : lora_A (r, in), lora_B (out, r); dW = B@A is (out,in) so
               M = dW.T = A.T @ B.T  ->  P = A.T (in,r), Q = B.T (r,out)
      custom : lora_a (in, r), lora_b (r, out) already in (in,out) order
               (NovelLoraLinear computes (h @ lora_a) @ lora_b)
    Scaling (alpha/r) is applied so magnitudes match what the model actually used.
    """
    out: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}

    peft_cfg = adapter_dir / "adapter_config.json"
    novel_cfg = adapter_dir / "novel_adapter_config.json"

    if peft_cfg.exists():
        cfg = json.loads(peft_cfg.read_text())
        scaling = cfg["lora_alpha"] / cfg["r"]
        sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
        pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        for k, v in sd.items():
            if ".lora_A" in k:
                pairs[k.split(".lora_A")[0]]["A"] = v
            elif ".lora_B" in k:
                pairs[k.split(".lora_B")[0]]["B"] = v
        for name, d in pairs.items():
            if "A" not in d or "B" not in d:
                continue
            P = d["A"].float().T.contiguous()  # (in, r)
            Q = (d["B"].float().T * scaling).contiguous()  # (r, out)
            out[name] = (P, Q, name.split(".")[-1])
        return out

    if novel_cfg.exists():
        cfg = json.loads(novel_cfg.read_text())
        scaling = cfg["alpha"] / cfg["rank"]
        sd = torch.load(str(adapter_dir / "novel_adapter.pt"), map_location="cpu", weights_only=False)
        pairs = defaultdict(dict)
        for k, v in sd.items():
            if k.endswith(".lora_a"):
                pairs[k[: -len(".lora_a")]]["a"] = v
            elif k.endswith(".lora_b"):
                pairs[k[: -len(".lora_b")]]["b"] = v
        for name, d in pairs.items():
            if "a" not in d or "b" not in d:
                continue
            P = d["a"].float().contiguous()  # (in, r)
            Q = (d["b"].float() * scaling).contiguous()  # (r, out)
            out[name] = (P, Q, name.split(".")[-1])
        return out

    raise FileNotFoundError(f"No adapter_config.json or novel_adapter_config.json in {adapter_dir}")


def gram_matrix(factors: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """G[i,j] = <M_i, M_j>_F for M = P@Q, computed without materialising M."""
    n = len(factors)
    G = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        Pi, Qi = factors[i]
        for j in range(i, n):
            Pj, Qj = factors[j]
            # <PiQi, PjQj> = trace(Qi^T Pi^T Pj Qj) = trace((Pi^T Pj)(Qj Qi^T))
            val = torch.trace((Pi.T @ Pj).double() @ (Qj @ Qi.T).double())
            G[i, j] = val
            G[j, i] = val
    return G


def build_basis_for_group(
    factors: list[tuple[torch.Tensor, torch.Tensor]], num_basis: int, rank_basis: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-`num_basis` principal delta-patterns across layers, each truncated to
    `rank_basis`. Returns (basis_u (k, in, rb), basis_v (k, rb, out))."""
    G = gram_matrix(factors)
    evals, evecs = torch.linalg.eigh(G)  # ascending
    order = torch.argsort(evals, descending=True)[:num_basis]
    in_f = factors[0][0].shape[0]
    out_f = factors[0][1].shape[1]

    bu = torch.zeros(num_basis, in_f, rank_basis)
    bv = torch.zeros(num_basis, rank_basis, out_f)

    for slot, idx in enumerate(order):
        w = evecs[:, idx].float()  # (L,) mixing weights over layers
        # E = sum_l w_l * P_l @ Q_l  =  [w_1 P_1 | ... ] @ [Q_1; ...]
        P_cat = torch.cat([w[i] * factors[i][0] for i in range(len(factors))], dim=1)  # (in, L*r)
        Q_cat = torch.cat([factors[i][1] for i in range(len(factors))], dim=0)  # (L*r, out)
        # rank-truncate E = P_cat @ Q_cat without forming (in, out)
        Qa, Ra = torch.linalg.qr(P_cat)  # Qa (in, L*r), Ra (L*r, L*r)
        small = Ra @ Q_cat  # (L*r, out)
        U_s, S_s, Vh_s = torch.linalg.svd(small, full_matrices=False)
        r = min(rank_basis, S_s.shape[0])
        sq = torch.sqrt(S_s[:r])
        u = (Qa @ U_s[:, :r]) * sq  # (in, r)
        v = Vh_s[:r, :] * sq.unsqueeze(1)  # (r, out)
        # normalise each basis element to unit Frobenius norm so that the
        # alpha/scaling calibration is predictable and independent of how big
        # the source adapter happened to be.
        fro = (u @ v).norm()
        if fro > 0:
            u = u / torch.sqrt(fro)
            v = v / torch.sqrt(fro)
        bu[slot, :, :r] = u
        bv[slot, :r, :] = v
    return bu, bv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapters", nargs="+", required=True, help="Adapter dirs to extract principal directions from.")
    ap.add_argument("--out", required=True, help="Output .pt path for the basis bank.")
    ap.add_argument("--num-basis", type=int, default=8)
    ap.add_argument("--rank-basis", type=int, default=8)
    args = ap.parse_args()

    groups: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    provenance = []
    for a in args.adapters:
        d = Path(a)
        if not d.is_absolute():
            d = REPO_ROOT / d
        fac = load_adapter_factors(d)
        provenance.append(str(d))
        for _name, (P, Q, mtype) in fac.items():
            key = f"{mtype}_{P.shape[0]}x{Q.shape[1]}"
            groups[key].append((P, Q))
        print(f"loaded {len(fac):4d} modules from {d.name}")

    print(f"\n{len(groups)} shape-groups (bank keys):")
    bank = {}
    for key, factors in sorted(groups.items()):
        bu, bv = build_basis_for_group(factors, args.num_basis, args.rank_basis)
        bank[key] = {"basis_u": bu, "basis_v": bv}
        print(f"  {key:24s} layers={len(factors):3d}  basis_u{tuple(bu.shape)}  basis_v{tuple(bv.shape)}")

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bank": bank, "num_basis": args.num_basis, "rank_basis": args.rank_basis, "sources": provenance}, out)
    total = sum(v["basis_u"].numel() + v["basis_v"].numel() for v in bank.values())
    print(f"\nsaved {out}  ({total:,} basis params, {total * 4 / 1e6:.1f} MB fp32 -- FROZEN, shared across tasks)")


if __name__ == "__main__":
    main()
