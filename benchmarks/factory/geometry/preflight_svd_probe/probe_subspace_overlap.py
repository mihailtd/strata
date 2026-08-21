"""Pre-Flight SVD Subspace Overlap Probe (Fast Subsecond CPU Version).

Mathematically verifies subspace orthogonality and cross-task basis transferability
between two fine-tuned LLM adapters BEFORE initiating expensive multi-task training or
shared-basis compression schemes.

USAGE:
    uv run scripts/probe_subspace_overlap.py \
        --adapter-a results/adapters/astral_qwen3.5_micro_lora_a256 \
        --adapter-b results/adapters/financial_planning_standard_lora \
        --k 32
"""

import argparse
import math
import sys
from pathlib import Path

import torch

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT))

from scripts.extract_svd_basis import load_adapter_factors  # noqa: E402


def extract_group_basis_fast(
    factors: list[tuple[torch.Tensor, torch.Tensor]], k: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast subsecond extraction of top-k basis components using SVD on concatenated factors."""
    in_f = factors[0][0].shape[0]
    out_f = factors[0][1].shape[1]

    # Concatenate all layer factors across depth
    # P_cat: (in, L*r), Q_cat: (L*r, out)
    P_cat = torch.cat([f[0].double() for f in factors], dim=1)
    Q_cat = torch.cat([f[1].double() for f in factors], dim=0)

    # QR decomposition to reduce problem size to (L*r, L*r)
    Qa, Ra = torch.linalg.qr(P_cat)  # Qa: (in, L*r), Ra: (L*r, L*r)
    small = Ra @ Q_cat  # (L*r, out)

    U_s, S_s, Vh_s = torch.linalg.svd(small, full_matrices=False)

    num_components = min(k, S_s.shape[0])
    bu = torch.zeros(num_components, in_f, 8, dtype=torch.float64)
    bv = torch.zeros(num_components, 8, out_f, dtype=torch.float64)

    for i in range(num_components):
        sq = math.sqrt(S_s[i].item())
        u_col = (Qa @ U_s[:, i : i + 1]) * sq  # (in, 1)
        v_row = Vh_s[i : i + 1, :] * sq  # (1, out)

        fro = (u_col @ v_row).norm().item()
        if fro > 0:
            u_col = u_col / math.sqrt(fro)
            v_row = v_row / math.sqrt(fro)

        bu[i, :, 0] = u_col.squeeze()
        bv[i, 0, :] = v_row.squeeze()

    return bu, bv


def evaluate_subspace_overlap(adapter_a_dir: Path, adapter_b_dir: Path, k: int = 32) -> dict[str, dict[str, float]]:
    """Computes cross-task subspace projection retention and cosine alignment."""
    factors_a = load_adapter_factors(adapter_a_dir)
    factors_b = load_adapter_factors(adapter_b_dir)

    groups_a: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    groups_b: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    for _name, (P, Q, mtype) in factors_a.items():
        key = f"{mtype}_{P.shape[0]}x{Q.shape[1]}"
        if key not in groups_a:
            groups_a[key] = []
        groups_a[key].append((P, Q))

    for _name, (P, Q, mtype) in factors_b.items():
        key = f"{mtype}_{P.shape[0]}x{Q.shape[1]}"
        if key not in groups_b:
            groups_b[key] = []
        groups_b[key].append((P, Q))

    results: dict[str, dict[str, float]] = {}

    for key in sorted(groups_a.keys()):
        if key not in groups_b:
            continue

        fac_a = groups_a[key]
        fac_b = groups_b[key]

        bu_a, bv_a = extract_group_basis_fast(fac_a, k=k)
        bu_b, bv_b = extract_group_basis_fast(fac_b, k=k)

        total_energy_b = 0.0
        projected_energy_b = 0.0

        for Pb, Qb in fac_b:
            Pb_d = Pb.double()
            Qb_d = Qb.double()
            norm_b_sq = torch.trace((Pb_d.T @ Pb_d) @ (Qb_d @ Qb_d.T)).item()
            total_energy_b += norm_b_sq

            for i in range(min(k, bu_a.shape[0])):
                u_i = bu_a[i, :, 0:1]  # (in, 1)
                v_i = bv_a[i, 0:1, :]  # (1, out)
                dot = torch.trace((Pb_d.T @ u_i) @ (v_i @ Qb_d.T)).item()
                projected_energy_b += dot**2

        retention_pct = (projected_energy_b / max(1e-12, total_energy_b)) * 100.0

        u0_a = bu_a[0, :, 0]
        v0_a = bv_a[0, 0, :]
        u0_b = bu_b[0, :, 0]
        v0_b = bv_b[0, 0, :]

        cos_u = (torch.dot(u0_a, u0_b) / (torch.norm(u0_a) * torch.norm(u0_b) + 1e-12)).item()
        cos_v = (torch.dot(v0_a, v0_b) / (torch.norm(v0_a) * torch.norm(v0_b) + 1e-12)).item()
        lead_cos = abs(cos_u * cos_v)

        dim = fac_a[0][0].shape[0] * fac_a[0][1].shape[1]
        random_cos_expected = 1.0 / math.sqrt(dim)

        # Random-chance floor. Projecting onto a k-dimensional subspace of an
        # (in*out)-dimensional matrix space captures k/(in*out) of the energy in
        # expectation. For k=32 and 2560x9216 that is 0.00014% -- so a raw
        # "retention < 1%" test can never fail, and firing on it says nothing.
        # The informative quantity is retention RELATIVE to this floor.
        floor_pct = 100.0 * min(k, bu_a.shape[0]) / dim

        results[key] = {
            "retained_energy_pct": retention_pct,
            "random_floor_pct": floor_pct,
            "x_above_floor": retention_pct / max(1e-30, floor_pct),
            "leading_basis_cosine": lead_cos,
            "random_expected_cosine": random_cos_expected,
            "num_layers": len(fac_a),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-Flight SVD Subspace Overlap Probe")
    parser.add_argument(
        "--adapter-a",
        required=True,
        help="Path to Adapter A directory (e.g. Astral LoRA)",
    )
    parser.add_argument(
        "--adapter-b",
        required=True,
        help="Path to Adapter B directory (e.g. Financial LoRA)",
    )
    parser.add_argument("--k", type=int, default=32, help="Number of basis components (default: 32)")
    args = parser.parse_args()

    dir_a = Path(args.adapter_a)
    dir_b = Path(args.adapter_b)
    if not dir_a.is_absolute():
        dir_a = REPO_ROOT / dir_a
    if not dir_b.is_absolute():
        dir_b = REPO_ROOT / dir_b

    print(f"🔬 PRE-FLIGHT SVD SUBSPACE OVERLAP PROBE (k={args.k}, CPU-only)\n" + "─" * 75)
    print(f"  Adapter A (Reference) : {dir_a.name}")
    print(f"  Adapter B (Target)    : {dir_b.name}\n" + "─" * 75)

    results = evaluate_subspace_overlap(dir_a, dir_b, k=args.k)

    total_energy_pct = 0.0
    total_count = 0

    print(f"{'Shape Group':<26s} {'Layers':>7s} {'Retained':>11s} {'Chance floor':>13s} {'x floor':>9s}")
    print("─" * 80)

    total_floor_pct = 0.0
    for group, metrics in sorted(results.items()):
        ret_pct = metrics["retained_energy_pct"]
        floor = metrics["random_floor_pct"]
        layers = metrics["num_layers"]

        total_energy_pct += ret_pct
        total_floor_pct += floor
        total_count += 1

        print(f"{group:<26s} {layers:>7d} {ret_pct:10.5f}% {floor:12.5f}% {metrics['x_above_floor']:8.2f}x")

    avg_retention = total_energy_pct / max(1, total_count)
    avg_floor = total_floor_pct / max(1, total_count)
    x_floor = avg_retention / max(1e-30, avg_floor)
    print("─" * 80)
    print(f"📊 RETENTION {avg_retention:.5f}%   CHANCE FLOOR {avg_floor:.5f}%   -> {x_floor:.2f}x above chance")

    # Thresholds are on x-above-chance, NOT raw retention. Calibrated on this
    # repo\'s own adapters (k=32):
    #     same adapter vs itself           26569x   (ceiling the metric reaches)
    #     astral a256 vs astral a128        7.15x   (same task -> real shared structure)
    #     astral vs financial_planning      1.28x   (different tasks -> chance)
    # A raw "< 1%" test called ALL THREE orthogonal, including the same-task
    # pair, because chance is ~0.0005% at this k and dimensionality.
    if x_floor < 2.0:
        verdict = (
            f"🔴 SUBSPACES AT CHANCE LEVEL ({x_floor:.2f}x floor)\n"
            "   VERDICT: No shared structure beyond random overlap.\n"
            "   RECOMMENDATION: Do NOT share a basis (Tucker / LoKr / MasterBasis).\n"
            "   Train and fold these adapters independently."
        )
    elif x_floor < 100.0:
        verdict = (
            f"⚠️ WEAK BUT REAL SHARED STRUCTURE ({x_floor:.2f}x floor)\n"
            "   Measurably above chance, but this repo\'s master_basis experiments\n"
            "   failed even WITH a task-relevant basis -- above-chance overlap has\n"
            "   not been sufficient for compression to pay here. Treat as negative\n"
            "   evidence for basis sharing unless you re-test it directly."
        )
    else:
        verdict = (
            f"🟢 STRONG SUBSPACE OVERLAP ({x_floor:.2f}x floor)\n"
            "   Substantial shared structure. Basis sharing is worth testing."
        )

    print("\n" + "=" * 75)
    print(verdict)
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
