"""Dynamic Alpha Calibration for LoRA Adapters (Chapter 6 §6.1.1.4).

Bounds deployment alpha from below by IEEE 754 bfloat16 mantissa merge precision
and from above by out-of-domain perturbation containment, deriving alpha_opt
in zero retraining time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from runtime.canon import REPO_ROOT

LAW_K = 0.167  # merge_rel_err_pct * (|dW|/|W|), fitted on the empirical alpha sweep
MERGE_ERR_FLOOR_PCT = 5.0  # Max acceptable merge error on bf16 mantissa
TARGET_DW_W_MIN = 0.040  # Minimum perturbation to maintain domain steering
TARGET_DW_W_MAX = 0.085  # Maximum perturbation before representation narrowing


def measure_adapter_perturbation(
    adapter_dir: Path | str,
    base_model: Any | None = None,
) -> dict[str, Any]:
    """Measures Frobenius norm ratio ||dW|| / ||W|| for a LoRA adapter."""
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"

    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing adapter artifacts in {adapter_dir}")

    cfg = json.loads(config_path.read_text())
    r = int(cfg.get("r", 8))
    alpha = float(cfg.get("lora_alpha", 128))

    sd = load_file(str(weights_path))

    # If live base model is provided in VRAM or RAM, use its exact parameters
    named_params: dict[str, torch.Tensor] = {}
    if base_model is not None:
        named_params = dict(base_model.named_parameters())

    num = 0.0
    den = 0.0
    pairs = 0

    # This whole block is a read-only norm measurement -- nothing here backprops, so
    # nothing needs gradient tracking. Without no_grad(), `W` (pulled straight from a
    # LIVE model's named_parameters() while it's resident for serving) carries
    # requires_grad=True, and float(W.float().norm() ** 2) triggered
    # "Converting a tensor with requires_grad=True to a scalar may lead to unexpected
    # behavior" on every calibration run against an in-memory model. `sd`'s tensors
    # (loaded straight from safetensors, never part of an autograd graph) were never
    # the actual source of the warning, but detaching them too costs nothing and
    # removes any doubt.
    with torch.no_grad():
        for k in sd:
            if "lora_A" not in k:
                continue
            kb = k.replace("lora_A", "lora_B")
            if kb not in sd:
                continue

            # dW = (alpha / r) * (B @ A)
            dW = (sd[kb].float().detach() @ sd[k].float().detach()) * (alpha / r)
            num += float(dW.norm() ** 2)

            if named_params:
                base_key = (
                    k.split("base_model.model.")[-1]
                    .replace(".lora_A.weight", ".weight")
                    .replace("lora_A.weight", "weight")
                )
                W = next(
                    (named_params[c] for c in (base_key, "model." + base_key) if c in named_params),
                    None,
                )
                if W is not None:
                    den += float(W.float().detach().norm() ** 2)
            pairs += 1

    # Fallback denominator references for Qwen3.5 adapted projection layers
    # Measured exact Frobenius norms across q, k, v, o, gate, up, down projections:
    # 0.8B: 205.556572, 2B: 323.400730, 4B: 501.783976, 9B: 882.049252, 27B: ~1419.0
    if den <= 0.0:
        base_name = str(cfg.get("base_model_name_or_path", "")).lower() + " " + adapter_dir.name.lower()
        if "0.8b" in base_name or "0_8b" in base_name:
            den = 205.556572**2
        elif "2b" in base_name:
            den = 323.400730**2
        elif "9b" in base_name:
            den = 882.049252**2
        elif "27b" in base_name:
            den = 1419.0**2
        else:
            den = 501.7839764855163**2

    dw_over_w = (num**0.5) / max(1e-30, den**0.5)

    return {
        "r": r,
        "alpha": alpha,
        "pairs": pairs,
        "dw_over_w": float(dw_over_w),
    }


def calibrate_adapter_alpha(
    adapter_dir: Path | str,
    alphas: list[int] | None = None,
    apply: bool = True,
    base_model: Any | None = None,
) -> dict[str, Any]:
    """Computes the full precision curve and optimal deployment alpha for an adapter."""
    if alphas is None:
        alphas = [16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160]

    adapter_dir = Path(adapter_dir)
    m = measure_adapter_perturbation(adapter_dir, base_model=base_model)
    ref_a = m["alpha"]
    ratio_ref = m["dw_over_w"]
    r = m["r"]

    curve = []
    for a in sorted(alphas):
        ratio = ratio_ref * (a / ref_a)
        err = LAW_K / max(1e-9, ratio)
        below_floor = err > MERGE_ERR_FLOOR_PCT
        in_target_band = TARGET_DW_W_MIN <= ratio <= TARGET_DW_W_MAX
        curve.append(
            {
                "alpha": int(a),
                "scaling": float(a / r),
                "dw_over_w": float(ratio),
                "merge_err_pct": float(err),
                "below_floor": bool(below_floor),
                "in_target_band": bool(in_target_band),
            }
        )

    admissible = [c for c in curve if not c["below_floor"]]

    # Pick optimal alpha:
    # Prefer point in target perturbation band [0.04, 0.085], closest to golden 0.071
    if admissible:
        best = min(
            admissible,
            key=lambda c: (
                0 if c["in_target_band"] else 1,
                abs(c["dw_over_w"] - 0.071),
            ),
        )
        alpha_opt = best["alpha"]
    else:
        alpha_opt = int(ref_a)
        best = next((c for c in curve if c["alpha"] == alpha_opt), curve[-1])

    alpha_min = next((c["alpha"] for c in curve if not c["below_floor"]), alphas[0])

    applied = False
    if apply and alpha_opt:
        cfg_path = adapter_dir / "adapter_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
            cfg["lora_alpha"] = int(alpha_opt)
            cfg_path.write_text(json.dumps(cfg, indent=2))
            applied = True

    result_data = {
        "adapter_name": adapter_dir.name,
        "adapter_dir": str(adapter_dir),
        "rank": r,
        "trained_alpha": int(ref_a),
        "trained_dw_over_w": float(ratio_ref),
        "alpha_min": int(alpha_min),
        "alpha_opt": int(alpha_opt),
        "applied": applied,
        "optimal_point": best,
        "curve": curve,
    }

    # Persist calibration report to results/calibrations/{adapter_name}.json
    try:
        calib_dir = REPO_ROOT / "results" / "calibrations"
        calib_dir.mkdir(parents=True, exist_ok=True)
        (calib_dir / f"{adapter_dir.name}.json").write_text(json.dumps(result_data, indent=2))
    except Exception:
        pass

    # Update SQLite database if factory.db exists
    try:
        import sqlite3

        db_path = REPO_ROOT / "results" / "factory.db"
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            conn.execute(
                """
                UPDATE training_runs
                SET alpha = ?, scaling = ?, final_dw_w = ?, predicted_merge_err = ?
                WHERE run_id LIKE ? OR domain LIKE ?
            """,
                (
                    int(alpha_opt),
                    float(alpha_opt / r),
                    float(best["dw_over_w"]),
                    float(best["merge_err_pct"]),
                    f"%{adapter_dir.name}%",
                    f"%{adapter_dir.name.split('_')[1] if len(adapter_dir.name.split('_')) > 1 else adapter_dir.name}%",
                ),
            )
            conn.commit()
            conn.close()
    except Exception:
        pass

    return result_data
