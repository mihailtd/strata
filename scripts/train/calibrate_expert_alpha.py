"""Per-adapter alpha calibration: bound alpha from BELOW by bf16 merge precision
and from ABOVE by held-out narrowing, then pick a point in the window.

WHY THIS IS NOT A GOLDEN CONSTANT
---------------------------------
alpha alone means nothing. What the hardware and the model both respond to is the
PERTURBATION MAGNITUDE:

    dW = (alpha / r) * B @ A          ->  |dW|/|W| is EXACTLY LINEAR in alpha

and |B@A| depends on the corpus, the step count and the rank -- it is only
knowable AFTER training. Two adapters trained with identical hyperparameters
measured |dW|/|W| = 0.0750 (postgresql v3) and 0.0741 (astral v3). A fixed
alpha=128 in the factory config applies one number to a quantity that varies per
adapter.

THE TWO BOUNDS
--------------
LOWER (analytic, free). Folding W + dW in bf16 truncates dW's low bits. The alpha
sweep fitted, across five points at rank 64:

    merge_rel_err_pct ~= 0.167 / (|dW|/|W|)        (matches all 5 to <5%)

Because |dW|/|W| is linear in alpha, ONE post-train measurement yields the whole
curve. No retraining, no GPU.

UPPER (empirical, one training run). Narrowing cannot be predicted analytically,
but alpha is a DEPLOYMENT knob: `from_dir` reads `lora_alpha` from
adapter_config.json and `_from_peft` derives scaling = alpha/rank at LOAD time.
So a single trained adapter can be evaluated at many alphas by rewriting one JSON
field -- 1 train + N evals instead of N trains + N evals.

WHAT THIS DELIBERATELY DOES
---------------------------
It evaluates at least one alpha BELOW the computed precision floor. The floor is a
prediction from a law fitted at a different rank; including a below-floor point
tests it instead of trusting it. If quality does not degrade below the floor, the
floor does not bind here and the write-up must say so.

    uv run --env-file .env python scripts/train/calibrate_expert_alpha.py \
        --adapter results/adapters/m2_postgresql_r8a128_v3 \
        --alphas 32 64 96 128
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from gnn_experiment.canon import REPO_ROOT as REPO  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

LAW_K = 0.167          # merge_rel_err_pct * (|dW|/|W|), fitted on the alpha sweep
MERGE_ERR_FLOOR_PCT = 5.0


def measure_dw_over_w(adapter_dir: Path) -> dict:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    sd = load_file(adapter_dir / "adapter_model.safetensors")
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    named = dict(model.named_parameters())
    num = den = 0.0
    pairs = 0
    for k in sd:
        if "lora_A" not in k:
            continue
        kb = k.replace("lora_A", "lora_B")
        if kb not in sd:
            continue
        dW = (sd[kb].float() @ sd[k].float()) * (alpha / r)
        base_key = (k.split("base_model.model.")[-1]
                     .replace(".lora_A.weight", ".weight").replace("lora_A.weight", "weight"))
        W = next((named[c] for c in (base_key, "model." + base_key) if c in named), None)
        if W is None:
            continue
        num += float(dW.norm() ** 2)
        den += float(W.float().norm() ** 2)
        pairs += 1
    del model
    assert pairs > 0, "no lora/base weight pairs matched -- cannot measure |dW|/|W|"
    return {"r": r, "alpha": alpha, "pairs": pairs,
            "dw_over_w": (num ** 0.5) / max(1e-30, den ** 0.5)}


def alpha_variant(src: Path, alpha: int, tmp: Path) -> Path:
    """A trained adapter re-scaled to a new deployment alpha. No retraining."""
    d = tmp / f"alpha_{alpha}"
    d.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, d / f.name)
    cfg = json.loads((d / "adapter_config.json").read_text())
    cfg["lora_alpha"] = alpha
    (d / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return d


def run_gate(pg: Path, astral: Path, log: Path) -> float | None:
    env = dict(os.environ, PG_ADAPTER=str(pg), ASTRAL_ADAPTER=str(astral), MAX_NEW="384")
    with open(log, "w") as fh:
        subprocess.run(
            [sys.executable, str(REPO / "benchmarks/factory/agentic/chained_holdout/bench_chained_holdout.py")],
            env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=2400, check=False)
    for line in log.read_text().splitlines():
        if "B oracle" in line and "mean=" in line:
            return float(line.split("mean=")[1].split()[0])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--peer", default="results/adapters/m2_astral_r8a128_v3",
                    help="held fixed while the target adapter's alpha varies")
    ap.add_argument("--alphas", type=int, nargs="+", default=[32, 64, 96, 128])
    ap.add_argument("--apply", action="store_true",
                    help="write the chosen alpha back into adapter_config.json")
    args = ap.parse_args()

    src = REPO / args.adapter
    peer = REPO / args.peer
    print(f"calibrating {src.name}  (peer held at {peer.name})", flush=True)

    m = measure_dw_over_w(src)
    ref_a, ratio_ref = m["alpha"], m["dw_over_w"]
    print(f"  r={m['r']} trained_alpha={ref_a} pairs={m['pairs']}  "
          f"|dW|/|W|={ratio_ref:.4f}", flush=True)

    curve = []
    for a in sorted(args.alphas):
        ratio = ratio_ref * (a / ref_a)
        err = LAW_K / max(1e-9, ratio)
        curve.append({"alpha": a, "scaling": a / m["r"], "dw_over_w": ratio,
                      "merge_err_pct": err, "below_floor": err > MERGE_ERR_FLOOR_PCT})
    a_min = next((c["alpha"] for c in curve if not c["below_floor"]), None)
    print(f"  precision floor ({MERGE_ERR_FLOOR_PCT}% merge err) -> alpha_min = {a_min}", flush=True)
    for c in curve:
        print(f"    alpha={c['alpha']:4d} s={c['scaling']:5.1f} |dW|/|W|={c['dw_over_w']:.4f} "
              f"merge_err={c['merge_err_pct']:6.2f}%"
              f"{'   <-- BELOW FLOOR (control)' if c['below_floor'] else ''}", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="alpha_cal_"))
    try:
        for c in curve:
            var = alpha_variant(src, c["alpha"], tmp)
            log = tmp / f"gate_{c['alpha']}.log"
            print(f"  evaluating alpha={c['alpha']} on the held-out gate...", flush=True)
            c["heldout_oracle"] = run_gate(var, peer, log)
            print(f"    alpha={c['alpha']:4d}  held-out oracle mean = {c['heldout_oracle']}", flush=True)
    finally:
        pass

    print("\n" + "=" * 74, flush=True)
    print(f"  {'alpha':>6} {'scaling':>8} {'merge_err%':>11} {'held-out':>9} {'admissible':>11}", flush=True)
    for c in curve:
        ok = (not c["below_floor"]) and c["heldout_oracle"] is not None
        print(f"  {c['alpha']:6d} {c['scaling']:8.1f} {c['merge_err_pct']:10.2f}% "
              f"{(c['heldout_oracle'] if c['heldout_oracle'] is not None else float('nan')):9.4f} "
              f"{'yes' if ok else 'no':>11}", flush=True)
    admissible = [c for c in curve if not c["below_floor"] and c["heldout_oracle"] is not None]
    best = max(admissible, key=lambda c: c["heldout_oracle"]) if admissible else None
    if best:
        print(f"\n  alpha_opt = {best['alpha']}  (held-out {best['heldout_oracle']:.4f}, "
              f"merge_err {best['merge_err_pct']:.2f}%)", flush=True)

    out = REPO / f"results/alpha_calibration_{src.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"adapter": src.name, "measured": m, "curve": curve,
                               "alpha_opt": best["alpha"] if best else None}, indent=2))
    print(f"  wrote {out}", flush=True)

    if args.apply and best:
        cfg_p = src / "adapter_config.json"
        cfg = json.loads(cfg_p.read_text())
        cfg["lora_alpha"] = best["alpha"]
        cfg_p.write_text(json.dumps(cfg, indent=2))
        print(f"  APPLIED lora_alpha={best['alpha']} to {cfg_p}", flush=True)


if __name__ == "__main__":
    main()
