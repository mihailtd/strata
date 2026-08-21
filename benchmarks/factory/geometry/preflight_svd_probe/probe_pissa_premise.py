"""Pre-flight probe: does PiSSA's premise hold on this model?

WHAT PiSSA CLAIMS
-----------------
Standard LoRA starts at B=0, A~gaussian, so dW=0 and the first steps are spent
finding a direction. PiSSA (Meng et al., 2024) instead initialises A and B from
the top-r SVD of the base weight W0, on the premise that the principal singular
directions of W0 are "the most critical parameter dimensions", so starting there
converges faster.

peft 0.20.0 implements this already: `LoraConfig(init_lora_weights="pissa")`.
There is nothing to build. The only open question is whether the premise holds
HERE -- and that is measurable on CPU, before spending a GPU-hour on training.

WHAT THIS PROBE MEASURES
------------------------
For each (layer, projection) it takes the ALREADY-TRAINED m2 adapter's update
dW = scaling * B @ A and asks: how much of dW's energy lies inside the top-r
singular subspace of W0 -- the exact subspace PiSSA would have initialised into?

    retention = || U_r^T dW V_r ||_F^2  /  || dW ||_F^2

against the random-chance floor for a two-sided projection onto an r-dim
subspace on each side:

    floor = (r / out_features) * (r / in_features)

The reported number is `x_above_floor = retention / floor`, matching the
convention already used by `probe_subspace_overlap.py` in this directory. That
convention exists because raw retention is uninterpretable: for r=8 on a
2560x9216 weight the floor is 2.7e-6, so "retention < 1%" can never fail and
firing on it says nothing.

It also profiles WHERE in W0's spectrum the update actually lives, by projecting
dW onto increasingly deep bands of W0's singular directions. If the energy is
flat across bands, the update is spectrally agnostic and "top-r is special" is
false for this model.

HOW TO READ THE RESULT
----------------------
  x_above_floor ~ 1     The trained update is no more aligned with W0's principal
                        subspace than a random matrix would be. PiSSA would be
                        initialising into a subspace the update does not use, so
                        expect no convergence benefit -- and possibly a cost,
                        since PiSSA also subtracts that subspace from the frozen
                        residual base weight.

  x_above_floor >> 1    The update concentrates where PiSSA initialises. The
                        premise holds; a training A/B is worth the GPU time.

WHAT IT DOES NOT PROVE
----------------------
This is correlational, not causal. It measures where a LoRA-initialised run
*ended up*, which is not necessarily where a PiSSA-initialised run would go --
different initialisations can converge to different optima. A null result here
is evidence that the premise does not hold for this model, not proof that PiSSA
cannot help. It is a cheap gate on an expensive experiment, which is exactly the
discipline TODO.md already prescribes: "Measure subspace overlap before building
any further shared-basis scheme."

USAGE:
    uv run --env-file .env python \
        benchmarks/factory/geometry/preflight_svd_probe/probe_pissa_premise.py \
        --adapter results/adapters/m2_astral_r8a128

Runs on GPU by default (all 32 layers in ~a minute); `--device cpu` is the
reference path. Because ROCm on this box has silently returned wrong numerics
before, the GPU path spot-checks modules against a CPU recompute and aborts on
disagreement -- see `--verify-device-agreement`.

NOTE ON SCHEDULING: heavy on whichever device it uses. TODO.md records that
batch-1 decode here is CPU-bound on kernel launches, so do not run this beside a
decode benchmark on either device -- it will skew that benchmark's tok/s.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT))

# Keep the footprint small; this is a probe, not a training job.
torch.set_num_threads(min(4, torch.get_num_threads()))

MODULE_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def find_base_shards(model_id: str) -> list[Path]:
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    repo = cache / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not repo.exists():
        raise SystemExit(f"Base model not found in HF cache: {repo}")
    snap = sorted(repo.iterdir())[-1]
    shards = sorted(snap.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"No safetensors under {snap}")
    return shards


def build_weight_index(shards: list[Path]) -> dict[str, tuple[Path, str]]:
    """Index base weights by the `layers.N.<module>.weight` SUFFIX.

    peft records targets as `base_model.model.model.layers.N.…` while this
    checkpoint stores them as `model.language_model.layers.N.…` (it is a hybrid
    stack — some layers are `linear_attn`, not `self_attn`). Matching on the
    suffix makes the probe independent of that prefix, and skips the `mtp.*`
    draft-head tensors, which share module names but are not what the adapter
    was folded into.
    """
    index: dict[str, tuple[Path, str]] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith("mtp."):
                    continue  # draft head, not the backbone the adapter targets
                m = re.search(r"(layers\.\d+\..*\.weight)$", k)
                if m:
                    index[m.group(1)] = (shard, k)
    return index


def load_weight(index: dict[str, tuple[Path, str]], name: str) -> torch.Tensor | None:
    m = re.search(r"(layers\.\d+\..*\.weight)$", name)
    if not m:
        return None
    hit = index.get(m.group(1))
    if hit is None:
        return None
    shard, real_name = hit
    with safe_open(str(shard), framework="pt") as f:
        return f.get_tensor(real_name).to(torch.float32)


def load_adapter_deltas(adapter_dir: Path) -> dict[str, torch.Tensor]:
    """{base_weight_name: dW (out,in)} with the model's real scaling applied."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]

    pairs: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if ".lora_A" in k:
                pairs[k.split(".lora_A")[0]]["A"] = f.get_tensor(k).to(torch.float32)
            elif ".lora_B" in k:
                pairs[k.split(".lora_B")[0]]["B"] = f.get_tensor(k).to(torch.float32)

    deltas: dict[str, torch.Tensor] = {}
    for mod, ab in pairs.items():
        if "A" not in ab or "B" not in ab:
            continue
        # peft: lora_A (r,in), lora_B (out,r) -> dW = scaling * B@A, shape (out,in)
        deltas[mod.replace("base_model.model.", "") + ".weight"] = scaling * (ab["B"] @ ab["A"])
    return deltas


def top_r_subspace(W: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """EXACT top-r left/right singular vectors, via the Gram matrix.

    Do NOT use `torch.svd_lowrank` here. These weights have a nearly flat
    spectrum (measured s[0]/s[7] = 2.05 on layer 0 down_proj), and randomised SVD
    cannot separate a top-r subspace that is barely distinguished from its
    neighbours. Measured consequence: five different seeds returned retentions
    spanning 2.56x (1.01e-05 .. 2.60e-05), and all of them UNDERESTIMATED the
    exact value of 1.79e-05 -> the randomised estimate reported ~1.3x above the
    chance floor where the truth is 6.59x. That error is large enough to invert
    this probe's verdict, so exactness is not optional.

    Exact and cheap: eigendecompose the Gram matrix on the SMALL side (2560 here,
    never 9216), then carry the result across:
        W (m,n), m <= n:  C = W W^T (m,m); eigh -> U_r, s;  V_r = W^T U_r / s
        W (m,n), m >  n:  C = W^T W (n,n); eigh -> V_r, s;  U_r = W V_r / s
    """
    m, n = W.shape
    if m <= n:
        C = W @ W.T                                   # (m, m)
        evals, evecs = torch.linalg.eigh(C)           # ascending
        U_r = evecs[:, -r:].flip(-1).contiguous()     # top-r, descending
        s = evals[-r:].flip(-1).clamp_min(0).sqrt()
        V_r = (W.T @ U_r) / s.clamp_min(1e-12)
        V_r = torch.linalg.qr(V_r)[0].contiguous()    # re-orthonormalise
    else:
        C = W.T @ W                                   # (n, n)
        evals, evecs = torch.linalg.eigh(C)
        V_r = evecs[:, -r:].flip(-1).contiguous()
        s = evals[-r:].flip(-1).clamp_min(0).sqrt()
        U_r = (W @ V_r) / s.clamp_min(1e-12)
        U_r = torch.linalg.qr(U_r)[0].contiguous()
    return U_r, V_r


def retention_in_subspace(dW: torch.Tensor, U_r: torch.Tensor, V_r: torch.Tensor) -> float:
    """Fraction of ||dW||_F^2 surviving the two-sided projection onto (U_r, V_r)."""
    total = dW.pow(2).sum().item()
    if total <= 0:
        return 0.0
    proj = U_r.T @ dW @ V_r
    return proj.pow(2).sum().item() / total


def spectral_band_profile(
    W: torch.Tensor, dW: torch.Tensor, bands: list[int]
) -> dict[str, float]:
    """Cumulative dW energy captured by W's top-k singular directions, per band.

    Compared against the chance floor for each band so the numbers are readable.
    """
    out: dict[str, float] = {}
    total = dW.pow(2).sum().item()
    if total <= 0:
        return out
    m, n = W.shape
    max_k = min(max(bands), min(W.shape))
    q = min(max_k + 16, min(W.shape))
    U, _, V = torch.svd_lowrank(W, q=q, niter=4)
    for k in bands:
        k = min(k, U.shape[1], V.shape[1])
        proj = U[:, :k].T @ dW @ V[:, :k]
        retained = proj.pow(2).sum().item() / total
        floor = (k / m) * (k / n)
        out[f"top_{k}"] = retained
        out[f"top_{k}_x_floor"] = retained / max(1e-30, floor)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="PiSSA premise pre-flight probe (CPU)")
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default="results/adapters/m2_astral_r8a128")
    ap.add_argument("--rank", type=int, default=None, help="defaults to the adapter's own r")
    ap.add_argument("--layers", type=int, nargs="+", default=list(range(32)))
    ap.add_argument("--bands", type=int, nargs="+", default=[8, 32, 128, 512])
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda is ~10x faster for the randomised SVDs; cpu is the reference",
    )
    ap.add_argument(
        "--verify-device-agreement",
        type=int,
        default=2,
        help="re-check N modules on CPU and assert the GPU agrees. ROCm has "
             "silently returned wrong results on this box before (AITER kernels "
             "emitting zeros at hidden=2560 on gfx1100), so a GPU speedup is only "
             "usable if it is checked. 0 disables.",
    )
    ap.add_argument("--out", default="results/pissa_premise_probe.json")
    args = ap.parse_args()

    device = torch.device(args.device)

    adapter_dir = REPO_ROOT / args.adapter
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r = args.rank or cfg["r"]

    print("=" * 88)
    print("  PiSSA PREMISE PRE-FLIGHT PROBE  (CPU only)")
    print("=" * 88)
    print(f"  Adapter : {args.adapter}  (r={cfg['r']}, alpha={cfg['lora_alpha']})")
    print(f"  Probing at rank r={r} -- the subspace PiSSA would initialise into")
    print(f"  Layers  : {args.layers}")

    print("\nIndexing base model shards...")
    index = build_weight_index(find_base_shards(args.model_id))
    deltas = load_adapter_deltas(adapter_dir)
    print(f"  {len(index)} base tensors, {len(deltas)} adapter deltas")

    rows = []
    per_type: dict[str, list[float]] = defaultdict(list)

    for name, dW in sorted(deltas.items()):
        m = re.search(r"layers\.(\d+)\.", name)
        if not m or int(m.group(1)) not in args.layers:
            continue
        mtype = next((t for t in MODULE_TYPES if f".{t}." in name), "other")

        W = load_weight(index, name)
        if W is None:
            print(f"  ! base weight missing for {name}")
            continue
        if W.shape != dW.shape:
            print(f"  ! shape mismatch {name}: W{tuple(W.shape)} dW{tuple(dW.shape)}")
            continue

        W = W.to(device)
        dW_d = dW.to(device)
        U_r, V_r = top_r_subspace(W, r)
        retained = retention_in_subspace(dW_d, U_r, V_r)
        floor = (r / W.shape[0]) * (r / W.shape[1])
        x_floor = retained / max(1e-30, floor)

        bands = spectral_band_profile(W, dW_d, args.bands)

        rows.append({
            "module": name,
            "layer": int(m.group(1)),
            "module_type": mtype,
            "shape": list(W.shape),
            "retained_in_top_r": retained,
            "random_floor": floor,
            "x_above_floor": x_floor,
            "bands": bands,
        })
        per_type[mtype].append(x_floor)
        print(f"  L{int(m.group(1)):<3} {mtype:<11} {str(tuple(W.shape)):<14} "
              f"retained={retained:.3e}  floor={floor:.3e}  {x_floor:7.2f}x floor")

    if not rows:
        raise SystemExit("No modules probed -- check --layers against the adapter's target modules.")

    # ROCm has silently produced wrong numerics on this box before, so a GPU
    # result is only trustworthy if spot-checked against the CPU reference.
    agreement = None
    if device.type != "cpu" and args.verify_device_agreement > 0:
        print(f"\nVerifying {args.verify_device_agreement} module(s) against a CPU recompute...")
        agreement = []
        for row in rows[: args.verify_device_agreement]:
            W_cpu = load_weight(index, row["module"])
            dW_cpu = deltas[row["module"]]
            U_c, V_c = top_r_subspace(W_cpu, r)
            ref = retention_in_subspace(dW_cpu, U_c, V_c)
            got = row["retained_in_top_r"]
            rel = abs(got - ref) / max(1e-30, ref)
            agreement.append({"module": row["module"], "gpu": got, "cpu": ref, "rel_diff": rel})
            status = "OK" if rel < 0.05 else "MISMATCH"
            print(f"  {row['module'][-40:]:<42} gpu={got:.3e} cpu={ref:.3e} rel={rel:.2%}  {status}")
        worst = max(a["rel_diff"] for a in agreement)
        if worst >= 0.05:
            raise SystemExit(
                f"GPU/CPU disagree by {worst:.1%} -- randomised SVD is stochastic but not "
                "that stochastic. Re-run with --device cpu and do not trust the GPU path here."
            )
        print(f"  agreement OK (worst {worst:.2%}); randomised SVD is seed-dependent so small drift is expected")

    all_x = [r_["x_above_floor"] for r_ in rows]
    all_x.sort()
    median_x = all_x[len(all_x) // 2]

    print("\n" + "=" * 88)
    print("  PER MODULE TYPE (median x above chance floor)")
    print("=" * 88)
    for t in MODULE_TYPES:
        if per_type[t]:
            v = sorted(per_type[t])
            print(f"  {t:<12} n={len(v):<3} median {v[len(v) // 2]:7.2f}x   "
                  f"range {min(v):.2f}-{max(v):.2f}")

    print("\n" + "=" * 88)
    print("  SPECTRAL BAND PROFILE (median x above floor, pooled)")
    print("=" * 88)
    for k in args.bands:
        key = f"top_{k}_x_floor"
        vals = sorted(r_["bands"][key] for r_ in rows if key in r_["bands"])
        if vals:
            print(f"  W0 top-{k:<4} directions: {vals[len(vals) // 2]:7.2f}x floor")

    print("\n" + "=" * 88)
    print("  VERDICT")
    print("=" * 88)
    print(f"  Median alignment of the trained update with W0's top-{r} subspace: "
          f"{median_x:.2f}x the random-chance floor.")
    if median_x < 2.0:
        verdict = "PREMISE DOES NOT HOLD"
        print(
            f"\n  => {verdict}. The update the model actually learned is essentially\n"
            "     no more aligned with W0's principal subspace than a random matrix.\n"
            "     PiSSA would initialise into a subspace this model does not use, so a\n"
            "     convergence benefit is not expected. Note PiSSA also SUBTRACTS that\n"
            "     subspace from the frozen residual, so it is not a free bet.\n"
            "     Recommendation: do not spend GPU time on a PiSSA training A/B on the\n"
            "     strength of the published claim alone."
        )
    elif median_x < 10.0:
        verdict = "WEAK / AMBIGUOUS"
        print(f"\n  => {verdict}. Some concentration, not decisive. A training A/B could go either way.")
    else:
        verdict = "PREMISE HOLDS"
        print(
            f"\n  => {verdict}. The update concentrates strongly in exactly the subspace\n"
            "     PiSSA initialises. A training A/B is worth the GPU time."
        )

    report = {
        "model_id": args.model_id,
        "adapter": args.adapter,
        "rank_probed": r,
        "layers": args.layers,
        "device": str(device),
        "device_agreement_check": agreement,
        "median_x_above_floor": median_x,
        "verdict": verdict,
        "per_module_type_median_x": {
            t: sorted(v)[len(v) // 2] for t, v in per_type.items() if v
        },
        "modules": rows,
        "caveat": (
            "Correlational: measures where a LoRA-initialised run ended up, which is "
            "not necessarily where a PiSSA-initialised run would go. A null result is "
            "evidence against the premise on this model, not proof PiSSA cannot help."
        ),
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
