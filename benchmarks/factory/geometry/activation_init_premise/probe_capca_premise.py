r"""Does the activation-covariance init premise hold here? -- one forward pass, no training.

THE CLAIM UNDER TEST
--------------------
`probe_pissa_premise.py` already killed the WEIGHT version: dW's energy sits only
1.37x above the random-chance floor inside W0's top-r singular subspace, so
initialising there is statistically indistinguishable from Gaussian init.

The proposed pivot is that W0 has no spikes but the TASK ACTIVATION covariance does:

    Sigma_task = E_{x~D}[ h(x) h(x)^T ]   over the INPUT to each adapted module

and that its top-8 eigenvectors capture 70%+ of activation energy, so LoRA's A should
be initialised into them (Covariate-Assisted PCA / CAPCA).

That is TWO claims, and passing the first without the second is worthless:

  1. SPIKE   Sigma_task is spiked -- top-8 energy >> 8/d chance floor.
  2. RELEVANCE  the subspace training ACTUALLY USES overlaps those spikes more than
     chance. Measured as retention of the trained adapter's input-side row space:

         retention = || P_8 A^T ||_F^2 / || A^T ||_F^2      chance floor = 8/d_in

     reported as x_above_floor, the same convention probe_pissa_premise.py and
     probe_subspace_overlap.py use, because raw retention is uninterpretable.

If (1) holds and (2) is at chance, activation-init aims at a subspace gradient descent
does not go to -- exactly the PiSSA verdict, one level up. That is the outcome this
probe exists to catch cheaply, before a training run is spent on it.

    uv run --env-file .env python \
        benchmarks/factory/geometry/activation_init_premise/probe_capca_premise.py
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from runtime.canon import CANON, REPO_ROOT, adapter_path  # noqa: E402
from runtime.novel_peft import FoldableExpert, set_hard_vram_cap  # noqa: E402

DOMAINS = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]
DATA_DIR = {"financial": "financial_planning"}
# Qwen3.5-4B is a hybrid: only layers 3, 7, 11, ... 31 carry self_attn; the rest are
# linear-attention (GatedDeltaNet) blocks with no q/k/v/o to adapt. MLP is on all 32.
# Probing "layer 4 q_proj" raised KeyError -- the layer exists, the module does not.
PROBE_SPECS = [(3, "self_attn.q_proj"), (15, "self_attn.q_proj"), (31, "self_attn.q_proj"),
               (4, "mlp.down_proj"), (12, "mlp.down_proj"),
               (20, "mlp.down_proj"), (28, "mlp.down_proj")]
RANK = 8


def load_prompts(domain: str, n: int) -> list[str]:
    d = DATA_DIR.get(domain, domain)
    out: list[str] = []
    for name in ("evaluation_data_disposition.jsonl", "training_data_v6.jsonl"):
        f = REPO_ROOT / "data" / d / name
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            p = r.get("prompt") or (r["messages"][0]["content"] if "messages" in r else None)
            if p:
                out.append(p)
            if len(out) >= n:
                return out
    return out


@torch.no_grad()
def collect_inputs(model, tok, prompts, keys) -> dict[str, torch.Tensor]:
    """Capture the INPUT to each probed module -- that is what A multiplies."""
    buf: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    handles = []

    def mk(key):
        def hook(_mod, args):
            buf[key].append(args[0].detach()[0].float().cpu())
        return hook

    mods = dict(model.named_modules())
    for k in keys:
        handles.append(mods[k].register_forward_pre_hook(mk(k)))
    try:
        for p in prompts:
            ids = tok(f"### Question:\n{p}\n\n### Answer:\n", return_tensors="pt",
                      truncation=True, max_length=512).to(model.device)
            model(**ids)
    finally:
        for h in handles:
            h.remove()
    return {k: torch.cat(v, dim=0) for k, v in buf.items() if v}


def spectrum_via_gram(H: torch.Tensor, k: int = RANK) -> dict:
    r"""Top-k spectrum and eigenvectors of Sigma = H^T H / n, WITHOUT forming a d x d matrix.

    WHY NOT eigh(Sigma) DIRECTLY
    ----------------------------
    The first version of this probe built the full d x d covariance and ran a float64
    eigendecomposition on it. For `down_proj`, d = 9216 while n = 1037 tokens, so the
    covariance has rank <= 1037 and that call spent ~8000 of its 9216 eigenvalues
    resolving structural zeros. Measured cost: ~40 minutes wall clock at 342% CPU with
    the GPU sitting idle holding the model resident.

    The Gram trick: H H^T is only n x n, shares every nonzero eigenvalue with H^T H, and
    its eigenvectors map across for free.

        G = H H^T            (n, n)     eigh -> mu_i, u_i
        eig(Sigma)  = mu_i / n
        eigvec(Sigma) = H^T u_i / sqrt(mu_i)

    Cost goes from O(d^3) = 7.8e11 to a BLAS matmul at O(n^2 d) = 1.0e10 plus O(n^3).

    LEDOIT-WOLF WITHOUT THE MATRIX
    ------------------------------
    The spherical target shifts every eigenvalue by the same amount and rotates
    nothing, so delta can be computed from the spectrum and the row norms alone:

        tr(S) = ||H||_F^2 / n            mu = tr(S)/p
        ||S||_F^2 = sum_i (mu_i/n)^2
        ||S - mu I||_F^2 = ||S||_F^2 - p*mu^2
        bbar2 = ( sum_t ||x_t||^4 - n ||S||_F^2 ) / n^2

    and because eigenVECTORS are unchanged by spherical shrinkage, `retention` is
    identical whether it is measured on S or on Sigma_LW. Only the energy fractions move.
    """
    n, d = H.shape
    Hd = H.double()
    G = Hd @ Hd.T                                            # (n, n)
    mu, U = torch.linalg.eigh(0.5 * (G + G.T))
    mu = mu.flip(0).clamp(min=0.0)                           # descending
    U = U.flip(1)

    lam = mu / n                                             # eigenvalues of S
    tr_S = float(Hd.pow(2).sum() / n)
    p = d
    mu_bar = tr_S / p
    S_fro2 = float(lam.pow(2).sum())
    d2 = (S_fro2 - p * mu_bar ** 2) / p
    # sum_t ||x_t x_t^T - S||_F^2 = sum_t ||x_t||^4 - n ||S||_F^2   (cross term is n||S||^2)
    sq = Hd.pow(2).sum(dim=1)                                # ||x_t||^2
    bbar2 = float(sq.pow(2).sum() - n * S_fro2) / (n * n * p)
    bbar2 = max(0.0, min(bbar2, d2)) if d2 > 0 else 0.0
    delta = float(bbar2 / d2) if d2 > 1e-30 else 0.0

    # top-k right eigenvectors, only the ones we need
    keep = int(min(k, (mu > mu[0] * 1e-12).sum())) if mu.numel() and mu[0] > 0 else 0
    if keep == 0:
        V = torch.zeros(d, k, dtype=torch.float64)
    else:
        V = (Hd.T @ U[:, :keep]) / mu[:keep].sqrt().unsqueeze(0)
        if keep < k:
            V = torch.cat([V, torch.zeros(d, k - keep, dtype=torch.float64)], dim=1)

    raw_topk = float(lam[:k].sum() / lam.sum()) if float(lam.sum()) > 0 else 0.0
    # shrunk spectrum: (1-delta)*lam_i + delta*mu_bar, over all p dims
    shrunk_top = float(((1 - delta) * lam[:k] + delta * mu_bar).sum())
    shrunk_tot = float((1 - delta) * tr_S + delta * mu_bar * p)
    return {"raw_topk_energy": raw_topk,
            "lw_topk_energy": shrunk_top / shrunk_tot if shrunk_tot > 0 else 0.0,
            "delta": delta, "eigvecs": V, "rank": keep}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/benchmarks/capca_premise.json")
    ap.add_argument("--prompts", type=int, default=32)
    args = ap.parse_args()

    set_hard_vram_cap(CANON.VRAM_CAP_GB)
    tok = AutoTokenizer.from_pretrained(CANON.BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CANON.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    model.eval()

    keys = [f"model.layers.{l}.{m}" for l, m in PROBE_SPECS]
    mods = dict(model.named_modules())
    missing = [k for k in keys if k not in mods]
    if missing:
        raise SystemExit(f"  no such modules: {missing}")
    t0 = time.time()
    print("=" * 100)
    print(f" CAPCA PREMISE   adapters={CANON.ADAPTER_VERSION}   {args.prompts} prompts/domain")
    print(" (1) is Sigma_task spiked?   (2) does the TRAINED adapter live in those spikes?")
    print("=" * 100)
    print(f"\n {'domain':14s} {'module':28s} {'n_tok':>6s} {'top8 energy':>12s} "
          f"{'chance':>8s} {'x_chance':>9s} {'LW delta':>9s} {'retention':>10s} {'x_floor':>8s}")
    print("-" * 100)

    rows = []
    for dom in DOMAINS:
        prompts = load_prompts(dom, args.prompts)
        if not prompts:
            print(f" {dom:14s} no prompts, skipped")
            continue
        acts = collect_inputs(model, tok, prompts, keys)
        try:
            expert = FoldableExpert.from_dir(adapter_path(dom), dom)
        except FileNotFoundError as e:
            print(f" {dom:14s} adapter missing ({str(e)[:40]}), spike test only")
            expert = None

        for k in keys:
            H = acts.get(k)
            if H is None:
                continue
            n, d = H.shape
            sp = spectrum_via_gram(H)
            top8, delta = sp["raw_topk_energy"], sp["delta"]
            chance = RANK / d

            ret = x_floor = float("nan")
            if expert is not None:
                f = expert.factors.get(k + ".weight")
                if f is not None:
                    _, A = f                      # A: (r, d_in) -- rows are what dW reads
                    At = A.detach().cpu().double().T
                    ret = float((sp["eigvecs"].T @ At).pow(2).sum() / At.pow(2).sum())
                    x_floor = ret / chance

            short = k.replace("model.layers.", "L").replace("self_attn.", "").replace("mlp.", "")
            print(f" {dom:14s} {short:28s} {n:6d} {top8:11.2%} {chance:8.2%} "
                  f"{top8/chance:8.1f}x {delta:9.4f} {ret:9.2%} {x_floor:7.2f}x", flush=True)
            rows.append({"domain": dom, "module": k, "n_tokens": n, "dim": d,
                         "top8_energy": top8, "lw_topk_energy": sp["lw_topk_energy"],
                         "chance": chance, "spike_x_chance": top8 / chance,
                         "lw_delta": delta, "sample_rank": sp["rank"],
                         "retention": ret, "x_above_floor": x_floor})
        print("-" * 100)

    import statistics as st
    spikes = [r["spike_x_chance"] for r in rows]
    floors = [r["x_above_floor"] for r in rows if r["x_above_floor"] == r["x_above_floor"]]
    print(f"\n  (1) SPIKE     median top-8 energy is {st.median(spikes):.1f}x the chance floor")
    print(f"  (2) RELEVANCE median trained-adapter retention is "
          f"{st.median(floors) if floors else float('nan'):.2f}x its floor")
    print("\n  For reference, probe_pissa_premise.py measured 1.37x on the WEIGHT version")
    print("  and that was judged indistinguishable from random init. Same bar applies.")

    p = REPO_ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"config": CANON.stamp() | {"prompts": args.prompts,
                                                        "rank": RANK},
                             "rows": rows, "elapsed_seconds": time.time() - t0}, indent=2))
    print(f"\n  {time.time()-t0:.0f}s   wrote {args.out}")


if __name__ == "__main__":
    main()
