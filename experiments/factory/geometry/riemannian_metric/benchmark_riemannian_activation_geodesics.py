r"""Geodesic distance between domains on REAL activation covariance -- the thread
`benchmark_riemannian_domain_distance.py` itself flags as open.

WHY THIS EXISTS
----------------
The weight-Gramian version of this benchmark is closed: it compares LoRA deltas
dW = s U V via a shared-subspace AIRM projection, is guarded by a rotation-
invariance self-test, and has been folded into DECISIONS.md as settled (flat,
backwards, already excised from the router). But that whole construction only
exists because a weight Gramian G = s^2 V V^T + U^T U is RANK-DEFICIENT (rank
<= 2r for shared r=8 adapters) and lives in an arbitrary per-adapter rank basis
-- the shared-subspace projection and its self-test are there to cancel a basis
ambiguity that is a fact about weight Gramians specifically.

Real per-token activation covariance Sigma_h = E_{x~D}[ h(x) h(x)^T ] has no such
ambiguity: h lives in the model's actual, fixed hidden-unit coordinates, the same
basis for every domain by construction. Comparing two domains' Sigma_h needs no
shared-subspace trick at all -- d_R(Sigma_a, Sigma_b) is already well-defined in
the natural basis. What it DOES have, that the weight version never did, is a
real sample count: n ~ 200-300 tokens against p = 2560 channels is the genuine
n < p regime `riemannian_covariance.ledoit_wolf_from_samples` was written for
and never called on until this script (see its docstring: "this is the branch
that belongs on ACTIVATION covariance... not usable on a weight Gramian").

WHAT THIS MEASURES
-------------------
For each of the 6 canonical domains, real forward hooks capture the real INPUT
activation to a set of shared modules across real domain-representative prompts
(same prompt loader as `activation_init_premise/probe_capca_premise.py`), on the
BASE model -- this is adapter-version-agnostic, unlike the weight version, since
it never touches a LoRA factor. Per (domain, module), the real Ledoit-Wolf
shrunk covariance is estimated from the real samples (delta from data, not
chosen), then AIRM geodesic distance is computed pairwise, directly, no
projection needed.

MODULES CHOSEN FOR COST, NOT CONVENIENCE
------------------------------------------
`down_proj`'s input dimension is 9216 (the MLP intermediate size) -- CAPCA's own
docstring measured a naive p=9216 eigendecomposition at ~40 minutes wall clock.
AIRM needs a full eigh of the actual p x p covariance (there is no Gram-trick
shortcut here, unlike CAPCA's top-k spectrum -- the WHOLE spectrum enters
matrix_log). So this probe uses only modules whose INPUT is the hidden size
(2560) or the attention concat width (4096): q/k/v_proj and mlp.{gate,up}_proj
(input 2560, eigh ~0.36s measured) and self_attn.o_proj (input 4096, ~1.8s
measured). down_proj is excluded -- not because it doesn't matter, but because
covering it needs the same Gram-trick-style dimensionality reduction CAPCA
already validates, which this script does not yet implement.

    uv run --env-file .env python \
        experiments/factory/geometry/riemannian_metric/benchmark_riemannian_activation_geodesics.py

Requires a GPU (real forward passes on Qwen3.5-4B). All the geodesic linear
algebra itself runs in float64 on CPU, same convention as the weight version.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from runtime.canon import CANON, REPO_ROOT  # noqa: E402
from runtime.novel_peft import set_hard_vram_cap  # noqa: E402
from runtime.riemannian_covariance import (  # noqa: E402
    airm_components,
    ledoit_wolf_from_samples,
    matrix_inv_sqrt,
)

DOMAINS = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]
DATA_DIR = {"financial": "financial_planning"}
PROBE_SPECS = [
    (3, "self_attn.q_proj"), (15, "self_attn.q_proj"), (31, "self_attn.q_proj"),
    (3, "self_attn.k_proj"), (15, "self_attn.k_proj"), (31, "self_attn.k_proj"),
    (3, "self_attn.v_proj"), (15, "self_attn.v_proj"), (31, "self_attn.v_proj"),
    (3, "self_attn.o_proj"), (15, "self_attn.o_proj"), (31, "self_attn.o_proj"),
    (4, "mlp.gate_proj"), (12, "mlp.gate_proj"), (20, "mlp.gate_proj"), (28, "mlp.gate_proj"),
    (4, "mlp.up_proj"), (12, "mlp.up_proj"), (20, "mlp.up_proj"), (28, "mlp.up_proj"),
]
N_PROMPTS = 32

# The same two measured-stacking-outcome files the weight version uses, for the
# same consistency check -- reused verbatim (domain identity, not adapter
# version, is what activation covariance depends on).
GROUND_TRUTH = {
    "stacked_4expert_matrix_2048.json": {
        "solo": {"astral": "Solo Astral", "postgresql": "Solo PostgreSQL", "duckdb": "Solo DuckDB"},
        "pairs": {("astral", "postgresql"): "ast+pg   [SHIP A]",
                  ("astral", "duckdb"): "ast+duck [SHIP B]",
                  ("postgresql", "duckdb"): "pg+duck  [collide]"},
        "base": "Base Model",
        "test_domains": {"financial_planning": "financial", "astral": "astral",
                         "postgresql": "postgresql", "duckdb": "duckdb"},
    },
    "stacked_experts_v4_all_clean.json": {
        "solo": {"astral": "ast", "postgresql": "pg", "financial": "fin"},
        "pairs": {("financial", "astral"): "fin+ast", ("financial", "postgresql"): "fin+pg"},
        "base": "base",
        "test_domains": {"financial_planning": "financial", "astral": "astral", "postgresql": "postgresql"},
    },
}


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


def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / den) if den > 1e-12 else 0.0


def print_matrix(title: str, M: np.ndarray, domains: list[str]) -> None:
    print(f"\n {title}")
    print(" " + "-" * (18 + 11 * len(domains)))
    print(f" {'':16s}" + "".join(f"{d[:9]:>11}" for d in domains))
    for i, d in enumerate(domains):
        print(f" {d:16s}" + "".join(
            "          ." if i == j else f"{M[i, j]:11.3f}"
            for j in range(len(domains))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/benchmarks/riemannian_activation_geodesics.json")
    ap.add_argument("--prompts", type=int, default=N_PROMPTS)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 92)
    print(" RIEMANNIAN GEODESIC DISTANCE BETWEEN DOMAINS -- REAL ACTIVATION COVARIANCE")
    print(f" base={CANON.BASE_MODEL}   {args.prompts} prompts/domain   {len(PROBE_SPECS)} modules")
    print("=" * 92)

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

    # domain -> module -> (Sigma_lw float64 p x p, delta, n_tokens)
    cov: dict[str, dict[str, tuple[torch.Tensor, float, int]]] = {}
    for dom in DOMAINS:
        prompts = load_prompts(dom, args.prompts)
        if not prompts:
            print(f"  {dom:14s} no prompts, skipped")
            continue
        acts = collect_inputs(model, tok, prompts, keys)
        cov[dom] = {}
        deltas = []
        for k in keys:
            H = acts.get(k)
            if H is None:
                continue
            Sigma, delta = ledoit_wolf_from_samples(H)
            cov[dom][k] = (Sigma.double(), delta, H.shape[0])
            deltas.append(delta)
        n_tok = next(iter(cov[dom].values()))[2] if cov[dom] else 0
        print(f"  {dom:14s} n_tokens={n_tok:4d}   median LW delta={np.median(deltas):.4f}"
              f"  (n<p regime: {n_tok} tokens vs p in {{2560, 4096}})")

    domains = [d for d in DOMAINS if d in cov]
    print(f"\n  self-check: same-domain self-distance d_R(Sigma_d, Sigma_d) should be ~0")
    worst_self = 0.0
    for dom in domains:
        for k, (S, _, _) in cov[dom].items():
            d = airm_components(S, S)["total"]
            worst_self = max(worst_self, d)
    print(f"  worst self-distance across all domains/modules: {worst_self:.2e} (must be < 1e-6)")
    if worst_self >= 1e-6:
        raise SystemExit("  ABORT: d_R(X, X) != 0 -- numerical issue, do not report these numbers.")

    n = len(domains)
    M = np.zeros((n, n))
    M_scale = np.zeros((n, n))
    M_shape = np.zeros((n, n))
    per_module_used = {k: 0 for k in keys}
    for i in range(n):
        for j in range(i + 1, n):
            tot = sc = sh = 0.0
            cnt = 0
            for k in keys:
                if k not in cov[domains[i]] or k not in cov[domains[j]]:
                    continue
                Sa, _, _ = cov[domains[i]][k]
                Sb, _, _ = cov[domains[j]][k]
                Ai = matrix_inv_sqrt(Sa)
                c = airm_components(Sa, Sb, A_inv_sqrt=Ai)
                tot += c["total"]; sc += c["scale"]; sh += c["shape"]; cnt += 1
                per_module_used[k] += 1
            M[i, j] = M[j, i] = tot / max(1, cnt)
            M_scale[i, j] = M_scale[j, i] = sc / max(1, cnt)
            M_shape[i, j] = M_shape[j, i] = sh / max(1, cnt)

    print_matrix(f"AIRM GEODESIC DISTANCE d_R  (real activation covariance, "
                 f"mean over {len(keys)} modules, LW-shrunk from real samples)", M, domains)
    off = M[np.triu_indices(n, 1)]
    print(f"\n  off-diagonal  min {off.min():.3f}  max {off.max():.3f}  "
          f"spread {off.max()-off.min():.3f}  ({100*(off.max()-off.min())/off.mean():.1f}% of the mean)")

    print("\n  scale vs shape -- is a domain's activation distribution FARTHER, or just BIGGER (more energetic)?")
    print(f"  {'pair':28s} {'d_R':>8s} {'shape':>8s} {'scale':>8s}")
    print("  " + "-" * 56)
    pair_detail = {}
    for i in range(n):
        for j in range(i + 1, n):
            name = f"{domains[i]}|{domains[j]}"
            pair_detail[name] = {"d_R": M[i, j], "shape": M_shape[i, j], "scale": M_scale[i, j]}
    for name, v in sorted(pair_detail.items(), key=lambda kv: -kv[1]["d_R"]):
        print(f"  {name:28s} {v['d_R']:8.3f} {v['shape']:8.3f} {v['scale']:8.3f}")

    # Consistency arm, same ground truth the weight version checks against.
    print("\n" + "=" * 92)
    print(" CONSISTENCY ARM -- activation d_R vs MEASURED stacking outcomes")
    print("=" * 92)
    rows = []
    for fname, spec in GROUND_TRUTH.items():
        path = REPO_ROOT / "results/benchmarks" / fname
        if not path.exists():
            print(f"  missing {fname}, skipped")
            continue
        means = json.loads(path.read_text())["means"]
        for (x, y), arm in spec["pairs"].items():
            if x not in cov or y not in cov or arm not in means:
                continue
            own, other = [], []
            for test_key, dom in spec["test_domains"].items():
                stacked = means[arm][test_key]
                if dom in (x, y):
                    solo = means[spec["solo"][dom]][test_key]
                    own.append(stacked - solo)
                else:
                    other.append(stacked - means[spec["base"]][test_key])
            i, j = domains.index(x), domains.index(y)
            rows.append({"pair": f"{x}+{y}", "run": fname.split(".")[0][:22],
                         "d_R": M[i, j], "shape": M_shape[i, j], "scale": M_scale[i, j],
                         "synergy": float(np.mean(own)) if own else float("nan"),
                         "collateral": float(np.mean(other)) if other else float("nan")})
    cons: dict = {"pairs": rows}
    if rows:
        print(f"\n {'pair':22s} {'d_R':>8s} {'shape':>8s} {'scale':>8s} "
              f"{'synergy':>9s} {'collateral':>11s}   run")
        print(" " + "-" * 88)
        for r in sorted(rows, key=lambda r: r["d_R"]):
            print(f" {r['pair']:22s} {r['d_R']:8.3f} {r['shape']:8.3f} {r['scale']:8.3f} "
                  f"{r['synergy']:+9.2f} {r['collateral']:+11.2f}   {r['run']}")
        dr = [r["d_R"] for r in rows]
        syn = [r["synergy"] for r in rows]
        col = [r["collateral"] for r in rows]
        rs_syn, rs_col = spearman(dr, syn), spearman(dr, col)
        print(f"\n  Spearman  d_R vs synergy     {rs_syn:+.3f}")
        print(f"  Spearman  d_R vs collateral   {rs_col:+.3f}")
        print(f"  n={len(rows)} pairs -- same tiny-n caveat as the weight version: this arm can only")
        print("  show a contradiction, never confirm a predictor, at n this small.")
        cons["spearman_synergy"] = rs_syn
        cons["spearman_collateral"] = rs_col
        cons["n"] = len(rows)
    else:
        print("  nothing measurable")

    # Cross-check against the weight-Gramian version's own saved result, if present.
    weight_path = REPO_ROOT / "results/benchmarks/riemannian_domain_geodesics.json"
    cross = {}
    if weight_path.exists():
        wdat = json.loads(weight_path.read_text())
        wdomains = wdat["domains"]
        wM = np.array(wdat["airm"])
        common = [d for d in domains if d in wdomains]
        a_vals, w_vals = [], []
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                di, dj = domains.index(common[i]), domains.index(common[j])
                wi, wj = wdomains.index(common[i]), wdomains.index(common[j])
                a_vals.append(M[di, dj]); w_vals.append(wM[wi, wj])
        rho = spearman(a_vals, w_vals) if len(a_vals) >= 3 else float("nan")
        print("\n" + "=" * 92)
        print(" CROSS-CHECK -- does activation-space d_R agree with weight-space d_R?")
        print("=" * 92)
        print(f"  {len(a_vals)} common pairs   Spearman rank-corr = {rho:+.3f}")
        print("  (Two totally different constructions -- real activations here, weight")
        print("   Gramians there. Agreement would be notable; disagreement is not a bug in")
        print("   either, since they measure different things about the same domains.)")
        cross = {"n_pairs": len(a_vals), "spearman": rho}

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": CANON.stamp() | {"prompts": args.prompts, "metric": "AIRM-activation-LW",
                                   "modules": keys},
        "domains": domains, "airm": M.tolist(), "airm_scale": M_scale.tolist(),
        "airm_shape": M_shape.tolist(), "pairs": pair_detail,
        "consistency": cons, "cross_check_vs_weight_space": cross,
        "elapsed_seconds": time.time() - t0,
    }, indent=2))
    print(f"\n  {time.time()-t0:.0f}s   wrote {args.out}")


if __name__ == "__main__":
    main()
