"""Stacking Merit & Conflict probe -- which adapters mix well, measured BEFORE stacking.

WHY THIS EXISTS
---------------
Two predictors of stacking damage are already refuted:

  * weight-space subspace overlap  -- 1.10-1.28x chance for EVERY cross-task pair
  * activation cosine (own domain) -- +0.0046 to +0.0176 for EVERY pair

Both are near-uniform, so neither can explain a 12pp spread between fin+pg
(-9.53pp) and ast+pg (+2.46pp). They measure whether adapters point in different
DIRECTIONS, and they all do. Direction was never the problem.

The surviving hypothesis has two terms:

  MERIT_X    = (solo gain over base on X's own domain)
               / (mean ||delta_X||/||h|| on OTHER domains)
             -> an adapter earns its slot when it carries more signal on its
                domain than noise on everyone else's.

  CONFLICT_XY = cosine(delta_X, delta_Y) measured ON THE SHARED DOMAIN's prompts
             -> two high-merit adapters can still fight if they fire on the SAME
                tokens toward DIFFERENT targets (QUALIFY vs DISTINCT ON).

The existing probe cannot produce either: it runs one adapter at a time against a
fixed prose set that lumps financial in with Byzantine history and Kafka, so there
is no per-domain breakdown and no pairwise term.

METHOD -- one base forward, all adapters compared on IDENTICAL h
----------------------------------------------------------------
delta = scale * (h @ A.T) @ B.T depends on h, and h depends on which adapters are
live. Loading each adapter separately gives each a DIFFERENT h, so the resulting
deltas are not comparable and a cosine between them is meaningless.

So: run the BASE model once per prompt, capture h at every LoRA target, then
compute every adapter's delta offline from those same h. That is both cheaper
(one forward instead of N) and the only way the cosine term is well-defined --
"given identical input state, what does each adapter emit, and do they agree?"

Energies are reported as ||delta||/||h|| so magnitudes are comparable across
domains whose hidden states differ in scale.

    uv run --env-file .env python benchmarks/factory/geometry/activation_inertia_probe/probe_stacking_merit.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import set_hard_vram_cap  # noqa: E402

ADAPTERS = {
    "ast": "results/adapters/m2_astral_r8a128_v4",
    "pg": "results/adapters/m2_postgresql_r8a128_v4",
    "duck": "results/adapters/m2_duckdb_r8a128_v4",
    "fin": "results/adapters/m2_financial_r8a128_v4",
    "ast_v5": "results/adapters/m2_astral_r8a128_v5",
    # v5b = L_inert on OUT-OF-DOMAIN replay (v5 penalised in-domain prompt
    # tokens and produced a uniform 0.843x with ASR unchanged at 0.96).
    "ast_v5b": "results/adapters/m2_astral_r8a128_v5b",
}

# Per-domain prompt sets. Financial gets its OWN bucket rather than being folded
# into "generic out-of-domain" -- the whole question is which PEER domain an
# adapter leaks into, and lumping peers together destroys exactly that signal.
PROMPTS = {
    "astral": [
        "How do I configure workspace dependencies and lockfiles using uv and pyproject.toml in Python 3.12?",
        "Write an async coroutine using asyncio.TaskGroup and type hints with ruff formatting rules.",
        "Demonstrate how to run isolated Python scripts with inline script metadata PEP 723 using uv run.",
        "Replace my pip + requirements.txt workflow with uv, and format the project with ruff.",
    ],
    "postgresql": [
        "How do I create an HNSW vector index in pgvector with cosine distance ops and query nearest neighbors?",
        "Write a query using DISTINCT ON with window functions and explain the plan with EXPLAIN ANALYZE.",
        "Demonstrate how to partition a PostgreSQL 17 table by range with BRIN indexing on timestamp columns.",
        "Return the top 3 products per category by revenue.",
    ],
    "duckdb": [
        "How do I query partitioned Parquet files from S3 with projection pushdown and hive_partitioning in DuckDB?",
        "Write a DuckDB query using FROM-first syntax, COLUMNS(*) regex aggregation, and EXCLUDE.",
        "Demonstrate zero-copy export from a DuckDB query to a Polars DataFrame using PyArrow record batches.",
        "Return the top 3 products per category by revenue.",
    ],
    "financial": [
        "Explain the difference between traditional Roth IRA contributions, 401(k) rollovers, and tax brackets.",
        "Calculate the net present value of a multi-period capital investment with inflation-adjusted cash flows.",
        "What asset allocation should a 45-year-old use for retirement in 20 years?",
        "Compare term life insurance against whole life for a family with two young children.",
    ],
    "general": [
        "Discuss the history of the Byzantine Empire and the architectural design of the Hagia Sophia.",
        "Write a short essay analyzing the themes of existentialism in Franz Kafka's The Metamorphosis.",
        "Explain how photosynthesis converts light energy into chemical energy in plants.",
        "Summarize the causes and consequences of the 1929 Wall Street crash.",
    ],
}

# NOTE: "Return the top 3 products per category by revenue." appears in BOTH the
# postgresql and duckdb sets deliberately. It is the collision prompt -- nothing
# in it names an engine, postgres wants DISTINCT ON, duckdb wants QUALIFY. It is
# the single most informative input for the CONFLICT term.


def load_adapter(path: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    """Return {module_path: (A, B)} plus the alpha/r scaling, straight from the
    safetensors. Deliberately NOT via PeftModel: attaching an adapter changes h,
    and every adapter must see the SAME h for the cosine term to mean anything."""
    cfg = json.loads((path / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    raw = load_file(path / "adapter_model.safetensors")

    factors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for k, v in raw.items():
        if ".lora_A.weight" in k:
            factors[k.split(".lora_A.weight")[0]]["A"] = v
        elif ".lora_B.weight" in k:
            factors[k.split(".lora_B.weight")[0]]["B"] = v

    out = {}
    for k, ab in factors.items():
        if "A" not in ab or "B" not in ab:
            continue
        # strip the PEFT wrapper prefix so keys match model.named_modules()
        name = k
        for prefix in ("base_model.model.", "base_model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        out[name] = (ab["A"], ab["B"])
    return out, scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="results/benchmarks/stacking_merit_probe.json")
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    dev = model.device

    adapters, scales = {}, {}
    for name, rel in ADAPTERS.items():
        p = REPO_ROOT / rel
        if not p.exists():
            print(f"  SKIP {name}: {rel} not found", flush=True)
            continue
        f, sc = load_adapter(p)
        adapters[name] = {k: (a.to(dev, torch.bfloat16), b.to(dev, torch.bfloat16))
                          for k, (a, b) in f.items()}
        scales[name] = sc
        print(f"  loaded {name:8s} {len(f)} modules  scale={sc:.1f}", flush=True)

    names = list(adapters)
    # every adapter must target the same modules or the comparison is apples/oranges
    common = set.intersection(*(set(a) for a in adapters.values()))
    print(f"  {len(common)} modules common to all {len(names)} adapters\n", flush=True)

    captured: dict[str, torch.Tensor] = {}
    hooks = []
    for mod_name, module in model.named_modules():
        if mod_name in common:
            def mk(n):
                def hook(m, inp, out):
                    captured[n] = inp[0].detach()
                return hook
            hooks.append(module.register_forward_hook(mk(mod_name)))

    # energy[adapter][domain] -> list of per-prompt mean ||delta||/||h||
    energy = defaultdict(lambda: defaultdict(list))
    # conflict[(x,y)][domain] -> list of per-prompt mean cosine
    conflict = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for domain, prompts in PROMPTS.items():
            for pi, prompt in enumerate(prompts):
                text = f"### Question:\n{prompt}\n\n### Answer:\n"
                ids = tok(text, return_tensors="pt").input_ids.to(dev)
                captured.clear()
                model(ids)

                # Statistics are accumulated PER MODULE and then averaged.
                # They cannot be concatenated across modules: q_proj/o_proj emit
                # 2560 dims while gate_proj/up_proj emit 9216, so a cat() on the
                # feature axis is both a shape error and conceptually wrong --
                # a cosine is only defined between vectors in the SAME space.
                e_acc: dict[str, list[float]] = {n: [] for n in names}
                c_acc: dict[tuple[str, str], list[float]] = {
                    pr: [] for pr in itertools.combinations(names, 2)
                }
                for mod_name in sorted(common):
                    h = captured.get(mod_name)
                    if h is None:
                        continue
                    hf = h.float().squeeze(0)                 # (seq, in_dim)
                    hnorm = hf.norm(dim=-1).clamp_min(1e-6)   # (seq,)

                    d: dict[str, torch.Tensor] = {}
                    for n in names:
                        A, B = adapters[n][mod_name]
                        d[n] = (hf.to(B.dtype) @ A.t() @ B.t()).float() * scales[n]
                        e_acc[n].append((d[n].norm(dim=-1) / hnorm).mean().item())

                    for x, y in itertools.combinations(names, 2):
                        cos = torch.nn.functional.cosine_similarity(d[x], d[y], dim=-1)
                        c_acc[(x, y)].append(cos.mean().item())

                if not e_acc[names[0]]:
                    continue
                for n in names:
                    energy[n][domain].append(sum(e_acc[n]) / len(e_acc[n]))
                for pr, vals in c_acc.items():
                    conflict[pr][domain].append(sum(vals) / len(vals))

            print(f"  {domain:12s} done ({len(prompts)} prompts)", flush=True)

    for h in hooks:
        h.remove()

    doms = list(PROMPTS)
    mean_energy = {n: {d: sum(v) / len(v) for d, v in energy[n].items()} for n in names}

    print("\n" + "=" * 92)
    print(" ACTIVATION ENERGY  ||delta|| / ||h||    (rows = adapter, cols = domain it is fed)")
    print("=" * 92)
    print(f" {'adapter':10s}" + "".join(f"{d:>13s}" for d in doms) + f"{'ASR':>10s}")
    def get_own_domain(adapter_key: str) -> str:
        for prefix, dom in [("ast", "astral"), ("pg", "postgresql"), ("duck", "duckdb"), ("fin", "financial")]:
            if adapter_key == prefix or adapter_key.startswith(f"{prefix}_"):
                return dom
        raise KeyError(f"Adapter '{adapter_key}' does not map to any known domain in {doms}")

    for n in names:
        row = mean_energy[n]
        own = get_own_domain(n)
        others = [row[d] for d in doms if d != own]
        assert others, f"No cross-domain peers found for {n} (own={own})"
        asr = row[own] / (sum(others) / len(others))
        print(f" {n:10s}" + "".join(f"{row[d]:13.4f}" for d in doms) + f"{asr:10.2f}x")

    print("\n" + "=" * 92)
    print(" CONFLICT  cosine(delta_x, delta_y)  -- per domain. High on a SHARED domain = they fight.")
    print("=" * 92)
    print(f" {'pair':16s}" + "".join(f"{d:>13s}" for d in doms))
    mean_conf = {}
    for (x, y), per in conflict.items():
        mean_conf[f"{x}|{y}"] = {d: sum(v) / len(v) for d, v in per.items()}
        print(f" {x + '+' + y:16s}" + "".join(f"{mean_conf[f'{x}|{y}'][d]:13.4f}" for d in doms))
    print("=" * 92 + "\n")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {"model": args.model_name, "adapters": ADAPTERS,
                   "n_modules": len(common), "prompts": PROMPTS},
        "energy": mean_energy,
        "conflict": mean_conf,
    }, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
