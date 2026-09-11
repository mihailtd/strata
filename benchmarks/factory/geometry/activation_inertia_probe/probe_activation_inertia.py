"""[SUPERSEDED DIAGNOSTIC]
NOTE: This probe is superseded by `probe_stacking_merit.py` (which provides full
per-domain peer breakdown, Merit vs Conflict decomposition, and multi-adapter coverage).

Historical Purpose:
Measures dynamic activation perturbation norms (||delta||_2 = ||(alpha/r) * B @ A @ h||_2)
across in-domain vs cross-domain tokens.
"""

import argparse
import json
import sys
from pathlib import Path
import torch
import torch.nn as nn

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.novel_peft import set_hard_vram_cap


PROMPTS_IN_DOMAIN = {
    "astral": [
        "How do I configure workspace dependencies and lockfiles using uv and pyproject.toml in Python 3.12?",
        "Write an async coroutine using asyncio.TaskGroup and type hints with ruff formatting rules.",
        "Demonstrate how to run isolated Python scripts with inline script metadata PEP 723 using uv run.",
    ],
    "postgresql": [
        "How do I create an HNSW vector index in pgvector with cosine distance ops and query nearest neighbors?",
        "Write a query using DISTINCT ON with window functions and explain the execution plan with EXPLAIN ANALYZE.",
        "Demonstrate how to partition a PostgreSQL 17 table by range with BRIN indexing on timestamp columns.",
    ],
    "duckdb": [
        "How do I query partitioned Parquet files from S3 with projection pushdown and hive_partitioning in DuckDB?",
        "Write a DuckDB query using FROM-first syntax, COLUMNS(*) regex aggregation, and EXCLUDE.",
        "Demonstrate zero-copy export from a DuckDB query to a Polars DataFrame using PyArrow record batches.",
    ],
}

PROMPTS_CROSS_DOMAIN = [
    "Explain the difference between traditional Roth IRA contributions, 401(k) rollovers, and tax brackets.",
    "Calculate the net present value of a multi-period capital investment with inflation-adjusted cash flows.",
    "Discuss the history of the Byzantine Empire and the architectural design of the Hagia Sophia in Constantinople.",
    "Write a short essay analyzing the themes of existentialism and alienation in Franz Kafka's The Metamorphosis.",
]


def hook_lora_layers(model):
    """Registers forward hooks on all LoRA linear layers to capture delta activations."""
    activations = []
    hooks = []

    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and "default" in module.lora_A:
            def make_hook(mod):
                def forward_hook(m, inp, out):
                    # inp[0] is hidden state h: (batch, seq_len, in_dim)
                    h = inp[0]
                    # Compute LoRA delta: delta = (alpha / r) * (h @ A.T @ B.T)
                    A = mod.lora_A["default"].weight.to(h.dtype)  # (r, in_dim)
                    B = mod.lora_B["default"].weight.to(h.dtype)  # (out_dim, r)
                    scale = mod.scaling["default"]
                    h_A = torch.matmul(h, A.t())
                    delta = torch.matmul(h_A, B.t()) * scale
                    token_norms = torch.norm(delta.float(), p=2, dim=-1)
                    activations.append(token_norms.squeeze(0).detach().cpu())
                return forward_hook

            h_obj = module.register_forward_hook(make_hook(module))
            hooks.append(h_obj)

    return activations, hooks


@torch.no_grad()
def measure_activation_norms(model, tok, adapter_path: str, domain: str = "astral"):
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    activations, hooks = hook_lora_layers(model)

    in_prompts = PROMPTS_IN_DOMAIN.get(domain, PROMPTS_IN_DOMAIN["astral"])
    out_prompts = PROMPTS_CROSS_DOMAIN

    # 1. In-Domain Measurement
    in_norms = []
    for p in in_prompts:
        activations.clear()
        text = f"### Question:\n{p}\n\n### Answer:\n"
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        _ = model(ids)
        # Average norm across all layers and token positions
        if activations:
            mean_step = torch.stack([a.mean() for a in activations]).mean().item()
            in_norms.append(mean_step)

    # 2. Out-of-Domain Measurement
    out_norms = []
    for p in out_prompts:
        activations.clear()
        text = f"### Question:\n{p}\n\n### Answer:\n"
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        _ = model(ids)
        if activations:
            mean_step = torch.stack([a.mean() for a in activations]).mean().item()
            out_norms.append(mean_step)

    for h in hooks:
        h.remove()

    mean_in = sum(in_norms) / max(1, len(in_norms))
    mean_out = sum(out_norms) / max(1, len(out_norms))
    asr = mean_in / max(1e-6, mean_out)
    suppression_pct = max(0.0, (1.0 - mean_out / max(1e-6, mean_in))) * 100.0

    return {
        "adapter": Path(adapter_path).name,
        "domain": domain,
        "mean_in_domain_norm": mean_in,
        "mean_out_domain_norm": mean_out,
        "activation_selectivity_ratio (ASR)": asr,
        "out_of_domain_suppression_pct": suppression_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Activation Inertia & ASR Probe")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-dir", default="results/adapters/m2_astral_r8a128_v4")
    parser.add_argument("--domain", default="astral", choices=["astral", "postgresql", "duckdb"])
    args = parser.parse_args()

    set_hard_vram_cap(22.0)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    base_model.eval()

    res = measure_activation_norms(base_model, tok, args.adapter_dir, args.domain)
    print("\n" + "=" * 70)
    print(f" 🔬 ACTIVATION INERTIA & SELECTIVITY PROBE: {res['adapter']}")
    print("=" * 70)
    print(f" Target Domain                : {res['domain']}")
    print(f" In-Domain Activation Norm    : {res['mean_in_domain_norm']:.4f}")
    print(f" Out-of-Domain Activation Norm: {res['mean_out_domain_norm']:.4f}")
    print(f" Activation Selectivity (ASR) : {res['activation_selectivity_ratio (ASR)']:.2f}x")
    print(f" Out-of-Domain Noise Filtered : {res['out_of_domain_suppression_pct']:.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
