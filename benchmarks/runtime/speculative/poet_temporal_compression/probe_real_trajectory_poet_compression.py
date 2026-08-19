r"""Real-Model Trajectory POET Dynamic Factor Compression & Logit Concordance Benchmark.

1. Generates 512-token realistic coding trajectories from Qwen3.5-4B.
2. Captures exact hidden states H in R^{T x d} across decoder layers [0, 8, 16, 24, 31].
3. Applies POET Dynamic Factor Model decomposition: H = F * Lambda^T + S.
4. Measures:
   - True singular value spectrum and variance explained on real language activations.
   - Compression Ratio vs Cosine Fidelity cos(h_t, \hat{h}_t).
   - Downstream Top-1 Token Logits Agreement & KL Divergence through final RMSNorm + LM Head.

Usage:
    CUDA_VISIBLE_DEVICES="" uv run python -u benchmarks/runtime/speculative/poet_temporal_compression/probe_real_trajectory_poet_compression.py [--out results/benchmarks/real_trajectory_poet_compression.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from gnn_experiment.canon import CANON, REPO_ROOT

torch.set_num_threads(2)


def capture_real_generation_trajectories(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    target_layers: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Generates autoregressive tokens and captures hidden state trajectories at target layers."""
    if target_layers is None:
        target_layers = [0, 8, 16, 24, 31]

    captured_states: dict[int, list[torch.Tensor]] = {l: [] for l in target_layers}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            # Output can be a tensor or tuple (hidden_states, ...)
            h = output[0] if isinstance(output, tuple) else output
            # Extract last token hidden state [batch, seq, dim] -> [dim]
            last_h = h[:, -1, :].detach().to(torch.float32).cpu()
            captured_states[layer_idx].append(last_h.squeeze(0))
        return hook

    # Register hooks on decoder layers
    for layer_idx in target_layers:
        hook_handle = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook_handle)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"  Generating {max_new_tokens} tokens to capture hidden state trajectories...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_time = time.time() - t0
    n_tokens = out.shape[1] - inputs.input_ids.shape[1]
    print(f"  Captured {n_tokens} tokens across {len(target_layers)} layers in {gen_time:.2f}s", flush=True)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Stack captured trajectories into H in R^{T x d}
    trajectories = {}
    for layer_idx, state_list in captured_states.items():
        if state_list:
            trajectories[layer_idx] = torch.stack(state_list, dim=0)  # [T, d]

    return trajectories


def decompose_poet(
    H: torch.Tensor,
    rank: int = 4,
    sparsity_target: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Applies POET Dynamic Factor Model decomposition: H = F * Lambda^T + S."""
    T, d = H.shape
    norm_orig = float(torch.norm(H).item()) + 1e-12

    # SVD for optimal low-rank factor decomposition
    U, S_vals, Vh = torch.linalg.svd(H, full_matrices=False)
    k = min(rank, len(S_vals), T)

    F_scores = U[:, :k] * S_vals[:k]  # [T, k]
    Lambda = Vh[:k, :].T              # [d, k]
    H_lowrank = F_scores @ Lambda.T   # [T, d]

    # Residual error
    E = H - H_lowrank  # [T, d]
    flat_abs = torch.abs(E).flatten()
    total_elements = flat_abs.numel()

    if sparsity_target > 0.0:
        k_elements = int(total_elements * sparsity_target)
        if k_elements > 0:
            threshold = torch.kthvalue(flat_abs, total_elements - k_elements + 1).values.item()
            S_sparse = torch.where(torch.abs(E) >= threshold, E, torch.zeros_like(E))
        else:
            S_sparse = torch.zeros_like(E)
    else:
        S_sparse = torch.zeros_like(E)

    H_reconstructed = H_lowrank + S_sparse

    # Metrics
    frob_error = float((torch.norm(H - H_reconstructed) / norm_orig).item())
    
    # Cosine similarity per time step
    cos_sims = F.cosine_similarity(H, H_reconstructed, dim=-1)
    mean_cos = float(torch.mean(cos_sims).item())
    min_cos = float(torch.min(cos_sims).item())

    # Memory Footprint
    dense_bytes = T * d * 2  # FP16 bytes
    sparse_entries = int(torch.sum(S_sparse != 0).item())
    poet_bytes = (T * k + d * k) * 2 + (sparse_entries * 6)  # float16 factor + (float16 val + int32 index)
    comp_ratio = dense_bytes / max(1, poet_bytes)

    # Variance explained by top k singular values
    total_var = float(torch.sum(S_vals**2).item())
    top_k_var = float(torch.sum(S_vals[:k]**2).item())
    var_explained = top_k_var / max(1e-12, total_var)

    meta = {
        "horizon_T": T,
        "dim_d": d,
        "rank": k,
        "sparsity_pct": float(sparse_entries / total_elements * 100.0),
        "var_explained": var_explained,
        "frob_error": frob_error,
        "mean_cosine": mean_cos,
        "min_cosine": min_cos,
        "dense_kb": dense_bytes / 1024.0,
        "poet_kb": poet_bytes / 1024.0,
        "compression_ratio": comp_ratio,
    }

    return F_scores, Lambda, S_sparse, meta


def evaluate_logit_concordance(
    model: AutoModelForCausalLM,
    H_dense: torch.Tensor,
    H_poet: torch.Tensor,
) -> dict[str, float]:
    """Computes Top-1 token agreement and KL divergence through final RMSNorm + LM Head."""
    # H: [T, d]
    with torch.no_grad():
        # Pass through model norm and lm_head
        h_dense_dev = H_dense.to(dtype=torch.bfloat16, device=model.device)
        h_poet_dev = H_poet.to(dtype=torch.bfloat16, device=model.device)

        normed_dense = model.model.norm(h_dense_dev)
        normed_poet = model.model.norm(h_poet_dev)

        logits_dense = model.lm_head(normed_dense).float()  # [T, vocab]
        logits_poet = model.lm_head(normed_poet).float()    # [T, vocab]

        # Top-1 token predictions
        top1_dense = torch.argmax(logits_dense, dim=-1)
        top1_poet = torch.argmax(logits_poet, dim=-1)

        top1_match = float(torch.mean((top1_dense == top1_poet).float()).item())

        # Top-5 agreement
        top5_dense = torch.topk(logits_dense, k=5, dim=-1).indices
        top5_poet = torch.topk(logits_poet, k=5, dim=-1).indices
        # Jaccard / Overlap across top-5
        overlap_count = 0
        for t in range(H_dense.shape[0]):
            s1 = set(top5_dense[t].tolist())
            s2 = set(top5_poet[t].tolist())
            overlap_count += len(s1 & s2) / 5.0
        top5_overlap = overlap_count / H_dense.shape[0]

        # KL divergence
        p_dense = F.softmax(logits_dense, dim=-1)
        log_p_poet = F.log_softmax(logits_poet, dim=-1)
        log_p_dense = F.log_softmax(logits_dense, dim=-1)
        kl_div = float(F.kl_div(log_p_poet, p_dense, log_target=False, reduction="batchmean").item())

    return {
        "top1_agreement": top1_match,
        "top5_overlap": top5_overlap,
        "kl_divergence": kl_div,
    }


def run_benchmark() -> dict[str, Any]:
    print("=" * 95, flush=True)
    print(" REAL-MODEL TRAJECTORY POET DYNAMIC FACTOR COMPRESSION BENCHMARK", flush=True)
    print(" Evaluating True Qwen3.5-4B Autoregressive State Sequences", flush=True)
    print("=" * 95, flush=True)

    # 1. Load Base Model
    print("\n[1/4] Loading base model and tokenizer...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(CANON.BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CANON.BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print(f"  Base model loaded in {time.time()-t0:.2f}s", flush=True)

    # 2. Capture Real Trajectories
    prompt = (
        "Write a high-performance asynchronous connection pool manager in Python for PostgreSQL using asyncpg. "
        "Include health check probes, automatic retry with exponential backoff, circuit breaker state machine, "
        "and comprehensive type annotations."
    )
    print("\n[2/4] Capturing real hidden-state activation trajectories (T=256 tokens)...", flush=True)
    target_layers = [0, 8, 16, 24, 31]
    trajectories = capture_real_generation_trajectories(
        model, tokenizer, prompt, max_new_tokens=256, target_layers=target_layers
    )

    # 3. POET Factor Decomposition Sweep
    print("\n[3/4] Running POET Factor Decomposition across Ranks & Sparsity Levels...", flush=True)
    ranks = [2, 4, 8, 16]
    sparsities = [0.0, 0.01, 0.02, 0.05]
    
    sweep_results = []

    for layer_idx, H in trajectories.items():
        print(f"\n--- Layer {layer_idx} (T={H.shape[0]}, d={H.shape[1]}) ---", flush=True)
        for r in ranks:
            for rho in sparsities:
                F_sc, Lam, S_sp, meta = decompose_poet(H, rank=r, sparsity_target=rho)
                
                # If last layer (L31), compute downstream logit concordance
                if layer_idx == 31:
                    H_recon = F_sc @ Lam.T + S_sp
                    logit_meta = evaluate_logit_concordance(model, H, H_recon)
                    meta |= logit_meta
                else:
                    logit_meta = {}

                sweep_results.append({"layer": layer_idx} | meta)

                logit_str = f" | Top1: {meta.get('top1_agreement', 0.0)*100:.1f}% | KL: {meta.get('kl_divergence', 0.0):.4f}" if layer_idx == 31 else ""
                print(
                    f"  Rank {r:2d}, Sparsity {rho*100:4.1f}% -> Comp: {meta['compression_ratio']:5.1f}x | "
                    f"Var: {meta['var_explained']*100:5.1f}% | Cos: {meta['mean_cosine']:.4f} (min: {meta['min_cosine']:.4f}){logit_str}",
                    flush=True,
                )

    # 4. Summary Table for Layer 31 (Final Logits Interface)
    print("\n" + "=" * 95, flush=True)
    print(f" LAYER 31 SPECULATIVE ROLLBACK FIDELITY (Real Qwen3.5-4B Trajectory, d=2560)", flush=True)
    print(" " + "─" * 93, flush=True)
    print(f" {'Rank (r)':<10} {'Sparsity (%)':<14} {'Comp Ratio':<12} {'Var Expl (%)':<14} {'Cosine':<10} {'Top-1 Match':<14} KL Div", flush=True)
    print(" " + "─" * 93, flush=True)
    for res in sweep_results:
        if res["layer"] == 31:
            r_str = f"r = {res['rank']}"
            sp_str = f"{res['sparsity_pct']:.1f}%"
            c_str = f"{res['compression_ratio']:.1f}x"
            v_str = f"{res['var_explained']*100:.1f}%"
            cos_str = f"{res['mean_cosine']:.4f}"
            t1_str = f"{res['top1_agreement']*100:.1f}%"
            kl_str = f"{res['kl_divergence']:.4f}"
            print(f" {r_str:<10} {sp_str:<14} {c_str:<12} {v_str:<14} {cos_str:<10} {t1_str:<14} {kl_str}", flush=True)
    print("=" * 95, flush=True)

    return {
        "metadata": CANON.stamp(),
        "prompt": prompt,
        "target_layers": target_layers,
        "sweep_results": sweep_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="results/benchmarks/real_trajectory_poet_compression.json",
        help="Output JSON artifact path.",
    )
    args = parser.parse_args()

    t_start = time.time()
    results = run_benchmark()
    elapsed = time.time() - t_start

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results | {"elapsed_seconds": elapsed}, indent=2))
    print(f"\n[Artifact] → {out_path}  (elapsed: {elapsed:.1f}s)\n", flush=True)


if __name__ == "__main__":
    main()
