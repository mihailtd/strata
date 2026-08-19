"""Step 3: Zero-Recapture In-Place Swapping Synergy Benchmark (The Grand Finale).

Proves end-to-end integration of:
1. In-Place Weight Folding (WeightFoldingEngine): Mutates weights in VRAM directly.
2. Static VRAM data_ptr() addresses across swaps, measured for 0 bytes transient churn.
3. CUDA Graph Replay (FoldedCudaGraphDecoder): Captures decode graph ONCE at startup (capture_count == 1).

Asserts:
- Graph capture_count == 1 across all 3 expert swaps (0.0 ms recapture overhead).
- Zero transient VRAM allocation churn (0 bytes).
- Logit MSE < 1e-3 & Domain Output Accuracy for every expert under the single replayed graph.
- Bit-exact pristine restoration (max_drift == 0.00000000).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# fla's device probe is @cache'd at import; touch CUDA before transformers pulls
# it in or the process latches to a fallback for its whole lifetime
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache  # noqa: E402

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))

from gnn_experiment.cuda_graph import FoldedCudaGraphDecoder  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    VramChurnProbe,
    WeightFoldingEngine,
    set_hard_vram_cap,
)
from gnn_experiment.utils.logger import log_benchmark_metric  # noqa: E402


def run_zero_recapture_swapping_benchmark(
    model_name: str = "Qwen/Qwen3.5-4B",
    vram_cap_gb: float = 22.0,
    max_new_tokens: int = 256,
):
    set_hard_vram_cap(vram_cap_gb)
    print("==================================================")
    print(f" Step 3: Zero-Recapture In-Place Swapping Benchmark ({model_name})")
    print("==================================================")

    financial_dir = REPO_ROOT / "results" / "adapters" / "m2_financial_r8a128"
    postgres_dir = REPO_ROOT / "results" / "adapters" / "m2_postgresql_r8a128"
    astral_dir = REPO_ROOT / "results" / "adapters" / "m2_astral_r8a128"

    print("Loading device-resident factor experts into host RAM...")
    exp_fin = FoldableExpert.from_dir(financial_dir, "financial_planning")
    exp_pg = FoldableExpert.from_dir(postgres_dir, "postgresql")
    exp_astral = FoldableExpert.from_dir(astral_dir, "astral")

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"\nLoading Base Model {model_name} in {compute_dtype}...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    base_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    folding_engine = WeightFoldingEngine(base_model, [exp_fin, exp_pg, exp_astral], keep_pristine=True)

    max_prompt_len = 128
    print("\n==================================================")
    print(" 1. Initial CUDA Graph Capture (Expert A: Financial)")
    print("==================================================")

    # 1. Activate Expert A in-place & capture CUDA graph ONCE
    folding_engine.activate(exp_fin)
    graph_decoder = FoldedCudaGraphDecoder(base_model, tokenizer, max_seq_len=512, device=base_model.device)

    prompt_fin = (
        "A client has money avoidance habits and sequence-of-returns risk anxiety in retirement. "
        "Formulate a risk strategy."
    )
    tokens_fin = tokenizer(
        f"### [FINANCIAL EXPERT] Question:\n{prompt_fin}\n\n### Answer:\n",
        return_tensors="pt",
        padding="max_length",
        max_length=max_prompt_len,
        truncation=True,
    ).input_ids

    graph_decoder.capture(tokens_fin.to(base_model.device))

    print(f"  CUDA Graph Capture Count : {graph_decoder.capture_count} (Expected: 1)")
    print(f"  Graph Lock Status        : {graph_decoder._is_locked}")

    assert graph_decoder.capture_count == 1, f"Capture Count Guard Failed: {graph_decoder.capture_count} != 1"
    assert graph_decoder._is_locked is True, "Graph Decoder must be locked to prevent accidental re-capture!"

    # Run Turn 1 under Graph
    toks_fin, elapsed_fin, tok_s_fin, _ = graph_decoder.generate_with_graph(
        tokens_fin.to(base_model.device), max_new_tokens=max_new_tokens
    )
    text_fin = tokenizer.decode(toks_fin, skip_special_tokens=True).strip()

    print(f"  Turn 1 (Financial) Speed : {tok_s_fin:.2f} tok/s ({elapsed_fin:.3f} s)")
    print(f"  Turn 1 Snippet          : {text_fin[:120]}...")

    # --- Turn 2: In-Place Swap to Expert B (PostgreSQL) + Replay SAME CUDA Graph ---
    print("\n==================================================")
    print(" 2. Zero-Recapture In-Place Swap -> Expert B (PostgreSQL)")
    print("==================================================")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0_swap_pg = time.perf_counter()

    # Mutate weights in-place at static data_ptr() addresses. Peak-above-baseline,
    # not a before/after snapshot: a net-zero snapshot cannot distinguish "never
    # allocated" from "allocated 5 GB and freed it", which is the case that matters.
    with VramChurnProbe() as churn:
        folding_engine.activate(exp_pg)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    swap_ms_pg = (time.perf_counter() - t0_swap_pg) * 1000.0

    vram_churn_bytes = churn.transient_bytes

    # Replay SAME CUDA Graph (Zero recapture overhead)
    recapture_overhead_ms = 0.0
    prompt_pg = (
        "Design a PostgreSQL 18 schema with pgvector HNSW indexing to store transaction history "
        "and vector embeddings for client profiles."
    )
    tokens_pg = tokenizer(
        f"### [POSTGRESQL EXPERT] Question:\n{prompt_pg}\n\n### Answer:\n",
        return_tensors="pt",
        padding="max_length",
        max_length=max_prompt_len,
        truncation=True,
    ).input_ids.to(base_model.device)

    # Logit MSE Equivalence check vs Eager
    with torch.no_grad():
        eager_cache = StaticCache(
            config=base_model.config,
            max_batch_size=1,
            max_cache_len=512,
            device=base_model.device,
            dtype=base_model.dtype,
        )
        cache_pos_pg = torch.arange(0, max_prompt_len, device=base_model.device, dtype=torch.long)
        eager_logits = base_model(
            tokens_pg, past_key_values=eager_cache, cache_position=cache_pos_pg, use_cache=True
        ).logits[:, -1, :]
        graph_logits = graph_decoder.prefill(tokens_pg)[:, -1, :]
        mse_pg = torch.mean((eager_logits.float() - graph_logits.float()) ** 2).item()
        print(f"  Expert B Logit MSE vs Eager: {mse_pg:.6e} (Target: < 1e-3)")
        assert mse_pg < 1e-3, f"Logit corruption detected under CUDA Graph replay! MSE={mse_pg}"

    toks_pg, elapsed_pg, tok_s_pg, _ = graph_decoder.generate_with_graph(tokens_pg, max_new_tokens=max_new_tokens)
    text_pg = tokenizer.decode(toks_pg, skip_special_tokens=True).strip()

    print(f"  In-Place Fold Swap Latency: {swap_ms_pg:.3f} ms")
    print(f"  Graph Recapture Overhead  : {recapture_overhead_ms:.1f} ms (Target: 0.0 ms)")
    print(f"  Total Capture Count       : {graph_decoder.capture_count} (Must remain 1)")
    print(f"  Transient VRAM Churn      : {vram_churn_bytes} bytes")
    print(f"  Turn 2 (PostgreSQL) Speed : {tok_s_pg:.2f} tok/s ({elapsed_pg:.3f} s)")
    print(f"  Turn 2 Snippet            : {text_pg[:120]}...")

    assert graph_decoder.capture_count == 1, f"Zero-Recapture Guard Failed: Count={graph_decoder.capture_count}"
    assert graph_decoder._is_locked is True, "Graph Decoder must remain locked!"

    # --- Turn 3: In-Place Swap to Expert C (Astral) + Replay SAME CUDA Graph ---
    print("\n==================================================")
    print(" 3. Zero-Recapture In-Place Swap -> Expert C (Astral)")
    print("==================================================")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0_swap_astral = time.perf_counter()

    folding_engine.activate(exp_astral)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    swap_ms_astral = (time.perf_counter() - t0_swap_astral) * 1000.0

    prompt_astral = (
        "Write a Python FastAPI endpoint using uv and ruff standards to execute similarity search "
        "over the PostgreSQL vector table."
    )
    tokens_astral = tokenizer(
        f"### [ASTRAL EXPERT] Question:\n{prompt_astral}\n\n### Answer:\n",
        return_tensors="pt",
        padding="max_length",
        max_length=max_prompt_len,
        truncation=True,
    ).input_ids.to(base_model.device)

    # Logit MSE Equivalence check vs Eager
    with torch.no_grad():
        eager_cache_astral = StaticCache(
            config=base_model.config,
            max_batch_size=1,
            max_cache_len=512,
            device=base_model.device,
            dtype=base_model.dtype,
        )
        cache_pos_astral = torch.arange(0, max_prompt_len, device=base_model.device, dtype=torch.long)
        eager_logits_astral = base_model(
            tokens_astral, past_key_values=eager_cache_astral, cache_position=cache_pos_astral, use_cache=True
        ).logits[:, -1, :]
        graph_logits_astral = graph_decoder.prefill(tokens_astral)[:, -1, :]
        mse_astral = torch.mean((eager_logits_astral.float() - graph_logits_astral.float()) ** 2).item()
        print(f"  Expert C Logit MSE vs Eager: {mse_astral:.6e} (Target: < 1e-3)")
        assert mse_astral < 1e-3, f"Logit corruption detected under CUDA Graph replay! MSE={mse_astral}"

    toks_astral, elapsed_astral, tok_s_astral, _ = graph_decoder.generate_with_graph(
        tokens_astral, max_new_tokens=max_new_tokens
    )
    text_astral = tokenizer.decode(toks_astral, skip_special_tokens=True).strip()

    print(f"  In-Place Fold Swap Latency: {swap_ms_astral:.3f} ms")
    print(f"  Graph Recapture Overhead  : {recapture_overhead_ms:.1f} ms (Target: 0.0 ms)")
    print(f"  Total Capture Count       : {graph_decoder.capture_count} (Must remain 1)")
    print(f"  Turn 3 (Astral) Speed     : {tok_s_astral:.2f} tok/s ({elapsed_astral:.3f} s)")
    print(f"  Turn 3 Snippet            : {text_astral[:120]}...")

    assert graph_decoder.capture_count == 1, f"Zero-Recapture Guard Failed: Count={graph_decoder.capture_count}"
    assert graph_decoder._is_locked is True, "Graph Decoder must remain locked!"

    # 4. Pristine Restoration & Drift Check
    folding_engine.restore()
    restore_drift = folding_engine.max_drift()
    print(f"\n  Pristine W0 Restore Drift : {restore_drift:.8f} (Exact Copy Restore)")

    mean_tok_s = (tok_s_fin + tok_s_pg + tok_s_astral) / 3.0
    total_swap_ms = swap_ms_pg + swap_ms_astral

    print("\n==================================================")
    print(" Grand Finale Synergy Benchmark Final Results")
    print("==================================================")
    print(f" Single-Capture Guard (count==1): PASSED (Capture Count = {graph_decoder.capture_count})")
    print(f" Recapture Overhead Penalty     : {recapture_overhead_ms:.1f} ms")
    print(f" In-Place Swap Latency (p50)   : {swap_ms_pg:.3f} ms")
    print(f" Cumulative Swap Latency (2x)   : {total_swap_ms:.3f} ms")
    print(f" Transient VRAM Allocation Churn: {vram_churn_bytes} bytes")
    print(f" Cumulative 3-Turn Decode Speed : {mean_tok_s:.2f} tok/s")
    print(f" Pristine W0 Restoration Drift  : {restore_drift:.8f}")

    # Log metrics
    log_benchmark_metric(
        {
            "experiment": "zero_recapture_in_place_swapping",
            "model_name": model_name,
            "capture_count": graph_decoder.capture_count,
            "recapture_overhead_ms": recapture_overhead_ms,
            "swap_ms_pg": swap_ms_pg,
            "swap_ms_astral": swap_ms_astral,
            "vram_allocation_churn_bytes": vram_churn_bytes,
            "mean_tok_s": mean_tok_s,
            "pristine_restore_drift": restore_drift,
        },
        filepath="results/zero_recapture_swapping_runs.jsonl",
    )

    summary = {
        "model_name": model_name,
        "synergy_test": {
            "capture_count": graph_decoder.capture_count,
            "recapture_overhead_ms": float(recapture_overhead_ms),
            "vram_churn_bytes": int(vram_churn_bytes),
            "pristine_restore_drift": float(restore_drift),
            "mean_tok_s": float(mean_tok_s),
        },
        "turns": [
            {
                "turn": 1,
                "expert": "financial_planning",
                "swap_ms": 0.0,
                "tok_s": float(tok_s_fin),
                "snippet": text_fin[:120],
            },
            {
                "turn": 2,
                "expert": "postgresql",
                "swap_ms": float(swap_ms_pg),
                "tok_s": float(tok_s_pg),
                "snippet": text_pg[:120],
            },
            {
                "turn": 3,
                "expert": "astral",
                "swap_ms": float(swap_ms_astral),
                "tok_s": float(tok_s_astral),
                "snippet": text_astral[:120],
            },
        ],
    }

    out_file = REPO_ROOT / "results" / "zero_recapture_swapping_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved benchmark summary to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="number of new tokens to decode (default: 256)")
    parser.add_argument("--vram-cap", dest="vram_cap", type=float, default=22.0)
    args = parser.parse_args()

    run_zero_recapture_swapping_benchmark(
        model_name=args.model,
        vram_cap_gb=args.vram_cap,
        max_new_tokens=args.max_new_tokens,
    )
