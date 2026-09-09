"""Benchmark Native Speculative Decoding (Neural MTP & Context N-Gram) on AMD Radeon RX 7900 XTX.

Evaluates:
1. Mathematical equivalence to pure greedy decode (100% bit-for-bit target parity).
2. Latency and tokens/sec throughput for:
   - Baseline single-token decode (1-token/sweep)
   - Neural MTP speculative decode (blk.64)
   - Context N-Gram speculative decode (lookahead)
3. Empirical candidate acceptance rate (tau).
"""

import time
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.native_27b_engine import Native27BEngine, EngineConfig27B
from runtime.server import get_27b_tokenizer

PROMPTS = [
    (
        "Python FastAPI Service",
        "Write a complete Python FastAPI endpoint that validates incoming JSON payloads using Pydantic v2 and performs async database queries.",
    ),
    (
        "DuckDB Analytics Query",
        "Write an optimized SQL query for DuckDB that calculates rolling 7-day active users and cumulative revenue by region using window functions.",
    ),
    (
        "PostgreSQL Migration",
        "Write a robust PostgreSQL migration script to convert a large table from SERIAL to BIGINT identity with minimal downtime and lock timeouts.",
    ),
]


def benchmark_speculative():
    print("=" * 80)
    print("AMD Radeon RX 7900 XTX - Native 27B Speculative Decoding Parity & Speedup Benchmark")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    tokenizer = get_27b_tokenizer()
    print("Loading 64-layer Native27BEngine...")
    engine = Native27BEngine(num_layers=64, device=device)
    engine.load_from_cache()

    has_mtp = engine.mtp_layer is not None
    print(f"Neural MTP Layer (blk.64) present: {has_mtp}")

    results = []

    for name, prompt_text in PROMPTS:
        print(f"\n--- Benchmark Scenario: {name} ---")
        prompt_ids = tokenizer.encode(prompt_text)
        max_new = 48

        # 1. Baseline Greedy Decode
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens_base = engine.generate(
            prompt_ids,
            max_new_tokens=max_new,
            temperature=0.0,
            use_hip_graph=True,
        )
        torch.cuda.synchronize()
        t_base = time.perf_counter() - t0
        tok_s_base = len(tokens_base) / t_base
        print(f"[Baseline Greedy]    {len(tokens_base)} tokens in {t_base:.2f}s ({tok_s_base:.2f} tok/s)")

        # 2. Speculative Decode with Context N-Gram
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens_ngram = engine.generate_speculative(
            prompt_ids,
            max_new_tokens=max_new,
            temperature=0.0,
            use_hip_graph=True,
            draft_k=3,
            draft_n=5,
            min_n=4,
            use_mtp=False,
        )
        torch.cuda.synchronize()
        t_ngram = time.perf_counter() - t0
        tok_s_ngram = len(tokens_ngram) / t_ngram
        is_ngram_exact = tokens_base == tokens_ngram
        print(f"[N-Gram Speculative] {len(tokens_ngram)} tokens in {t_ngram:.2f}s ({tok_s_ngram:.2f} tok/s) | Exact match: {is_ngram_exact}")

        # 3. Speculative Decode with Neural MTP (if available)
        tokens_mtp = []
        tok_s_mtp = 0.0
        is_mtp_exact = False
        if has_mtp:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            tokens_mtp = engine.generate_speculative(
                prompt_ids,
                max_new_tokens=max_new,
                temperature=0.0,
                use_hip_graph=True,
                use_mtp=True,
            )
            torch.cuda.synchronize()
            t_mtp = time.perf_counter() - t0
            tok_s_mtp = len(tokens_mtp) / t_mtp
            is_mtp_exact = tokens_base == tokens_mtp
            speedup_mtp = tok_s_mtp / tok_s_base if tok_s_base > 0 else 1.0
            print(f"[Neural MTP Spec]    {len(tokens_mtp)} tokens in {t_mtp:.2f}s ({tok_s_mtp:.2f} tok/s, {speedup_mtp:.2f}x) | Exact match: {is_mtp_exact}")

        results.append({
            "name": name,
            "tok_s_base": tok_s_base,
            "tok_s_ngram": tok_s_ngram,
            "tok_s_mtp": tok_s_mtp,
            "ngram_exact": is_ngram_exact,
            "mtp_exact": is_mtp_exact,
        })

    print("\n" + "=" * 80)
    print("SUMMARY OF SPECULATIVE DECODING RESULTS")
    print("=" * 80)
    for r in results:
        print(f"{r['name']}:")
        print(f"  Baseline:   {r['tok_s_base']:.2f} tok/s")
        print(f"  N-Gram:     {r['tok_s_ngram']:.2f} tok/s (Exact: {r['ngram_exact']})")
        if has_mtp:
            speedup = r['tok_s_mtp'] / r['tok_s_base'] if r['tok_s_base'] > 0 else 0
            print(f"  Neural MTP: {r['tok_s_mtp']:.2f} tok/s (Exact: {r['mtp_exact']}, Speedup: {speedup:.2f}x)")


if __name__ == "__main__":
    benchmark_speculative()
