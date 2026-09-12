#!/usr/bin/env python3
"""Multi-Turn Dashboard Benchmark Suite for Speculative Decoding & LoRA Weight Folding.

Simulates sequential user clicks on the 4 dashboard domain prompts in a single context window:
  1. Astral (Python tooling / ruff & ty)
  2. PostgreSQL (pgvector hnsw indexing)
  3. FastAPI (Modern Python web async endpoints)
  4. DuckDB (Parquet analytics & window functions)

Evaluates Turn-by-Turn in a Shared Context Window:
  - Mode A: Eager Greedy Baseline (Autoregressive single-step greedy forward)
  - Mode B: BucketedSpeculativeDecoder (K=2, StaticCache + CUDA/HIP Graphs + Selective State Ring Buffer)

Metrics:
  - Exact token match / Fidelity (100% Bit-Exact target)
  - End-to-end latency (seconds)
  - Throughput (tokens/second)
  - Speculative draft acceptance rate (tau)
"""

from __future__ import annotations

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.bucketed_speculative import BucketedSpeculativeDecoder
from runtime.mtp_draft import Qwen35MTPDraftHead
from runtime.novel_peft import FoldableExpert, WeightFoldingEngine
from runtime.server import format_prompt, ChatMessage
from runtime.canon import REPO_ROOT

PREDEFINED_PROMPTS = [
    ("astral", "Add ruff and ty as dev dependencies, then format and lint the whole codebase."),
    ("postgresql", "We store product descriptions in Postgres and want 'find me similar products' without standing up new infrastructure."),
    ("python_web", "Write an async FastAPI endpoint with Pydantic request and response models and dependency injection."),
    ("duckdb", "Aggregate a directory of parquet files and return the top 3 rows per group."),
]

def eager_greedy_generate(model, input_ids: torch.Tensor, max_new_tokens: int, stop_ids: set[int],
                           cache=None) -> tuple[list[int], float]:
    """Ground-truth greedy autoregressive generation using StaticCache incremental steps.

    Uses the SAME incremental computation path as the speculative decoder: the SAME
    StaticCache instance (``cache``) that the speculative decoder uses (and whose
    SSM recurrent-state buffers are already bound to the model's GatedDeltaNet layers).
    This ensures that both the baseline and spec decoding run in identical SSM
    recurrent-step mode.

    Important: the caller MUST reset ``cache`` before calling this function so that
    the baseline starts from a clean state. This function does NOT reset the cache
    itself, because the caller may want to inspect or share the cache.
    """
    device = input_ids.device
    cur = input_ids.shape[1]
    gen_tokens: list[int] = []

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        # Prefill
        out = model(
            input_ids,
            past_key_values=cache,
            cache_position=torch.arange(0, cur, device=device),
            use_cache=True,
        )
        nxt = torch.argmax(out.logits[0, -1, :]).item()
        gen_tokens.append(nxt)
        pos = cur

        if nxt not in stop_ids:
            for _ in range(max_new_tokens - 1):
                tok_t = torch.tensor([[nxt]], device=device, dtype=torch.long)
                cache_pos = torch.tensor([pos], device=device)
                out = model(tok_t, past_key_values=cache, cache_position=cache_pos, use_cache=True)
                nxt = torch.argmax(out.logits[0, -1, :]).item()
                gen_tokens.append(nxt)
                pos += 1
                if nxt in stop_ids:
                    break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return gen_tokens, elapsed



def run_benchmark():
    model_id = "Qwen/Qwen3.5-9B"
    print("=" * 90)
    print(f"MULTI-TURN CHAT SPECULATIVE DECODING FIDELITY BENCHMARK: {model_id}")
    print("=" * 90)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    print(f"Loading tokenizer & model ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    ).eval()

    # Load All 4 Experts
    adapter_paths = {
        "astral": REPO_ROOT / "results" / "adapters" / "m2_astral_r8a128_v7_9b",
        "postgresql": REPO_ROOT / "results" / "adapters" / "m2_postgresql_r8a128_v7_9b",
        "python_web": REPO_ROOT / "results" / "adapters" / "m2_python_web_r8a128_v7_9b",
        "duckdb": REPO_ROOT / "results" / "adapters" / "m2_duckdb_r8a128_v7_9b",
    }
    experts = [
        FoldableExpert.from_dir(path, name)
        for name, path in adapter_paths.items()
        if path.exists()
    ]
    print(f"Loaded {len(experts)} foldable domain experts: {[e.name for e in experts]}")

    engine = WeightFoldingEngine(model, experts, keep_pristine=True)
    draft_head = Qwen35MTPDraftHead(model, model_id).eval()
    engine.register_draft_head(draft_head)

    # Initial activation for graph capture
    engine.activate(experts[0])

    print("\nInitializing and capturing BucketedSpeculativeDecoder (K=2, max_seq_len=2048)...")
    spec = BucketedSpeculativeDecoder(model, tokenizer, draft_head, k=2, max_seq_len=2048)
    dummy_prompt = format_prompt([ChatMessage(role="user", content="Hello")], thinking_effort="off")
    dummy_ids = tokenizer.encode(dummy_prompt, return_tensors="pt").to(device)
    spec.capture(dummy_ids)
    print("Graph capture complete.\n")

    stop_ids = {tokenizer.eos_token_id}
    for special_token in ("<|im_end|>", "<|endoftext|>"):
        s_id = tokenizer.convert_tokens_to_ids(special_token)
        if isinstance(s_id, int) and s_id > 0:
            stop_ids.add(s_id)

    # --------------------------------------------------------------------------
    # SEQUENTIAL MULTI-TURN EVALUATION
    # --------------------------------------------------------------------------
    history: list[ChatMessage] = []
    comparison_results: list[dict] = []

    print("=" * 90)
    print("EXECUTING MULTI-TURN CHAT (4 TURNS IN SHARED CONTEXT WINDOW)")
    print("=" * 90)

    for turn_idx, (expert_name, text) in enumerate(PREDEFINED_PROMPTS, start=1):
        history.append(ChatMessage(role="user", content=text))
        prompt_str = format_prompt(history, thinking_effort="off")
        input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)

        exp = next(e for e in experts if e.name == expert_name)

        # 1. Ground Truth Greedy Execution (same StaticCache instance as spec decoder)
        # Reset spec.cache to a clean state before running the greedy baseline so that
        # both runs start from the same empty SSM recurrent state. The GatedDeltaNet
        # layers are bound to spec.cache's SSM buffers (captured in the CUDA graphs);
        # using a separate StaticCache object would miss those bindings and silently
        # use stale recurrent state, causing token divergence even for short contexts.
        engine.activate(exp)
        spec.cache.reset()
        greedy_toks, eager_elapsed = eager_greedy_generate(
            model, input_ids, max_new_tokens=150, stop_ids=stop_ids, cache=spec.cache
        )
        greedy_tok_s = len(greedy_toks) / max(1e-9, eager_elapsed)


        # 2. Bucketed Speculative Graph Execution
        spec_toks, spec_elapsed, stats = spec.generate(
            input_ids,
            max_new_tokens=150,
            stop_ids=stop_ids,
            engine=engine,
            expert=exp,
        )
        spec_tok_s = stats.get("tok_s", len(spec_toks) / max(1e-9, spec_elapsed))
        tau = stats.get("tau", 0.0)

        # Compare outputs
        min_len = min(len(greedy_toks), len(spec_toks))
        token_match = (greedy_toks == spec_toks)
        match_pct = (sum(1 for g, s in zip(greedy_toks, spec_toks) if g == s) / max(1, min_len)) * 100.0

        resp_text = tokenizer.decode(spec_toks, skip_special_tokens=True)
        print(f"\n--- [Turn {turn_idx}/4] Domain: {expert_name.upper()} ---")
        print(f"Prompt: {text}")
        print(f"  Greedy (Eager):   {greedy_tok_s:.2f} tok/s ({len(greedy_toks)} tokens, {eager_elapsed:.3f}s)")
        print(f"  Speculative:      {spec_tok_s:.2f} tok/s ({len(spec_toks)} tokens, {spec_elapsed:.3f}s)")
        print(f"  Acceptance (tau): {tau:.2f} / 2.00")
        print(f"  Fidelity:         {'100% BIT-EXACT MATCH' if token_match else f'{match_pct:.1f}% MATCH'}")
        print(f"Response Preview:\n{resp_text[:160]}...\n")

        # Advance multi-turn conversation
        history.append(ChatMessage(role="assistant", content=resp_text))

        comparison_results.append({
            "turn": turn_idx,
            "expert": expert_name,
            "greedy_toks": len(greedy_toks),
            "greedy_tok_s": greedy_tok_s,
            "spec_toks": len(spec_toks),
            "spec_tok_s": spec_tok_s,
            "tau": tau,
            "match": token_match,
            "match_pct": match_pct,
        })

    # --------------------------------------------------------------------------
    # SUMMARY MATRIX
    # --------------------------------------------------------------------------
    print("=" * 90)
    print("BENCHMARK SUMMARY MATRIX")
    print("=" * 90)
    print(f"{'Turn':<6} | {'Expert':<12} | {'Greedy tok/s':<14} | {'Spec tok/s':<12} | {'Acceptance (tau)':<18} | {'Fidelity':<16}")
    print("-" * 90)

    all_passed = True
    for r in comparison_results:
        status_str = "100% BIT-EXACT" if r["match"] else f"{r['match_pct']:.1f}%"
        if not r["match"]:
            all_passed = False
        print(f"{r['turn']:<6} | {r['expert']:<12} | {r['greedy_tok_s']:<14.2f} | {r['spec_tok_s']:<12.2f} | {r['tau']:<18.2f} | {status_str:<16}")

    print("=" * 90)
    if all_passed:
        print(">>> SUCCESS: 100% BIT-EXACT GREEDY FIDELITY VERIFIED ACROSS ALL 4 TURNS! <<<")
    else:
        print(">>> Discrepancies detected in benchmark matrix. <<<")
    print("=" * 90)

if __name__ == "__main__":
    run_benchmark()
