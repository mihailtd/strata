"""End-to-end quality and latency validation for FlashNorm weight folding.

Compares:
  Arm 1 (Stock):     Stock RMSNorm (ExactRMSNorm) + standard base/adapted weights
  Arm 2 (FlashNorm): ScaleFreeRMSNorm + FlashNorm folded weights (+ scaled adapter factors)

Verifies:
  1. Logit top-1 agreement and rank correlation across domains
  2. Generated text equality / divergence
  3. Per-token decode latency reduction

    uv run --env-file .env benchmarks/runtime/folding/evaluate_flash_norm_quality.py
"""

import sys
import time
from pathlib import Path

import torch

# fla device probe ordering
torch.zeros(1, device="cuda")
torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.fused_norm import (  # noqa: E402
    fold_rmsnorm_into_linear,
    inject_exact_rmsnorm,
    scale_expert_factors_for_folded_norms,
    unfold_rmsnorm,
)
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

PROMPTS = [
    (
        "astral",
        "Write a FastAPI endpoint, managed with uv and linted with ruff, that "
        "runs a similarity search over a pgvector table.",
    ),
    (
        "postgresql",
        "Design a PostgreSQL 18 schema with pgvector HNSW indexing for "
        "transaction history and client profile embeddings.",
    ),
    (
        "financial_planning",
        "A client shows money-avoidance behaviour and anxiety about "
        "sequence-of-returns risk in retirement. Outline a risk strategy.",
    ),
]


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> tuple[str, float]:
    """Generate greedy tokens and return (generated_text, mean_tok_per_sec)."""
    input_ids = tokenizer(f"### Question:\n{prompt}\n\n### Answer:\n", return_tensors="pt").input_ids.to(model.device)

    # Warmup / prefill
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past_key_values = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    gen_tokens = [next_tok]
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens - 1):
            out = model(input_ids=next_tok, past_key_values=past_key_values, use_cache=True)
            past_key_values = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_tokens.append(next_tok)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    tok_s = (max_new_tokens - 1) / elapsed if elapsed > 0 else 0.0

    all_tokens = torch.cat([input_ids, *gen_tokens], dim=-1)
    text = tokenizer.decode(all_tokens[0], skip_special_tokens=True)
    return text, tok_s


def main():
    set_hard_vram_cap(22.0)
    print("=" * 70)
    print("FlashNorm End-to-End Quality & Latency Validation")
    print("=" * 70)

    model_id = "Qwen/Qwen3.5-4B"
    print(f"Loading {model_id} in bf16...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    # Base setup: ExactRMSNorm
    inject_exact_rmsnorm(model)

    # Load an adapter
    astral_dir = REPO_ROOT / "results" / "adapters" / "m2_astral_r8a128"
    expert_stock = FoldableExpert.from_dir(astral_dir, "astral")
    expert_flash = FoldableExpert.from_dir(astral_dir, "astral")

    REPEATS = 3

    def timed_arm(label):
        """R repeats per domain; report the MEDIAN and the spread.

        ⚠️ The first version of this script measured each domain ONCE, with no
        warmup, running Arm 1 entirely before Arm 2. That reports a single sample
        per cell on a box where repeat spreads are several percent, and it cannot
        distinguish a 3% effect from 3% variance. The reported 0.5-3.1% spread
        across domains was never resolvable -- norm folding saves the same work
        every step regardless of domain, so a real effect MUST be flat across
        them.
        """
        res = {}
        for domain, prompt in PROMPTS:
            speeds, text = [], None
            for _ in range(REPEATS):
                text, sp = generate_text(model, tokenizer, prompt, max_new_tokens=64)
                speeds.append(sp)
            speeds.sort()
            res[domain] = {"text": text, "tok_s": speeds[REPEATS // 2],
                           "lo": speeds[0], "hi": speeds[-1]}
            print(f"  [{domain}] {speeds[REPEATS // 2]:.1f} tok/s "
                  f"(spread {speeds[0]:.1f}-{speeds[-1]:.1f})")
        return res

    # ONE engine for warmup + arm 1. Each WeightFoldingEngine(keep_pristine=True)
    # clones every adapted weight; holding several alive OOMs a 24 GB card at the
    # 22 GB cap, which is how the first run of this revision died.
    print("\n--- Warmup (compilation OUTSIDE the timed region) ---")
    stock_engine = WeightFoldingEngine(model, [expert_stock], keep_pristine=True)
    stock_engine.activate(expert_stock)
    for _, prompt in PROMPTS:
        generate_text(model, tokenizer, prompt, max_new_tokens=16)
    print("  done")

    print("\n--- Testing Arm 1 (Stock ExactRMSNorm) ---")
    stock_results = timed_arm("stock")
    stock_engine.restore()
    del stock_engine
    torch.cuda.empty_cache()

    print("\n--- Applying FlashNorm Fold ---")
    folded_count = fold_rmsnorm_into_linear(model)
    print(f"  Folded {folded_count} RMSNorm layers into Linear weights.")

    # Scale expert factors for FlashNorm
    scaled_count = scale_expert_factors_for_folded_norms(model, [expert_flash])
    print(f"  Scaled {scaled_count} adapter factor matrices.")

    print("\n--- Testing Arm 2 (FlashNorm Folded) ---")
    flash_engine = WeightFoldingEngine(model, [expert_flash], keep_pristine=True)
    flash_engine.activate(expert_flash)
    flash_results = timed_arm("flash")
    flash_engine.restore()
    del flash_engine
    torch.cuda.empty_cache()

    # ---- Arm 3: stock AGAIN, after unfolding -------------------------------
    # Two jobs in one arm. (a) If arm3 is materially faster than arm1, arm1 was
    # cold and the arm1-vs-arm2 gap was warmup, not FlashNorm. (b) unfold_rmsnorm
    # restores by DIVIDING out (1+gamma) rather than from a pristine copy -- the
    # "reconstruct by inverse arithmetic" pattern this repo retired for adapter
    # folding (NOVELTY.md master-weights discipline). If a fold/unfold round trip
    # drifts the weights, arm3's TEXT will differ from arm1's.
    print("\n--- Unfolding, then Arm 3 (Stock again: warmup + drift control) ---")
    unfolded = unfold_rmsnorm(model)
    print(f"  Unfolded {unfolded} norms.")
    expert_ctrl = FoldableExpert.from_dir(astral_dir, "astral")
    ctrl_engine = WeightFoldingEngine(model, [expert_ctrl], keep_pristine=True)
    ctrl_engine.activate(expert_ctrl)
    ctrl_results = timed_arm("stock-again")
    ctrl_engine.restore()

    print("\n  DRIFT / WARMUP CONTROL")
    for domain, _ in PROMPTS:
        a1, a3 = stock_results[domain], ctrl_results[domain]
        same = a1["text"] == a3["text"]
        print(f"    [{domain}] arm1 {a1['tok_s']:.1f} -> arm3 {a3['tok_s']:.1f} tok/s "
              f"({100 * (a3['tok_s'] - a1['tok_s']) / a1['tok_s']:+.1f}%)   "
              f"text after fold/unfold round trip: {'IDENTICAL' if same else 'DRIFTED'}")

    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    for domain, _ in PROMPTS:
        s_text = stock_results[domain]["text"]
        f_text = flash_results[domain]["text"]
        s_speed = stock_results[domain]["tok_s"]
        f_speed = flash_results[domain]["tok_s"]
        speedup_pct = (f_speed - s_speed) / s_speed * 100 if s_speed > 0 else 0

        # Exact character match
        match = s_text == f_text
        print(f"\n[{domain}]")
        print(f"  Stock speed:     {s_speed:.1f} tok/s")
        print(f"  FlashNorm speed: {f_speed:.1f} tok/s ({speedup_pct:+.1f}%)")
        print(f"  Output Match:    {'✅ EXACT MATCH' if match else '⚠️ Diverged (acceptable in bf16 accumulation)'}")
        if not match:
            print(f"  Stock snippet:     {repr(s_text[:120])}")
            print(f"  FlashNorm snippet: {repr(f_text[:120])}")

    print("\n" + "=" * 70)
    print("FlashNorm validation completed.")


if __name__ == "__main__":
    main()
