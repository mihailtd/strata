"""Empirical Benchmark: Single-Pass Range Statistic Speculative Early-Exit Gating.

Grounding: Chapter 8 (Outlier Detection Based on Range Statistic Empirical Evaluation
and Comparisons, Dallah, Sulieman, Alzaatreh).

Experimental Design:
Evaluates speculative decoding efficiency across 3 gating strategies:
- Arm A: Baseline Blind Fixed Speculation (K=4, no gating)
- Arm B: Full Softmax Entropy Gating (2-pass reduction + exp + log)
- Arm C: Single-Pass Range Statistic Gate (O(1) register top-8 range spread)

Evaluates on mixed-entropy token streams (simulating code generation with predictable
syntax keywords interspersed with high-entropy variable names).

Measures:
- Acceptance Yield (tau, effective tokens accepted per verification round)
- Wasted Draft Token Pruning Ratio (% of low-confidence drafts pruned before verification)
- Gating Decision Latency (microseconds per draft step)
- Net Speculative Decode Velocity (tok/s)
"""

import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F

from runtime.range_statistic_gate import RangeStatisticGate

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_softmax_entropy(logits: torch.Tensor) -> float:
    """Standard 2-pass softmax entropy calculation."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return float(entropy.squeeze().item())


def run_range_gating_benchmark():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Single-Pass Range Statistic Speculative Gating Benchmark on: {device}")

    vocab_size = 152064  # Qwen 3.5 vocab size
    spec_k = 4
    n_rounds = 500  # 500 speculative verification rounds

    # Generate synthetic validation stream simulating realistic LLM decoding:
    # 70% predictable tokens (high range / low entropy, e.g. "SELECT", "import", "def", "return")
    # 30% ambiguous tokens (low range / high entropy, e.g. arbitrary identifiers, punctuation choice)
    stream_types = []
    stream_logits = []
    stream_targets = []

    for _ in range(n_rounds):
        round_logits = []
        round_targets = []
        for step in range(spec_k):
            is_predictable = torch.rand(1).item() < 0.70
            logits = torch.randn(1, vocab_size, device=device) * 0.5
            target = torch.randint(0, vocab_size, (1,), device=device)

            if is_predictable:
                # Top candidate dominates
                logits[0, target] = 12.0 + torch.randn(1, device=device) * 1.0
            else:
                # Flat distribution among top 8 candidates
                alt_tokens = torch.randint(0, vocab_size, (8,), device=device)
                logits[0, alt_tokens] = 4.0 + torch.randn(8, device=device) * 0.5

            round_logits.append(logits)
            round_targets.append(target)
            stream_types.append(is_predictable)

        stream_logits.append(round_logits)
        stream_targets.append(round_targets)

    # Base Model Verification Cost model (AMD RX 7900 XTX empirical model from DECISIONS.md §24):
    # Base decode time for 1 token: ~28.7 ms
    # Speculative verification cost for (K+1) chunk: 28.7 * (1.0 + 0.15 * K) ms
    t_base_single = 28.7  # ms

    gate_range = RangeStatisticGate(top_m=8, threshold=5.0, mode="extreme_range")
    entropy_threshold = 2.0  # Nats

    arms = [
        ("Arm_A_Blind_Fixed_K4", "Blind Fixed-Width Speculation (K=4, No Gating)"),
        ("Arm_B_Softmax_Entropy_Gate", "Softmax Entropy Gating (2-Pass Softmax + Log)"),
        ("Arm_C_Single_Pass_Range_Gate", "Single-Pass Range Statistic Gating (Chapter 8, O(1))"),
    ]

    benchmark_results = {
        "benchmark": "Single-Pass Range Statistic Speculative Gating",
        "grounding": "Chapter 8 (Outlier Detection Based on Range Statistics, Dallah et al.)",
        "device": str(device),
        "vocab_size": vocab_size,
        "spec_k": spec_k,
        "n_rounds": n_rounds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": {},
    }

    for arm_id, arm_desc in arms:
        print(f"\n--- Evaluating {arm_id}: {arm_desc} ---")

        total_drafted_tokens = 0
        total_accepted_tokens = 0
        total_wasted_drafts_avoided = 0
        total_gating_time_us = 0.0
        total_round_latency_ms = 0.0

        for r_idx in range(n_rounds):
            round_l = stream_logits[r_idx]
            round_t = stream_targets[r_idx]

            drafted_tokens = []
            drafted_logits = []

            # Step 1: Draft tokens with gating
            for step in range(spec_k):
                l = round_l[step]
                pred_tok = torch.argmax(l, dim=-1).item()

                if arm_id == "Arm_A_Blind_Fixed_K4":
                    # No gating, always draft all K
                    drafted_tokens.append(pred_tok)
                    drafted_logits.append(l)

                elif arm_id == "Arm_B_Softmax_Entropy_Gate":
                    # Softmax Entropy check
                    t0 = time.perf_counter()
                    ent = compute_softmax_entropy(l)
                    t_gate_us = (time.perf_counter() - t0) * 1e6
                    total_gating_time_us += t_gate_us

                    if step > 0 and ent > entropy_threshold:
                        # Abort early
                        total_wasted_drafts_avoided += (spec_k - step)
                        break

                    drafted_tokens.append(pred_tok)
                    drafted_logits.append(l)

                elif arm_id == "Arm_C_Single_Pass_Range_Gate":
                    # Single-Pass Range check in registers
                    t0 = time.perf_counter()
                    should_abort, _ = gate_range.should_early_exit(l, step_idx=step)
                    t_gate_us = (time.perf_counter() - t0) * 1e6
                    total_gating_time_us += t_gate_us

                    if step > 0 and should_abort:
                        # Abort early
                        total_wasted_drafts_avoided += (spec_k - step)
                        break

                    drafted_tokens.append(pred_tok)
                    drafted_logits.append(l)

            # Step 2: Verification pass on actual drafted length k_actual
            k_actual = len(drafted_tokens)
            total_drafted_tokens += k_actual

            # Sequential speculative verify
            n_acc = 0
            for step in range(k_actual):
                if drafted_tokens[step] == round_t[step].item():
                    n_acc += 1
                else:
                    break

            # tau = 1 (bonus verified base token) + accepted drafts
            tau_round = 1.0 + n_acc
            total_accepted_tokens += tau_round

            # Verification round cost based on actual dynamic chunk width k_actual
            # Draft step cost: ~1.2 ms per drafted token
            # Verification cost: t_base_single * (1.0 + 0.15 * k_actual)
            round_cost = (k_actual * 1.2) + (t_base_single * (1.0 + 0.15 * k_actual))
            total_round_latency_ms += round_cost

        # Compute summary metrics
        mean_tau = total_accepted_tokens / n_rounds
        mean_gating_latency_us = total_gating_time_us / max(1, total_drafted_tokens)
        prune_ratio = (total_wasted_drafts_avoided / (n_rounds * spec_k)) * 100.0
        tok_per_sec = (total_accepted_tokens / (total_round_latency_ms / 1000.0))

        benchmark_results["arms"][arm_id] = {
            "name": arm_desc,
            "mean_tau_acceptance": round(mean_tau, 3),
            "wasted_draft_pruning_pct": round(prune_ratio, 1),
            "gating_decision_latency_us": round(mean_gating_latency_us, 2),
            "decode_throughput_tok_s": round(tok_per_sec, 2),
            "total_latency_ms": round(total_round_latency_ms, 2),
        }

        print(f"  Mean Acceptance Yield (tau): {mean_tau:.2f} tok/round")
        print(f"  Wasted Drafts Pruned: {prune_ratio:.1f}%")
        print(f"  Gating Overhead: {mean_gating_latency_us:.2f} µs/step")
        print(f"  Decode Velocity: {tok_per_sec:.2f} tok/s")

    # Measure speedup of Range Gating over Blind Speculation
    blind_toks = benchmark_results["arms"]["Arm_A_Blind_Fixed_K4"]["decode_throughput_tok_s"]
    range_toks = benchmark_results["arms"]["Arm_C_Single_Pass_Range_Gate"]["decode_throughput_tok_s"]
    speedup = round(range_toks / blind_toks, 3)
    benchmark_results["comparison"] = {
        "range_over_blind_speedup": speedup,
        "latency_reduction_us": round(
            benchmark_results["arms"]["Arm_B_Softmax_Entropy_Gate"]["gating_decision_latency_us"]
            - benchmark_results["arms"]["Arm_C_Single_Pass_Range_Gate"]["gating_decision_latency_us"],
            2,
        ),
    }

    print(f"\n[+] Single-Pass Range Gating Speedup vs Blind Fixed: {speedup}x")
    output_path = RESULTS_DIR / "range_speculative_gating.json"
    with open(output_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"[+] Telemetry successfully exported to: {output_path}")
    return benchmark_results


if __name__ == "__main__":
    run_range_gating_benchmark()
