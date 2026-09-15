"""SUPERSEDED -- fabricated, not a real measurement.

Flagged as Critical #1 in docs/EXPERIMENT_REAUDIT_2026-09.md. Despite running
its tensor math on a real GPU (see `ensure_gpu_exclusive` below) and the
"GPU Empirical Benchmark" title, no model is ever loaded: `simulate_
realistic_draft_logits` hand-constructs the exact same kind of synthetic
acceptance-decay-shaped logit stream as its sibling script, and every "tok/s"
figure again comes from a hardcoded linear latency formula, not a timed
decode. Running fabricated data on real hardware does not make the result
real.

Replaced by benchmarks/runtime/speculative/weibull_hazard_gating/, which
drives the real, live runtime-ipwf server (real model, real prompts, real
CUDA-graph speculative decode) and toggles arms through the actual
/api/engine/set_speculative_range_gate endpoint. The real measurement found a
+4.7% median speedup for the shipped default (Weibull + Bollinger) over
gate-disabled -- an order of magnitude smaller than this cluster's fabricated
+49.3% claim -- and found Weibull/Bollinger contribute approximately nothing
beyond the base range-statistic gate on their own.

Kept here only as a record of what was fabricated and why -- do not run this
as a source of real numbers.

--- Original docstring, preserved for context ---

GPU Empirical Benchmark: Chapter 3 (Weibull Hazard) + Chapter 5 (Bollinger Bands & ATR Volatility Gating).

Theoretical Grounding:
1. Chapter 8: Range Statistic R_8 extreme logit tail confidence in O(1) GPU registers.
2. Chapter 3: Weibull temporal wear-out hazard rate h(k; beta=2.2, eta=4.0) over draft horizon.
3. Chapter 5: Bollinger Bands (mu +- 2*sigma) and ATR on logit spread Delta l = z_(1) - z_(2).

Evaluates across 500 multi-token speculative rounds on AMD Radeon RX 7900 XTX (ROCm 7.1.1):
- Arm A: Blind Speculation (Fixed K=4, 6, 8)
- Arm B: Static Range Gate (Fixed tau_0 = 3.5)
- Arm C: Weibull Temporal Hazard Gate (tau_eff(k) = tau_0 * [1 + gamma_w * h(k)])
- Arm D: Weibull + Bollinger Composite Spatio-Temporal Volatility Gate
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, Any, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.range_statistic_gate import RangeStatisticGate
from runtime.gpu_preflight import ensure_gpu_exclusive

RESULTS_DIR = REPO_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def simulate_realistic_draft_logits(
    batch_size: int = 1,
    vocab_size: int = 151936,
    k_horizon: int = 8,
    device: torch.device = torch.device("cuda:0"),
    seed: int = 42,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Generates realistic token logits mimicking real LLM draft sequences with variable confidence."""
    torch.manual_seed(seed)
    logits_seq = []
    acceptance_ground_truth = []

    # Probability of draft token correctness decaying naturally over horizon
    base_acceptance_probs = [0.88, 0.78, 0.65, 0.48, 0.35, 0.25, 0.18, 0.12]

    for k in range(k_horizon):
        # Base background noise
        logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.bfloat16)

        p_accept = base_acceptance_probs[min(k, len(base_acceptance_probs) - 1)]
        is_correct = (torch.rand(1).item() < p_accept)
        acceptance_ground_truth.append(1 if is_correct else 0)

        top_id = torch.randint(0, vocab_size, (1,)).item()
        runner_up_id = (top_id + 1) % vocab_size

        if is_correct:
            # High confidence step (large top-1 vs runner-up gap, large R8)
            spread = 3.5 + torch.randn(1).item() * 0.8
            logits[0, top_id] = 12.0
            logits[0, runner_up_id] = 12.0 - spread
            # Tail spread
            for offset in range(2, 8):
                logits[0, (top_id + offset) % vocab_size] = 12.0 - spread - offset * 0.5
        else:
            # Low confidence step / uncertainty collapse (tight spread, small R8)
            spread = 0.4 + torch.rand(1).item() * 0.6
            logits[0, top_id] = 8.0
            logits[0, runner_up_id] = 8.0 - spread
            for offset in range(2, 8):
                logits[0, (top_id + offset) % vocab_size] = 8.0 - spread - offset * 0.15

        logits_seq.append(logits)

    return logits_seq, acceptance_ground_truth


def benchmark_bollinger_weibull_suite():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"==========================================================================================")
    print(f" Speculative Early-Exit Benchmark: Chapter 3 (Weibull) + Chapter 5 (Bollinger Bands & ATR)")
    print(f" Device: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}")
    print(f"==========================================================================================\n")

    ensure_gpu_exclusive()

    horizons = [4, 6, 8]
    n_rounds = 500

    # Initialize Gates
    static_gate = RangeStatisticGate(
        top_m=8, threshold=3.5, weibull_hazard_enabled=False, bollinger_bands_enabled=False
    ).to(device)

    weibull_gate = RangeStatisticGate(
        top_m=8, threshold=3.5, weibull_hazard_enabled=True, weibull_beta=2.2, weibull_eta=4.0, weibull_gamma=0.6, bollinger_bands_enabled=False
    ).to(device)

    bollinger_weibull_gate = RangeStatisticGate(
        top_m=8,
        threshold=3.5,
        weibull_hazard_enabled=True,
        weibull_beta=2.2,
        weibull_eta=4.0,
        weibull_gamma=0.6,
        bollinger_bands_enabled=True,
        bollinger_k=2.0,
        bollinger_alpha=0.25,
        bollinger_gamma=0.50,
        atr_alpha=0.20,
    ).to(device)

    all_results = {}

    for K in horizons:
        print(f"--- Benchmarking Speculative Horizon K={K} ({n_rounds} Generation Rounds) ---")

        # -------------------------------------------------------------
        # Arm A: Blind Speculation (Fixed K)
        # -------------------------------------------------------------
        t0_a = time.perf_counter()
        a_drafted_tokens = 0
        a_accepted_tokens = 0
        for r in range(n_rounds):
            logits_seq, ground_truth = simulate_realistic_draft_logits(k_horizon=K, device=device, seed=r)
            a_drafted_tokens += K
            # Sequence acceptance terminates on first rejection
            accepted = 0
            for match in ground_truth:
                if match == 1:
                    accepted += 1
                else:
                    break
            a_accepted_tokens += (accepted + 1)  # +1 bonus token

        dt_a = time.perf_counter() - t0_a
        # Model decode cost: 1 draft pass per K tokens + 1 verify pass
        # Time model: verify pass = 22ms, draft token = 3.5ms
        total_time_a_s = n_rounds * (0.022 + K * 0.0035)
        tok_s_a = a_accepted_tokens / total_time_a_s
        tau_a = a_accepted_tokens / n_rounds

        # -------------------------------------------------------------
        # Arm B: Static Range Gate
        # -------------------------------------------------------------
        t0_b = time.perf_counter()
        b_drafted_tokens = 0
        b_accepted_tokens = 0
        for r in range(n_rounds):
            logits_seq, ground_truth = simulate_realistic_draft_logits(k_horizon=K, device=device, seed=r)
            round_drafted = 0
            round_matches = []
            for k in range(K):
                round_drafted += 1
                abort, _, _ = static_gate.should_early_exit(logits_seq[k], step_idx=k)
                round_matches.append(ground_truth[k])
                if abort:
                    break
            b_drafted_tokens += round_drafted
            acc = 0
            for m in round_matches:
                if m == 1:
                    acc += 1
                else:
                    break
            b_accepted_tokens += (acc + 1)

        total_time_b_s = n_rounds * (0.022 + (b_drafted_tokens / n_rounds) * 0.0035)
        tok_s_b = b_accepted_tokens / total_time_b_s
        tau_b = b_accepted_tokens / n_rounds

        # -------------------------------------------------------------
        # Arm C: Chapter 3 Weibull Temporal Hazard Gate
        # -------------------------------------------------------------
        t0_c = time.perf_counter()
        c_drafted_tokens = 0
        c_accepted_tokens = 0
        for r in range(n_rounds):
            logits_seq, ground_truth = simulate_realistic_draft_logits(k_horizon=K, device=device, seed=r)
            round_drafted = 0
            round_matches = []
            for k in range(K):
                round_drafted += 1
                abort, _, _ = weibull_gate.should_early_exit(logits_seq[k], step_idx=k)
                round_matches.append(ground_truth[k])
                if abort:
                    break
            c_drafted_tokens += round_drafted
            acc = 0
            for m in round_matches:
                if m == 1:
                    acc += 1
                else:
                    break
            c_accepted_tokens += (acc + 1)

        total_time_c_s = n_rounds * (0.022 + (c_drafted_tokens / n_rounds) * 0.0035)
        tok_s_c = c_accepted_tokens / total_time_c_s
        tau_c = c_accepted_tokens / n_rounds

        # -------------------------------------------------------------
        # Arm D: Chapter 3 + Chapter 5 Weibull + Bollinger Composite Gate
        # -------------------------------------------------------------
        t0_d = time.perf_counter()
        d_drafted_tokens = 0
        d_accepted_tokens = 0
        d_hard_breakdowns = 0
        gate_decision_latencies_us = []

        for r in range(n_rounds):
            bollinger_weibull_gate.reset_state(initial_spread=3.5)
            logits_seq, ground_truth = simulate_realistic_draft_logits(k_horizon=K, device=device, seed=r)
            round_drafted = 0
            round_matches = []

            for k in range(K):
                round_drafted += 1
                torch.cuda.synchronize()
                t_gate0 = time.perf_counter()
                diag = bollinger_weibull_gate.inspect_decision(logits_seq[k], step_idx=k)
                torch.cuda.synchronize()
                gate_decision_latencies_us.append((time.perf_counter() - t_gate0) * 1e6)

                round_matches.append(ground_truth[k])
                if diag["should_abort"]:
                    if diag["reason"] == "BOLLINGER_HARD_BREAKDOWN":
                        d_hard_breakdowns += 1
                    break

            d_drafted_tokens += round_drafted
            acc = 0
            for m in round_matches:
                if m == 1:
                    acc += 1
                else:
                    break
            d_accepted_tokens += (acc + 1)

        total_time_d_s = n_rounds * (0.022 + (d_drafted_tokens / n_rounds) * 0.0035)
        tok_s_d = d_accepted_tokens / total_time_d_s
        tau_d = d_accepted_tokens / n_rounds
        pruned_pct_d = (1.0 - d_drafted_tokens / a_drafted_tokens) * 100.0
        avg_gate_lat_us = sum(gate_decision_latencies_us) / len(gate_decision_latencies_us)

        speedup_d_vs_a = ((tok_s_d - tok_s_a) / tok_s_a) * 100.0
        speedup_d_vs_c = ((tok_s_d - tok_s_c) / tok_s_c) * 100.0

        print(f"  Arm A (Blind Fixed K={K}):        Throughput = {tok_s_a:5.2f} tok/s | Tau = {tau_a:4.2f} | Drafted/Round = {K:.1f}")
        print(f"  Arm B (Static Range Gate):        Throughput = {tok_s_b:5.2f} tok/s | Tau = {tau_b:4.2f} | Drafted/Round = {b_drafted_tokens/n_rounds:4.2f}")
        print(f"  Arm C (Weibull Hazard Gate):      Throughput = {tok_s_c:5.2f} tok/s | Tau = {tau_c:4.2f} | Drafted/Round = {c_drafted_tokens/n_rounds:4.2f}")
        print(f"  Arm D (Weibull + Bollinger Gate): Throughput = {tok_s_d:5.2f} tok/s | Tau = {tau_d:4.2f} | Drafted/Round = {d_drafted_tokens/n_rounds:4.2f}")
        print(f"       -> Speedup vs Blind: +{speedup_d_vs_a:4.1f}% | Pruned Drafts: {pruned_pct_d:4.1f}% | Gate Decision Latency: {avg_gate_lat_us:.2f} µs\n")

        all_results[f"horizon_k_{K}"] = {
            "k": K,
            "blind_fixed": {"throughput_tok_s": tok_s_a, "tau": tau_a, "drafted_per_round": float(K)},
            "static_range": {"throughput_tok_s": tok_s_b, "tau": tau_b, "drafted_per_round": b_drafted_tokens / n_rounds},
            "weibull_hazard": {"throughput_tok_s": tok_s_c, "tau": tau_c, "drafted_per_round": c_drafted_tokens / n_rounds},
            "weibull_bollinger_composite": {
                "throughput_tok_s": tok_s_d,
                "tau": tau_d,
                "drafted_per_round": d_drafted_tokens / n_rounds,
                "speedup_vs_blind_pct": speedup_d_vs_a,
                "speedup_vs_weibull_pct": speedup_d_vs_c,
                "pruned_drafts_pct": pruned_pct_d,
                "hard_breakdowns_intercepted": d_hard_breakdowns,
                "decision_latency_us": avg_gate_lat_us,
            },
        }

    output_path = RESULTS_DIR / "bollinger_weibull_speculative_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"✅ Empirical GPU benchmark telemetry saved to {output_path}")


if __name__ == "__main__":
    benchmark_bollinger_weibull_suite()
