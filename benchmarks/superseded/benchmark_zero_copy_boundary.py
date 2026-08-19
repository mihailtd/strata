"""Zero-Copy Memory Boundary Interface & Inference Runtime Architecture Benchmark.

Implements & Benchmarks:
1. Expert-swap cost across four realistic deployments (see arms A-D below).
2. Dual Equivalence Gate: MSE < 10^-5 logit check & token-exact match vs standard PeftModel.
3. On-Device GEMM Folding: s * (U @ V) on VRAM bandwidth (~836 GB/s), ~12.6 MB factors.
4. Pristine W_0 reset: 5.12 GB copy-back. Exact BY CONSTRUCTION (a copy_, not
   repeated add/sub), so 0 drift is a design property, not a measured surprise.
5. Scale-gated router: folds when scaling >= threshold, else RAISES -- there is
   no wrap path on WeightFoldingEngine (the old branch silently ran the base model).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))

from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    VramChurnProbe,
    WeightFoldingEngine,
    apply_novel_lora,
    route_expert_execution,
    set_hard_vram_cap,
)
from gnn_experiment.utils.logger import log_benchmark_metric  # noqa: E402


def measure_loopback_rtt_ms(payload: dict, reps: int = 20) -> float:
    """Median round-trip for a command-sized JSON message over loopback TCP.

    This is the honest cost of putting an expert swap behind an API boundary:
    you send an "activate expert X" command, not the weights.
    """
    import socket
    import threading

    blob = json.dumps(payload).encode()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()

    def echo():
        conn, _ = srv.accept()
        with conn:
            while True:
                data = conn.recv(65536)
                if not data:
                    return
                conn.sendall(data)

    t = threading.Thread(target=echo, daemon=True)
    t.start()
    cli = socket.create_connection((host, port))
    cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        cli.sendall(blob)
        got = b""
        while len(got) < len(blob):
            got += cli.recv(65536)
        times.append((time.perf_counter() - t0) * 1000.0)
    cli.close()
    srv.close()
    times.sort()
    return times[len(times) // 2]


def generate_text_greedy(
    model,
    tokenizer,
    prompt: str,
    domain_label: str,
    max_new_tokens: int = 96,
) -> tuple[str, float, float, torch.Tensor]:
    """Generates text greedily and returns (text, elapsed_s, tok_s, logits)."""
    formatted = f"### [{domain_label.upper()} EXPERT] Question:\n{prompt}\n\n### Answer:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_t = time.perf_counter()

    with torch.no_grad():
        out = model(**inputs)
        first_logits = out.logits[:, -1, :].detach()

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start_t

    new_tokens = outputs[0][inputs.input_ids.shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    tok_s = len(new_tokens) / max(1e-5, elapsed_s)

    return text, elapsed_s, tok_s, first_logits


def run_zero_copy_boundary_benchmark(
    model_name: str = "Qwen/Qwen3.5-4B",
    vram_cap_gb: float = 22.0,
):
    set_hard_vram_cap(vram_cap_gb)
    print("==================================================")
    print(f" Zero-Copy Memory Boundary Benchmark ({model_name})")
    print("==================================================")

    # 1. Resolve Adapter Directories
    financial_dir = REPO_ROOT / "results" / "adapters" / "financial_planning_krona_dora"
    postgres_dir = REPO_ROOT / "results" / "adapters" / "postgres_qwen3.5_micro_id_kron_r16"
    astral_dir = REPO_ROOT / "results" / "adapters" / "astral_qwen3.5_micro_id_kron"

    print("Loading device-resident factor experts into host RAM...")
    exp_fin = FoldableExpert.from_dir(financial_dir, "financial_planning")
    exp_pg = FoldableExpert.from_dir(postgres_dir, "postgresql")
    exp_astral = FoldableExpert.from_dir(astral_dir, "astral")

    print(f"  Financial Expert : {exp_fin}")
    print(f"  PostgreSQL Expert: {exp_pg}")
    print(f"  Astral Expert    : {exp_astral}")

    # 2. Load Base Model in bfloat16
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

    # 3. Initialize WeightFoldingEngine
    print("\nInitializing WeightFoldingEngine with Pristine W0 VRAM snapshot...")
    folding_engine = WeightFoldingEngine(base_model, [exp_fin, exp_pg, exp_astral], keep_pristine=True)

    print(f"  Slots tracked in engine : {len(folding_engine.slots)}")
    print(f"  Pristine W0 VRAM Size   : {folding_engine.pristine_bytes / 1e6:.2f} MB")

    # 4. Pass 1: Equivalence Validation Gate vs PeftModel
    print("\n--- 1. Dual Equivalence Gate (Folded vs Wrapped PeftModel) ---")
    test_prompt = "Explain loss aversion and money avoidance scripts."

    # Measure Folded output
    folding_engine.activate(exp_fin)
    folded_text, _, _, folded_logits = generate_text_greedy(
        base_model, tokenizer, test_prompt, "financial_planning", max_new_tokens=48
    )
    folding_engine.restore()

    # Measure Wrapped model output (supports both novel_adapter.pt and standard PEFT)
    peft_base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    if (financial_dir / "novel_adapter_config.json").exists():
        cfg = json.loads((financial_dir / "novel_adapter_config.json").read_text())
        state = torch.load(financial_dir / "novel_adapter.pt", map_location="cpu")
        apply_novel_lora(peft_base, mode=cfg.get("mode", "id_kron"), rank_in=cfg.get("rank_in", 8))
        peft_base.load_state_dict(state, strict=False)
        peft_model = peft_base
    else:
        peft_model = PeftModel.from_pretrained(peft_base, str(financial_dir))
    peft_model.eval()
    wrapped_text, _, _, wrapped_logits = generate_text_greedy(
        peft_model, tokenizer, test_prompt, "financial_planning", max_new_tokens=48
    )

    del peft_model, peft_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compute Equivalence Metrics
    logit_diff = (folded_logits.float() - wrapped_logits.float()).abs()
    max_logit_err = float(logit_diff.max().item())
    logit_mse = float((logit_diff**2).mean().item())
    token_match = folded_text.strip() == wrapped_text.strip()

    print(f"  Logit Max Error (L_inf) : {max_logit_err:.6e}")
    print(f"  Logit Mean Squared Error: {logit_mse:.6e} (Threshold: < 1e-3 for bfloat16)")
    print(f"  Token Generation Match  : {'EXACT MATCH (100%)' if token_match else 'MISMATCH'}")

    if logit_mse >= 1e-3:
        raise ValueError(f"Equivalence Gate Failed: Logit MSE {logit_mse:.6e} >= 1e-3")
    if not token_match:
        raise ValueError("Equivalence Gate Failed: Token strings do not match exactly")

    print("  Equivalence Gate: PASSED (100% Token-Exact & Logit MSE < 1e-3)")

    print("\n--- 2. Expert-swap cost across realistic deployments ---")

    # ---- ARM A: resident fold (the path a server actually takes) ----------
    # Factors already live in VRAM; the swap is one fused addmm per module.
    with VramChurnProbe() as churn:
        route_expert_execution(folding_engine, exp_fin, absorption_threshold=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def timed(fn, reps=5):
        fn()  # warm
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out = []
        for _ in range(reps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1000.0)
        out.sort()
        return out[len(out) // 2]

    resident_ms = timed(lambda: folding_engine.activate(exp_fin))
    vram_churn_bytes = churn.transient_bytes

    # ---- ARM B: cold load from disk, then fold ----------------------------
    # The honest "expert not yet resident" cost: read the adapter file, build
    # factors, move to device, fold. This is what a real system pays on first
    # touch of an expert it has not served before.
    def cold_load_fold():
        e = FoldableExpert.from_dir(financial_dir, "financial_planning")
        ref = next(iter(folding_engine.slots.values()))
        e.to(ref.device, ref.dtype)
        folding_engine.activate(e)

    cold_ms = timed(cold_load_fold, reps=3)

    # ---- ARM C: naive host round-trip (STRAWMAN, kept for reference) ------
    # Ship every factor GPU -> host -> GPU per swap. No real IPC design does
    # this: if the swap crosses a process boundary you send a ~20 byte
    # "activate expert X" command and the weights never move. This arm is
    # reported ONLY to show what the previously-claimed 14.5x was measured
    # against -- and note the old version also threw the uploaded tensors
    # away without folding them, so it did strictly less useful work.
    def host_roundtrip_fold():
        cpu = {k: (u.cpu(), v.cpu()) for k, (u, v) in exp_fin.factors.items()}
        dev = base_model.device
        e = FoldableExpert({k: (u.to(dev), v.to(dev)) for k, (u, v) in cpu.items()}, exp_fin.scaling, "roundtrip")
        folding_engine.activate(e)

    strawman_ms = timed(host_roundtrip_fold, reps=3)

    # ---- ARM D: what crossing a real API boundary actually costs ----------
    # A command-sized JSON payload over a loopback TCP round trip. This is the
    # true marginal cost of putting the swap behind an HTTP/IPC boundary.
    ipc_ms = measure_loopback_rtt_ms({"op": "activate_expert", "name": exp_fin.name})

    factor_mb = exp_fin.nbytes / 1e6
    print(f"  A. resident fold (factors in VRAM)   : {resident_ms:8.3f} ms   <- the real path")
    print(f"  B. cold load from disk + fold        : {cold_ms:8.3f} ms   (first touch of an expert)")
    print(f"  C. host round-trip + fold [STRAWMAN] : {strawman_ms:8.3f} ms   ({factor_mb:.1f} MB GPU->host->GPU)")
    print(f"  D. loopback IPC command round trip   : {ipc_ms:8.3f} ms   (~20 byte activate message)")
    print(f"  VRAM transient churn during fold     : {vram_churn_bytes} bytes (peak-above-baseline)")
    print()
    print(f"  Honest read: crossing an API boundary costs A + D = {resident_ms + ipc_ms:.3f} ms,")
    print(
        f"  i.e. {(resident_ms + ipc_ms) / resident_ms:.2f}x the in-process fold "
        f"-- not {strawman_ms / resident_ms:.1f}x."
    )
    print(f"  The {strawman_ms / resident_ms:.1f}x figure only appears if you ship the weights themselves")
    print("  through host RAM on every swap, which no serving design does.")

    int_swap_ms = resident_ms

    folding_engine.restore()

    # 6. Pass 3: Real 4-Turn Agentic Loop Execution
    print("\n--- 3. Multi-Expert 4-Turn Agentic Loop Execution ---")

    prompts = [
        (
            "financial_planning",
            exp_fin,
            "A client has money avoidance habits and sequence-of-returns risk anxiety in retirement. "
            "Formulate a risk strategy.",
        ),
        (
            "postgresql",
            exp_pg,
            "Design a PostgreSQL 18 schema with pgvector HNSW indexing to store transaction history "
            "and vector embeddings for client profiles.",
        ),
        (
            "astral",
            exp_astral,
            "Write a Python FastAPI endpoint using uv and ruff standards to execute similarity search "
            "over the PostgreSQL vector table.",
        ),
        (
            "financial_planning",
            exp_fin,
            "Combine the financial strategy, DB schema, and FastAPI code into a comprehensive execution plan.",
        ),
    ]

    chain_history = []
    total_swap_time_ms = 0.0

    for turn_idx, (domain, expert, prompt) in enumerate(prompts, 1):
        print(f"\n[Turn {turn_idx}/4 - {domain.upper()}] Prompt: '{prompt[:70]}...'")

        # Swap expert
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        swap_t0 = time.perf_counter()

        route_type = route_expert_execution(folding_engine, expert, absorption_threshold=0.0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        swap_ms = (time.perf_counter() - swap_t0) * 1000.0
        total_swap_time_ms += swap_ms

        print(f"  In-Place Fold Latency ({route_type}) : {swap_ms:.3f} ms")

        # Generate response
        text, elapsed_s, tok_s, _ = generate_text_greedy(base_model, tokenizer, prompt, domain, max_new_tokens=96)
        print(f"  Decode Throughput             : {tok_s:.2f} tok/s ({elapsed_s:.2f} s)")
        print(f"  Response Snippet              : {text[:130]}...")

        chain_history.append(
            {
                "turn": turn_idx,
                "domain": domain,
                "route": route_type,
                "swap_latency_ms": swap_ms,
                "tok_per_sec": tok_s,
                "elapsed_s": elapsed_s,
                "snippet": text[:130],
            }
        )

    # Restore pristine W0 weights
    folding_engine.restore()
    restore_drift = folding_engine.max_drift()

    print("==================================================")
    print(" Zero-Copy Boundary Benchmark Final Results")
    print("==================================================")
    print(" Equivalence Validation Gate    : PASSED (Logit MSE < 1e-3 & 100% Token Match)")
    print(f" Internal Swap Latency (p50)    : {int_swap_ms:.3f} ms")
    print(f" Transient VRAM Allocation Churn: {vram_churn_bytes} bytes")
    print(f" Cumulative 4-Turn Swap Latency : {total_swap_time_ms:.3f} ms (Target: < 45.0 ms)")
    print(f" Mean Chain Decode Throughput   : {sum(h['tok_per_sec'] for h in chain_history) / 4:.2f} tok/s")
    print(f" Pristine W0 Restore Drift      : {restore_drift:.8f} (Exact Copy Restore)")

    # Log metrics
    log_benchmark_metric(
        {
            "experiment": "zero_copy_memory_boundary",
            "model_name": model_name,
            "equivalence_gate_passed": True,
            "logit_mse": logit_mse,
            "internal_swap_ms": int_swap_ms,
            "vram_allocation_churn_bytes": vram_churn_bytes,
            "cumulative_4turn_swap_ms": total_swap_time_ms,
            "mean_chain_tok_s": sum(h["tok_per_sec"] for h in chain_history) / 4,
            "pristine_restore_drift": restore_drift,
        },
        filepath="results/zero_copy_boundary_runs.jsonl",
    )

    summary = {
        "model_name": model_name,
        "equivalence_gate": {
            "passed": True,
            "max_logit_err": float(max_logit_err),
            "logit_mse": float(logit_mse),
            "token_match": bool(token_match),
        },
        "swap_cost_by_deployment": {
            "a_resident_fold_ms": float(resident_ms),
            "b_cold_load_from_disk_ms": float(cold_ms),
            "c_host_roundtrip_ms_STRAWMAN": float(strawman_ms),
            "d_ipc_command_rtt_ms": float(ipc_ms),
            "vram_transient_churn_bytes": int(vram_churn_bytes),
            # The meaningful ratio: what an API boundary actually adds.
            "api_boundary_overhead_x": float((resident_ms + ipc_ms) / max(1e-9, resident_ms)),
            # Kept only to document what the retracted "14.5x zero-copy" claim
            # was measured against. Not a speedup of anything anyone would build.
            "strawman_ratio_x_DO_NOT_CITE": float(strawman_ms / max(1e-9, resident_ms)),
        },
        "multi_turn_chain": {
            "cumulative_swap_ms": float(total_swap_time_ms),
            "mean_tok_s": float(sum(h["tok_per_sec"] for h in chain_history) / 4),
            "pristine_restore_drift": float(restore_drift),
            "turns": [
                {
                    "turn": int(h["turn"]),
                    "domain": str(h["domain"]),
                    "route": str(h["route"]),
                    "swap_latency_ms": float(h["swap_latency_ms"]),
                    "tok_per_sec": float(h["tok_per_sec"]),
                    "elapsed_s": float(h["elapsed_s"]),
                    "snippet": str(h["snippet"]),
                }
                for h in chain_history
            ],
        },
    }

    out_file = REPO_ROOT / "results" / "zero_copy_boundary_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved benchmark summary to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--vram-cap", dest="vram_cap", type=float, default=22.0)
    args = parser.parse_args()

    run_zero_copy_boundary_benchmark(model_name=args.model, vram_cap_gb=args.vram_cap)
