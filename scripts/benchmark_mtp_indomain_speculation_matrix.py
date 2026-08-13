"""In-Domain Speculative Draft Acceptance & Speedup Matrix Audit (3 x 3 Grid, 9 Cells)

Audits EAGLE-style speculative decoding (K=4) vs autoregressive baseline (K=1)
across a 3x3 matrix of 3 folded domain experts (astral, postgresql, financial_planning)
and 3 domain prompt datasets (20 prompts per domain, 180 total generation runs).

USAGE:
    LD_PRELOAD=/opt/rocm-7.2.0/lib/libhsa-runtime64.so PYTHONUNBUFFERED=1 PYTHONPATH=. \
    .venv/bin/python scripts/benchmark_mtp_indomain_speculation_matrix.py \
        --model-name Qwen/Qwen3.5-4B \
        --out results/mtp_indomain_speculation_matrix.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.mtp_draft import (  # noqa: E402
    Qwen35MTPDraftHead,
    restore_state,
    snapshot_state,
)
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

# Stock LoRA adapters (r=8, a=128)
# Stock LoRA r=8 alpha=128 (scaling 16 -- LoRA's measured peak on astral).
# A previous version of this script used astral_sweep_a128 / pg_sweep_a128 /
# fin_sweep_a128 and described them as "Stock LoRA (r=8, alpha=128)". They are
# id_kron (rank_in=8, rank_out=8 -> rank_total=64, scaling 2.0), which violated
# this audit's own pre-flight rule ("do not use raw id_kron adapters").
EXPERT_ADAPTERS = {
    "astral": "results/adapters/m2_astral_r8a128",
    "postgresql": "results/adapters/m2_postgresql_r8a128",
    "financial_planning": "results/adapters/m2_financial_r8a128",
}

# Domain prompt datasets (20 prompts each)
PROMPT_FILES = {
    "astral": "data/astral/evaluation_data.jsonl",
    "postgresql": "data/postgresql/evaluation_data.jsonl",
    "financial_planning": "data/financial_planning/evaluation_data.jsonl",
}


def load_domain_prompts(questions_file: str) -> list[str]:
    path = Path(questions_file)
    if not path.is_absolute():
        path = REPO_ROOT / path
    prompts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                p = data["prompt"]
                formatted = f"### Question:\n{p}\n\n### Answer:\n"
                prompts.append(formatted)
    return prompts


@torch.no_grad()
def chunked_reference(model, tokenizer, prompt: str, toks: list[int]) -> list[int]:
    """Greedy continuation according to the CHUNKED kernel -- the path verification uses.

    Speculative verification runs the multi-token chunked kernel while plain decode
    runs the single-token recurrent kernel, and those disagree numerically (measured:
    1 of 3 prompts diverges at token 9/32 with no speculation involved). So
    token-exactness against autoregressive output is NOT achievable here, and the
    honest reference for a speculative decoder is the path its verifier computes.
    """
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    full = torch.cat([ids, torch.tensor([toks], device=ids.device)], dim=-1)
    logits = model(full, use_cache=False).logits
    return torch.argmax(logits[0, ids.shape[1] - 1 : -1, :], -1).tolist()


@torch.no_grad()
def run_autoregressive(model, tokenizer, prompt: str, n_new: int = 32) -> tuple[list[int], float]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model(ids, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]

    for _ in range(n_new - 1):
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks.append(nxt.item())

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return toks, elapsed


@torch.no_grad()
def run_speculative(
    model, tokenizer, head, prompt: str, n_new: int = 32, k: int = 4
) -> tuple[list[int], float, dict]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model(ids, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    hids = [out.hidden_states[-1]]
    seq = ids
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    pos = ids.shape[1]
    stats = {"steps": 0, "drafted": 0, "accepted": 0, "full_accepts": 0}

    while len(toks) < n_new:
        H = torch.cat(hids, dim=1)
        dcache = head.prefill(H, seq)
        draft = head.draft(H[:, -1:, :], nxt, k=k, start_pos=pos - 1, cache=dcache)

        snap = snapshot_state(cache)
        chunk = torch.cat([nxt, draft], dim=-1)
        out = model(chunk, past_key_values=cache, use_cache=True, output_hidden_states=True)
        target = torch.argmax(out.logits[0], -1)

        n_acc = 0
        for i in range(k):
            if draft[0, i].item() == target[i].item():
                n_acc += 1
            else:
                break

        stats["steps"] += 1
        stats["drafted"] += k
        stats["accepted"] += n_acc
        stats["full_accepts"] += int(n_acc == k)

        committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
        if n_acc < k:
            restore_state(cache, snap)
            out = model(committed, past_key_values=cache, use_cache=True, output_hidden_states=True)
            new_h = out.hidden_states[-1]
        else:
            new_h = out.hidden_states[-1][:, : n_acc + 1, :]

        bonus = target[n_acc].item()
        for t in draft[0, :n_acc].tolist() + [bonus]:
            if len(toks) < n_new:
                toks.append(t)

        hids.append(new_h)
        seq = torch.cat([seq, committed], dim=-1)
        pos += n_acc + 1
        nxt = torch.tensor([[bonus]], device=ids.device)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return toks[:n_new], elapsed, stats


def main():
    parser = argparse.ArgumentParser(
        description="In-Domain Speculative Draft Acceptance & Speedup Matrix Audit"
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--tokens-per-prompt", type=int, default=32)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--vram-cap-gb", type=float, default=22.0)
    parser.add_argument("--repeats", type=int, default=3, help="interleaved repeats per cell")
    parser.add_argument("--draft-cost", type=float, default=0.4, help="draft cost in units of one auto step")
    parser.add_argument("--p-partial", type=float, default=0.9, help="P(not all K drafts accepted)")
    parser.add_argument(
        "--out",
        default="results/mtp_indomain_speculation_matrix.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(
        f"🔬 AUDITING IN-DOMAIN SPECULATIVE matrix (3x3 Grid, K={args.k}) ON {device} ({dev_name})\n"
        + "═" * 90
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {args.model_name} onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()

    print("Building MTP draft head...")
    head = Qwen35MTPDraftHead(model, args.model_name)

    # Load all 3 prompt datasets
    datasets = {}
    for domain, filepath in PROMPT_FILES.items():
        prompts = load_domain_prompts(filepath)
        datasets[domain] = prompts
        print(f"Loaded {len(prompts)} prompts for domain '{domain}' from {filepath}")

    # Load all 3 expert adapters
    experts = {}
    for domain, adapter_rel in EXPERT_ADAPTERS.items():
        adapter_path = REPO_ROOT / adapter_rel
        if adapter_path.exists():
            experts[domain] = FoldableExpert.from_dir(adapter_path, domain)
            print(f"Loaded expert adapter '{domain}' from {adapter_path}")
        else:
            raise FileNotFoundError(f"Expert adapter for {domain} not found at {adapter_path}")

    folding_engine = WeightFoldingEngine(model, list(experts.values()), keep_pristine=True)

    matrix_results = {}
    grid_domains = ["astral", "postgresql", "financial_planning"]

    print("\n" + "═" * 90)
    print("🚀 EXECUTING 3x3 IN-DOMAIN SPECULATION MATRIX")
    print("═" * 90)

    for exp_domain in grid_domains:
        # Fold expert
        expert = experts[exp_domain]
        folding_engine.activate(expert)
        print(f"\n▶ FOLDED EXPERT: [{exp_domain.upper()}]")
        print("─" * 90)

        matrix_results[exp_domain] = {}

        for prompt_domain in grid_domains:
            prompts = datasets[prompt_domain]
            is_indomain_diagonal = (exp_domain == prompt_domain)
            diag_label = " (PROD DIAGONAL 🔥)" if is_indomain_diagonal else ""
            print(f"  • Evaluating on prompt set [{prompt_domain}]{diag_label} ({len(prompts)} prompts)...")

            # Warmup
            _ = run_autoregressive(model, tokenizer, prompts[0], n_new=4)
            _ = run_speculative(model, tokenizer, head, prompts[0], n_new=4, k=args.k)

            # REPEATS. A previous version measured each cell ONCE and then
            # disabled speculation for a domain on a 3.9% difference. This same
            # class of ratio has measured 1.17x/1.19x/1.31x/1.52x across runs on
            # this box, so a single shot cannot resolve a few percent. Arms stay
            # interleaved per prompt so machine drift hits both equally.
            tot_drafted = tot_accepted = tot_steps = 0
            exact_vs_chunked = 0
            auto_rates, spec_rates = [], []

            for rep in range(args.repeats):
                a_tok = a_time = s_tok = s_time = 0.0
                for p in prompts:
                    toks_auto, time_auto = run_autoregressive(
                        model, tokenizer, p, n_new=args.tokens_per_prompt
                    )
                    a_tok += len(toks_auto)
                    a_time += time_auto

                    toks_spec, time_spec, stats = run_speculative(
                        model, tokenizer, head, p, n_new=args.tokens_per_prompt, k=args.k
                    )
                    s_tok += len(toks_spec)
                    s_time += time_spec

                    if rep == 0:
                        tot_drafted += stats["drafted"]
                        tot_accepted += stats["accepted"]
                        tot_steps += stats["steps"]
                        # CORRECTNESS GATE: greedy speculation must reproduce the
                        # path its verifier computes. Speed numbers from a decoder
                        # that emits different text are not comparable.
                        ref = chunked_reference(model, tokenizer, p, toks_spec)
                        exact_vs_chunked += int(toks_spec == ref[: len(toks_spec)])

                auto_rates.append(a_tok / max(1e-5, a_time))
                spec_rates.append(s_tok / max(1e-5, s_time))

            ratios = sorted(sp / au for sp, au in zip(spec_rates, auto_rates, strict=True))
            speedup = ratios[len(ratios) // 2]
            auto_tok_s = sorted(auto_rates)[len(auto_rates) // 2]
            spec_tok_s = sorted(spec_rates)[len(spec_rates) // 2]
            tau = tot_accepted / max(1, tot_steps)
            accept_pct = (tot_accepted / max(1, tot_drafted)) * 100.0
            exact_pct = 100.0 * exact_vs_chunked / len(prompts)
            straddles_one = ratios[0] < 1.0 < ratios[-1]

            # Predicted speedup from the break-even model, so the model and the
            # measurement can be reconciled rather than silently disagreeing.
            predicted = (tau + 1) / (args.draft_cost + 1.19 * (1 + args.p_partial))

            matrix_results[exp_domain][prompt_domain] = {
                "expert_folded": exp_domain,
                "prompt_domain": prompt_domain,
                "is_indomain_diagonal": is_indomain_diagonal,
                "auto_tok_s": auto_tok_s,
                "spec_tok_s": spec_tok_s,
                "speedup_multiplier": speedup,
                "speedup_all_reps": ratios,
                "speedup_spread": ratios[-1] - ratios[0],
                "straddles_unity": straddles_one,
                "predicted_speedup_from_tau": predicted,
                "tau_accepted_per_step": tau,
                "accept_pct": accept_pct,
                "exact_vs_chunked_pct": exact_pct,
                "repeats": args.repeats,
                "tot_drafted": tot_drafted,
                "tot_accepted": tot_accepted,
                "tot_steps": tot_steps,
            }

            flag = "  ⚠️ SPREAD STRADDLES 1.0 -- not resolvable" if straddles_one else ""
            print(
                f"      tau={tau:.2f}  ratio={speedup:.3f} (reps {ratios[0]:.3f}-{ratios[-1]:.3f})"
                f"  predicted={predicted:.3f}  exact-vs-chunked={exact_pct:.0f}%{flag}"
            )

            if speedup >= 1.05 and tau >= 1.39:
                status_symbol = "✅ WIN"
            elif speedup >= 0.98:
                status_symbol = "⚠️ PARITY"
            else:
                status_symbol = "🔴 SLOWDOWN"

            print(
                f"    -> Auto: {auto_tok_s:5.2f} tok/s | Spec: {spec_tok_s:5.2f} tok/s | "
                f"Speedup: {speedup:5.2f}x | τ: {tau:4.2f} accepted toks ({accept_pct:5.1f}%) | {status_symbol}"
            )

        # Restore backbone to pristine state before next expert fold
        folding_engine.restore()

    # Formulate production gateway routing rules
    speculation_config = {}
    for domain in grid_domains:
        diag_cell = matrix_results[domain][domain]
        # Production gate requires speedup > 1.0 and tau >= 1.39
        is_beneficial = (
            diag_cell["speedup_multiplier"] > 1.00 and diag_cell["tau_accepted_per_step"] >= 1.39
        )
        speculation_config[domain] = is_beneficial

    print("\n" + "═" * 90)
    print("📊 FINAL IN-DOMAIN SPECULATIVE DECODING MATRIX (K=4)")
    print("═" * 90)
    header_str = (
        f"{'Folded Expert':<16s} {'Prompts':<16s} {'Mean τ':<9s} {'Accept %':<9s} "
        f"{'Net Speedup':<13s} {'Gate'}"
    )
    print(header_str)
    print("─" * 90)

    for exp_domain in grid_domains:
        for prompt_domain in grid_domains:
            cell = matrix_results[exp_domain][prompt_domain]
            is_diag = cell["is_indomain_diagonal"]
            diag_tag = " (DIAG 🔥)" if is_diag else ""
            gate_status = (
                "ENABLED ✅"
                if speculation_config[exp_domain]
                else "DISABLED ❌"
            )
            if not is_diag:
                gate_status = "—"

            print(
                f"{exp_domain:<20s} {prompt_domain + diag_tag:<20s} "
                f"{cell['tau_accepted_per_step']:<10.2f} {cell['accept_pct']:<10.1f}% "
                f"{cell['speedup_multiplier']:<14.2f}x {gate_status}"
            )

    print("\n" + "═" * 90)
    print("🎯 PRODUCTION DYNAMIC SPECULATION ROUTER CONFIGURATION")
    print("═" * 90)
    print(json.dumps(speculation_config, indent=2))
    print("═" * 90)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "k_draft": args.k,
        "tokens_per_prompt": args.tokens_per_prompt,
        "matrix": matrix_results,
        "speculation_router_config": speculation_config,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved matrix results to {out_path}")


if __name__ == "__main__":
    main()
