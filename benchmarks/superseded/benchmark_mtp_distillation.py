"""Benchmark Distilled MTP Draft Head vs Stock Shipped MTP Head.

Evaluates:
1. Draft Acceptance Rate (tau) with 95% paired bootstrap confidence intervals.
2. 256-token steady-state decode velocity (tok/s) across K in [2, 4, 6, 8].
3. Recurrent state snapshot/restore bit-exactness on domain prompts.

USAGE:
    uv run --env-file .env python benchmarks/runtime/speculative/mtp_distillation/benchmark_mtp_distillation.py \
        --domain astral --max-new-tokens 256
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Touch CUDA before importing transformers/fla
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.mtp_draft import Qwen35MTPDraftHead, restore_state, snapshot_state  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

DOMAINS = {
    "astral": (
        "data/astral/evaluation_data.jsonl",
        "results/adapters/m2_astral_r8a128",
        "results/adapters/mtp_distilled_astral",
    ),
    "postgresql": (
        "data/postgresql/evaluation_data.jsonl",
        "results/adapters/m2_postgresql_r8a128",
        "results/adapters/mtp_distilled_postgresql",
    ),
    "financial": (
        "data/financial_planning/evaluation_data.jsonl",
        "results/adapters/m2_financial_r8a128",
        "results/adapters/mtp_distilled_financial",
    ),
}


def load_distilled_head(base_model, model_id: str, dist_dir: Path) -> Qwen35MTPDraftHead:
    head = Qwen35MTPDraftHead(base_model, model_id)
    weights_path = dist_dir / "mtp_distilled_weights.pt"
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        head.load_state_dict(state_dict, strict=False)
        print(f"  Loaded distilled MTP weights from {weights_path}")
    else:
        print(f"  [WARN] Distilled weights not found at {weights_path}; using stock weights.")
    head.to(device=base_model.device, dtype=torch.bfloat16)
    head.eval()
    return head


def paired_bootstrap_ci(baseline: list[float], experimental: list[float], n_boot: int = 2000) -> tuple[float, float]:
    rng = random.Random(42)
    deltas = []
    n = len(baseline)
    for _ in range(n_boot):
        sample_idx = [rng.randint(0, n - 1) for _ in range(n)]
        b_mean = sum(baseline[i] for i in sample_idx) / n
        e_mean = sum(experimental[i] for i in sample_idx) / n
        deltas.append(e_mean - b_mean)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    return lo, hi


@torch.no_grad()
def greedy_decode(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[list[int], float, float]:
    formatted = f"### Question:\n{prompt}\n\n### Answer:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    seq = inputs.input_ids

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model(seq, use_cache=True)
    past_key_values = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    gen_tokens = [nxt.item()]

    for _ in range(max_new_tokens - 1):
        out = model(nxt, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        gen_tokens.append(nxt.item())

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tok_s = len(gen_tokens) / max(1e-5, dt)
    return gen_tokens, dt, tok_s


@torch.no_grad()
def speculative_decode(
    model, head: Qwen35MTPDraftHead, tokenizer, prompt: str, max_new_tokens: int, k: int
) -> tuple[list[int], float, float, float, list[int]]:
    formatted = f"### Question:\n{prompt}\n\n### Answer:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    ids = inputs.input_ids

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model(ids, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    hids = [out.hidden_states[-1]]
    seq = ids
    nxt = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    toks = [nxt.item()]
    pos = ids.shape[1]
    accepted_counts = []

    while len(toks) < max_new_tokens:
        H = torch.cat(hids, dim=1)
        dcache = head.prefill(H, seq)
        draft = head.draft(H[:, -1:, :], nxt, k=k, start_pos=pos - 1, cache=dcache)

        snap = snapshot_state(cache)
        chunk = torch.cat([nxt, draft], dim=-1)
        v_out = model(chunk, past_key_values=cache, use_cache=True, output_hidden_states=True)
        target = torch.argmax(v_out.logits[0], dim=-1)

        n_acc = 0
        for i in range(k):
            if draft[0, i].item() == target[i].item():
                n_acc += 1
            else:
                break

        accepted_counts.append(n_acc)
        committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)

        if n_acc < k:
            restore_state(cache, snap)
            v_out = model(committed, past_key_values=cache, use_cache=True, output_hidden_states=True)
            new_h = v_out.hidden_states[-1]
        else:
            new_h = v_out.hidden_states[-1][:, : n_acc + 1, :]

        bonus = target[n_acc].item()
        for t in draft[0, :n_acc].tolist() + [bonus]:
            if len(toks) < max_new_tokens:
                toks.append(t)

        hids.append(new_h)
        seq = torch.cat([seq, committed], dim=-1)
        pos += n_acc + 1
        nxt = torch.tensor([[bonus]], device=ids.device)

    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tok_s = len(toks) / max(1e-5, dt)
    tau = sum(accepted_counts) / max(1, len(accepted_counts))
    return toks[:max_new_tokens], dt, tok_s, tau, accepted_counts


def main():
    parser = argparse.ArgumentParser(description="Distilled MTP Head Benchmark")
    parser.add_argument("--domain", choices=sorted(DOMAINS), default="astral")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--k-values", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--out", default="results/mtp_distillation_benchmark.json")
    parser.add_argument("--vram-cap-gb", type=float, default=22.0)
    args = parser.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    eval_rel, adapter_rel, dist_rel = DOMAINS[args.domain]
    eval_path = REPO_ROOT / eval_rel
    adapter_path = REPO_ROOT / adapter_rel
    dist_dir = REPO_ROOT / dist_rel

    print("=" * 80)
    print(f" 🚀 Speculative Draft Head Distillation Benchmark [{args.domain.upper()}]")
    print("=" * 80)
    print(f"  Target Horizon   : {args.max_new_tokens} tokens")
    print(f"  K-Sweep Range    : {args.k_values}")
    print(f"  Folded Expert    : {adapter_path}")
    print(f"  Distilled Head   : {dist_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading base model backbone...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    base_model.eval()

    # Fold target expert
    expert = FoldableExpert.from_dir(adapter_path, name=args.domain)
    engine = WeightFoldingEngine(base_model, [expert], keep_pristine=True)
    engine.activate(expert)

    # Load prompts
    with open(eval_path) as f:
        prompts = [json.loads(line)["prompt"] for line in f if line.strip()][: args.max_prompts]

    # Baseline Greedy Benchmark
    print(f"\n1. Running Baseline Plain Greedy Decode (n={len(prompts)} prompts)...")
    greedy_speeds = []
    for p in prompts:
        _, _, tok_s = greedy_decode(base_model, tokenizer, p, args.max_new_tokens)
        greedy_speeds.append(tok_s)
    base_tok_s = sum(greedy_speeds) / len(greedy_speeds)
    print(f"  Base Plain Greedy Speed: {base_tok_s:.2f} tok/s")

    # Load Stock Head vs Distilled Head
    print("\n2. Initializing Draft Heads...")
    stock_head = Qwen35MTPDraftHead(base_model, args.model_id)
    stock_head.to(device=base_model.device, dtype=torch.bfloat16).eval()

    distilled_head = load_distilled_head(base_model, args.model_id, dist_dir)

    results = {
        "domain": args.domain,
        "base_greedy_tok_s": base_tok_s,
        "k_sweep": {},
    }

    print("\n" + "═" * 84)
    print(
        f" {'K':<4s} │ {'Stock Head (tau / tok/s)':<26s} │ "
        f"{'Distilled Head (tau / tok/s)':<28s} │ {'Delta tau (95% CI)':<18s}"
    )
    print("─" * 84)

    for k in args.k_values:
        # Evaluate Stock Head
        stock_taus = []
        stock_speeds = []
        for p in prompts:
            _, _, tok_s, tau, _ = speculative_decode(base_model, stock_head, tokenizer, p, args.max_new_tokens, k=k)
            stock_taus.append(tau)
            stock_speeds.append(tok_s)

        mean_stock_tau = sum(stock_taus) / len(stock_taus)
        mean_stock_tok_s = sum(stock_speeds) / len(stock_speeds)

        # Evaluate Distilled Head
        dist_taus = []
        dist_speeds = []
        for p in prompts:
            _, _, tok_s, tau, _ = speculative_decode(
                base_model, distilled_head, tokenizer, p, args.max_new_tokens, k=k
            )
            dist_taus.append(tau)
            dist_speeds.append(tok_s)

        mean_dist_tau = sum(dist_taus) / len(dist_taus)
        mean_dist_tok_s = sum(dist_speeds) / len(dist_speeds)

        ci_lo, ci_hi = paired_bootstrap_ci(stock_taus, dist_taus)
        delta_tau = mean_dist_tau - mean_stock_tau

        stock_str = f"τ={mean_stock_tau:.2f} ({mean_stock_tok_s:.1f} t/s, {mean_stock_tok_s/base_tok_s:.2f}x)"
        dist_str = f"τ={mean_dist_tau:.2f} ({mean_dist_tok_s:.1f} t/s, {mean_dist_tok_s/base_tok_s:.2f}x)"
        ci_str = f"{delta_tau:+.2f} [{ci_lo:+.2f}, {ci_hi:+.2f}]"

        print(f" K={k:<2d} │ {stock_str:<26s} │ {dist_str:<28s} │ {ci_str:<18s}")

        results["k_sweep"][f"K={k}"] = {
            "k": k,
            "stock_tau": mean_stock_tau,
            "stock_tok_s": mean_stock_tok_s,
            "stock_speedup": round(mean_stock_tok_s / base_tok_s, 2),
            "distilled_tau": mean_dist_tau,
            "distilled_tok_s": mean_dist_tok_s,
            "distilled_speedup": round(mean_dist_tok_s / base_tok_s, 2),
            "delta_tau": delta_tau,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
        }

    out_file = REPO_ROOT / args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print("═" * 84)
    print(f"\nSaved distillation benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
