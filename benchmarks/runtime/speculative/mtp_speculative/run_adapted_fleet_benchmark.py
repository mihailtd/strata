"""Run Side-by-Side MTP Speculative Benchmark: Unadapted vs Domain-Adapted Draft Head.

Evaluates speculative decoding performance across all 6 canonical v7 domains
with both the stock MTP head and the domain-adapted MTP micro-adapter.

Usage:
    uv run --env-file .env python \
        benchmarks/runtime/speculative/mtp_speculative/run_adapted_fleet_benchmark.py
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.canon import CANON, REPO_ROOT, adapter_path
from runtime.gpu_preflight import ensure_gpu_exclusive
from runtime.mtp_draft import (
    Qwen35MTPDraftHead,
    fold_mtp_adapter,
    mtp_adapter_path,
    restore_state,
    snapshot_state,
)
from runtime.novel_peft import (
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

DOMAINS = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]

# Domain-specific canonical prompts for realistic domain decoding
DOMAIN_PROMPTS = {
    "astral": [
        "### Question:\nHow do I add a dependency with uv?\n\n### Answer:\n",
        "### Question:\nHow do I configure ruff for modern Python linting and formatting?\n\n### Answer:\n",
    ],
    "postgresql": [
        "### Question:\nWhat is pgvector used for in PostgreSQL?\n\n### Answer:\n",
        "### Question:\nWrite a PostgreSQL query with CTEs and window functions.\n\n### Answer:\n",
    ],
    "duckdb": [
        "### Question:\nHow do I query a parquet file directly in DuckDB?\n\n### Answer:\n",
        "### Question:\nDemonstrate DuckDB spatial or full text search extension.\n\n### Answer:\n",
    ],
    "financial": [
        "### Question:\nExplain loss aversion in one sentence.\n\n### Answer:\n",
        "### Question:\nExplain the difference between traditional and Roth IRA tax treatment.\n\n### Answer:\n",
    ],
    "python_modern": [
        "### Question:\nWrite a modern Python async context manager.\n\n### Answer:\n",
        "### Question:\nWrite a type-safe dataclass using modern Python 3.12 syntax.\n\n### Answer:\n",
    ],
    "python_web": [
        "### Question:\nCreate a FastAPI endpoint with Pydantic validation.\n\n### Answer:\n",
        "### Question:\nWrite a modern async FastAPI dependency injection handler.\n\n### Answer:\n",
    ],
}


@torch.no_grad()
def plain_greedy(model, tok, ids, n_new):
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    for _ in range(n_new - 1):
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks.append(nxt.item())
    return toks


@torch.no_grad()
def speculative(model, tok, head, ids, n_new, k):
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

    return toks[:n_new], stats


def main():
    ensure_gpu_exclusive()
    set_hard_vram_cap(CANON.VRAM_CAP_GB)

    print("=" * 100)
    print("🚀 FLEET SPECULATIVE DECODING BENCHMARK: UNADAPTED vs DOMAIN-ADAPTED MTP HEAD")
    print("=" * 100)

    tok = AutoTokenizer.from_pretrained(CANON.BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CANON.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    experts = [FoldableExpert.from_dir(adapter_path(d), d) for d in DOMAINS]
    engine = WeightFoldingEngine(model, experts, keep_pristine=True)

    head = Qwen35MTPDraftHead(model, CANON.BASE_MODEL)
    head.eval()

    tokens = 192  # standard evaluation budget
    k_eval = 4    # canonical speculative lookahead horizon

    summary_rows = []

    # Snapshot pristine head weights
    pristine_head_weights = {k: p.detach().clone() for k, p in head.named_parameters() if not k.startswith("_")}

    for domain in DOMAINS:
        prompts = DOMAIN_PROMPTS[domain]
        expert = next(e for e in experts if e.name == domain)

        # 1. Activate domain expert in backbone
        engine.activate(expert)

        # Measure baseline greedy throughput
        base_rates = []
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(model.device)
            plain_greedy(model, tok, ids, 4)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            plain_greedy(model, tok, ids, tokens)
            torch.cuda.synchronize()
            base_rates.append(tokens / (time.perf_counter() - t0))
        base_tok_s = sum(base_rates) / len(base_rates)

        # 2. Arm A: Unadapted Stock MTP Head
        # Restore pristine head weights
        for k_name, p_data in pristine_head_weights.items():
            dict(head.named_parameters())[k_name].data.copy_(p_data)

        unadapted_rates, unadapted_accs = [], []
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(model.device)
            speculative(model, tok, head, ids, 4, k_eval)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, st = speculative(model, tok, head, ids, tokens, k_eval)
            torch.cuda.synchronize()
            unadapted_rates.append(tokens / (time.perf_counter() - t0))
            unadapted_accs.append(st)

        unadapted_tok_s = sum(unadapted_rates) / len(unadapted_rates)
        unadapted_d = sum(a["drafted"] for a in unadapted_accs)
        unadapted_a = sum(a["accepted"] for a in unadapted_accs)
        unadapted_steps = sum(a["steps"] for a in unadapted_accs)
        unadapted_pct = 100 * unadapted_a / unadapted_d
        unadapted_tau = unadapted_a / unadapted_steps

        # 3. Arm B: Domain-Adapted MTP Head
        mtp_dir = mtp_adapter_path(domain, "v7")
        fold_mtp_adapter(head, mtp_dir, pristine=pristine_head_weights)

        adapted_rates, adapted_accs = [], []
        for p in prompts:
            ids = tok(p, return_tensors="pt").input_ids.to(model.device)
            speculative(model, tok, head, ids, 4, k_eval)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, st = speculative(model, tok, head, ids, tokens, k_eval)
            torch.cuda.synchronize()
            adapted_rates.append(tokens / (time.perf_counter() - t0))
            adapted_accs.append(st)

        adapted_tok_s = sum(adapted_rates) / len(adapted_rates)
        adapted_d = sum(a["drafted"] for a in adapted_accs)
        adapted_a = sum(a["accepted"] for a in adapted_accs)
        adapted_steps = sum(a["steps"] for a in adapted_accs)
        adapted_pct = 100 * adapted_a / adapted_d
        adapted_tau = adapted_a / adapted_steps

        summary_rows.append({
            "domain": domain,
            "greedy_tok_s": base_tok_s,
            "unadapted_tok_s": unadapted_tok_s,
            "unadapted_speedup": unadapted_tok_s / base_tok_s,
            "unadapted_pct": unadapted_pct,
            "unadapted_tau": unadapted_tau,
            "adapted_tok_s": adapted_tok_s,
            "adapted_speedup": adapted_tok_s / base_tok_s,
            "adapted_pct": adapted_pct,
            "adapted_tau": adapted_tau,
            "delta_tok_s": adapted_tok_s - unadapted_tok_s,
            "delta_pct": adapted_pct - unadapted_pct,
        })

        print(
            f"[{domain:14s}] Greedy: {base_tok_s:5.2f} tok/s | "
            f"Stock MTP: {unadapted_tok_s:5.2f} tok/s ({unadapted_pct:4.1f}%, {unadapted_tok_s/base_tok_s:4.2f}x) -> "
            f"Adapted MTP: {adapted_tok_s:5.2f} tok/s ({adapted_pct:4.1f}%, {adapted_tok_s/base_tok_s:4.2f}x) "
            f"[{'+' if adapted_tok_s >= unadapted_tok_s else ''}{adapted_tok_s - unadapted_tok_s:+.2f} tok/s]"
        )

    # Restore baseline
    engine.restore_pristine()

    print("\n" + "=" * 105)
    print(f"{'Domain':<15s} | {'Greedy':<9s} | {'Stock MTP Head':<22s} | {'Adapted MTP Head':<22s} | {'Gain':<15s}")
    print(f"{'':<15s} | {'(tok/s)':<9s} | {'tok/s (acc%, speedup)':<22s} | {'tok/s (acc%, speedup)':<22s} | {'Δ tok/s (Δ acc%)':<15s}")
    print("-" * 105)
    for r in summary_rows:
        stock_str = f"{r['unadapted_tok_s']:5.2f} ({r['unadapted_pct']:4.1f}%, {r['unadapted_speedup']:4.2f}x)"
        adapt_str = f"{r['adapted_tok_s']:5.2f} ({r['adapted_pct']:4.1f}%, {r['adapted_speedup']:4.2f}x)"
        gain_str = f"{r['delta_tok_s']:+5.2f} tok/s ({r['delta_pct']:+4.1f}%)"
        print(f"{r['domain']:<15s} | {r['greedy_tok_s']:6.2f}   | {stock_str:<22s} | {adapt_str:<22s} | {gain_str:<15s}")
    print("=" * 105)

    out_file = REPO_ROOT / "results/benchmarks/mtp_adapted_fleet_comparison.json"
    out_file.write_text(json.dumps(summary_rows, indent=2))
    print(f"\nArtifact saved to: {out_file}")


if __name__ == "__main__":
    main()
