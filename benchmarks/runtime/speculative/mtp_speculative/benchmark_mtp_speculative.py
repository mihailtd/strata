"""End-to-End MTP Speculative Decoding K-Sweep Benchmark (Qwen 3.5 4B).

Audits EAGLE-style speculative decoding using the native Multi-Token Prediction (MTP)
draft head across draft horizons K in [2, 4, 6, 8] at 256-token steady state.

Key Mechanics:
1. Shipped MTP Draft Head (1-layer EAGLE attention) drafts K tokens from committed hidden states.
2. 52.5 MB Fixed-Size Recurrent State Snapshot & Restore for bit-exact rollback on partial rejection.
3. Fast chunked verification via active `flash-linear-attention` Triton kernels on AMD ROCm (gfx1100).

For full technical documentation, verification cost models, and analysis, see:
    benchmarks/runtime/speculative/mtp_speculative/README.md

Usage:
    uv run --env-file .env python \
        benchmarks/runtime/speculative/mtp_speculative/benchmark_mtp_speculative.py --tokens 256 --k 2 4 6 8
"""

import os
import sys

import argparse
import json
import time
from pathlib import Path

import torch

# fla's device probe is @cache'd at import; touch CUDA before transformers pulls
# it in or the process latches to a fallback for its whole lifetime
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.gpu_preflight import ensure_gpu_exclusive  # noqa: E402

from runtime.mtp_draft import (  # noqa: E402
    Qwen35MTPDraftHead,
    fold_mtp_adapter,
    mtp_adapter_path,
    restore_state,
    snapshot_state,
    state_nbytes,
)
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

from runtime.canon import adapter_path

ADAPTERS = {
    "astral": str(adapter_path("astral")),
    "postgresql": str(adapter_path("postgresql")),
    "duckdb": str(adapter_path("duckdb")),
    "financial": str(adapter_path("financial")),
    "python_modern": str(adapter_path("python_modern")),
    "python_web": str(adapter_path("python_web")),
}

PROMPTS = [
    "### Question:\nHow do I add a dependency with uv?\n\n### Answer:\n",
    "### Question:\nWhat is pgvector used for in PostgreSQL?\n\n### Answer:\n",
    "### Question:\nHow do I query a parquet file directly in DuckDB?\n\n### Answer:\n",
    "### Question:\nExplain loss aversion in one sentence.\n\n### Answer:\n",
    "### Question:\nWrite a modern Python async context manager.\n\n### Answer:\n",
    "### Question:\nCreate a FastAPI endpoint with Pydantic validation.\n\n### Answer:\n",
]


@torch.no_grad()
def plain_greedy(model, tok, ids, n_new):
    # Let the model build its own cache. A bare DynamicCache() fails here: the
    # hybrid model needs LinearAttentionLayer entries for its 24 GatedDeltaNet
    # layers, and constructing the cache manually yields an empty layer list
    # (IndexError inside update_conv_state on the first decode step).
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
    """Speculative decode with the MTP head, keeping the head's context intact."""
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

        committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)  # positions pos .. pos+n_acc
        if n_acc < k:
            # state advanced by k+1, only n_acc+1 tokens are real -> restore and redo
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


@torch.no_grad()
def chunked_reference(model, ids, toks):
    """What the CHUNKED kernel says the greedy continuation is."""
    full = torch.cat([ids, torch.tensor([toks], device=ids.device)], dim=-1)
    T = ids.shape[1]
    logits = model(full, use_cache=False).logits
    return torch.argmax(logits[0, T - 1 : -1, :], -1).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--adapter", default="none", choices=["none", *ADAPTERS])
    ap.add_argument("--adapt-mtp-head", action="store_true", help="Fold domain-adapted MTP micro-adapter into head")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ensure_gpu_exclusive()
    set_hard_vram_cap(args.vram_cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    head = Qwen35MTPDraftHead(model, args.model_name)

    engine = None
    if args.adapter != "none":
        expert = FoldableExpert.from_dir(Path(ADAPTERS[args.adapter]), args.adapter)
        engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
        engine.activate(expert)
        print(f"folded domain expert into backbone: {args.adapter}")

        if args.adapt_mtp_head:
            mtp_p = mtp_adapter_path(args.adapter, "v7")
            if mtp_p.exists():
                fold_mtp_adapter(head, mtp_p)
                print(f"folded domain micro-adapter into MTP draft head: {mtp_p.name}\n")
            else:
                print(f"⚠️ MTP adapter not found: {mtp_p}; using unadapted head\n")
        else:
            print("using unadapted baseline MTP draft head\n")

    default_suffix = "_adapted_head" if args.adapt_mtp_head else ""
    out_path = args.out or f"results/benchmarks/mtp_speculative_{args.adapter}{default_suffix}.json"

    ids0 = tok(PROMPTS[0], return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        c = model(ids0, use_cache=True).past_key_values
    print(f"recurrent+kv state snapshot size: {state_nbytes(c) / 1e6:.1f} MB\n")

    # baseline
    base_rates = []
    refs = {}
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(model.device)
        plain_greedy(model, tok, ids, 4)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        refs[p] = plain_greedy(model, tok, ids, args.tokens)
        torch.cuda.synchronize()
        base_rates.append(args.tokens / (time.perf_counter() - t0))
    base = sum(base_rates) / len(base_rates)
    print(f"baseline plain greedy: {base:.2f} tok/s\n")

    results = {"adapter": args.adapter, "baseline_tok_s": base, "arms": {}}
    print(
        f"{'K':>3s} {'tok/s':>8s} {'speedup':>8s} {'accept%':>8s} {'acc/step':>9s} "
        f"{'vs 1tok':>8s} {'vs chunk':>9s}"
    )
    for k in args.k:
        rates, accs, exact, exact_chunk = [], [], 0, 0
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(model.device)
            speculative(model, tok, head, ids, 4, k)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            got, st = speculative(model, tok, head, ids, args.tokens, k)
            torch.cuda.synchronize()
            rates.append(args.tokens / (time.perf_counter() - t0))
            accs.append(st)
            exact += int(got == refs[p])
            chunk_ref = chunked_reference(model, ids, got)
            exact_chunk += int(got == chunk_ref)
        r = sum(rates) / len(rates)
        tot_d = sum(a["drafted"] for a in accs)
        tot_a = sum(a["accepted"] for a in accs)
        steps = sum(a["steps"] for a in accs)
        flag = "OK" if exact_chunk == len(PROMPTS) else "*** DIVERGED ***"
        print(
            f"{k:3d} {r:7.2f} {r / base:7.2f}x {100 * tot_a / tot_d:7.1f}% "
            f"{tot_a / steps:8.2f} {exact}/{len(PROMPTS)} {exact_chunk}/{len(PROMPTS)} {flag}"
        )
        results["arms"][k] = {
            "tok_s": r,
            "speedup": r / base,
            "accept_pct": 100 * tot_a / tot_d,
            "accepted_per_step": tot_a / steps,
            "exact_vs_single_token": exact,
            "exact_vs_chunked": exact_chunk,
            "n_prompts": len(PROMPTS),
        }

    out = REPO_ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
