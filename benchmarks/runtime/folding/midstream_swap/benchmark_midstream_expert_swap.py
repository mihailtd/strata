"""Can an expert be swapped MID-GENERATION, and does the carried state survive it?

THE CLAIM
---------
"Mid-Stream Cross-Domain Expert Hot-Swapping (In-Place Folding + ReplaySSM)":
when an agent writing Python hits an inline SQL block, fold to the postgres expert
mid-stream and keep the single response stream going, without resetting the KV
cache or re-forwarding the prefix.

TWO THINGS TO SEPARATE BEFORE MEASURING ANYTHING
------------------------------------------------
1. **ReplaySSM is not what enables this.** The KV cache is a SEPARATE OBJECT from
   the model weights. `WeightFoldingEngine.activate()` mutates weights in place
   and never touches the cache handed to `forward()`. So a mid-stream swap already
   works with folding alone — there is nothing for a rollback primitive to "hold
   stable", because nothing is being rolled back. ReplaySSM is a REWIND mechanism;
   hot-swapping wants to CONTINUE. (Where it would genuinely help: undoing a swap
   that went badly. That is a different, weaker claim.)

2. **The real risk is stale state**, and it is measurable. After swapping A -> B:
     - every KV entry was computed under A's weights; B's queries now attend to
       A's keys
     - the SSM recurrent state is a compressed summary of prior tokens under A's
       dynamics, now consumed by B's

WHAT THIS MEASURES
------------------
    ARM A   expert A throughout                     (no swap)
    ARM B   expert B throughout                     (no swap, the "ideal" for the
                                                     second half)
    ARM C   A for the prefix, swap to B mid-stream, CARRY the cache
    ARM D   A for the prefix, swap to B, but RE-PREFILL the prefix under B
            (discards the carried state)

    C vs D is the whole experiment. D is what you get by paying the full prefix
    cost; C is what the carried state gives you for free. If C ~ D, the stale
    state is benign and the capability is real. If C diverges toward A (or toward
    incoherence), the carried state is contaminated and the swap needs a reset.

    Post-swap text is scored with the repo's own postgres term lists, so the
    number is comparable to every other quality figure here.

    uv run --env-file .env python \
        benchmarks/runtime/folding/midstream_swap/benchmark_midstream_expert_swap.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "runtime" / "folding"))

from evaluate_folded_vs_wrapped import (  # noqa: E402
    POSTGRES_LEGACY_TERMS,
    POSTGRES_MODERN_TERMS,
)

from runtime.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

# Prompts that genuinely cross a domain boundary mid-answer: Python tooling first
# (astral's territory), then an embedded SQL / pgvector block (postgres's).
PROMPTS = [
    "### Question:\nWrite a Python script, managed with uv, that connects to "
    "PostgreSQL and finds the 5 most similar documents to a query. Show the "
    "SQL it runs.\n\n### Answer:\n",
    "### Question:\nSet up a Python project with uv that stores document "
    "embeddings in PostgreSQL. Include the table definition and the lookup "
    "query.\n\n### Answer:\n",
    "### Question:\nUsing uv and ruff for tooling, write the data layer for a "
    "recommendation service backed by PostgreSQL similarity search. Include the "
    "schema.\n\n### Answer:\n",
]

ASTRAL = "results/adapters/m2_astral_r8a128"
POSTGRES = "results/adapters/m2_postgresql_r8a128"


def score_pg(text: str) -> float:
    low = text.lower()
    g = sum(len(re.findall(p, low)) for p in POSTGRES_MODERN_TERMS)
    b = sum(len(re.findall(p, low)) for p in POSTGRES_LEGACY_TERMS)
    return (g / max(1, g + b)) * 100.0 if (g + b) else 0.0


@torch.no_grad()
def gen(model, tok, prompt: str, n: int, engine=None, expert_b=None, swap_at: int | None = None):
    """Greedy decode, optionally folding to `expert_b` after `swap_at` tokens.

    The cache is NOT rebuilt at the swap -- that is the whole point of the claim.
    """
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    eos = tok.eos_token_id
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    swap_ms = 0.0

    while len(toks) < n and toks[-1] != eos:
        if swap_at is not None and len(toks) == swap_at and engine is not None:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            engine.activate(expert_b)
            torch.cuda.synchronize()
            swap_ms = 1000.0 * (time.perf_counter() - t0)
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks.append(nxt.item())
    return toks, swap_ms, cache


@torch.no_grad()
def gen_reprefill(model, tok, prompt: str, prefix_toks: list[int], n_more: int):
    """ARM D: rebuild the cache from scratch under the NEW expert.

    Feeds prompt + the prefix already generated, so the model sees identical text
    but with every KV entry and the recurrent state computed under expert B.
    """
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    full = torch.cat([ids, torch.tensor([prefix_toks], device=ids.device)], dim=-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(full, use_cache=True)
    torch.cuda.synchronize()
    reprefill_ms = 1000.0 * (time.perf_counter() - t0)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    eos = tok.eos_token_id
    toks = [nxt.item()]
    while len(toks) < n_more and toks[-1] != eos:
        out = model(nxt, past_key_values=cache, use_cache=True)
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks.append(nxt.item())
    return toks, reprefill_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--tokens", type=int, default=96)
    ap.add_argument("--swap-at", type=int, default=32, help="token index of the domain switch")
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/midstream_swap.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  MID-STREAM EXPERT SWAP — does the carried state survive it?")
    print("=" * 100)
    print(f"  device={dev}  tokens={args.tokens}  swap at token {args.swap_at}\n")

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    ).eval()

    ast = FoldableExpert.from_dir(REPO_ROOT / ASTRAL, "astral")
    pg = FoldableExpert.from_dir(REPO_ROOT / POSTGRES, "postgresql")
    engine = WeightFoldingEngine(model, [ast, pg], keep_pristine=True)

    # warmup
    engine.activate(ast)
    gen(model, tok, PROMPTS[0], 8)
    engine.restore()

    rows = []
    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- prompt {i + 1}/{len(PROMPTS)} ---")

        engine.activate(ast)
        a_toks, _, _ = gen(model, tok, prompt, args.tokens)

        engine.restore()
        engine.activate(pg)
        b_toks, _, _ = gen(model, tok, prompt, args.tokens)

        engine.restore()
        engine.activate(ast)
        c_toks, swap_ms, _ = gen(model, tok, prompt, args.tokens,
                                 engine=engine, expert_b=pg, swap_at=args.swap_at)

        # ARM D: same prefix, but rebuild the cache under postgres
        engine.restore()
        engine.activate(pg)
        d_tail, reprefill_ms = gen_reprefill(
            model, tok, prompt, c_toks[: args.swap_at], args.tokens - args.swap_at
        )
        engine.restore()

        def tail_text(t):
            return tok.decode(t, skip_special_tokens=True)

        c_tail = c_toks[args.swap_at:]
        sa = score_pg(tail_text(a_toks[args.swap_at:]))
        sb = score_pg(tail_text(b_toks[args.swap_at:]))
        sc = score_pg(tail_text(c_tail))
        sd = score_pg(tail_text(d_tail))
        cd_exact = c_tail[: len(d_tail)] == d_tail[: len(c_tail)]
        div = next((j for j, (x, y) in enumerate(zip(c_tail, d_tail, strict=False)) if x != y), None)

        print(f"    post-swap postgres score   A(astral only) {sa:5.1f}   B(pg only) {sb:5.1f}"
              f"   C(swap, carried) {sc:5.1f}   D(swap, re-prefilled) {sd:5.1f}")
        print(f"    swap cost {swap_ms:6.2f} ms   vs re-prefill {reprefill_ms:7.2f} ms"
              f"   ({reprefill_ms / max(1e-9, swap_ms):.1f}x)")
        print(f"    C vs D tokens identical: {'YES' if cd_exact else f'NO (diverge at {div})'}")
        rows.append({
            "prompt": i, "score_a_astral": sa, "score_b_pg": sb,
            "score_c_carried": sc, "score_d_reprefilled": sd,
            "swap_ms": swap_ms, "reprefill_ms": reprefill_ms,
            "c_equals_d": cd_exact, "first_divergence": div,
            "text_c": tail_text(c_tail), "text_d": tail_text(d_tail),
        })

    n = len(rows)
    m = {k: sum(r[k] for r in rows) / n for k in
         ("score_a_astral", "score_b_pg", "score_c_carried", "score_d_reprefilled",
          "swap_ms", "reprefill_ms")}

    print("\n" + "=" * 100)
    print("  RESULT")
    print("=" * 100)
    print(f"    post-swap postgres score, mean of {n}")
    print(f"      A  astral throughout        {m['score_a_astral']:6.2f}")
    print(f"      B  postgres throughout      {m['score_b_pg']:6.2f}   <- the ceiling")
    print(f"      C  swap, state CARRIED      {m['score_c_carried']:6.2f}   <- the claim")
    print(f"      D  swap, state RE-PREFILLED {m['score_d_reprefilled']:6.2f}   <- carried-state control")
    print()
    print(f"    swap {m['swap_ms']:.2f} ms vs re-prefill {m['reprefill_ms']:.2f} ms "
          f"({m['reprefill_ms'] / max(1e-9, m['swap_ms']):.1f}x cheaper)")
    print(f"    C == D on {sum(r['c_equals_d'] for r in rows)}/{n} prompts")
    print()
    if m["score_c_carried"] >= m["score_d_reprefilled"] - 5:
        print("    => CARRIED STATE IS BENIGN. The swap works without a reset, and the")
        print("       expensive re-prefill buys nothing measurable.")
    else:
        print("    => CARRIED STATE IS CONTAMINATED. The swap needs a prefix reset,")
        print("       which removes most of the advantage.")
    print("\n    NOTE: this needs NO ring buffer. The cache is a separate object from")
    print("    the weights, so in-place folding alone delivers the swap. ReplaySSM")
    print("    would only matter for UNDOING a swap.")

    out_p = REPO_ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"device": dev, "swap_at": args.swap_at,
                                 "tokens": args.tokens, "means": m, "rows": rows}, indent=2))
    print(f"\n  Saved -> {out_p.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
