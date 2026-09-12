r"""Which adapter is responsible for a bad answer -- solo arms vs the stack.

WHY
---
The morphing studio returned a frozen dataclass to "format this package with ruff
and uv". Two hypotheses survive the corpus audit and they demand OPPOSITE work:

  A. DOMINANCE   astral solo answers it, the stack does not
                 -> python_modern's corpus (76.4% one answer-form) is the problem
  B. ABSENCE     astral solo also fails
                 -> astral never learned it (only 7 records contain [tool.ruff])
                    and rewriting python_modern would be rewriting the wrong file

You cannot tell them apart from the stacked output alone, which is why this exists
before any retraining. Arms per prompt: base, each solo, and the stack.

Greedy, CANON.MAX_NEW_TOKENS. No scoring -- a regex scorer is what produced the
NATIVE=1.0 on an invalid TOML schema (§44). Read the text.

    uv run --env-file .env python \
        benchmarks/factory/agentic/attribution/bench_adapter_attribution.py
"""

from __future__ import annotations

import argparse
import json
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from runtime.canon import CANON, REPO_ROOT, adapter_path  # noqa: E402
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

# An arm entry is "domain" (CANON version) or "domain@ver" to pin a generation.
# Pinning is how the corpus rebuild gets attributed: python_modern@v6 trained on the
# monoculture corpus (24 unique disposition answers), python_modern@v7 on the rebuilt
# one (1418 records, 0.0% duplicate answers). Same prompt, same partner, one variable.
CASES = [
    {"id": "ruff_old", "prompt": "How do I format this package with ruff and uv in pyproject.toml?",
     "arms": [[], ["astral"], ["python_modern@v6"], ["python_modern@v7"],
              ["astral", "python_modern@v6"], ["astral", "python_modern@v7"]],
     "note": "the exact prompt that failed in the studio"},
    {"id": "ruff_new", "prompt": "Add ruff and ty as dev dependencies, then format and lint the whole codebase.",
     "arms": [[], ["astral"], ["python_modern@v6"], ["python_modern@v7"],
              ["astral", "python_modern@v6"], ["astral", "python_modern@v7"]],
     "note": "same intent, unambiguous wording -- separates wording from adapter"},
    {"id": "asyncpg", "prompt": "Now write the asyncpg connection pool for PostgreSQL and similarity search using the pgvector <=> operator.",
     "arms": [[], ["postgresql"], ["postgresql", "python_modern@v6"],
              ["postgresql", "python_modern@v7"]],
     "note": "asyncpg has 0 records in the postgresql v5 corpus; 453 in v6"},
]


@torch.no_grad()
def answer(model, tok, prompt: str, max_new: int) -> tuple[str, int, float]:
    ids = tok(f"### Question:\n{prompt}\n\n### Answer:\n", return_tensors="pt").to(model.device)
    t0 = time.time()
    gen = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         stop_strings=["### Question"], tokenizer=tok)
    dt = time.time() - t0
    new = gen[0][ids["input_ids"].shape[1]:]
    txt = tok.decode(new, skip_special_tokens=True)
    return re.split(r"#+\s*Question", txt)[0].strip(), int(new.numel()), dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/benchmarks/adapter_attribution.json")
    ap.add_argument("--max-new-tokens", type=int, default=CANON.MAX_NEW_TOKENS)
    args = ap.parse_args()

    set_hard_vram_cap(CANON.VRAM_CAP_GB)
    tok = AutoTokenizer.from_pretrained(CANON.BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CANON.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    model.eval()

    wanted = sorted({a for c in CASES for arm in c["arms"] for a in arm})
    experts = {}
    for spec in wanted:
        dom, _, ver = spec.partition("@")
        experts[spec] = FoldableExpert.from_dir(adapter_path(dom, version=ver or None), spec)
        print(f"  loaded {spec:24s} <- {adapter_path(dom, version=ver or None).name}")
    engine = WeightFoldingEngine(model, experts.values(), keep_pristine=True)

    print("=" * 92, flush=True)
    print(f" ADAPTER ATTRIBUTION   adapters={CANON.ADAPTER_VERSION}  "
          f"max_new_tokens={args.max_new_tokens}  greedy", flush=True)
    print("=" * 92, flush=True)

    rows, t0 = [], time.time()
    for case in CASES:
        print(f"\n{'='*92}\n {case['id']}  --  {case['note']}\n Q: {case['prompt']}\n{'='*92}", flush=True)
        for arm in case["arms"]:
            label = "base" if not arm else " + ".join(arm)
            if arm:
                engine.activate_many([experts[d] for d in arm], scale_mode="surgical")
            else:
                engine.restore()
            txt, n, dt = answer(model, tok, case["prompt"], args.max_new_tokens)
            rows.append({"case": case["id"], "arm": label, "prompt": case["prompt"],
                         "tokens": n, "seconds": round(dt, 1), "text": txt})
            print(f"\n----- [{label}]  {n} tok, {dt:.0f}s, {n/max(dt,1e-9):.1f} tok/s "
                  f"{'-'*max(0, 40-len(label))}", flush=True)
            print(txt[:1400], flush=True)
            (REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
            (REPO_ROOT / args.out).write_text(json.dumps(
                {"config": CANON.stamp(), "rows": rows}, indent=2))

    engine.restore()
    print(f"\n\n  {time.time()-t0:.0f}s   wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
