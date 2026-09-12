"""A/B one prompt across base / system-prompt / every expert. Side by side.

For trying an idea in 60 seconds instead of committing to a 4-hour benchmark.

    uv run --env-file .env python apps/runtime/ab.py "your prompt here"
    uv run --env-file .env python apps/runtime/ab.py -f prompt.txt --experts astral
    uv run --env-file .env python apps/runtime/ab.py "..." --system "This project uses uv."

WHY A SYSTEM PROMPT HERE IS **CONTEXT**, NOT INSTRUCTIONS
---------------------------------------------------------
The disposition benchmark's system-prompt arm was written as a cheat sheet -- it
listed `uv add`, `uv lock`, `read_parquet`, `pgvector` and told the model never to
use pip. That is not a baseline, it is an answer key, and the model only had to
echo it back. It is why that arm "won".

A fair system prompt states the SITUATION and stops:

    "This project uses uv and ruff."          <- context, fair
    "Use uv add. Never use pip."              <- answer key, not a baseline

--system takes whatever you give it. Just know which of the two you are testing.
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from runtime.canon import CANON, DOMAINS, REPO_ROOT, adapter_path  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "apps"))
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

SKIP = {"merged_sql", "merged_all"}  # not per-domain; ask for them explicitly


@torch.no_grad()
def gen(model, tok, prompt: str, system: str, max_new: int) -> tuple[str, int, float]:
    pre = f"{system}\n\n" if system else ""
    ids = tok(f"{pre}### Question:\n{prompt}\n\n### Answer:\n", return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         stop_strings=["### Question"], tokenizer=tok)
    dt = time.perf_counter() - t0
    new = out[0][ids["input_ids"].shape[1]:]
    txt = tok.decode(new, skip_special_tokens=True)
    return re.split(r"#+\s*Question", txt)[0].strip(), int(new.numel()), dt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="the prompt (or use -f)")
    ap.add_argument("-f", "--file", help="read the prompt from a file")
    ap.add_argument("--system", default="", help="system text prepended to every arm")
    ap.add_argument("--experts", nargs="*", default=None,
                    help=f"which experts to fold (default: all of {[d for d in DOMAINS if d not in SKIP]})")
    ap.add_argument("--max-new-tokens", type=int, default=CANON.MAX_NEW_TOKENS)
    ap.add_argument("--stack", action="store_true",
                    help="also run every named expert folded TOGETHER")
    args = ap.parse_args()

    prompt = open(args.file).read().strip() if args.file else args.prompt
    if not prompt:
        ap.error("give a prompt, or -f FILE")

    names = args.experts if args.experts is not None else [d for d in DOMAINS if d not in SKIP]

    set_hard_vram_cap(CANON.VRAM_CAP_GB)
    tok = AutoTokenizer.from_pretrained(CANON.BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CANON.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    model.eval()

    experts = {n: FoldableExpert.from_dir(adapter_path(n), n) for n in names}
    engine = WeightFoldingEngine(model, experts.values(), keep_pristine=True)

    arms: list[tuple[str, list, str]] = [("base", [], "")]
    if args.system:
        arms.append(("base+system", [], args.system))
    arms += [(n, [experts[n]], "") for n in names]
    if args.stack and len(names) > 1:
        arms.append(("+".join(names), [experts[n] for n in names], ""))

    print("=" * 100)
    print(f"PROMPT: {prompt}")
    if args.system:
        print(f"SYSTEM: {args.system}")
    print(f"(greedy, max_new_tokens={args.max_new_tokens}, {CANON.ADAPTER_VERSION} adapters)")
    print("=" * 100)

    rows = []
    for label, exps, sysmsg in arms:
        engine.restore() if not exps else engine.activate_many(exps)
        txt, ntok, dt = gen(model, tok, prompt, sysmsg, args.max_new_tokens)
        rows.append((label, ntok, dt, txt))
        print(f"\n\033[1m--- {label}  ({ntok} tok, {dt:.1f}s, {ntok/max(dt,1e-9):.1f} tok/s) "
              f"{'-' * max(0, 60 - len(label))}\033[0m")
        print(txt)

    engine.restore()
    print("\n" + "=" * 100)
    print(f" {'arm':28s} {'tokens':>8s} {'secs':>7s} {'tok/s':>8s}")
    for label, ntok, dt, _ in rows:
        print(f" {label:28s} {ntok:8d} {dt:7.1f} {ntok/max(dt,1e-9):8.1f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
