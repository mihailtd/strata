"""Controlled head-to-head: stock LoRA vs id_kron, matched scaling, each at its peak.

REPLACES a retracted audit that compared four PRE-EXISTING adapters of unknown
provenance. That audit reported id_kron at "r=64, 49.8M" when the adapter it
loaded was rank_total=8 / 6.26M (inverting its own headline -- id_kron used 41%
FEWER params than the LoRA that beat it), had no training records for any arm,
and left effective scaling uncontrolled at 32.0 vs 2.0.

WHAT IS CONTROLLED HERE
-----------------------
Every adapter is trained fresh: astral (815 records), 150 steps, batch 2,
lr 2e-4, bf16. The only variables are architecture and effective scaling.

**Scaling is matched, not alpha.** Effective scaling is alpha/rank_total, so
comparing alpha=256 at r=8 against alpha=32 at rank_total=16 compares scaling 32
against scaling 2 -- which is what the retracted audit did. Here each
architecture is swept over the SAME scalings and compared at its own peak.

    stock LoRA r=8        rank_total=8   10.62M params
    id_kron rin=8,rout=1  rank_total=8    6.26M params
    id_kron rin=16,rout=1 rank_total=16  12.42M params

TRAINING-LOSS CONTEXT (measured; loss is NOT quality here -- see below)

    scaling      0.25    0.5     1.0     2.0     8.0    16.0    32.0
    LoRA       1.0646  1.0502  1.0371  1.0274  1.0290  1.0556  1.1464
    idk rt8    1.0357  1.0444  1.0918  1.2067  7.6679  8.0602  7.7288
    idk rt16   1.0337  1.0574    ...   1.2799  7.2853  7.0332  7.9349

Two things already follow from loss alone:
  * LoRA is stable across a 128x scaling range; id_kron diverges between
    scaling 2 and 8 at this LR, independent of rank. That is a real robustness
    difference and it is not a subtle ranking question -- loss ~7.7 is a broken
    model, so those points are excluded from scoring.
  * The loss curves CROSS (id_kron lower at scaling 0.25, LoRA lower at >=1.0),
    which is exactly why a single-point comparison can yield either verdict.

But this repo has repeatedly measured training loss ANTI-CORRELATED with
adherence -- the astral alpha sweep peaked in quality where loss was mid-range.
So the ranking below comes from the rubric, not the loss.

    uv run --env-file .env scripts/eval_controlled_headtohead.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch

if torch.cuda.is_available():  # fla's device probe is @cache'd at import
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.eval.eval_suite import LEGACY_TERMS, MODERN_TERMS  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    load_novel_adapter,
    set_hard_vram_cap,
    unwrap_novel_lora,
)

QUESTIONS = "data/astral/evaluation_data.jsonl"
# (label, kind, rank_total, alpha) -- diverged points (scaling >= 8 for id_kron)
# are excluded: loss ~7-8 means the model is broken, not merely worse.
ARMS = (
    [("lora", "peft", 8, a) for a in (2, 4, 8, 16, 64, 128, 256)]
    + [("idk_rt8", "novel", 8, a) for a in (2, 4, 8, 16)]
    + [("idk_rt16", "novel", 16, a) for a in (4, 8, 16, 32)]
)
DIRS = {"lora": "ctl_lora_r8_a{}", "idk_rt8": "ctl_idk_rt8_a{}", "idk_rt16": "ctl_idk_rt16_a{}"}


@torch.no_grad()
def score(model, tok, questions, max_new_tokens):
    out = []
    for q in questions:
        ids = tok(f"### Question:\n{q['prompt']}\n\n### Answer:\n", return_tensors="pt").to(model.device)
        gen = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            stop_strings=["### Question"],
            tokenizer=tok,
        )
        txt = tok.decode(gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        txt = re.split(r"#+\s*Question", txt)[0].strip().lower()
        g = sum(len(re.findall(p, txt)) for p in MODERN_TERMS)
        b = sum(len(re.findall(p, txt)) for p in LEGACY_TERMS)
        out.append((g / max(1, g + b)) * 100.0 if (g + b) else 0.0)
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/controlled_headtohead.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    questions = [json.loads(x) for x in (REPO_ROOT / QUESTIONS).read_text().splitlines() if x.strip()]

    def fresh():
        m = AutoModelForCausalLM.from_pretrained(
            args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
        )
        m.eval()
        return m

    model = fresh()
    base = score(model, tok, questions, args.max_new_tokens)
    print(f"base (no adapter): {base:.2f}%\n")
    print(f"  {'arm':10s} {'scaling':>8s} {'alpha':>6s} {'score':>8s}")

    rows = []
    for kind_label, kind, rt, alpha in ARMS:
        d = REPO_ROOT / "results" / "adapters" / DIRS[kind_label].format(alpha)
        if not d.exists():
            print(f"  {kind_label:10s} {alpha / rt:8.2f} {alpha:6d}   MISSING {d.name}")
            continue
        if kind == "novel":
            load_novel_adapter(model, d)
            s = score(model, tok, questions, args.max_new_tokens)
            unwrap_novel_lora(model)
        else:
            from peft import PeftModel

            pm = PeftModel.from_pretrained(model, str(d))
            pm.eval()
            s = score(pm, tok, questions, args.max_new_tokens)
            pm.unload()
        rows.append({"arm": kind_label, "rank_total": rt, "alpha": alpha, "scaling": alpha / rt, "score": s})
        print(f"  {kind_label:10s} {alpha / rt:8.2f} {alpha:6d} {s:7.2f}%", flush=True)

    print("\n" + "=" * 70)
    print(" PEAK PER ARCHITECTURE (the only fair comparison)")
    print("=" * 70)
    print(f" base {base:.2f}%\n")
    print(f" {'architecture':12s} {'params':>9s} {'best scaling':>13s} {'peak':>8s} {'vs base':>9s}")
    params = {"lora": "10.62M", "idk_rt8": "6.26M", "idk_rt16": "12.42M"}
    best = {}
    for a in ("lora", "idk_rt8", "idk_rt16"):
        sub = [r for r in rows if r["arm"] == a]
        if not sub:
            continue
        b = max(sub, key=lambda r: r["score"])
        best[a] = b
        print(f" {a:12s} {params[a]:>9s} {b['scaling']:13.2f} {b['score']:7.2f}% {b['score'] - base:+8.2f}pp")

    if "lora" in best:
        print()
        for a in ("idk_rt8", "idk_rt16"):
            if a in best:
                d = best[a]["score"] - best["lora"]["score"]
                verdict = "beats" if d > 0 else "loses to"
                print(f" {a} {verdict} stock LoRA by {abs(d):.2f}pp at matched-scaling peaks")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"base": base, "rows": rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
