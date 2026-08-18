"""TIER 3 — Architectural Adherence & Anti-Pattern Rejection.

Tier 1 (held-out gate) asks whether the adapter damaged the base. Tier 2
(in-domain) asks whether it can solve the problem. Neither measures the reason
the adapter exists: the base already solves most standard problems but carries a
decade of bad habits.

This scores WHICH PATTERN was chosen, which sidesteps a confound that made the
Tier-2 result hard to read. There, base scored 0.067 on Python, most likely
because it emitted prose rather than a fenced block -- so the measurement
conflated output-format compliance with capability. Here, an answer with no
detectable pattern is recorded as NO SIGNAL and excluded from the rate, and the
fenced-block rate is reported separately so format and taste never mix.

TRAINED vs NOVEL
----------------
Reporting one number would repeat the contamination mistake:

    trained  the corpus teaches this fix explicitly (NOT EXISTS 223 hits,
             BETWEEN 309, IDENTITY 164, lower() 390, NULLIF 106). A high score
             proves fine-tuning works -- nothing more.
    novel    ZERO hits in either corpus. This is the score that speaks to taste.

STATISTICS
----------
n is small (14 traps), so this uses McNemar's exact test on DISCORDANT PAIRS
(base wrong / expert right, versus base right / expert wrong) rather than a
bootstrap CI, which would imply precision the sample cannot support.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from math import comb
from pathlib import Path

REPO = Path("/home/mihai/gnn-experiment")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

T0 = time.perf_counter()


def stage(m): print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)


import torch  # noqa: E402

torch.zeros(1, device="cuda"); torch.cuda.synchronize()
from traps import TRAPS  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gnn_experiment.cuda_graph import FoldedCudaGraphDecoder  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

MODEL = "Qwen/Qwen3.5-4B"
ADAPTERS = {
    "postgresql": os.environ.get("PG_ADAPTER", "results/adapters/m2_postgresql_r8a128_v4"),
    "astral": os.environ.get("ASTRAL_ADAPTER", "results/adapters/m2_astral_r8a128_v4"),
}
MAX_NEW = int(os.environ.get("MAX_NEW", "1536"))
MAX_SEQ = 8192
FENCE = re.compile(r"```(?:sql|python|py|bash|sh|shell)?\s*(.*?)```", re.S | re.I)


def extract(t):
    m = FENCE.findall(t)
    return (max(m, key=len) if m else t).strip(), bool(m)


def mcnemar_exact(b01: int, b10: int) -> float:
    """two-sided exact p over the discordant pairs only."""
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    set_hard_vram_cap(22.0)
    stage("loading model")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    experts = {k: FoldableExpert.from_dir(REPO / v, k) for k, v in ADAPTERS.items()}
    engine = WeightFoldingEngine(model, list(experts.values()), keep_pristine=True)
    dec = FoldedCudaGraphDecoder(model, tok, max_seq_len=MAX_SEQ, device=model.device)
    dec.capture(tok("hi", return_tensors="pt").input_ids.to(model.device))
    stage(f"ready — {len(TRAPS)} traps")

    def gen(prompt, expert):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        out, _, _, _ = dec.generate_with_graph(ids, engine=engine, expert=expert,
                                               max_new_tokens=MAX_NEW)
        raw = tok.decode(out, skip_special_tokens=True)
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1]
        return extract(raw.strip())

    rows = []
    for t in TRAPS:
        exp_obj = experts["postgresql"] if t.lang == "sql" else experts["astral"]
        rec = {"id": t.id, "tier": t.tier, "lang": t.lang, "method": t.method}
        for arm, e in (("base", None), ("expert", exp_obj)):
            code, fenced = gen(t.prompt, e)
            bad, good = bool(t.bad(code)), bool(t.good(code))
            verdict = "no_signal" if not (bad or good) else ("bad" if bad else "good")
            rec[arm] = {"verdict": verdict, "fenced": fenced, "bad": bad, "good": good}
        rows.append(rec)
        stage(f"  {t.id:16s} [{t.tier:7s}] base={rec['base']['verdict']:9s} "
              f"expert={rec['expert']['verdict']:9s}")

    stage("=" * 74)
    for tier in ("trained", "novel", "ALL"):
        sub = [r for r in rows if tier == "ALL" or r["tier"] == tier]
        scored = [r for r in sub
                  if r["base"]["verdict"] != "no_signal" and r["expert"]["verdict"] != "no_signal"]
        if not scored:
            stage(f"  {tier:8s} no scorable items")
            continue
        b = sum(r["base"]["verdict"] == "good" for r in scored)
        e = sum(r["expert"]["verdict"] == "good" for r in scored)
        b01 = sum(r["base"]["verdict"] != "good" and r["expert"]["verdict"] == "good"
                  for r in scored)                       # expert fixes it
        b10 = sum(r["base"]["verdict"] == "good" and r["expert"]["verdict"] != "good"
                  for r in scored)                       # expert breaks it
        p = mcnemar_exact(b01, b10)
        stage(f"  {tier:8s} n={len(scored):2d}  base={b}/{len(scored)} ({b/len(scored):.3f})  "
              f"expert={e}/{len(scored)} ({e/len(scored):.3f})  "
              f"discordant +{b01}/-{b10}  McNemar p={p:.4f}"
              f"{'  SIGNIFICANT' if p < 0.05 else ''}")
        dropped = len(sub) - len(scored)
        if dropped:
            stage(f"           ({dropped} excluded as NO SIGNAL — no pattern detected)")

    for arm in ("base", "expert"):
        f = sum(r[arm]["fenced"] for r in rows)
        stage(f"  fenced-block rate {arm:6s}: {f}/{len(rows)} ({f/len(rows):.3f})")
    stage("  (format compliance is reported separately from taste, on purpose)")

    out = REPO / "results/benchmarks/adherence_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "adapters": ADAPTERS}, indent=2))
    stage(f"wrote {out}")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
