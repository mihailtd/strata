"""Compare adapter generations THROUGH THE SERVING PATH: base vs vOLD vs vNEW.

Why a new instrument
--------------------
`evals/factory/agentic/disposition/bench_disposition_v2.py` prompts every arm --
base included -- as `### Question: ... ### Answer:`, the format the legacy
adapters were trained on and the served model never sees. It therefore measured
each adapter in the one position it was fitted to. This script sends every
request through `runtime-next` with the real Qwen3.5 chat template (thinking on,
as served) and swaps adapters per request, so all arms are measured exactly as a
user would get them.

Two axes, because either alone can hide the other (MEASURED_FINDINGS §12):

  correctness  HumanEval pass@1 (164 problems), paired exact McNemar vs base and
               between generations. The "did it lose intelligence" axis.
  style        held-out disposition items (expects/avoid regexes), scored on the
               ANSWER after `</think>` only -- reasoning that names a rejected
               tool ("uv instead of pip because...") must not trip `avoid`, the
               scoring artifact the v6 changelog entry documents. For
               agentic_coding, which has no expects/avoid set, the style axis is
               SEARCH/REPLACE format adherence on its 179 held-out prompts
               (verified: 0 of them appear in training corpus v6).
  + ruff findings on PASSING HumanEval solutions (UP,C4,SIM,PTH,PERF,RET), the
    style signal that is independent of the regex eval sets.

Requests go through `/v1/chat/completions/batch` (one adapter per batch, which
is exactly the per-arm shape). Every arm uses the same path and batch size, so
the batch-shape GEMM non-associativity documented in §16 applies equally to all
arms.

Every raw completion is stored. A number without the text behind it cannot be
audited, which is how three separate harness bugs survived in this repo.

Usage (server started from the repo root, single-engine rule):
    python3 evals/factory/compare_adapter_generations.py --size 4b --old v7 --new v9
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from math import comb
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.humaneval.dataset import load_humaneval_problems  # noqa: E402
from benchmarks.humaneval.executor import HumanEvalExecutor, clean_code  # noqa: E402

DOMAINS = ["agentic_coding", "python_modern", "python_web", "astral", "duckdb", "postgresql", "financial"]
# Corpus directory differs from the adapter stem only for financial.
EVAL_FILE = {
    "python_modern": "data/python_modern/evaluation_data_disposition.jsonl",
    "python_web": "data/python_web/evaluation_data_disposition.jsonl",
    "astral": "data/astral/evaluation_data_disposition.jsonl",
    "duckdb": "data/duckdb/evaluation_data_disposition.jsonl",
    "postgresql": "data/postgresql/evaluation_data_disposition.jsonl",
    "financial": "data/financial_planning/evaluation_data_disposition.jsonl",
    "agentic_coding": "apps/factory/data/agentic_coding/eval_data_v3.jsonl",
}
HE_SYSTEM = (
    "You are an expert Python coding assistant. "
    "Complete the requested function cleanly. "
    "Output valid Python code without Markdown explanations if possible, or inside a ```python block."
)
RUFF_RULES = "UP,C4,SIM,PTH,PERF,RET"
# The latest SHIPPED generation per domain, which is what a new generation must
# beat. agentic_coding shipped a v8 on corpus v6; every other domain is at v7.
# Verified from results/benchmarks/server_4b_humaneval.log, which pre-loaded
# m2_python_modern_r8a128_v7 and m2_agentic_coding_r8a128_v8 for §12.
OLD_OVERRIDES = {"agentic_coding": "v8"}
POINTS = {"NATIVE": 1.0, "MIXED": 0.5, "MANUAL": 0.0, "NEITHER": 0.0}
# A well-formed aider-style edit block: SEARCH text, separator, REPLACE text, close.
SEARCH_REPLACE = re.compile(r"<<<<<<< SEARCH\n.*?\n=======\n.*?\n>>>>>>>", re.DOTALL)


def old_of(domain: str, old: str) -> str:
    """The baseline arm. Overrides exist only for the shipped generation (v7):
    comparing v9 to v10 needs no substitution."""
    return OLD_OVERRIDES.get(domain, old) if old == "v7" else old


def adapter_name(domain: str, version: str, size: str) -> str:
    suffix = "" if size == "4b" else f"_{size.replace('.', '_')}"
    return f"m2_{domain}_r8a128_{version}{suffix}"


def answer_part(text: str) -> tuple[str, bool]:
    """(answer after the last </think>, whether the think block closed)."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip(), True
    return "", False


def batch_generate(url: str, prompts: list[list[dict]], adapter: str | None, max_tokens: int,
                   batch_size: int) -> list[dict]:
    out: list[dict] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start: start + batch_size]
        # ALWAYS name the adapter, "base" included: runtime-next's adapter state is
        # sticky -- a request with no `adapter` field runs under whatever the
        # previous request folded in, so an implicit base arm would silently be
        # the last adapter measured.
        payload: dict = {"batch": [{"messages": m} for m in chunk], "max_tokens": max_tokens,
                         "temperature": 0.0, "adapter": adapter or "base"}
        req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
                      method="POST")
        try:
            with urlopen(req, timeout=3600) as resp:
                body = json.loads(resp.read())
        except HTTPError as exc:
            raise RuntimeError(f"{adapter or 'base'}: {exc.read().decode(errors='replace')}") from exc
        for r in sorted(body["responses"], key=lambda r: r["index"]):
            out.append({"text": r["message"]["content"], "finish_reason": r["finish_reason"],
                        "tokens": r.get("usage", {}).get("completion_tokens")})
        print(f"    {adapter or 'base'}: {len(out)}/{len(prompts)}", flush=True)
    return out


def classify(text: str, item: dict) -> str:
    exp = any(re.search(p, text, re.I) for p in item.get("expects", []))
    avo = any(re.search(p, text, re.I) for p in item.get("avoid", []))
    return "NATIVE" if exp and not avo else "MIXED" if exp else "MANUAL" if avo else "NEITHER"


def ruff_findings(code: str) -> int | None:
    ruff = shutil.which("ruff") or str(REPO_ROOT / ".venv" / "bin" / "ruff")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([ruff, "check", "--select", RUFF_RULES, "--output-format", "json", "--no-cache",
                            "--isolated", path], capture_output=True, text=True, timeout=30)
        return len(json.loads(r.stdout or "[]"))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    finally:
        Path(path).unlink(missing_ok=True)


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_first": only_a, "only_second": only_b, "p_exact": p}


def run_humaneval(url: str, adapter: str | None, args: argparse.Namespace, problems, executor) -> dict:
    ask = "Please complete the following Python function:\n\n```python\n{}\n```"
    prompts = [[{"role": "system", "content": HE_SYSTEM}, {"role": "user", "content": ask.format(p.prompt)}]
               for p in problems]
    gens = batch_generate(url, prompts, adapter, args.max_tokens, args.batch_size)
    rows = [{"task_id": p.task_id, "finish_reason": g["finish_reason"], "tokens": g["tokens"], "completion": g["text"]}
            for p, g in zip(problems, gens, strict=True)]
    return score_humaneval(rows, problems, executor)


def score_humaneval(rows: list[dict], problems, executor) -> dict:
    """Execute stored completions. Separate from generation so a scoring fix can
    be re-applied offline (`--rescore`) without regenerating on the GPU.

    `executor.execute` takes the RAW completion: it runs `clean_code` itself.
    Passing pre-cleaned code cleans it twice, and the second pass -- finding
    `def <entry_point>` in fence-less text -- cuts off everything above it,
    imports included. That failed correct solutions with NameError (`math`,
    `Counter`) and read as a 12-point accuracy drop in the first run of this
    script (2026-09-24), which was briefly misattributed to batched decode.
    """
    by_id = {p.task_id: p for p in problems}
    for r in rows:
        p = by_id[r["task_id"]]
        res = executor.execute(p, r["completion"])
        r["passed"] = bool(res.passed)
        r["error_tail"] = "" if res.passed else (res.error or "").strip().splitlines()[-1:]
        r["ruff"] = ruff_findings(clean_code(r["completion"], p.entry_point, p.prompt)) if res.passed else None
    passed = [r["passed"] for r in rows]
    ruff = [r["ruff"] for r in rows if r["ruff"] is not None]
    return {"pass": sum(passed), "n": len(rows), "passed_vector": passed,
            "truncated": sum(r["finish_reason"] == "length" for r in rows),
            "ruff_findings_per_passing": round(sum(ruff) / len(ruff), 3) if ruff else None,
            "ruff_clean_share": round(sum(x == 0 for x in ruff) / len(ruff), 3) if ruff else None,
            "median_tokens": sorted(r["tokens"] or 0 for r in rows)[len(rows) // 2], "rows": rows}


def run_style(url: str, domain: str, adapter: str | None, args: argparse.Namespace) -> dict:
    items = [json.loads(line) for line in (REPO_ROOT / EVAL_FILE[domain]).read_text().splitlines() if line.strip()]
    if domain == "agentic_coding":
        prompts = [[{"role": "user", "content": it["messages"][0]["content"]}] for it in items]
    else:
        prompts = [[{"role": "user", "content": it["prompt"]}] for it in items]
    gens = batch_generate(url, prompts, adapter, args.max_tokens, args.batch_size)
    rows = []
    for it, g in zip(items, gens, strict=True):
        ans, closed = answer_part(g["text"])
        if domain == "agentic_coding":
            verdict = "FORMAT_OK" if SEARCH_REPLACE.search(ans) else "FORMAT_BAD"
        else:
            verdict = classify(ans, it) if closed else "NEITHER"
        rows.append({"id": it.get("id"), "verdict": verdict, "think_closed": closed,
                     "finish_reason": g["finish_reason"], "tokens": g["tokens"], "completion": g["text"]})
    c = Counter(r["verdict"] for r in rows)
    if domain == "agentic_coding":
        score = c["FORMAT_OK"] / len(rows)
    else:
        score = sum(POINTS[r["verdict"]] for r in rows) / len(rows)
    return {"score": round(score, 4), "counts": dict(c), "n": len(rows),
            "think_unclosed": sum(not r["think_closed"] for r in rows),
            "median_tokens": sorted(r["tokens"] or 0 for r in rows)[len(rows) // 2], "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b", help="4b, 9b, 0.8b, 2b (adapter suffix convention)")
    ap.add_argument("--old", default="v7", help="baseline generation (per-domain overrides: OLD_OVERRIDES)")
    ap.add_argument("--new", default="v9")
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--url", default="http://127.0.0.1:8003/v1/chat/completions/batch")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--skip-humaneval", action="store_true")
    ap.add_argument("--base-only", action="store_true",
                    help="measure only the served model itself (HumanEval), no adapters -- for a whole-model "
                         "variant such as a fine-tune served via RUNTIME_NEXT_MODEL_DIR")
    ap.add_argument("--rescore", action="store_true",
                    help="re-execute every stored HumanEval completion with the current scorer, then report")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--old-humaneval-domains", default=None,
                    help="comma list: run HumanEval on the OLD arm only for these domains (default: all). "
                         "Skipping leaves a real gap in the comparison; it is a time-saving choice, not a "
                         "redundancy (the 9B v7s' '~7e-6 geometry' turned out to be bad telemetry).")
    ap.add_argument("--base-from", type=Path, default=None,
                    help="reuse arms (base, and any arm already measured, e.g. the old generation) from another "
                         "report of the same size instead of regenerating them. Valid only while prompts are "
                         "byte-identical across engine builds: verified 2026-09-24 for the single-turn "
                         "system+user shape every arm here uses (chat_template_goldens.json).")
    args = ap.parse_args()

    out = args.out or REPO_ROOT / f"results/benchmarks/adapter_generations_{args.old}_vs_{args.new}_{args.size}.json"
    report: dict = json.loads(out.read_text()) if out.exists() else {}
    report["config"] = {
        "size": args.size, "old": args.old, "old_overrides": OLD_OVERRIDES, "new": args.new,
        "max_tokens": args.max_tokens, "batch_size": args.batch_size, "ruff_rules": RUFF_RULES,
        "path": "runtime-next /v1/chat/completions/batch, real Qwen3.5 chat template, thinking on, greedy",
    }
    report.setdefault("arms", {})

    def save() -> None:
        out.write_text(json.dumps(report, indent=1))

    problems = load_humaneval_problems()
    executor = HumanEvalExecutor()
    t0 = time.time()

    if args.rescore:
        for name, arm in report["arms"].items():
            if "humaneval" in arm:
                before = arm["humaneval"]["pass"]
                arm["humaneval"] = score_humaneval(arm["humaneval"]["rows"], problems, executor)
                print(f"  rescored {name}: {before} -> {arm['humaneval']['pass']}", flush=True)
        save()

    # Base first: it is shared by every domain comparison. Base does not depend on
    # the generations being compared, so another report's base arm is reusable.
    if args.base_from:
        donor = json.loads(args.base_from.read_text())["arms"]
        reused = [name for name in donor if name not in report["arms"]]
        for name in reused:
            report["arms"][name] = donor[name]
        report["config"]["arms_reused_from"] = {"file": str(args.base_from), "arms": reused}
    if "base" not in report["arms"]:
        report["arms"]["base"] = {"style": {}}
    base = report["arms"]["base"]
    if not args.skip_humaneval and "humaneval" not in base:
        print("== base / humaneval", flush=True)
        base["humaneval"] = run_humaneval(args.url, None, args, problems, executor)
        save()

    if args.base_only:
        save()
        he = base.get("humaneval", {})
        print(f"base-only: HumanEval {he.get('pass')}/{he.get('n')}  ruff/pass {he.get('ruff_findings_per_passing')}")
        return 0

    for domain in args.domains.split(","):
        if domain not in base["style"]:
            print(f"== base / style {domain}", flush=True)
            base["style"][domain] = run_style(args.url, domain, None, args)
            save()
        old_version = old_of(domain, args.old)
        for version in (old_version, args.new):
            name = adapter_name(domain, version, args.size)
            if not (REPO_ROOT / "results/adapters" / name / "adapter_model.safetensors").exists():
                print(f"  SKIP {name}: no weights", flush=True)
                continue
            arm = report["arms"].setdefault(name, {"domain": domain, "version": version})
            if "style" not in arm:
                print(f"== {name} / style", flush=True)
                arm["style"] = run_style(args.url, domain, name, args)
                save()
            skip_old = (version != args.new and args.old_humaneval_domains is not None
                        and domain not in args.old_humaneval_domains.split(","))
            if not args.skip_humaneval and not skip_old and "humaneval" not in arm:
                print(f"== {name} / humaneval", flush=True)
                arm["humaneval"] = run_humaneval(args.url, name, args, problems, executor)
                save()

    # Paired statistics, computed from stored vectors so a resumed run is consistent.
    stats = {}
    base_vec = base.get("humaneval", {}).get("passed_vector")
    for domain in args.domains.split(","):
        old_name = adapter_name(domain, old_of(domain, args.old), args.size)
        old = report["arms"].get(old_name, {}).get("humaneval")
        new = report["arms"].get(adapter_name(domain, args.new, args.size), {}).get("humaneval")
        s = {}
        if base_vec and old:
            s["base_vs_old"] = mcnemar(base_vec, old["passed_vector"])
        if base_vec and new:
            s["base_vs_new"] = mcnemar(base_vec, new["passed_vector"])
        if old and new:
            s["old_vs_new"] = mcnemar(old["passed_vector"], new["passed_vector"])
        stats[domain] = s
    report["paired_humaneval"] = stats
    save()

    print(f"\n{'arm':42s} {'HE pass':>8s} {'trunc':>5s} {'ruff/pass':>9s} {'style':>7s} {'n':>4s}")
    for name, arm in report["arms"].items():
        he = arm.get("humaneval", {})
        if name == "base":
            for d, st in arm["style"].items():
                print(f"{'base [' + d + ']':42s} {he.get('pass', ''):>8} {he.get('truncated', ''):>5} "
                      f"{str(he.get('ruff_findings_per_passing', '')):>9s} {st['score']:7.3f} {st['n']:4d}")
            continue
        st = arm.get("style", {})
        ruff = str(he.get("ruff_findings_per_passing", ""))
        print(f"{name:42s} {he.get('pass', ''):>8} {he.get('truncated', ''):>5} {ruff:>9s} "
              f"{st.get('score', float('nan')):7.3f} {st.get('n', 0):4d}")
    print(f"\n{time.time() - t0:.0f}s  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
