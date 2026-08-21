"""DuckDB v4 Expert Evaluation Harness.

Evaluates Base Model vs Solo DuckDB v4 (and multi-expert combinations)
across 50 held-out DuckDB evaluation questions with regex expectation scoring.
"""

import argparse
import json
import re
import sys
from pathlib import Path
import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.canon import REPO_ROOT, adapter_path  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from runtime.novel_peft import FoldableExpert, WeightFoldingEngine, set_hard_vram_cap

EVAL_FILE = REPO_ROOT / "data/duckdb/evaluation_data.jsonl"
DUCKDB_ADAPTER_DIR = adapter_path("duckdb")


def score_question(q: dict, generated_text: str) -> float:
    expects = q.get("expects", [])
    if not expects:
        return 100.0 if len(generated_text.strip()) > 20 else 0.0
    matches = sum(1 for p in expects if re.search(p, generated_text, re.IGNORECASE))
    return (matches / len(expects)) * 100.0


@torch.no_grad()
def evaluate_condition(model, tok, questions, max_new_tokens=2048):
    scores = []
    per_q = []
    for q in questions:
        prompt_text = f"### Question:\n{q['prompt']}\n\n### Answer:\n"
        ids = tok(prompt_text, return_tensors="pt").input_ids.to(model.device)
        gen = model.generate(
            input_ids=ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            stop_strings=["### Question"],
            tokenizer=tok,
        )
        gen_tokens = gen[0][ids.shape[1]:]
        txt = tok.decode(gen_tokens, skip_special_tokens=True).strip()
        txt = re.split(r"#+\s*Question", txt)[0].strip()
        s = score_question(q, txt)
        scores.append(s)
        per_q.append({"id": q["id"], "category": q.get("category", ""), "score": s, "generated": txt[:200]})
    return sum(scores) / max(1, len(scores)), per_q


def main():
    parser = argparse.ArgumentParser(description="DuckDB Expert Evaluation Benchmark")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--adapter-dir", default=str(DUCKDB_ADAPTER_DIR))
    parser.add_argument("--out", default="results/benchmarks/duckdb_v4_evaluation.json")
    args = parser.parse_args()

    set_hard_vram_cap(22.0)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    questions = [json.loads(line) for line in EVAL_FILE.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} held-out DuckDB evaluation questions from {EVAL_FILE}")

    # 1. Evaluate Base Model
    print("Evaluating Base Model (no adapters)...")
    base_score, base_per_q = evaluate_condition(model, tok, questions, args.max_new_tokens)
    print(f"Base Model DuckDB Pass Rate: {base_score:.2f}%")

    # 2. Evaluate Solo DuckDB v4 Adapter if available
    adapter_path = Path(args.adapter_dir)
    results = {
        "base_score": base_score,
        "base_per_q": base_per_q,
    }

    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        print(f"\nLoading DuckDB adapter from: {adapter_path}")
        duckdb_expert = FoldableExpert.from_dir(adapter_path, "duckdb")
        engine = WeightFoldingEngine(model, [duckdb_expert], keep_pristine=True)
        engine.activate(duckdb_expert)

        print("Evaluating Solo DuckDB v4 Adapter...")
        expert_score, expert_per_q = evaluate_condition(model, tok, questions, args.max_new_tokens)
        delta_pp = expert_score - base_score
        print(f"Solo DuckDB v4 Pass Rate: {expert_score:.2f}% (Delta: {delta_pp:+.2f}pp)")
        
        results["duckdb_v4_score"] = expert_score
        results["delta_pp"] = delta_pp
        results["expert_per_q"] = expert_per_q
        engine.restore()
    else:
        print(f"Adapter not yet trained at {adapter_path}. Run training first.")

    out_file = REPO_ROOT / args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nWrote benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
