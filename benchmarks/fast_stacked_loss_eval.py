"""[INVALID / SUPERSEDED DIAGNOSTIC]

WARNING & INVALIDATION NOTICE:
-------------------------------
This script was drafted as a loss diagnostic, but evaluating loss across evaluation files lacking
isolated completion fields evaluates prompt tokens. Scoring prompt cross-entropy acts as an
uncalibrated norm meter that mechanically penalizes any weight perturbation (|W_expert| > 0)
and produces the pathological recommendation to delete all adapters.

For valid multi-expert evaluation:
- Use execution gating (`benchmarks/multi_turn_execution_benchmark.py` or `benchmarks/applied_execution_gate.py`)
- Or use completion-only loss with strict prompt masking (`-100`) against ground-truth outputs.
"""

import json
import math
import sys
from pathlib import Path
import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.novel_peft import FoldableExpert, WeightFoldingEngine, set_hard_vram_cap

ANSWER_MARKER = "\n\n### Answer:\n"

EXPERTS_V4 = {
    "fin": "results/adapters/m2_financial_r8a128_v4",
    "ast": "results/adapters/m2_astral_r8a128_v4",
    "pg": "results/adapters/m2_postgresql_r8a128_v4",
}


def load_heldout_records():
    """Loads held-out records containing prompt + verified answer."""
    # We load reserved family records from v3 (which were removed from v4)
    # Plus financial v3 test samples
    data = {"financial": [], "astral": [], "postgresql": []}
    
    # 1. PostgreSQL reserved families
    pg_v3 = REPO_ROOT / "data/postgresql/training_data_v3.jsonl"
    if pg_v3.exists():
        for line in pg_v3.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            fam = r.get("meta", {}).get("family", "")
            if any(k in fam for k in ["distinct_on", "filter_clause", "ordinality", "lateral", "percentile"]):
                data["postgresql"].append(r["text"])

    # 2. Astral reserved families
    ast_v3 = REPO_ROOT / "data/astral/training_data_v3.jsonl"
    if ast_v3.exists():
        for line in ast_v3.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            fam = r.get("meta", {}).get("family", "")
            if any(k in fam for k in ["partial_curry", "singledispatch", "protocol_slots", "taskgroup"]):
                data["astral"].append(r["text"])

    # 3. Financial held-out samples (last 100 records of financial v3)
    fin_v3 = REPO_ROOT / "data/financial_planning/training_data_v3.jsonl"
    if fin_v3.exists():
        fin_lines = [json.loads(x)["text"] for x in fin_v3.read_text().splitlines() if x.strip()]
        data["financial"] = fin_lines[-100:]  # held-out slice

    return data


@torch.no_grad()
def evaluate_completion_only_loss(model, tok, texts):
    """Computes cross-entropy loss STRICTLY on completion tokens (prompt masked to -100)."""
    total_loss = 0.0
    total_completion_tokens = 0

    for text in texts:
        if ANSWER_MARKER not in text:
            continue
        prompt, _, completion = text.partition(ANSWER_MARKER)
        prompt_with_marker = prompt + ANSWER_MARKER
        
        prompt_ids = tok(prompt_with_marker, return_tensors="pt", add_special_tokens=True)["input_ids"]
        full_ids = tok(text, return_tensors="pt", add_special_tokens=True)["input_ids"].to(model.device)
        
        prompt_len = prompt_ids.shape[1]
        full_len = full_ids.shape[1]
        
        if full_len <= prompt_len:
            continue
            
        labels = full_ids.clone()
        labels[:, :prompt_len] = -100  # MASK PROMPT TO -100
        
        outputs = model(input_ids=full_ids, labels=labels)
        loss = outputs.loss.item()
        
        n_comp_tokens = full_len - prompt_len
        total_loss += loss * n_comp_tokens
        total_completion_tokens += n_comp_tokens

    avg_loss = total_loss / max(1, total_completion_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl, total_completion_tokens


def main():
    set_hard_vram_cap(22.0)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    experts = {n: FoldableExpert.from_dir(REPO_ROOT / p, n) for n, p in EXPERTS_V4.items()}
    engine = WeightFoldingEngine(model, experts.values(), keep_pristine=True)

    data = load_heldout_records()
    print("=" * 88)
    print(" 🎯 HELD-OUT COMPLETION-ONLY LOSS & PERPLEXITY DIAGNOSTIC (Prompt Masked to -100)")
    print("=" * 88)
    for d, texts in data.items():
        print(f" Loaded {len(texts)} held-out completion records for domain: {d}")
    print("-" * 88)
    print(f" {'Condition':<28s} {'Financial Loss (PPL)':>18s} {'Astral Loss (PPL)':>18s} {'PostgreSQL Loss (PPL)':>20s}")
    print("-" * 88)

    conditions = [
        ("Base Model (No Adapters)", []),
        ("Solo Financial (α=128)", [("fin", 1.0)]),
        ("Solo Astral (α=128)", [("ast", 1.0)]),
        ("Solo PostgreSQL (α=128)", [("pg", 1.0)]),
        ("3-Way Unscaled (α=128)", [("fin", 1.0), ("ast", 1.0), ("pg", 1.0)]),
        ("3-Way √K-Scaled (α=74)", [("fin", 1.0 / math.sqrt(3)), ("ast", 1.0 / math.sqrt(3)), ("pg", 1.0 / math.sqrt(3))]),
        ("3-Way 1/K-Scaled (α=43)", [("fin", 1.0 / 3.0), ("ast", 1.0 / 3.0), ("pg", 1.0 / 3.0)]),
    ]

    for label, active_specs in conditions:
        if not active_specs:
            engine.restore()
        else:
            active_experts = []
            orig_scalings = []
            for name, mult in active_specs:
                e = experts[name]
                orig_scalings.append((e, e.scaling))
                e.scaling = e.scaling * mult
                active_experts.append(e)

            engine.activate_many(active_experts)

            for e, orig in orig_scalings:
                e.scaling = orig

        row_str = []
        for d in ["financial", "astral", "postgresql"]:
            loss_val, ppl_val, n_toks = evaluate_completion_only_loss(model, tok, data[d])
            row_str.append(f"{loss_val:.3f} ({ppl_val:5.1f})")

        print(f" {label:<28s} {row_str[0]:>18s} {row_str[1]:>18s} {row_str[2]:>20s}")

    engine.restore()
    print("=" * 88)


if __name__ == "__main__":
    main()
