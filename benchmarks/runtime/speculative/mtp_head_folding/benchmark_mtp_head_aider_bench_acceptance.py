"""Real MTP draft-head acceptance rate (tau) on real aider-bench prompts.

THE QUESTION
------------
`benchmark_mtp_head_adapter_acceptance.py` (this directory) measured the
stock, un-adapted MTP head's real acceptance rate on `data/astral/
evaluation_data.jsonl` prompts (a general-domain Q&A set). The legacy
Python fleet's own real K=4 sweep (docs/DECISIONS.md sec.61-65) found
two CODING-heavy domains -- python_modern, python_web -- are net losses
or breakeven (0.99x/1.02x) for speculative decoding with this SAME stock
head, while general-domain/financial-heavy domains win clearly
(1.13x-1.52x). runtime-next's own actual target workload right now is
agentic coding (aider-bench), which sits in exactly the domain the fleet
data says is weakest. This script measures the real, honest number on
the real thing, instead of assuming either "the fleet numbers transfer"
or "they don't."

METHOD
------
Identical methodology to `benchmark_mtp_head_adapter_acceptance.py`
(greedy_truth / accepted_prefixes, unadapted head only -- domain-tuning
the head is a repeatedly-refuted dead end, docs/DECISIONS.md sec.5/63/69,
not re-attempted here), but the 10 prompts are the REAL aider-bench task
instructions (benchmarks/aider_bench/tasks/*/.docs/instructions.md),
built into the exact same system+user prompt template
`benchmarks/aider_bench/harness_direct.py`'s real "whole" edit-format
path sends to a real server (confirmed from
`results/benchmarks/aider_10tasks_base_qwen35_4b.json`'s own recorded
config: edit_format=whole), run through the tokenizer's real chat
template (these are chat messages a real client would POST to
/chat/completions, not raw completion text).

`truth_tokens` is set well above the original script's default (200 vs
40) and `offsets` spread further out (up to 180) specifically to reach
past this model's real thinking-mode preamble into the actual generated
CODE -- the region the python_modern/python_web fleet weakness is about,
not the preamble.

    uv run --env-file .env benchmarks/runtime/speculative/mtp_head_folding/benchmark_mtp_head_aider_bench_acceptance.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.canon import REPO_ROOT  # noqa: E402
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.mtp_draft import Qwen35MTPDraftHead  # noqa: E402
from runtime.novel_peft import set_hard_vram_cap  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "aider_bench"))
from tasks import load_tasks  # noqa: E402

BREAK_EVEN_TAU = 1.39

SYSTEM_PROMPT_WHOLE = (
    "You are an expert software engineer. Follow the user's instructions carefully.\n"
    "Implement the requested solution for `{target_file_name}`.\n"
    "Respond ONLY with the complete updated Python code inside a ```python ... ``` code block."
)
USER_PROMPT_WHOLE = (
    "Instructions:\n{instructions}\n\n"
    "Starting stub in `{target_file_name}`:\n"
    "```python\n{stub_code}\n```\n\n"
    "Please implement the full code for `{target_file_name}`."
)


@torch.no_grad()
def greedy_truth(model, ids, n_new):
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
def accepted_prefixes(model, head, ids, truth, k, offsets):
    seq = torch.cat([ids, torch.tensor([truth], device=ids.device)], dim=1)
    hidden = model(seq, output_hidden_states=True, use_cache=False).hidden_states[-1]
    T = ids.shape[1]
    rows = []
    for off in offsets:
        t = T - 1 + off
        if t + 2 + k > seq.shape[1]:
            continue
        cache = head.prefill(hidden[:, : t + 1, :], seq[:, : t + 1])
        nxt = seq[:, t + 1 : t + 2]
        draft = head.draft(hidden[:, t : t + 1, :], nxt, k=k, start_pos=t, cache=cache)
        tgt = seq[0, t + 2 : t + 2 + k]
        acc = 0
        for i in range(k):
            if draft[0, i].item() == tgt[i].item():
                acc += 1
            else:
                break
        rows.append(acc)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--truth-tokens", type=int, default=200)
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, 20, 60, 120, 180])
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/benchmarks/mtp_head_aider_bench_acceptance.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    head = Qwen35MTPDraftHead(model, args.model_name)

    tasks = load_tasks()
    print(f"{len(tasks)} real aider-bench tasks x {len(args.offsets)} offsets = "
          f"{len(tasks) * len(args.offsets)} draft events (upper bound), K={args.k}, "
          f"truth_tokens={args.truth_tokens}\n")

    print("Generating shared truth sequences from the backbone, real aider-bench prompts...")
    truths = []
    for task in tasks:
        system_prompt = SYSTEM_PROMPT_WHOLE.format(target_file_name=task.target_file_name)
        user_prompt = USER_PROMPT_WHOLE.format(
            instructions=task.instructions, target_file_name=task.target_file_name, stub_code=task.stub_code
        )
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        templated = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        ids = (templated.input_ids if hasattr(templated, "input_ids") else templated).to(model.device)
        truth = greedy_truth(model, ids, args.truth_tokens)
        truths.append((task.name, ids, truth))
        print(f"  {task.name}: prompt {ids.shape[1]} tokens, generated {len(truth)} real truth tokens")

    accs = []
    per_task = {}
    for name, ids, truth in truths:
        rows = accepted_prefixes(model, head, ids, truth, args.k, args.offsets)
        accs += rows
        per_task[name] = rows
        if rows:
            print(f"  {name:20s} accepted-per-round={rows}  tau={sum(rows) / len(rows):.3f}")
        else:
            print(f"  {name:20s} no valid offsets (truth too short)")

    tau = sum(accs) / len(accs)
    accept_rate_pct = 100 * sum(accs) / (len(accs) * args.k)
    print("\n" + "=" * 84)
    print(f" Real MTP head acceptance on real aider-bench prompts (K={args.k}, break-even tau={BREAK_EVEN_TAU})")
    print("=" * 84)
    print(f" tau (mean accepted/round) = {tau:.3f}   accept_rate = {accept_rate_pct:.1f}%   n_draft_events = {len(accs)}")
    print(f" {'above' if tau >= BREAK_EVEN_TAU else 'BELOW'} the repo's own real break-even (1.39)")
    print(" fleet reference (docs/DECISIONS.md sec.61-65, K=4): base 1.46x/65.6% accept, "
          "python_modern 0.99x/41.1% accept, python_web 1.02x/40.1% accept")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "k": args.k,
        "truth_tokens": args.truth_tokens,
        "offsets": args.offsets,
        "tau_mean_accepted": tau,
        "accept_rate_pct": accept_rate_pct,
        "n_draft_events": len(accs),
        "above_break_even": tau >= BREAK_EVEN_TAU,
        "per_task_accepted": per_task,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
