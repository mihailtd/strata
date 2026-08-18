"""Would Qwen3.5-0.8B be a good enough drafter for the 4B target?

WHY THIS AND NOT A BUILD
------------------------
The 0.8B draft path was tried before and recorded FAILED in CURRENT.md, but the
failure was a library refusal, not a measurement:

    ValueError: assisted generation is not supported with stateful models,
                such as Qwen3_5ForCausalLM

transformers declines because it cannot roll back the 24 GatedDeltaNet recurrent
states by truncation. Our own loop rolls them back by in-place copy
(_snapshot_ssm/_restore_ssm), so that specific blocker does not apply to us.

What actually decides the question is ACCEPTANCE, and that needs no speculative
machinery at all. Under greedy decoding the target's own continuation IS what
verification would produce, so we can simulate acceptance exactly with plain
forward passes:

    1. generate the target's greedy continuation
    2. at sampled positions, let the 0.8B draft K tokens autoregressively
    3. count the matching prefix -- that is n_acc, exactly as the verifier would

BREAK-EVEN THIS IS MEASURED AGAINST
-----------------------------------
Costing in units of one single-token 4B forward, per speculative step:
    verify (width K+1)          1.27   (measured, DECISIONS §22 correction)
    commit re-forward           ~1.1   (the hybrid tax; §18 could not remove it)
    K drafts on 0.8B            ~1.0-1.4
    total                       ~3.4   -> need tau + 1 > 3.4, i.e. tau > ~2.4

The current MTP head drafts for ~0.47 units, break-even tau > 1.84, and delivers
1.65 -> 0.78x. A 0.8B drafter RAISES the bar while raising tau, so it only pays
if acceptance is genuinely high.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/mihai/gnn-experiment")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

T0 = time.perf_counter()


def stage(m: str) -> None:
    print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)


import torch  # noqa: E402

torch.zeros(1, device="cuda"); torch.cuda.synchronize()
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

TARGET = "Qwen/Qwen3.5-4B"
DRAFT = "Qwen/Qwen3.5-0.8B"
ADAPTER = "results/adapters/m2_astral_r8a128"
K = int(os.environ.get("K", "4"))
GEN = int(os.environ.get("GEN", "256"))
N_PROMPTS = int(os.environ.get("N_PROMPTS", "6"))
STRIDE = int(os.environ.get("STRIDE", "8"))  # sample every Nth position as a chunk start


def vram() -> str:
    return f"alloc={torch.cuda.memory_allocated() / 2**30:.2f}G"


@torch.no_grad()
def greedy(model, ids, n, stop_ids):
    """Target's greedy continuation -- this is exactly what a verifier would emit."""
    out = model(ids, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    for _ in range(n - 1):
        if toks[-1] in stop_ids:
            break
        out = model(nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        toks.append(nxt.item())
    return toks


@torch.no_grad()
def draft_k(dmodel, prefix_ids, k):
    """0.8B drafts k tokens autoregressively, conditioned on its OWN outputs.

    Rebuilt from scratch per sampled start: the 0.8B is hybrid too, so its
    recurrent state cannot be cropped, and correctness matters more than speed in
    a probe that only runs a few hundred times.
    """
    out = dmodel(prefix_ids, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    drafted = [nxt.item()]
    for _ in range(k - 1):
        out = dmodel(nxt, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
        drafted.append(nxt.item())
    return drafted


def main() -> None:
    set_hard_vram_cap(22.0)
    stage("loading tokenizer + target 4B")
    tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    stage(f"target loaded {vram()}")

    stage("loading draft 0.8B")
    dmodel = AutoModelForCausalLM.from_pretrained(
        DRAFT, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    stage(f"draft loaded {vram()}")

    stop_ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid > 0:
            stop_ids.add(tid)

    rows = [json.loads(x) for x in
            (REPO_ROOT / "data/astral/evaluation_data.jsonl").read_text().splitlines() if x.strip()]
    prompts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
               for r in rows[:N_PROMPTS]]

    expert = FoldableExpert.from_dir(REPO_ROOT / ADAPTER, "astral")
    engine = WeightFoldingEngine(target, [expert], keep_pristine=True)
    stage(f"folding engine ready {vram()}")

    results = {}
    for cfg in ("base", "astral-folded"):
        if cfg == "astral-folded":
            engine.activate(expert)
        stage(f"===== target = {cfg} =====")
        hits = [0] * K          # matches at draft index i
        trials = 0              # chunks sampled
        n_acc_all: list[int] = []
        for pi, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt").input_ids.to(target.device)
            cont = greedy(target, ids, GEN, stop_ids)
            full = torch.cat([ids, torch.tensor([cont], device=target.device)], dim=-1)
            starts = range(ids.shape[1], ids.shape[1] + len(cont) - K, STRIDE)
            for s in starts:
                prefix = full[:, :s]
                drafted = draft_k(dmodel, prefix, K)
                actual = full[0, s:s + K].tolist()
                n_acc = 0
                for i in range(K):
                    if drafted[i] == actual[i]:
                        hits[i] += 1
                        n_acc += 1
                    else:
                        break
                n_acc_all.append(n_acc)
                trials += 1
            stage(f"  prompt {pi + 1}/{len(prompts)}: gen={len(cont)} chunks={trials}")

        tau = sum(n_acc_all) / max(1, trials)
        results[cfg] = {
            "tau": tau, "chunks": trials,
            "alpha_by_index": [h / max(1, trials) for h in hits],
            "alpha_first": hits[0] / max(1, trials),
        }
        stage(f"  {cfg}: tau={tau:.3f} alpha_first={results[cfg]['alpha_first']:.3f} "
              f"chain={[f'{h / max(1, trials):.3f}' for h in hits]}")

    stage("=" * 78)
    for cfg, r in results.items():
        tau = r["tau"]
        # emitted per step = tau + 1; cost ~3.4 units (see module docstring)
        for label, cost in (("optimistic 3.1", 3.1), ("central 3.4", 3.4), ("pessimistic 3.9", 3.9)):
            stage(f"  {cfg:14s} tau={tau:.3f}  cost={label:16s} -> "
                  f"{(tau + 1) / cost:5.3f}x vs autoregressive")
        stage(f"  {cfg:14s} break-even tau at central cost = 2.40  "
              f"({'CLEARS' if tau > 2.4 else 'MISSES'} it)")
    out = Path("/tmp/claude-1000/-home-mihai-gnn-experiment/"
               "e1416be8-5f2b-4f25-8921-27360ecd3639/scratchpad/draft_acceptance.json")
    out.write_text(json.dumps(results, indent=2))
    stage(f"wrote {out}")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
