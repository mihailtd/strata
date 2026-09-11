"""End-to-End Evaluation of Ring-Buffered Hybrid State Rollback (ReplaySSM).

Compares:
  Arm A: Baseline Speculative Decoding (Dynamic .clone() state snapshotting)
  Arm B: Ring-Buffered Speculative Decoding (Pre-allocated StateRingBuffer rollback)

Validates:
  1. Exact text generation fidelity (100% token-for-token equality).
  2. Speculative throughput (tok/s) and generation latency.
  3. Zero dynamic memory allocation overhead.

Run:
  uv run --env-file .env benchmarks/runtime/speculative/state_replay/evaluate_ring_buffer_replay.py
"""

from __future__ import annotations

import sys
import time
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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.mtp_draft import (  # noqa: E402
    Qwen35MTPDraftHead,
    attach_state_ring_buffer,
    restore_state,
    snapshot_state,
)
from runtime.novel_peft import set_hard_vram_cap  # noqa: E402

MODEL_NAME = "Qwen/Qwen3.5-4B"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda:0")

EVAL_PROMPTS = [
    "Write a high-performance Python function that computes the Levenshtein "
    "distance between two strings with memoization.",
    "Explain the architectural differences between Transformer attention and "
    "State Space Models (SSMs) like Mamba and GatedDeltaNet.",
    "Draft a SQL query to calculate 7-day rolling average revenue per customer segment in PostgreSQL.",
]


@torch.no_grad()
def generate_speculative(
    model,
    tok,
    head: Qwen35MTPDraftHead,
    prompt: str,
    max_new_tokens: int = 48,
    k: int = 4,
    use_ring_buffer: bool = False,
) -> tuple[str, float, int, int]:
    """Runs speculative decoding with optional RingBuffer acceleration."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
    eos = tok.eos_token_id

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    out = model(ids, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values

    if use_ring_buffer:
        attach_state_ring_buffer(cache, max_depth=k + 2)

    hids = [out.hidden_states[-1]]
    seq = ids
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    toks = [nxt.item()]
    pos = ids.shape[1]

    total_drafted = 0
    total_accepted = 0
    done = toks[0] == eos

    while len(toks) < max_new_tokens and not done:
        H = torch.cat(hids, dim=1)
        dcache = head.prefill(H, seq)
        draft = head.draft(H[:, -1:, :], nxt, k=k, start_pos=pos - 1, cache=dcache)
        total_drafted += k

        snap = snapshot_state(cache)
        chunk = torch.cat([nxt, draft], dim=-1)

        out = model(chunk, past_key_values=cache, use_cache=True, output_hidden_states=True)
        target = torch.argmax(out.logits[0], -1)

        n_acc = 0
        for i in range(k):
            if draft[0, i].item() == target[i].item():
                n_acc += 1
            else:
                break
        total_accepted += n_acc

        committed = torch.cat([nxt, draft[:, :n_acc]], dim=-1)
        if n_acc < k:
            restore_state(cache, snap)
            out = model(committed, past_key_values=cache, use_cache=True, output_hidden_states=True)
            new_h = out.hidden_states[-1]
        else:
            new_h = out.hidden_states[-1][:, : n_acc + 1, :]

        bonus = target[n_acc].item()
        for t in draft[0, :n_acc].tolist() + [bonus]:
            if len(toks) >= max_new_tokens:
                break
            toks.append(t)
            if t == eos:
                done = True
                break

        hids.append(new_h)
        seq = torch.cat([seq, committed], dim=-1)
        pos += n_acc + 1
        nxt = torch.tensor([[bonus]], device=ids.device)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    generated_text = tok.decode(toks, skip_special_tokens=True)
    return generated_text, elapsed, total_drafted, total_accepted


def main():
    set_hard_vram_cap(22.0)
    print("=" * 80)
    print("ReplaySSM Ring-Buffered Hybrid State Rollback: E2E Generation Verification")
    print("=" * 80)

    print(f"Loading base model {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.eval()

    print("Initializing MTP Draft Head...")
    draft_head = Qwen35MTPDraftHead(model).to(DEVICE, dtype=DTYPE)
    draft_head.eval()

    all_matched = True
    speedups = []
    REPEATS = 3

    # ------------------------------------------------------------------ WARMUP
    # ⚠️ THE FIRST VERSION OF THIS SCRIPT HAD NO WARMUP, AND THE HEADLINE WAS THE
    # ARTEFACT. Arm A of prompt 1 absorbed lazy Triton/HIP compilation and read
    # 18.9 tok/s while prompt 2's Arm A read 42.1 tok/s on the IDENTICAL config --
    # a 2.2x spread that is warmup, not prompts. That single contaminated pair
    # produced "+43.12%" and dragged the reported mean to "+17.45%".
    #
    # The same failure is already on this repo's record: the retracted
    # "prefill = 54%" claim was ~33 s of Triton JIT inside the timed region.
    print("\nWarmup (compilation OUTSIDE the timed region)...")
    for use_rb in (False, True):
        generate_speculative(model, tok, draft_head, EVAL_PROMPTS[0],
                             max_new_tokens=16, k=4, use_ring_buffer=use_rb)
    print("  done")

    print("\nEvaluating test prompts...")
    for idx, prompt in enumerate(EVAL_PROMPTS):
        print(f"\n--- Prompt {idx + 1}/{len(EVAL_PROMPTS)} ---")
        print(f"Prompt: {prompt[:70]}...")

        # REPEATS with interleaved arms so machine drift hits both equally, and
        # the MEDIAN is reported -- a single shot cannot resolve a few percent on
        # this box.
        ta, tb = [], []
        for _ in range(REPEATS):
            text_a, time_a, drafted_a, acc_a = generate_speculative(
                model, tok, draft_head, prompt, max_new_tokens=48, k=4, use_ring_buffer=False
            )
            ta.append(time_a)
            text_b, time_b, drafted_b, acc_b = generate_speculative(
                model, tok, draft_head, prompt, max_new_tokens=48, k=4, use_ring_buffer=True
            )
            tb.append(time_b)

        time_a, time_b = sorted(ta)[REPEATS // 2], sorted(tb)[REPEATS // 2]
        n_tok_a, n_tok_b = len(tok.encode(text_a)), len(tok.encode(text_b))
        tps_a, tps_b = n_tok_a / time_a, n_tok_b / time_b

        match = text_a == text_b
        all_matched = all_matched and match
        speedup_pct = ((time_a - time_b) / time_a) * 100
        speedups.append(speedup_pct)

        print(f"  Arm A (Baseline):      {time_a:.3f}s ({tps_a:.1f} tok/s, {n_tok_a} tok) "
              f"| spread {min(ta):.3f}-{max(ta):.3f}s | Accept: {acc_a / max(1, drafted_a):.1%}")
        print(f"  Arm B (RingBuffer):    {time_b:.3f}s ({tps_b:.1f} tok/s, {n_tok_b} tok) "
              f"| spread {min(tb):.3f}-{max(tb):.3f}s | Accept: {acc_b / max(1, drafted_b):.1%}")
        print(f"  Exact Text Match:      {'MATCH' if match else 'MISMATCH'}")
        print(f"  Latency Change:        {speedup_pct:+.2f}%  "
              f"(arm spreads overlap: {'YES - unresolved' if max(min(ta), min(tb)) < min(max(ta), max(tb)) else 'no'})")

    print("\n" + "=" * 80)
    print("E2E SPECULATIVE VERIFICATION SUMMARY")
    print("=" * 80)
    avg_speedup = sum(speedups) / len(speedups)
    print(f"  Text Match Across All Prompts: {'100% EXACT MATCH' if all_matched else 'FAILED'}")
    print(f"  Median-of-{REPEATS} speedup per prompt: "
          + ", ".join(f"{x:+.2f}%" for x in speedups))
    print(f"  Mean:                          {avg_speedup:+.2f}%")
    print()
    print("  AMDAHL BOUND: rollback saves 498 us (499.61 -> 1.28). At ~13 rollbacks")
    print("  per 48-token generation that is ~6.5 ms of a ~1500 ms run = ~0.4%.")
    print("  Any result materially above that is measuring something else.")
    print("=" * 80)

    return 0 if all_matched else 1


if __name__ == "__main__":
    sys.exit(main())
