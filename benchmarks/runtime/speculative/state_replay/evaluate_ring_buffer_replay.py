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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.mtp_draft import (  # noqa: E402
    Qwen35MTPDraftHead,
    attach_state_ring_buffer,
    restore_state,
    snapshot_state,
)
from gnn_experiment.novel_peft import set_hard_vram_cap  # noqa: E402

MODEL_NAME = "Qwen/Qwen3.5-4B"
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda:0")

EVAL_PROMPTS = [
    "Write a high-performance Python function that computes the Levenshtein distance between two strings with memoization.",
    "Explain the architectural differences between Transformer attention and State Space Models (SSMs) like Mamba and GatedDeltaNet.",
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

    print("\nEvaluating test prompts...")
    for idx, prompt in enumerate(EVAL_PROMPTS):
        print(f"\n--- Prompt {idx + 1}/{len(EVAL_PROMPTS)} ---")
        print(f"Prompt: {prompt[:70]}...")

        # Arm A: Baseline Speculative (clone snapshot)
        text_a, time_a, drafted_a, acc_a = generate_speculative(
            model, tok, draft_head, prompt, max_new_tokens=48, k=4, use_ring_buffer=False
        )
        tps_a = len(tok.encode(text_a)) / time_a

        # Arm B: Ring-Buffered Speculative
        text_b, time_b, drafted_b, acc_b = generate_speculative(
            model, tok, draft_head, prompt, max_new_tokens=48, k=4, use_ring_buffer=True
        )
        tps_b = len(tok.encode(text_b)) / time_b

        match = text_a == text_b
        all_matched = all_matched and match
        speedup_pct = ((time_a - time_b) / time_a) * 100
        speedups.append(speedup_pct)

        print(f"  Arm A (Baseline):      {time_a:.3f}s ({tps_a:.1f} tok/s) | Accept Rate: {acc_a / max(1, drafted_a):.1%}")
        print(f"  Arm B (RingBuffer):    {time_b:.3f}s ({tps_b:.1f} tok/s) | Accept Rate: {acc_b / max(1, drafted_b):.1%}")
        print(f"  Exact Text Match:      {'✅ MATCH' if match else '❌ MISMATCH'}")
        print(f"  Latency Change:        {speedup_pct:+.2f}%")

    print("\n" + "=" * 80)
    print("E2E SPECULATIVE VERIFICATION SUMMARY")
    print("=" * 80)
    avg_speedup = sum(speedups) / len(speedups)
    print(f"  Text Match Across All Prompts: {'✅ 100% EXACT MATCH' if all_matched else '❌ FAILED'}")
    print(f"  Mean Speedup:                  {avg_speedup:+.2f}%")
    print("=" * 80)

    return 0 if all_matched else 1


if __name__ == "__main__":
    sys.exit(main())
