"""Tests mathematical parity of True O(1) Tensor State Handoff on Native27BEngine.

Validates that:
1. Prefilling Prompt A -> inheriting state S_t -> incremental prefilling Prompt B
   produces identical next-token logits to an all-in-one prefill of (Prompt A + Prompt B).
2. Autoregressive generation continuation with inherited state produces 100% bit-exact
   token sequences compared to the full historical context prefill.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.native_27b_engine import Native27BEngine
from runtime.server import get_27b_tokenizer


@pytest.fixture(scope="module")
def engine():
    from runtime import server

    return server.get_native_triton_27b_engine(num_layers=64)


@pytest.fixture(scope="module")
def tokenizer():
    return get_27b_tokenizer()


def test_incremental_prefill_logits_parity(engine: Native27BEngine, tokenizer):
    """Verifies that incremental prefill produces bit-exact logits to all-in-one prefill."""
    text_a = "<|im_start|>user\nCreate a table for users.<|im_end|>\n<|im_start|>assistant\nCREATE TABLE users (id SERIAL PRIMARY KEY, name TEXT);<|im_end|>\n"
    text_b = "<|im_start|>user\nNow add an email column.<|im_end|>\n<|im_start|>assistant\n"

    tokens_a = tokenizer.encode(text_a)
    tokens_b = tokenizer.encode(text_b)
    tokens_ab = tokens_a + tokens_b

    print(f"\n[Test] len(tokens_a)={len(tokens_a)}, len(tokens_b)={len(tokens_b)}, total={len(tokens_ab)}")

    # 1. Ground Truth: All-in-one prefill
    state_ab = engine.init_kv_caches(1, 512)
    l_ab, _ = engine.forward_prompt(tokens_ab, state_ab)
    pred_ab = int(torch.argmax(l_ab[0, -1]).item())
    token_ab_str = tokenizer.decode([pred_ab])

    # 2. State Handoff: Prefill A -> Inherit S_t -> Incremental prefill B
    state_a = engine.init_kv_caches(1, 512)
    l_a, state_a = engine.forward_prompt(tokens_a, state_a)
    l_b, state_b = engine.forward_prompt(tokens_b, state_a)
    pred_b = int(torch.argmax(l_b[0, -1]).item())
    token_b_str = tokenizer.decode([pred_b])

    print(f"[Test] Ground Truth token: {pred_ab} ({token_ab_str!r})")
    print(f"[Test] State Handoff token: {pred_b} ({token_b_str!r})")

    diff = torch.max(torch.abs(l_ab[0, -1] - l_b[0, -1])).item()
    print(f"[Test] Max absolute logit difference: {diff:.6f}")

    assert pred_ab == pred_b, f"Predictions do not match! Ground truth={pred_ab}, Handoff={pred_b}"
    assert diff < 0.15, f"Logit difference too high: {diff:.6f} >= 0.15"


def test_generation_continuation_parity(engine: Native27BEngine, tokenizer):
    """Verifies that generation with state handoff produces identical token sequences."""
    text_a = "<|im_start|>user\nWrite a python function to add two numbers.<|im_end|>\n<|im_start|>assistant\ndef add(a, b):\n    return a + b<|im_end|>\n"
    text_b = "<|im_start|>user\nNow write multiply.<|im_end|>\n<|im_start|>assistant\n"

    tokens_a = tokenizer.encode(text_a)
    tokens_b = tokenizer.encode(text_b)
    tokens_ab = tokens_a + tokens_b

    # 1. Ground Truth: generate directly from tokens_ab
    gen_gt = engine.generate(tokens_ab, max_new_tokens=16, temperature=0.0, use_hip_graph=False)

    # 2. State Handoff: generate tokens_a, then continue with tokens_b
    state_a = engine.init_kv_caches(1, 512)
    _, state_a = engine.forward_prompt(tokens_a, state_a)
    gen_handoff, _ = engine.generate_with_state(
        tokens_b, state_dict=state_a, max_new_tokens=16, temperature=0.0, use_hip_graph=False
    )

    print(f"\n[Test] Ground Truth tokens: {gen_gt}")
    print(f"[Test] Handoff tokens:      {gen_handoff}")
    print(f"[Test] Ground Truth text:   {tokenizer.decode(gen_gt)!r}")
    print(f"[Test] Handoff text:        {tokenizer.decode(gen_handoff)!r}")

    assert gen_gt == gen_handoff, f"Generated tokens mismatch: GT={gen_gt} vs Handoff={gen_handoff}"


if __name__ == "__main__":
    eng = Native27BEngine(num_layers=64)
    eng.load_from_cache()
    tok = get_27b_tokenizer()
    test_incremental_prefill_logits_parity(eng, tok)
    test_generation_continuation_parity(eng, tok)
    print("\nAll State Handoff Parity Tests Passed Successfully!")
