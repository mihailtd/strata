"""Unit tests for Category 1 SOTA Hygiene Rules (AMD_SPECIFFIC.md & Operational Rules).

Covers:
1. Action 1.1: Ban INT4 KV cache (enforces BF16/FP16/INT8, rejects sub-byte 4-bit KV cache).
2. Action 3.1: Context Scrubbing (strips <think>...</think> intermediate reasoning blocks from history).
3. Action 2.3: Deterministic Attention Backend Pinning (pins PyTorch SDPA flags to eliminate reduction drift).
4. Section 5: Numerical QA Validation Probes (KLD calculation and Top-1 argmax agreement).
5. Phase 1: Environment & GPU Preflight Exclusivity Guard.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from runtime.canon import (
    CANON,
    configure_deterministic_attention,
    validate_kv_cache_precision,
)
from runtime.gpu_preflight import (
    check_gpu_availability,
    ensure_gpu_exclusive,
    get_gpu_vram_info,
)
from runtime.server import (
    ChatMessage,
    extract_thinking_and_content,
    format_prompt,
    scrub_thinking_blocks,
)


# =============================================================================
# 1. Action 1.1: Banning INT4 KV Cache
# =============================================================================

def test_kv_cache_allowed_precisions():
    """Verify standard production precisions (bfloat16, float16, int8) are accepted."""
    assert validate_kv_cache_precision("bfloat16") == "bfloat16"
    assert validate_kv_cache_precision("float16") == "float16"
    assert validate_kv_cache_precision("int8") == "int8"
    assert validate_kv_cache_precision("BFLOAT16") == "bfloat16"


@pytest.mark.parametrize("banned_dtype", ["int4", "fp4", "nf4", "q4_0", "q4_1", "4bit", "INT4", "k_int4"])
def test_banning_int4_kv_cache_raises_error(banned_dtype: str):
    """Enforce strict ban on INT4 KV cache to prevent 40k token cliff regressions."""
    with pytest.raises(ValueError, match="BANNED_PRECISION"):
        validate_kv_cache_precision(banned_dtype)


def test_canon_kv_cache_default_is_bfloat16():
    """Verify canonical global default is bfloat16."""
    assert CANON.KV_CACHE_DTYPE == "bfloat16"
    assert validate_kv_cache_precision(CANON.KV_CACHE_DTYPE) == "bfloat16"


# =============================================================================
# 2. Action 3.1: Context Scrubbing (<think> Block Stripping)
# =============================================================================

def test_scrub_thinking_blocks_clean_tags():
    """Verify standard <think>...</think> blocks are completely stripped."""
    raw = "<think>\nLet me analyze the SQL schema and indexes.\nWe need pgvector.\n</think>\nSELECT * FROM embeddings;"
    expected = "SELECT * FROM embeddings;"
    assert scrub_thinking_blocks(raw) == expected


def test_scrub_thinking_blocks_multiline_and_code():
    """Verify multiline reasoning containing code, quotes, and markdown is cleanly stripped."""
    raw = """<think>
```python
def draft():
    return "test"
```
The user wants an async endpoint with `@app.get("/items")`.
</think>
from fastapi import FastAPI
app = FastAPI()"""
    expected = "from fastapi import FastAPI\napp = FastAPI()"
    assert scrub_thinking_blocks(raw) == expected


def test_scrub_thinking_blocks_unclosed_and_orphaned():
    """Verify unclosed or orphaned think tags do not corrupt text."""
    assert scrub_thinking_blocks("<think>partial reasoning") == "partial reasoning"
    assert scrub_thinking_blocks("answer text</think>") == "answer text"
    assert scrub_thinking_blocks("plain content without think") == "plain content without think"
    assert scrub_thinking_blocks("") == ""


def test_format_prompt_scrubs_prior_assistant_turns():
    """Verify format_prompt removes reasoning traces from historical assistant messages."""
    messages = [
        ChatMessage(role="user", content="Step 1: create table"),
        ChatMessage(
            role="assistant",
            content="<think>\nNeed uuid and vector columns.\n</think>\nCREATE TABLE items (id UUID, emb VECTOR(1536));",
        ),
        ChatMessage(role="user", content="Step 2: add query"),
    ]

    formatted = format_prompt(messages, thinking_effort="low")

    # Historical assistant turn MUST NOT contain historical thinking scratchpad
    assert "Need uuid and vector columns" not in formatted
    assert "CREATE TABLE items (id UUID, emb VECTOR(1536));" in formatted

    # Historical user turns must remain intact
    assert "Step 1: create table" in formatted
    assert "Step 2: add query" in formatted

    # The prompt initiates a new assistant turn with fresh <think> start
    assert formatted.endswith("<|im_start|>assistant\n<think>\n")


def test_extract_thinking_and_content_separation():
    """Verify extract_thinking_and_content splits reasoning from final text."""
    text = "<think>\nComputing cosine distance metric\n</think>\nORDER BY emb <=> target LIMIT 10;"
    reasoning, content = extract_thinking_and_content(text)
    assert reasoning == "Computing cosine distance metric"
    assert content == "ORDER BY emb <=> target LIMIT 10;"


# =============================================================================
# 3. Action 2.3: Deterministic Attention Backend Pinning
# =============================================================================

def test_configure_deterministic_attention_flags():
    """Verify configure_deterministic_attention returns proper deterministic config."""
    cfg = configure_deterministic_attention()
    assert isinstance(cfg, dict)
    assert "deterministic" in cfg
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_flash_sdp"):
        assert cfg["flash_sdp"] is True
        assert cfg["mem_efficient_sdp"] is True
        assert cfg["math_sdp"] is False
        assert cfg["deterministic"] is True


def test_canon_attention_backend_is_sdpa():
    """Verify canonical attention backend is pinned to SDPA."""
    assert CANON.ATTENTION_BACKEND == "sdpa"


# =============================================================================
# 4. Section 5: Numerical QA Validation Probes (KLD & Top-1 Argmax Agreement)
# =============================================================================

def compute_kld_in_fp64(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Computes Kullback-Leibler Divergence D_KL(P || Q) in double precision."""
    p_probs = F.softmax(p_logits.to(torch.float64), dim=-1)
    q_log_probs = F.log_softmax(q_logits.to(torch.float64), dim=-1)
    p_log_probs = F.log_softmax(p_logits.to(torch.float64), dim=-1)
    # D_KL(P || Q) = sum(P * (log P - log Q))
    kld = torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1).mean().item()
    return float(max(0.0, kld))


def compute_top1_agreement(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Computes fraction of positions where greedy argmax predictions match."""
    p_top1 = torch.argmax(p_logits, dim=-1)
    q_top1 = torch.argmax(q_logits, dim=-1)
    matches = (p_top1 == q_top1).float().mean().item()
    return float(matches)


def test_kld_and_top1_numerical_probe_identical_distributions():
    """Identical distributions must have KLD = 0.0 and Top-1 agreement = 1.0."""
    torch.manual_seed(42)
    logits = torch.randn(10, 128, dtype=torch.bfloat16)
    kld = compute_kld_in_fp64(logits, logits)
    top1 = compute_top1_agreement(logits, logits)

    assert kld == pytest.approx(0.0, abs=1e-7)
    assert top1 == pytest.approx(1.0, abs=1e-7)


def test_kld_and_top1_numerical_probe_small_perturbation():
    """Small numerical noise (e.g. BF16 rounding) must stay well below KLD < 0.01 threshold."""
    torch.manual_seed(42)
    base_logits = torch.randn(32, 1024, dtype=torch.float32)
    noisy_logits = base_logits + torch.randn_like(base_logits) * 0.01

    kld = compute_kld_in_fp64(base_logits, noisy_logits)
    top1 = compute_top1_agreement(base_logits, noisy_logits)

    assert kld < 0.01, f"KLD exceeded safety threshold: {kld:.6f}"
    assert top1 >= 0.95, f"Top-1 agreement fell below 95%: {top1:.4f}"


def test_kld_and_top1_numerical_probe_detects_severe_divergence():
    """Severe quantization noise (simulating INT4 collapse) triggers KLD > 0.01 failure."""
    torch.manual_seed(42)
    base_logits = torch.randn(32, 1024, dtype=torch.float32)
    corrupted_logits = base_logits + torch.randn_like(base_logits) * 1.5

    kld = compute_kld_in_fp64(base_logits, corrupted_logits)
    top1 = compute_top1_agreement(base_logits, corrupted_logits)

    assert kld > 0.01, f"Expected KLD failure on corrupt logits, got: {kld:.6f}"
    assert top1 < 0.95, f"Expected Top-1 agreement failure, got: {top1:.4f}"


# =============================================================================
# 5. Phase 1: Environment & GPU Preflight Exclusivity Guard
# =============================================================================

def test_gpu_vram_info_and_availability():
    """Verify GPU status inquiry and availability checking execute safely."""
    info = get_gpu_vram_info()
    assert isinstance(info, dict)
    assert "used_gb" in info
    assert "free_gb" in info
    assert "total_gb" in info

    avail = check_gpu_availability()
    assert isinstance(avail, dict)
    assert "is_clean" in avail


def test_ensure_gpu_exclusive_warning_mode():
    """Verify ensure_gpu_exclusive with exit_on_conflict=False returns boolean."""
    is_safe = ensure_gpu_exclusive(exit_on_conflict=False)
    assert isinstance(is_safe, bool)
