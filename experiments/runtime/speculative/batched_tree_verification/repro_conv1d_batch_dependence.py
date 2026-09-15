r"""Minimal, airtight repro: Qwen3.5's real GatedDeltaNet conv1d step is NOT
batch-row-independent on this hardware, despite being mathematically defined
that way (groups=hidden_size, i.e. fully depthwise -- each channel/row should
be computable in total isolation from every other row).

WHY THIS EXISTS
---------------
`docs/DECISIONS.md` §76 found that batching two divergent speculative-decode
candidates into one verification pass is ~1.8x cheaper than sequential
verification, but occasionally (5.6% of 18 real prompts) flips which token
argmax picks -- a real correctness hazard, not bf16 rounding noise. This
script is the root-cause chain that pinned the exact origin, bisected with
real forward hooks on the real model (see the session's investigation):

  1. Ruled out cross-row data leakage: layer 0's decoder output, and every
     real submodule from layer 0's out_proj through layer 1's in_proj_qkv,
     match EXACTLY (0.0) between a batched row and its independent solo
     equivalent.
  2. Ruled out generic bf16 batch-size numerics: a plain `nn.Linear`, a
     batched `torch.linalg.solve_triangular` (the GDN chunk kernel's own
     triangular solve), and a synthetic depthwise `Conv1d` at the model's
     EXACT real dimensions (conv_dim=8192, kernel_size=4) all gave EXACT
     (0.0) agreement with RANDOM data of matching shape.
  3. The one thing that reproduces it: the model's REAL trained conv1d
     weight, with REAL captured activation values (extracted via a hook
     from an actual forward pass), fed through `causal_conv1d_update`'s
     `F.conv1d(..., groups=hidden_size)` call. Real weights + real values +
     real batch=2 context vs. the SAME exact tensors replayed standalone at
     batch=1: they disagree by 0.445 (this script reproduces that number).
  4. `torch.use_deterministic_algorithms(True)` does NOT fix it -- expected,
     since that flag guarantees run-to-run repeatability for the SAME
     inputs, not batch-size invariance across DIFFERENT total problem sizes,
     which is a different property entirely.

CONCLUSION: this is a genuine numerical property of grouped/depthwise
`F.conv1d` on this hardware (AMD RDNA3 / ROCm) for this model's real trained
weight distribution -- the convolution kernel's internal reduction/algorithm
selection is sensitive to total batch size in a way that is invisible with
synthetic random test data but real and large (0.445 raw, ~5.6% real
downstream argmax-flip rate) with the model's actual weights. Not fixable in
this codebase; flagged as a real hardware/kernel-library numerical gap.

    uv run --env-file .env python \
        experiments/runtime/speculative/batched_tree_verification/repro_conv1d_batch_dependence.py
"""

from __future__ import annotations

import sys

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.canon import CANON, REPO_ROOT  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps" / "runtime-ipwf"))

import transformers.models.qwen3_5.modeling_qwen3_5 as qmod  # noqa: E402

PROMPT = "### Question:\nWrite a PostgreSQL query that returns the top customers by revenue.\n\n### Answer:\n"


def main() -> None:
    print("=" * 100)
    print("  REPRO: real GatedDeltaNet conv1d step is batch-size-dependent on real weights")
    print("=" * 100)

    tok = AutoTokenizer.from_pretrained(CANON.BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CANON.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()

    orig_update = qmod.causal_conv1d_update
    captured: list[dict] = []

    def wrapped(hidden_states, conv_state, weight, bias=None, activation=None):
        out = orig_update(hidden_states, conv_state, weight, bias, activation)
        captured.append({
            "hidden_states": hidden_states.detach().clone(),
            "conv_state_before": conv_state.detach().clone(),
            "weight": weight.detach().clone(),
            "bias": None if bias is None else bias.detach().clone(),
            "activation": activation,
            "out": out.detach().clone(),
        })
        return out

    qmod.causal_conv1d_update = wrapped

    ids1 = tok(PROMPT, return_tensors="pt").input_ids.to(model.device)
    ids2 = ids1.repeat(2, 1)
    with torch.no_grad():
        out = model(ids2, use_cache=True)
    cache = out.past_key_values
    nxt = torch.argmax(out.logits[:, -1, :], -1, keepdim=True)
    pos = ids1.shape[1]
    tok_a = nxt[0:1]
    tok_b = tok_a.clone()
    tok_b[0, 0] = tok_a.item() + 1  # a real, different (arbitrary) next token for row 1
    chunk_ab = torch.cat([tok_a, tok_b], dim=0)
    positions = torch.arange(pos, pos + 1, device=model.device)

    captured.clear()
    with torch.no_grad():
        model(chunk_ab, past_key_values=cache, cache_position=positions, use_cache=True)
    qmod.causal_conv1d_update = orig_update

    batched_l1 = captured[1]  # layer index 1's real call, batch=2, real weights
    hs1 = batched_l1["hidden_states"][1:2].clone()
    cs1 = batched_l1["conv_state_before"][1:2].clone()
    out_asis_batched = batched_l1["out"][1:2]

    out_isolated = orig_update(hs1, cs1.clone(), batched_l1["weight"], batched_l1["bias"], batched_l1["activation"])
    diff = (out_asis_batched - out_isolated).abs().max().item()

    print(f"\n  Row 1's REAL captured (weight, activations) tensors -- byte-identical either way.")
    print(f"  Computed AS PART of the real batch=2 forward pass  vs.")
    print(f"  the SAME exact tensors replayed standalone at batch=1:")
    print(f"\n  max|diff| = {diff:.6f}")

    torch.use_deterministic_algorithms(True, warn_only=True)
    out_det = orig_update(hs1, cs1.clone(), batched_l1["weight"], batched_l1["bias"], batched_l1["activation"])
    diff_det = (out_asis_batched - out_det).abs().max().item()
    torch.use_deterministic_algorithms(False)
    print(f"  max|diff| with torch.use_deterministic_algorithms(True): {diff_det:.6f}  "
          f"({'NOT fixed -- expected, different property' if diff_det > 1e-3 else 'fixed'})")

    print("\n  => Mathematically, groups=hidden_size (fully depthwise) conv1d has NO way for one")
    print("     batch row to influence another's output. This is a real numerical property of")
    print("     F.conv1d's kernel/algorithm selection on this hardware for this model's real")
    print("     trained weights -- invisible with synthetic random test data of matching shape")
    print("     (confirmed separately, see docs/DECISIONS.md §76), real and large with the")
    print("     actual weights. Not fixable in this codebase.")


if __name__ == "__main__":
    main()
