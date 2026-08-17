"""Sizing the ops Liger does NOT cover — is a custom Triton kernel justified?

WHY NOW
-------
Liger measured **47.5% faster** on the real training workload (`docs/DECISIONS.md`
§12), which resets the prior on fusion: custom kernels on gfx1100 are high-leverage
when aimed at the right op. Liger covers fused_linear_cross_entropy, rms_norm and
swiglu. It explicitly does NOT cover:

  * RoPE   — `apply_liger_kernel_to_qwen3_5(rope=True)` raises NotImplementedError
             for Qwen3.5's hybrid GatedDeltaNet/attention mix. Blanket opt-out.
  * The GatedDeltaNet linear-attention layers, forward and backward.

If those are a meaningful slice of the step, there is a mathematically justified
Triton target. If they are 2%, there is not.

WHY NOT torch.profiler
----------------------
It returns **zero CUDA events** on this ROCm build (no roctracer) — a previous
benchmark printed an empty kernel table under "total GPU self time 0.0 ms" and
nothing flagged it. So timing here uses two mechanisms that do not depend on
profiler support:

  1. CUDA EVENTS on EVERY module, reduced to EXCLUSIVE time by subtracting each
     module's direct children. Events record asynchronously and are read after a
     single synchronize, so the stream is not serialised. Hooking only LEAVES --
     the first version of this file -- reports GatedDeltaNet as 0.0%, because it
     is a COMPOSITE module whose scan runs as functional code in its own forward.
     That is a measurement hole that reads exactly like a finding.

  2. A FUNCTION PATCH for RoPE. `apply_rotary_pos_emb` is a free function, not a
     module, so module hooks miss it completely — hooking alone would have
     reported RoPE as 0% and "proved" it not worth fusing. The patch is verified
     to have bound (it raises if the symbol is not found) rather than silently
     measuring nothing.

Both forward and backward are timed: the backward pass is where a fused kernel for
a linear-attention layer would earn most of its keep, and it is roughly 40% of the
step on this rig.

    uv run --env-file .env python \
        benchmarks/factory/training_profile/benchmark_unfused_op_sizing.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import set_hard_vram_cap  # noqa: E402


class EventTimer:
    """Accumulates GPU time per bucket using CUDA events (no stream serialisation)."""

    def __init__(self):
        self.pairs: dict[str, list[tuple]] = defaultdict(list)
        self.open: dict[str, torch.cuda.Event] = {}

    def start(self, bucket: str) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.open[bucket] = e

    def stop(self, bucket: str) -> None:
        s = self.open.pop(bucket, None)
        if s is None:
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.pairs[bucket].append((s, e))

    def totals(self) -> dict[str, float]:
        torch.cuda.synchronize()
        out = {}
        for k, v in self.pairs.items():
            out[k] = sum(s.elapsed_time(e) for s, e in v)
        return out

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.pairs.items()}


# Disjoint sibling classes inside a decoder layer. Timing these INCLUSIVELY needs
# no tree arithmetic: they do not nest inside one another, so their windows are
# non-overlapping and each is directly a share of the step.
#
# ⚠️ TWO EARLIER APPROACHES IN THIS FILE BOTH PRODUCED FALSE ZEROS.
#   (a) Leaf-only hooking reported GatedDeltaNet = 0.0%, because it is a COMPOSITE
#       module whose scan runs as functional code in its own forward. Only 240.5 ms
#       of a 459.3 ms step was attributed and nothing flagged the gap.
#   (b) Hooking every module and subtracting direct children broke on CONTAINER
#       modules: `Qwen3_5TextModel`'s child `layers` is a ModuleList with no
#       forward, so it never times, subtracting it removes nothing, and the parent
#       keeps its whole subtree. That summed to 239% of the step with -643 ms
#       "unattributed" — incoherent, and only visible because the row was printed.
# The whitelist below avoids both by never relying on parent/child bookkeeping.
WHITELIST = (
    "Qwen3_5GatedDeltaNet",          # linear-attention layer (fusion candidate)
    "Qwen3_5Attention",              # full-attention layer
    "LigerQwen3MoeSwiGLUMLP",        # MLP (Liger-fused)
    "LigerRMSNormForQwen3Next",      # norms (Liger-fused)
    "FusedRMSNormGated",             # GatedDeltaNet's internal gated norm
    "Conv1d",                        # GatedDeltaNet's short conv
)


def install_module_hooks(model, timer: EventTimer, handles: list) -> dict[str, int]:
    """Hook only the whitelisted, mutually non-nesting classes. Inclusive timing.

    Backward is NOT hooked here: `register_full_backward_hook` does not reliably
    fire on multi-input/output modules and silently reported
    `bwd::Qwen3_5GatedDeltaNet = 0.00%`. The backward share is obtained by
    ablation instead (forward-only vs forward+backward), which cannot silently
    return zero.
    """
    counts: dict[str, int] = defaultdict(int)
    for _name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls not in WHITELIST:
            continue
        counts[cls] += 1
        handles.append(mod.register_forward_pre_hook(
            lambda m, i, c=cls: timer.start(f"fwd::{c}::{id(m)}")))
        handles.append(mod.register_forward_hook(
            lambda m, i, o, c=cls: timer.stop(f"fwd::{c}::{id(m)}")))
    return dict(counts)


def patch_rope(timer: EventTimer) -> tuple[object, str]:
    """Time apply_rotary_pos_emb, which is a FUNCTION and invisible to hooks.

    Raises if the symbol cannot be found, rather than leaving RoPE silently
    unmeasured and reporting it as 0% of the step.
    """
    import importlib

    for modpath in (
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "transformers.models.qwen3.modeling_qwen3",
        "transformers.models.qwen2.modeling_qwen2",
    ):
        try:
            m = importlib.import_module(modpath)
        except ModuleNotFoundError:
            continue
        fn = getattr(m, "apply_rotary_pos_emb", None)
        if fn is None:
            continue

        def timed(*a, _fn=fn, **kw):
            timer.start("fwd::apply_rotary_pos_emb")
            r = _fn(*a, **kw)
            timer.stop("fwd::apply_rotary_pos_emb")
            return r

        m.apply_rotary_pos_emb = timed
        return m, modpath
    raise RuntimeError(
        "apply_rotary_pos_emb not found in any candidate modeling module — RoPE "
        "would be silently unmeasured. Refusing to report a 0% that is an artefact."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128,
                    help="astral's real mean is 109 tokens; 512 is NOT representative")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/unfused_op_sizing.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  SIZING THE UNFUSED OPS — RoPE and GatedDeltaNet, forward and backward")
    print("=" * 100)
    print(f"  device={dev}  batch={args.batch}  seq_len={args.seq_len}  steps={args.steps}")
    print("  (Liger ON — the production config. seq_len matches the real corpus, not 512.)\n")

    from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    apply_liger_kernel_to_qwen3_5()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    ))
    model.train()

    ids = torch.randint(1000, 40000, (args.batch, args.seq_len), device="cuda")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

    def step():
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(4):
        step()
    torch.cuda.synchronize()

    # clean wall clock, no instrumentation
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    clean_ms = 1000.0 * (time.perf_counter() - t0) / args.steps
    print(f"  clean step time: {clean_ms:.1f} ms\n")

    # --- forward-only vs full step: the backward share, by ablation ----------
    # Module backward hooks silently returned 0.00% for GatedDeltaNet, so the
    # backward split is measured by NOT running backward rather than by trusting
    # a hook that can fail quietly.
    def fwd_only():
        with torch.no_grad():
            model(input_ids=ids, labels=ids)

    for _ in range(3):
        fwd_only()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        fwd_only()
    torch.cuda.synchronize()
    fwd_ms = 1000.0 * (time.perf_counter() - t0) / args.steps
    bwd_opt_ms = clean_ms - fwd_ms
    print(f"  forward-only  : {fwd_ms:6.1f} ms  ({100 * fwd_ms / clean_ms:.0f}% of step)")
    print(f"  backward+optim: {bwd_opt_ms:6.1f} ms  ({100 * bwd_opt_ms / clean_ms:.0f}%)\n")

    timer = EventTimer()
    handles: list = []
    counts = install_module_hooks(model, timer, handles)
    _, ropemod = patch_rope(timer)
    print(f"  RoPE patched in {ropemod}")
    print("  hooked: " + ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())) + "\n")

    for _ in range(args.steps):
        step()
    totals = timer.totals()
    for h in handles:
        h.remove()

    by_class: dict[str, float] = defaultdict(float)
    for k, v in totals.items():
        parts = k.split("::")
        cls = parts[1] if len(parts) > 1 else k
        by_class[cls] += v / args.steps

    print("=" * 100)
    print(f"  FORWARD GPU TIME BY LAYER TYPE  (inclusive; of a {clean_ms:.1f} ms step, "
          f"forward = {fwd_ms:.1f} ms)")
    print("=" * 100)
    print(f"  {'class':<40}{'ms/step':>10}{'% of fwd':>11}{'% of step':>12}")
    print("  " + "-" * 96)
    rows = []
    for cls in sorted(by_class, key=lambda x: -by_class[x]):
        ms = by_class[cls]
        print(f"  {cls:<40}{ms:>10.2f}{100 * ms / fwd_ms:>10.1f}%{100 * ms / clean_ms:>11.1f}%")
        rows.append({"class": cls, "ms_per_step": ms,
                     "pct_of_forward": 100 * ms / fwd_ms,
                     "pct_of_step": 100 * ms / clean_ms})

    gdn = by_class.get("Qwen3_5GatedDeltaNet", 0.0)
    rope_fn = by_class.get("apply_rotary_pos_emb", 0.0)
    attn = by_class.get("Qwen3_5Attention", 0.0)

    # GatedDeltaNet's backward is not directly attributable, but its forward share
    # bounds it: scale by the measured backward/forward ratio for the whole step.
    bwd_ratio = bwd_opt_ms / fwd_ms
    gdn_total_est = gdn * (1 + bwd_ratio)
    rope_total_est = rope_fn * (1 + bwd_ratio)

    print("\n" + "=" * 100)
    print("  THE TWO CANDIDATE TARGETS")
    print("=" * 100)
    print(f"    GatedDeltaNet forward (incl. its conv/norm/proj) : {gdn:6.2f} ms  "
          f"{100 * gdn / clean_ms:5.2f}% of step")
    print(f"    RoPE  apply_rotary_pos_emb                       : {rope_fn:6.2f} ms  "
          f"{100 * rope_fn / clean_ms:5.2f}% of step")
    print(f"    (full attention layers, for scale)               : {attn:6.2f} ms  "
          f"{100 * attn / clean_ms:5.2f}% of step")
    print(f"\n  Backward is {bwd_ratio:.2f}x forward on this step. Scaling each candidate by")
    print("  (1 + that ratio) gives an UPPER BOUND on its total footprint, assuming its")
    print("  backward is proportional to its forward — module backward hooks are not")
    print("  reliable here, so this is a bound, not a measurement:")
    for name, est in (("GatedDeltaNet", gdn_total_est), ("RoPE", rope_total_est)):
        pct = 100 * est / clean_ms
        ceil = 1 / (1 - pct / 100) if pct < 100 else float("inf")
        print(f"    {name:<16} <= {pct:5.2f}% of step  ->  perfect fusion caps at {ceil:.3f}x")
    print("\n    Training only — never touches user-facing serving latency.")

    report = {
        "device": dev, "batch": args.batch, "seq_len": args.seq_len,
        "clean_ms_per_step": clean_ms,
        "forward_ms_per_step": fwd_ms,
        "backward_optim_ms_per_step": bwd_opt_ms,
        "backward_to_forward_ratio": bwd_ratio,
        "gated_delta_fwd_ms": gdn,
        "gated_delta_pct_of_step_fwd": 100 * gdn / clean_ms,
        "gated_delta_upper_bound_pct": 100 * gdn_total_est / clean_ms,
        "rope_fn_ms": rope_fn,
        "rope_upper_bound_pct": 100 * rope_total_est / clean_ms,
        "by_class": rows,
        "hooked_counts": counts,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
