"""What is left to fuse in a training step, after Liger?

THE CLAIM UNDER TEST
--------------------
"Fused Triton Backprop (Faster Training on ROCm)" is on the roadmap as a 3+ week
kernel project. But this repo's m2 methodology ALREADY applies Liger's fused
kernels — `liger_fused_kernels: true` on every m2 adapter — covering
fused_linear_cross_entropy, rms_norm and swiglu. So the real question is not
"would fusion help" but "how much is left after the fusion already in place".

Two measurements bound it:

  1. LIGER ON vs OFF, wall clock. What the existing fusion is worth here. This is
     also the best available estimate of the scale of the remaining opportunity:
     if the fused kernels already applied buy X%, a second round of fusion on the
     leftovers is very unlikely to buy more than X%.

  2. KERNEL BREAKDOWN via torch.profiler. Where the step time actually sits, and
     specifically what share is in ops Liger does NOT cover -- notably RoPE, which
     Liger refuses for Qwen3.5 (`NotImplementedError`, the hybrid GatedDeltaNet /
     attention mix), and the GatedDeltaNet backward itself.

⚠️ THE SYNTHETIC A/B IN THIS FILE GAVE THE WRONG ANSWER. READ THIS FIRST.

At the fixed shape below (batch 2 x seq 512, random tokens, EVERY position
contributing to cross-entropy) this benchmark measures:

    liger_on   874.3 ms/step   14.99 GB
    liger_off  689.4 ms/step   19.34 GB   -> "Liger is 0.789x, i.e. 27% SLOWER"

A matched A/B on the REAL trainer, real data, both arms in-session, reverses it:

    liger      127.6 s / 150 steps   final EMA 1.0717
    noliger    188.2 s / 150 steps   final EMA 1.0704   -> Liger is 47.5% FASTER
                                                           (losses equivalent,
                                                            delta -0.0013 vs
                                                            noise floor 0.0173)

The factory does not run this shape. Real batches are dynamically padded to a
MEAN OF 109 TOKENS for astral with the prompt masked out, not 512 fully-labelled
positions. **Trust `results/loss_curves/astral_{liger,noliger}_insession.json`
over the synthetic numbers here**, and treat this file as a phase breakdown and a
VRAM measurement rather than a verdict on Liger.

Everything downstream of that is timed on the real trainer path so the numbers
describe the factory as it actually runs.

    uv run --env-file .env python \
        benchmarks/factory/training_profile/benchmark_training_step_profile.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import set_hard_vram_cap  # noqa: E402

DATA = "data/astral/training_data.jsonl"


def build(model_id: str, use_liger: bool, rank: int, alpha: int):
    """Liger patches at CLASS level, so it must be applied BEFORE from_pretrained.

    Reversing that order makes the patch silently do nothing -- the trainer
    documents this and it is the difference between measuring Liger and measuring
    nothing at all.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if use_liger:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_5

        apply_liger_kernel_to_qwen3_5()

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    return get_peft_model(model, cfg)


def step_batch(tok_ids: torch.Tensor, model, opt) -> None:
    out = model(input_ids=tok_ids, labels=tok_ids)
    out.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def timed_steps(model, ids, n: int, warmup: int = 5) -> float:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    for _ in range(warmup):
        step_batch(ids, model, opt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step_batch(ids, model, opt)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def run_single_arm(args, use_liger: bool) -> None:
    """One arm, one process. Emits ARMRESULT <json> for the parent to parse."""
    set_hard_vram_cap(args.vram_cap_gb)
    ids = torch.randint(1000, 40000, (args.batch, args.seq_len), device="cuda")
    model = build(args.model_id, use_liger, args.rank, args.alpha)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    s = timed_steps(model, ids, args.steps)
    phases = phase_breakdown(model, ids, args.steps)
    print("ARMRESULT " + json.dumps({
        "s_per_step": s,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "liger": use_liger,
        "phases_s": phases,
    }))


def phase_breakdown(model, ids, n: int) -> dict:
    """forward / backward / optimizer, synchronised.

    The torch profiler returns zero CUDA events on this ROCm build (no
    roctracer), so the first version of this benchmark printed an empty kernel
    table under the heading "total GPU self time 0.0 ms". Manual phase timing
    does not depend on profiler support and answers the question that actually
    matters: which third of the step a new kernel would have to attack.
    """
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    for _ in range(3):
        step_batch(ids, model, opt)
    torch.cuda.synchronize()
    acc = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = model(input_ids=ids, labels=ids)
        torch.cuda.synchronize()
        acc["forward"] += time.perf_counter() - t

        t = time.perf_counter()
        out.loss.backward()
        torch.cuda.synchronize()
        acc["backward"] += time.perf_counter() - t

        t = time.perf_counter()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        acc["optimizer"] += time.perf_counter() - t
    return {k: v / n for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-liger", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--arm-no-liger", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/training_step_profile.json")
    args = ap.parse_args()

    if args.arm_liger or args.arm_no_liger:
        run_single_arm(args, use_liger=args.arm_liger)
        return

    set_hard_vram_cap(args.vram_cap_gb)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 100)
    print("  TRAINING STEP PROFILE — what is left to fuse after Liger?")
    print("=" * 100)
    print(f"  device={dev}  batch={args.batch}  seq_len={args.seq_len}  "
          f"r={args.rank} alpha={args.alpha}  steps={args.steps}\n")

    report: dict = {"device": dev, "batch": args.batch, "seq_len": args.seq_len, "arms": {}}

    # --- 1. Liger on/off, ONE ARM PER PROCESS -------------------------------
    #
    # apply_liger_kernel_to_qwen3_5() rebinds at CLASS level and there is no
    # unpatch. Running both arms in one process therefore measures Liger against
    # ITSELF: the first arm patches the class, the second inherits the patch.
    # That is exactly what happened on the first run of this benchmark — it
    # reported "Liger is worth 0.994x and -0.01 GB", which is physically
    # impossible, since fused_linear_cross_entropy alone avoids materialising a
    # 2x512x248320 bf16 logits tensor (~508 MB, plus its gradient). Each arm now
    # gets a fresh interpreter.
    import subprocess

    for arm, flag in (("liger_on", "--arm-liger"), ("liger_off", "--arm-no-liger")):
        cmd = [sys.executable, __file__, flag,
               "--model-id", args.model_id, "--rank", str(args.rank),
               "--alpha", str(args.alpha), "--seq-len", str(args.seq_len),
               "--batch", str(args.batch), "--steps", str(args.steps),
               "--vram-cap-gb", str(args.vram_cap_gb)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        line = [x for x in res.stdout.splitlines() if x.startswith("ARMRESULT ")][-1]
        payload = json.loads(line[len("ARMRESULT "):])
        report["arms"][arm] = payload
        print(f"  {arm:<12} {payload['s_per_step'] * 1000:8.1f} ms/step   "
              f"peak VRAM {payload['peak_vram_gb']:5.2f} GB   (fresh process)")

    on, off = report["arms"]["liger_on"], report["arms"]["liger_off"]
    speedup = off["s_per_step"] / on["s_per_step"]
    vram_saved = off["peak_vram_gb"] - on["peak_vram_gb"]
    report["liger_speedup"] = speedup
    report["liger_vram_saved_gb"] = vram_saved
    print(f"\n  Liger is worth {speedup:.3f}x wall clock and {vram_saved:+.2f} GB peak VRAM.")
    print("  That is the scale of what ALREADY-APPLIED fusion buys on this rig, and the")
    print("  best available prior on what a second round could add.")

    # --- 2. where the step time sits ---------------------------------------
    print("\n" + "=" * 100)
    print("  PHASE BREAKDOWN — which third of the step would a new kernel attack?")
    print("=" * 100)
    print("  (torch.profiler returns zero CUDA events on this ROCm build, so this is")
    print("   manual synchronised timing rather than a kernel-level trace.)\n")
    print(f"  {'arm':<12}{'forward':>12}{'backward':>12}{'optimizer':>12}{'sum':>10}")
    print("  " + "-" * 96)
    for arm in ("liger_on", "liger_off"):
        ph = report["arms"][arm].get("phases_s", {})
        tot = sum(ph.values()) or 1e-9
        print(f"  {arm:<12}"
              + "".join(f"{1000 * ph.get(k, 0):>9.1f} ms" for k in ("forward", "backward", "optimizer"))
              + f"{1000 * tot:>8.1f} ms")
        report["arms"][arm]["phase_pct"] = {k: 100.0 * v / tot for k, v in ph.items()}
    on_ph = report["arms"]["liger_on"].get("phase_pct", {})
    if on_ph:
        print("\n  Liger ON shares: "
              + "  ".join(f"{k} {v:.1f}%" for k, v in on_ph.items()))
        print(f"\n  A further fusion project must beat {speedup:.3f}x (what Liger already")
        print("  buys) while attacking the same forward+backward that Liger has covered.")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
