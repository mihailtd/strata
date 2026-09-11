"""Where does K-token verification time actually go, and what is the break-even?

CORRECTS AN EARLIER VERSION OF THIS SCRIPT, which reported that K=4 verification
costs 0.94x a single-token step -- i.e. that multi-token verification is CHEAPER
than single-token decode. It is not. Two defects produced that:

1. It called `model(base_tokens[:, :k], use_cache=False)`: a standalone K-token
   forward with NO KV cache and NO context. Real verification appends K draft
   tokens to a cache built from a ~128-token prompt and advances the recurrent
   state from its current value. With no cache the measurement is dominated by
   the fixed ~8 GB weight read and is therefore flat in K BY CONSTRUCTION.
   Measured side by side:

        method                     K=1      K=2      K=4      K=8
        use_cache=False (wrong)   1.00x    1.03x    0.99x    0.96x
        warm cache (correct)      1.00x    1.53x    1.52x    1.46x

2. Its attribution table was ALL ZEROS, and "no short-conv penalty" was
   concluded from `conv_time_ms == 0.0`. The cause: on this torch/ROCm build
   `evt.device_time_total` EXISTS but is always 0 (only CPU times populate), so
   `getattr(evt, "device_time_total", <fallback>)` reads the zero and never
   falls back. torch.profiler cannot attribute GPU kernel time here at all.
   This version uses CUDA events on module hooks instead, which does work.

BREAK-EVEN IS NOT THE COST RATIO. Accepting M drafts yields M+1 tokens, and a
partial acceptance costs a further state re-advance. With verify(K) = r*t1 and
re-advance on the same chunked path:

    required E[M]  =  d/t1 + r*(1 + P_partial) - 1

where d is the draft cost. Reporting tau = r understates the requirement badly.

    uv run --env-file .env scripts/profile_mtp_verification_path.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if torch.cuda.is_available():  # fla's device probe is @cache'd at import
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM  # noqa: E402

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.novel_peft import set_hard_vram_cap  # noqa: E402


class ModuleTimer:
    """CUDA-event timing around a module's forward, summed over all instances.

    torch.profiler reports 0 device time on this ROCm build, so kernel-level
    attribution has to come from explicit events instead.
    """

    def __init__(self):
        self.pairs = []
        self._start = None

    def pre(self, *_):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._start = e

    def post(self, *_):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        if self._start is not None:
            self.pairs.append((self._start, e))
            self._start = None

    def total_ms(self):
        torch.cuda.synchronize()
        t = sum(a.elapsed_time(b) for a, b in self.pairs)
        self.pairs = []
        return t


@torch.no_grad()
def verification_ms(model, prompt_len, k, reps=20):
    """Median ms to append k tokens to a WARM cache -- the real verification cost."""
    ids = torch.randint(1000, 50000, (1, prompt_len), device=model.device)
    cache = model(ids, use_cache=True).past_key_values
    step = torch.randint(1000, 50000, (1, k), device=model.device)
    for _ in range(5):
        model(step, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(step, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


@torch.no_grad()
def attribute(model, prompt_len, k, reps=10):
    """Split linear-attention time into conv vs the rest, via module hooks."""
    la_t, conv_t = ModuleTimer(), ModuleTimer()
    handles = []
    for lay in model.model.layers:
        if not hasattr(lay, "linear_attn"):
            continue
        la = lay.linear_attn
        handles += [la.register_forward_pre_hook(la_t.pre), la.register_forward_hook(la_t.post)]
        handles += [la.conv1d.register_forward_pre_hook(conv_t.pre), la.conv1d.register_forward_hook(conv_t.post)]

    ids = torch.randint(1000, 50000, (1, prompt_len), device=model.device)
    cache = model(ids, use_cache=True).past_key_values
    step = torch.randint(1000, 50000, (1, k), device=model.device)
    for _ in range(3):
        model(step, past_key_values=cache, use_cache=True)
    la_t.total_ms(), conv_t.total_ms()  # discard warmup

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        model(step, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / reps * 1000
    la_ms, conv_ms = la_t.total_ms() / reps, conv_t.total_ms() / reps
    for h in handles:
        h.remove()
    return {
        "wall_ms": wall,
        "linear_attn_ms": la_ms,
        "conv_ms": conv_ms,
        "linear_attn_pct": 100 * la_ms / wall,
        "conv_pct": 100 * conv_ms / wall,
        "conv_pct_of_linear_attn": 100 * conv_ms / max(1e-9, la_ms),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--draft-ms", type=float, default=3.3, help="measured per-step MTP draft cost")
    ap.add_argument("--p-partial", type=float, default=0.9, help="P(not all K drafts accepted)")
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/fla_verification_profile.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    fla = M.chunk_gated_delta_rule is not None
    print(f"fla chunk kernel active: {fla}   |  causal_conv1d: {M.causal_conv1d_fn is not None}\n")

    print("VERIFICATION COST (append K tokens to a warm cache -- the real quantity)")
    print(f"  {'K':>3s} {'ms':>8s} {'vs K=1':>8s} {'required E[M] to break even':>29s}")
    t1 = None
    rows = {}
    for k in args.ks:
        ms = verification_ms(model, args.prompt_len, k)
        if t1 is None:
            t1 = ms
        r = ms / t1
        need = args.draft_ms / t1 + r * (1 + args.p_partial) - 1
        rows[k] = {"ms": ms, "ratio_vs_k1": r, "required_accepted_tokens": need}
        print(f"  {k:3d} {ms:7.2f} {r:7.2f}x {need:28.2f}")

    print("\nATTRIBUTION at K=4 (CUDA events on module hooks; torch.profiler")
    print("reports 0 device time on this ROCm build and cannot be used)")
    att = attribute(model, args.prompt_len, 4)
    print(f"  wall              {att['wall_ms']:7.2f} ms")
    print(f"  linear_attn total {att['linear_attn_ms']:7.2f} ms  ({att['linear_attn_pct']:.1f}% of wall)")
    print(f"  of which conv1d   {att['conv_ms']:7.2f} ms  ({att['conv_pct']:.1f}% of wall, "
          f"{att['conv_pct_of_linear_attn']:.1f}% of linear_attn)")
    print("\n  causal_conv1d is NOT installed, so conv1d runs the torch fallback.")
    print("  The number above is what that fallback actually costs -- it is the")
    print("  measurement the previous version reported as 0.0 because its")
    print("  profiler returned nothing.")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "fla_active": fla,
                "causal_conv1d_available": M.causal_conv1d_fn is not None,
                "method": "append K tokens to warm KV+recurrent cache",
                "k_sweep": rows,
                "attribution_k4": att,
                "break_even_formula": "E[M] >= draft_ms/t1 + ratio*(1+P_partial) - 1",
                "assumptions": {"draft_ms": args.draft_ms, "p_partial": args.p_partial},
            },
            indent=2,
        )
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
