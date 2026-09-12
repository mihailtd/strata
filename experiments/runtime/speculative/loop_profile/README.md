# Speculative loop profile — sizing a fused verification kernel before writing one

**Verdict: do not build SonicSampler-style fused verification. The fusable share
is 1.5% of the loop, an Amdahl ceiling of 1.016×.** The real cost is somewhere
nobody was looking: the **rollback re-forward at 29.4%**.

The roadmap carried "Fused Speculative Verification (SonicSampler-style)" as a
1–2 week ROCm/Triton project, justified by the claim that speculative decoding
"wastes critical time bouncing draft tokens between CPU and GPU". That claim was
never measured on this stack. It is false here, and it is false in both
directions — see the negative result below.

```bash
uv run --env-file .env python \
    benchmarks/runtime/speculative/loop_profile/benchmark_speculative_loop_profile.py
```

Results: [`results/speculative_loop_profile.json`](../../../../results/speculative_loop_profile.json)

---

## Where the time actually goes

K=4, astral expert folded, 8 prompts × 64 tokens, 89 speculative steps, τ=2.02:

| phase | % of loop | ms/call | calls | |
| :--- | ---: | ---: | ---: | :--- |
| `verify_fwd` | **43.9%** | 36.36 | 89 | the chunked verification forward |
| `commit_fwd` | **29.4%** | 34.96 | **62** | **the rollback re-forward** |
| `draft_gen` | 14.4% | 11.88 | 89 | K draft tokens from the MTP head |
| `prompt_prefill` | 4.3% | 39.55 | 8 | |
| `draft_prefill` | 2.9% | 2.36 | 89 | |
| `snapshot` | 1.1% | 0.94 | 89 | 52.5 MB recurrent state copy |
| `accept_calc` | **1.0%** | 0.81 | 89 | ← the fusion target |
| `bookkeeping` | 0.9% | 0.70 | 89 | |
| `restore` | 0.7% | 0.87 | 62 | |
| `argmax` | **0.6%** | 0.46 | 89 | ← the fusion target |
| `hidden_cat` | 0.5% | 0.38 | 89 | |
| `chunk_build` / `commit_build` | 0.2% each | 0.17 | 89 | |

Per-phase timing needs a `cuda.synchronize()` around each phase, which inflates
the total by **1.20×** (7.36 s instrumented vs 6.16 s clean). The percentages are
shares of the inflated whole and are reported as such rather than passed off as
wall clock.

## The fusion target is 1.5%

`argmax` + `accept_calc` = **1.5%** of the loop. Making them *entirely free* caps
at **1.016×**, before the prefill share of a real agent turn is applied on top —
at 3,000 in / 150 out decode is 87% of the turn, so the end-to-end ceiling is
**~1.014×** for one to two weeks of Triton work on an architecture where AITER's
RDNA3 support is already partial.

### The negative result that settles it

The premise was CPU↔GPU round trips, so the obvious fix was implemented and timed
rather than argued about. The Python accept loop reads back through `.item()`:

```python
for i in range(k):
    if draft[0, i].item() == target[i].item():   # 2 syncs per token
```

replaced by a single GPU reduction with one sync:

```python
match = (draft[0, :k] == target[:k]).int()
n_acc = int(torch.cumprod(match, 0).sum())       # 1 sync
```

| arm | syncs | syncs/step | clean wall clock |
| :--- | ---: | ---: | ---: |
| `cpu_item_loop` | 484 | 5.4 | **6.16 s** |
| `gpu_single_sync` | 89 | 1.0 | 6.26 s |

**Cutting syncs 5.4× made it 1.65% SLOWER.** Two reasons, both of which
generalise: the `.item()` loop **short-circuits** on the first mismatch, so at
τ=2.02 it averages 5.4 syncs per step rather than the 8 a K=4 worst case suggests;
and `cumprod`+`sum` launch several small kernels whose overhead exceeds the syncs
they remove. There is no CPU-bouncing problem here to fuse away.

## What the profile actually points at

**`commit_fwd` is 29.4% of the loop and was not on anyone's list.** On a partial
accept the chunked forward has already advanced the recurrent state past the
rejected tokens, so the loop restores the 52.5 MB snapshot and re-runs a full
forward over just the committed prefix. At τ=2.02 that happens on **62 of 89
steps (70%)**, and it costs 34.96 ms — essentially as much as the verification
forward it is repairing (36.36 ms).

So a speculative step usually costs **two full model forwards**, not one.

Eliminating it entirely would cap at **1.42× on the loop** (~1.33× end-to-end on a
3,000/150 turn) — roughly 25× the ceiling of the fused-verification project it
displaces. It cannot be removed for free: it exists because the GatedDeltaNet
chunked kernel returns only the final recurrent state, so there is no way to ask
for the state as of position `n_acc`. Making the chunked scan emit intermediate
per-position states is a genuine kernel project, and it is a **different** one
from SonicSampler.

**This is a lead, not a result.** Whether that kernel is feasible on gfx1100
without `fla` is unknown, and the 1.42× is an upper bound assuming the re-forward
becomes free rather than cheaper. Size it before building it — which is the point
of this benchmark existing at all.

## Caveats

- One domain (astral), one K (4), one τ regime (2.02). `commit_fwd`'s share is a
  direct function of the partial-accept rate, so it falls as τ rises toward K and
  rises as τ drops. A τ sweep would bound it properly.
- 89 steps is enough to separate 43.9% from 1.0% but not to resolve the sub-1%
  phases against each other.
- Timing is on the folded astral expert with `fla` unavailable (the PyTorch
  fallback path), which is this rig's real configuration but inflates the chunked
  forward relative to a machine with working linear-attention kernels. If `fla`
  ever builds here, **re-run this before trusting any of the shares** — a faster
  `verify_fwd` raises everything else's percentage, including the fusion target.
