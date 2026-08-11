# Project Execution Plan

> **Revisit Billboard Impostors when having 10–20 distinct domain adapters for "hot-swapping"!** It requires building a fixed-slot tensor copy infrastructure from scratch, and without a fleet of distinct domain adapters (SQL, Rust, YAML), you would be testing infrastructure without a real workload.

---

## ⚠️ Status of This Document (2026-08-11)

This plan was written before several of its premises were measured. Corrections,
all from real runs on this repo's hardware — see `## Measured Reality` below for
the full numbers:

- **Velocity-Masked SFT is dead as a pillar.** The "59.95% → 83.70%" figure this
  document is built on came from runs where the gate was provably inert (0% of
  layers ever masked, due to a hook bug). Under a controlled A/B it neither
  improves adherence nor saves compute. Sections describing it as a load-bearing
  component are struck through below rather than deleted, so the reasoning trail
  survives.
- **The ~3 MB micro-adapter target is real as arithmetic, unmet in practice.**
  Kronecker factorization really does give ~3.2 MB on this model's shapes, but
  every adapter measured at that scale collapses in quality (LoKr 1.2-2.1 MB →
  14-19% adherence vs base 11.94%). Nothing yet reaches ~3 MB with usable
  quality; the best small result is `id_kron_r8` at **12.6 MB** (4x over target).
- **The architecture benchmark is confounded — twice.** Baseline on-disk sizes
  were inflated ~2x (fp32 assumed vs real bf16), and the custom-path variants
  carry an **~18x inflated effective update** from a kaiming fan_in bug. Fixing
  only the init drops `custom_standard` from 58.17% to 37.55%, i.e. to peft
  LoRA's level. **No custom-vs-peft architecture claim in this document is
  currently supported.** See `## Measured Reality`.
- **Update magnitude, not factorization, is the dominant lever measured so far**
  (+20.6 pp from an ~18x larger effective update). The baseline must be
  magnitude-tuned before any architecture is compared against it.
- **The entry point is NOT a HIP kernel.** `peft` 0.20.0 already ships DoRA
  (`LoraConfig(use_dora=True)`) and Kronecker (`LoKrConfig`). The core
  quality-vs-size question is answerable in a few PyTorch training runs, before
  committing to weeks of C++/HIP work.
- **A local peft patch was found and vendored.** peft's LoKr has no 4-bit
  dispatch and crashes on a quantized base. The fix had been applied by editing
  `.venv/.../peft/tuners/lokr/layer.py` in place — which, because uv *hardlinks*
  its global cache into venvs (verified: same inode, 4 links), had silently
  mutated `~/.cache/uv` and propagated to `training/.venv` as well. Now restored
  to pristine upstream (peft matches its install-record hashes) with the fix
  living in `src/gnn_experiment/peft_compat.py`.

---

## 📐 Measured Reality (supersedes claims below)

Everything in this section is from real runs/arithmetic on Qwen3.5-4B on this machine, greedy-decoded eval (`do_sample=False`), 20-question astral benchmark:

### Architecture Benchmark & Payload Sizes

> ⚠️ **This table is CONFOUNDED. Do not draw architecture conclusions from it.**
> Two systematic errors were found in it, both flattering the custom
> architectures over the peft baselines. Corrected analysis directly below.

Adherence values below are correct (verified against MLflow). **On-disk sizes
and all custom-vs-peft comparisons are not.**

| Scheme | Trainable Params | On-Disk (as claimed) | **On-Disk (measured)** | Adherence % |
| :--- | :---: | :---: | :---: | :---: |
| **Base Model** | 0 | — | — | **11.94%** |
| `id_kron_r16` | 12.4M | 24.9 MB | **24.92 MB** ✅ | **59.32%** |
| `custom_standard` | 10.6M | 40.0 MB | **21.32 MB** ❌ 88% inflated | **58.17%** |
| `id_kron_r8` | 6.26M | 12.0 MB | **12.61 MB** ✅ | **51.44%** |
| PEFT LoRA (r=8, α=16) | 10.6M | 40.47 MB | **21.27 MB** ❌ 90% inflated | **40.47%** |
| **`custom_standard_fixedinit`** | 10.6M | — | **21.32 MB** | **37.55%** |
| PEFT DoRA | 11.4M | 40.98 MB | **22.83 MB** ❌ 79% inflated | **33.92%** |
| KroTucker (r=64) | 85.0M | 170.0 MB | **170.07 MB** ✅ | **19.70%** |
| PEFT LoKr (r=32) | 1.04M | 2.12 MB | **2.12 MB** ✅ | **18.92%** |
| PEFT LoKr (r=8) | 585K | 1.22 MB | **1.22 MB** ✅ | **14.42%** |

#### 🔴 Error 1 — baseline sizes inflated ~2x (mixed dtype conventions)

Every *new* architecture got its real bf16 on-disk size; every *baseline* got an
fp32-assumed size (10.6M x 4 bytes ~= 42 MB instead of the actual 10.6M x 2 =
21.3 MB). Note PEFT LoRA's "40.47 MB" is literally its adherence value (40.47%)
copy-pasted into the size column.

Consequence: **"id_kron_r16 beats custom_standard while saving 40.7% payload
(24.9 vs 40.0 MB)" inverts.** Real: 24.92 vs 21.32 MB — id_kron_r16 is **17%
LARGER**, and has more parameters (12.4M vs 10.6M). Beating a smaller-capacity
adapter by 1.15 pp while using 17% more capacity is not architectural evidence.
Likewise `id_kron_r8`'s "70.4% payload reduction" is really **41%** (12.61 vs
21.32 MB), for **-6.73 pp** adherence — a legitimate size/quality tradeoff point,
but not a free win, and 12.6 MB is 4x over the ~3 MB micro-adapter target.

#### 🔴 Error 2 — the custom path has an ~18x inflated update (init bug)

`NovelLoraLinear` stores its down-projection **transposed** vs peft: `lora_a` is
`(in_features, rank)` where peft's `lora_A` is `(rank, in_features)`.
`nn.init.kaiming_uniform_` derives fan_in from `tensor.size(1)`, so on this
layout it reads fan_in = **rank (8)** instead of **in_features (2560)** and
over-initialises by `sqrt(2560/8)`. Three independent measurements agree:

| | ratio |
| --- | --- |
| Predicted `sqrt(in/rank)` | 17.889x |
| Measured at init | 17.824x |
| Measured in trained ΔW (custom vs peft) | **18.294x** |

`lora_b` is zero-init so the starting delta is 0 and nothing errors — but
`∂L/∂B ∝ A`, so B grows ~18x faster. The two implementations also converge to
**essentially orthogonal solutions**: mean cosine(ΔW_peft, ΔW_custom) = 0.0012,
with |cos| > 0.1 on **0 of 128** modules.

**Controlled test** — `custom_standard_fixedinit` is identical to
`custom_standard` except the init uses the true fan_in:

| variant | init | final loss | adherence |
| --- | --- | --- | --- |
| `custom_standard` | buggy (fan_in=8) | 1.1396 | **58.17%** |
| `custom_standard_fixedinit` | correct (fan_in=2560) | 1.0074 | **37.55%** |
| PEFT LoRA | correct (fan_in=2560) | 1.0083 | **40.47%** |

Fixing one line drops custom_standard **58.17% → 37.55%**, landing next to peft
LoRA, and its training loss moves to within **0.0009** of peft's (from a 0.13
gap). **The two implementations are equivalent once init matches; the entire
17.7 pp "custom beats peft" advantage was the inflated update.**

All custom-path variants inherit this — verified: `id_kron` (`w_a` as
`(in_sub, r2)`) and `krotucker` (`tucker_a` as `(in_f, rank)`) use the identical
transposed pattern.

#### ✅ What survives

- Comparisons **within** the custom family (`id_kron_r16` vs `custom_standard`)
  are fair — same inflated init. But 1.15 pp with 17% more params is inside
  seed noise, which has never been measured here.
- **DoRA underperforms LoRA on this task** (33.92% vs 40.47%) at 77% more wall
  time (327 s vs 185 s) — contradicts this document's "DoRA ~90% quality" premise.
- **LoKr's collapse is real, not a bug.** Investigated directly: peft's LoKr has
  no 4-bit dispatch and crashed on the quantized base (`RuntimeError: shape
  '[11796480, 1]' is invalid for input of size 23592960`; 11796480 = 9216*2560/2,
  the packed-nibble byte count). That was patched before the recorded runs, and
  the saved adapter is fully trained with a delta **1.8x larger** than LoRA's
  (0.738 vs 0.410 mean Frobenius) applied at the correct orientation. So the low
  score is a structural-expressiveness limit, **not** the "3-factor gradient
  vanishing" claimed — gradients that vanish do not produce a larger delta than
  the baseline. (Patch now vendored: `src/gnn_experiment/peft_compat.py`.)

#### The real finding

An ~18x larger effective update is worth **+20.6 pp** adherence at 150 steps /
lr 2e-4. That means the peft-default configuration is badly under-powered for
this task and step budget, and **update magnitude (α/r or LR) — not
factorization — is the dominant lever measured so far.** Until the baseline is
magnitude-tuned, any architecture comparison mostly measures which arm
accidentally had the larger effective learning rate.

### Velocity-Masked SFT — CLAIM FALSIFIED 🔴

Controlled three-arm comparison, identical everything except layer selection.

> ⚠️ Adherence in THIS table was measured with the old **sampled** decoding and
> is therefore NOT comparable to the greedy numbers in the table above (where
> `custom_standard` = 58.17%, not 63.82%). The three arms here are still
> comparable *to each other* — all sampled, all same config. All three also
> share the inflated-init bug described above.

| Arm | Train loss | Wall s | Quiet % | Set churn % | Adherence (sampled) |
| --- | --- | --- | --- | --- | --- |
| `custom_standard` (no masking) | 1.1396 | 182.6 | 0 | — | **63.82%** |
| `velocity` (bottom-25% by Δh_l) | 1.1135 | 198.1 | 25.0 | 0.0 | 61.73% |
| `random_mask` (random 25%, control) | 1.1040 | 180.7 | 25.0 | 76.2 | 60.85% |

1. **The velocity ranking carries no signal.** Random selection lands within 0.9 pp
   of velocity — i.e. choosing layers by measured Δh_l is not distinguishable from
   choosing them by coin flip.
2. **It was never stochastic-depth regularization.** Churn is 0.013: the *same* 8
   layers (`[12,13,14,16,17,24,25,29]`) are masked on every single step. It is a
   static capacity cut to 24/32 layers, not a dynamic regularizer.
3. **No speedup is physically available.** The adapter is 0.437% of parameters;
   masking 25% of its layers skips ~0.1% of compute, while the frozen base still
   runs full forward+backward through every layer. Measured: velocity was *slower*
   (198.1 s vs 182.6 s) because the measurement hooks cost more than the skipped
   math saves. It therefore cannot "recover DoRA's ~20% overhead".
4. **The original 83.70% never demonstrated the technique.** It was produced with
   `skip_recompute=True` under non-reentrant gradient checkpointing, where the
   hook's `torch.is_grad_enabled()` test misfires and records nothing — the gate
   was inert for that entire run.

### Eval harness — FIXED ⚠️

The benchmark previously used sampled decoding (`do_sample=True, temperature=0.2`).
Two *mechanically identical* configs scored 76.96% and 63.05% on the postgres set —
a 13.9 pp swing of pure decoding noise, larger than any real effect measured. Eval
is now greedy (`do_sample=False`) everywhere. **All adherence numbers recorded
before this change are not comparable to numbers after it.**

---

## ✅ Execution Plan (gated, cheapest-decisive-first)

The original Step 1 below opens with 2–3 weeks of C++/HIP kernel work. That is the
wrong entry point: `peft` 0.20.0 already ships **DoRA** (`LoraConfig(use_dora=True)`)
and **Kronecker** (`LoKrConfig`), so the core quality-vs-size question is answerable
in a handful of PyTorch training runs first. Each step below has an explicit gate —
if the gate fails, the steps after it are not worth doing.

| Step | Work | Cost | Gate to proceed |
| --- | --- | --- | --- |
| **A. Re-baseline (greedy)** | astral base + LoRA `custom_standard` | ~10 min | none — mandatory, every pre-greedy number is incomparable |
| **B. DoRA baseline** | `LoraConfig(use_dora=True)`, no new code | ~10 min | does DoRA actually beat plain LoRA here? (this doc claims ~90% vs ~80%) |
| **C. LoKr baseline** ⭐ | `LoKrConfig`, no new code; **measure real saved adapter size** | ~10 min | **decisive**: does Kronecker hold adherence at ~3 MB, or collapse the way Tucker did? |
| **D. KronA + DoRA** | `LoKrConfig` has no `use_dora` field — add a magnitude vector to LoKr in PyTorch | ~1 day | only if C holds quality but loses some to plain LoRA/DoRA |
| **E. Fused HIP kernel** | `krona_dora_fused.hip` (Step 1 below) | 2–3 weeks | only if C/D prove quality at ~3 MB. Reframed honestly as a **speed/VRAM optimization**, not a capability unlock — C/D already deliver the small adapters |
| **F. Billboard Impostor** | Step 3 below; **blocked** on a hot-swap primitive rewrite (`load_novel_adapter` re-wraps the module tree and corrupts on a second call — confirmed by a real crash) | — | only after E, and after the fleet of distinct domain adapters exists |

**A + C together cost ~20 minutes of GPU time and test the premise that the entire
2–3 week kernel effort rests on.** If LoKr's adherence collapses at 3 MB, the
micro-adapter thesis dies there for the price of two training runs.

---

## 🗺️ System Overview

The three-component architecture that resolves the multi-tenant LLM serving trilemma:

$$\underbrace{\cancel{\text{Velocity-Masked SFT}}}_{\text{FALSIFIED — see Measured Reality}} + \underbrace{\text{KronA + DoRA}}_{\text{\~{}3 MB High-Rank Payload}} + \underbrace{\text{Billboard Impostor Router}}_{\text{0ms In-Flight Handoff}}$$

```text
                            TRAINING PHASE
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. [STRUCK] Velocity-Masked SFT (Dynamic Gradient Regularizer)             │
│    FALSIFIED: ranking carries no signal (random control matches it), the   │
│    masked set is static (churn 0.013 -> not stochastic depth), and no      │
│    speedup is possible (adapter is 0.437% of params). See Measured Reality.│
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 2. Fused KronA + DoRA HIP Kernel (RDNA3 / gfx1100)                         │
│    - Combines Kronecker high-rank updates (A ⊗ B) + DoRA magnitude (m)     │
│    - Computes B · X · A^T in GPU registers (Zero VRAM allocation spikes)   │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
                        [~3 MB High-Rank Micro-Adapters]
                                      │
                               INFERENCE PHASE
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. Billboard Impostor Engine (Zero-Stall Routing)                          │
│    - 100 KB VRAM Centroid Proxy (0 ms Token 1 Launch)                      │
│    - ~10 ms PCIe Background Stream of ~3 MB Micro-Payload                  │
│    - In-Flight Slot Swap in Captured Execution Graph                       │
└────────────────────────────────────────────────────────────────────────────┘
```

### Why This Combination Is Novel

Parts of these techniques exist in isolation across research literature, but combining them into a unified, zero-stall execution engine on local/consumer hardware has never been done before:

| Component | What Exists in Literature / Industry | What THIS System Achieves | Novel? |
| --- | --- | --- | --- |
| **Adapter Compression** | Standard DoRA (heavy ~30 MB files) OR Tucker (ultra-tiny ~1.5 MB, but **collapsed adherence** due to global basis sharing). | **KronA + DoRA (~3 MB payload)**: Decouples magnitude ($m$) from high-rank Kronecker direction ($A \otimes B$), maintaining **full fine-tuning adherence** while cutting file size by 80%+. | 🟢 **Yes.** Solves Tucker's mathematical failure mode at micro-adapter scales. |
| ~~**Training Regularization**~~ | Standard drop-out or layer-dropping. | ~~**Velocity-Masked SFT**: Bypasses backward passes on quiet layers ($\Delta h_l$)~~ | 🔴 **FALSIFIED.** A random-selection control matches it within 0.9 pp; masked set is static (churn 0.013); no speedup possible. The 83.70% came from a run where the gate was inert. |
| **GPU Execution Kernel** | Eager PyTorch (unrolls Kronecker matrices into VRAM, causing memory spikes). | **Fused C++/HIP Kernel (`gfx1100`)**: Computes $B \cdot X \cdot A^T$ directly in GPU registers/LDS with zero intermediate VRAM allocations. | 🟢 **Yes.** High-rank register-level Kronecker execution on RDNA3. |
| **Adapter Hot-Swapping** | vLLM / SGLang (PCIe transfer stalls or heavy VRAM caching). | **Billboard Impostor Engine**: Generates Token 1 instantly via a 100 KB VRAM proxy (0 ms launch) while streaming the ~3 MB payload in ~10 ms in the background for an in-flight slot swap. | 🟢 **Yes.** Completely masks the physical PCIe transfer window. |

### The Serving Trilemma This Solves

Until now, the entire AI serving industry operated under a strict **trilemma** — forced to choose two and suffer the third:

```text
              THE SERVING TRILEMMA

              [1. High Quality]
               (Full-Rank Rules)
                    /\
                   /  \
                  /    \
                 /  ??  \
                /        \
[2. Tiny Payload] ──────── [3. Zero Hot-Swap Latency]
 (2-4 MB / Edge)            (Instant Token 1 Launch)
```

**Why existing solutions failed:**

1. **Option A: High Quality + Zero Latency (vLLM / SGLang / S-LoRA):** Keep standard ~30 MB adapters permanently resident in GPU VRAM. Failure: requires **gigabytes of VRAM caching**, starving the KV cache on consumer hardware.
2. **Option B: Tiny Payload + Zero Latency:** Compress adapters to ~1.5 MB using global factor sharing. Failure: **catastrophic adherence collapse** — global shared basis destroys layer-specific subspace rotations.
3. **Option C: High Quality + Tiny Payload:** Offload small adapters to host RAM/disk, stream over PCIe on demand. Failure: **execution stalls** — every domain switch pays a PCIe streaming and graph allocation penalty before Token 1 can generate.

**The new reality:**

```text
             THE NEW REALITY (THIS SYSTEM)

               [1. High Fine-Tuning Quality]
                 (DoRA Magnitude + High-Rank KronA)
                             │
                             │  Achieved Simultaneously!
                             ▼
  [2. ~3 MB Micro-Payload] ───── [3. 0ms Initial Launch Delay]
    (Stores 50+ Experts in <150MB)  (Impostor Proxy + In-Flight Swap)
```

---

## 📊 Comparison: This System vs. Bleeding Edge

| Architectural Axis | Bleeding Edge (vLLM + DoRA / LoRA) | Unified Architecture |
| --- | --- | --- |
| **Adapter Artifact Size** | **~20 MB – 60 MB** per adapter | 🟢 **~2 MB – 4 MB** (80%+ reduction via A⊗B) |
| **Training Speed** | Baseline (DoRA adds ~20% norm overhead) | ⚪ **Unproven** — ~~Velocity-Masking~~ cannot recover this (falsified; adapter is 0.437% of compute). DoRA's real overhead on this rig is still unmeasured. |
| **Training VRAM Allocation** | High (PyTorch unrolls temporary matrices) | 🟢 **Zero Allocation Spikes** (Fused HIP kernel in registers) |
| **Initial Token Launch Delay** | **~100 ms – 200 ms** (PCIe streaming stall) | 🟢 **0 ms** (Instant execution via VRAM Centroid Impostor) |
| **PCIe Transfer Latency** | ~100 ms – 150 ms | 🟢 **~10 ms – 15 ms** (Micro-payload streams invisibly) |
| **Multi-Tenant VRAM Density** | ~1 GB – 2.5 GB for 50 domain adapters | 🟢 **<150 MB** for 50 domain adapters |
| **Downstream Adherence / Quality** | High (~83%–90%+) | 🟢 **High** (Matches full fine-tuning via DoRA magnitude m) |

### Why 200 ms Actually Matters (Not Just a Micro-Optimization)

In production agentic workflows, a ~200 ms stall creates a massive **systemic wall**:

- **Compound Stutter in Agent Loops:** Modern AI pipelines run multi-step agent loops (_Router → SQL Generator → Validator → YAML Formatter → Python Executor_). Every domain switch in a standard framework pays the PCIe penalty. Across a 10-step agent execution: **2 to 3 seconds of raw idle GPU time**. This system eliminates it entirely.
- **VRAM Multi-Tenancy Wall:** To avoid PCIe stalls, frameworks like vLLM/SGLang keep 30–60 MB adapters permanently in VRAM. Holding 50–100 adapters consumes gigabytes, directly starving the **KV Cache** and crushing batch throughput. This system stores 50 experts in **<150 MB**.

The final breakthrough in 3 sentences:

> 1. **Compression without adherence collapse:** By using **DoRA magnitude vectors (m)** alongside **Kronecker directional products (A⊗B)**, full fine-tuning quality is maintained on strict domain rules at a **~3 MB file size**.
> 2. ~~**Training overhead eliminated:** Velocity-Masking (Δh_l) skips backward passes on quiet layers — a regularizer that boosts adherence to 83.70% while cutting DoRA's training time penalty.~~ **🔴 FALSIFIED — see Measured Reality.** The claim requires masking adapter layers to save meaningful compute; the adapter is 0.437% of parameters, so it cannot. Measured: masking made training *slower*.
> 3. **Physical latency masked:** Token 1 executes instantly via a **100 KB VRAM Impostor Proxy** while the ~3 MB micro-payload streams over PCIe (~10 ms) and swaps directly into a pre-allocated **fused HIP execution slot (gfx1100)** — making hot-swapping completely invisible.

---

---

## Step 1: Build the `krona_dora_fused.hip` Kernel

**Goal:** A fused C++/HIP kernel for RDNA3 (`gfx1100`) that computes the KronA + DoRA forward and backward passes in GPU registers/LDS without any intermediate VRAM allocation.

### 1.1 — Why `w4a16.hip` Cannot Be Reused

The existing `w4a16.hip` kernel is a **W4A16 GEMV (Matrix-Vector) Decode Kernel**:
- **Input:** Packed 4-bit weights + FP16 activation vector (B=1).
- **Operation:** Dequantize 4-bit nibbles in GPU registers → fused dot product → output vector.
- **Direction:** Forward-pass single-token decode **only**.

KronA + DoRA Training requires three entirely different GPU operations:

1. **The Kronecker Pass (A⊗B):** Computing Y = (A⊗B)X, which reshapes inputs into higher-dimensional tensors.
2. **The DoRA Magnitude Norm (||W||_c):** Calculating L2 norms across matrix columns dynamically during every forward pass.
3. **The Backward Pass (∂L/∂A, ∂L/∂B, ∂L/∂m):** Computing gradients for A, B, and m through autograd hooks.

> `w4a16.hip` solved a **VRAM-read bandwidth bottleneck for 4-bit inference**. KronA + DoRA training needs a **fused matrix-transformation kernel for 16-bit/32-bit gradients**. They are two completely different HIP C++ codebases.

### 1.2 — The Core Mathematical Identity

Instead of PyTorch expanding (A⊗B) into a temporary high-rank matrix in VRAM, the fused HIP kernel exploits the Kronecker product identity:

$$(A \otimes B) \vec{x} \equiv \text{vec}\left( B \cdot \text{mat}(\vec{x}) \cdot A^T \right)$$

This means the full Kronecker expansion **never materializes in VRAM**. Instead, the kernel:

1. **Reads** A ∈ ℝ^(p×q), B ∈ ℝ^(r×s), X ∈ ℝ^(qs) from VRAM into thread-block registers / LDS.
2. **Reshapes** X into a (s × q) matrix in shared memory.
3. **Computes** B · X_mat · A^T in-place across wave fronts — output shape (r × p) — directly in registers.
4. **Writes** the flattened result Y ∈ ℝ^(rp) to VRAM.

### 1.3 — What the Fused Kernel Does (3 Operations, 1 Pass)

1. **Zero Tensor Allocation:** Computes B·X·A^T in small GPU shared memory (LDS) or registers without expanding A⊗B into VRAM.
2. **Fused DoRA Normalization:** Computes column-wise norms and directional updates in the same pass as the GEMM, bypassing PyTorch's intermediate tensor allocations.
3. **Fused Backward Pass:** Evaluates partial derivatives for A, B, and m directly in GPU registers.

### 1.4 — KronA + DoRA Architecture (Why Not DoRA Alone)

NVIDIA's **DoRA** (Weight-Decomposed Low-Rank Adaptation) splits a weight update into scalar magnitude (m) and directional matrix (V), parameterized by standard LoRA:

$$W = m \cdot \frac{W_0 + A \cdot B}{\Vert{}W_0 + A \cdot B\Vert{}}$$

- **What DoRA gives you:** Higher accuracy/adherence. Separating direction from magnitude behaves closer to full fine-tuning.
- **What DoRA does NOT give you:** Smaller file sizes. DoRA still uses standard low-rank A·B for direction. Parameter count is virtually identical to standard LoRA (**~20 MB to 60 MB** per adapter on 8B models).

> DoRA alone gives the _quality_, but leaves the exact same heavy **~25 MB payload** per adapter. Streaming across PCIe in 10 ms is impossible, and storing 50 DoRA adapters still eats gigabytes of VRAM.

**The hybrid construction — KronA + DoRA (m + A⊗B):**

- **KronA alone (A⊗B):** Compresses parameter footprints by 80%+ (~3 MB per adapter), but can suffer from optimization/scaling instability during SFT.
- **DoRA alone (m + A·B):** Excellent accuracy, heavy parameter footprints (~25 MB).
- **KronA + DoRA (m + A⊗B):** Kronecker products for the directional update (A⊗B) + DoRA's magnitude vector (m) for scaling. **Full fine-tuning adherence at a ~3 MB payload size.**

The merge formula for production deployment:

$$W_{\text{final}} = m \cdot \frac{W_0 + (A \otimes B)}{\Vert{}W_0 + (A \otimes B)\Vert{}_F}$$

**Structural comparison:**

| Architectural Dimension | Standard LoRA | Tucker (DO NOT USE) | KronA + DoRA Hybrid |
| --- | --- | --- | --- |
| **Tool Adherence & Quality** | High (~83%) | 🔴 **Collapsed** (Shared Basis Bottleneck) | 🟢 **Matches Full Fine-Tuning** |
| **Expressive Matrix Rank** | Low Rank (rank r) | Restricted Low Rank | 🟢 **High / Full Rank** (A⊗B) |
| **Adapter Checkpoint Size** | ~21 MB | ~1.5 MB | 🟢 **~2 MB – 5 MB** |
| **Layer Independence** | 100% Independent | 🔴 Forced Global Shared Basis | 🟢 **100% Layer-Independent** |
| **Inference Overhead** | 0 ms (Merged) | 0 ms (Merged) | 🟢 **0 ms (Merged)** |

Tucker Factorization is **off the table and will not be mentioned again**. All benchmarks and architectural comparisons are strictly evaluated against: **Standard LoRA (A·B)**, **DoRA (NVIDIA)**, and **vLLM/SGLang multi-adapter serving frameworks**.

### 1.5 — ~~Velocity-Masked SFT as the Training Regularizer~~ 🔴 FALSIFIED

> **This entire subsection is retained only as a record of a disproven hypothesis.**
> Do not build on it. The measurements that killed it are in `## Measured Reality`
> at the top of this document. Summary of why it fails, in the order the failures
> were found:
>
> 1. **The gate was inert when the headline number was produced.** The 59.95% →
>    83.70% result came from runs with `skip_recompute=True` under non-reentrant
>    gradient checkpointing, where the hook's `torch.is_grad_enabled()` recompute
>    test is always true and therefore skips every measurement. 0% of layers were
>    ever masked. That run demonstrated nothing about velocity masking.
> 2. **It is not stochastic depth.** With the hook fixed, quiet-set churn is
>    **0.013** — the identical 8 layers (`[12,13,14,16,17,24,25,29]`) are masked
>    every single step. It is a static capacity cut to 24/32 layers wearing a
>    dynamic-regularizer costume.
> 3. **The ranking carries no signal.** A random-selection control arm (same mask
>    count, redrawn each step, churn 76.2%) scored 60.85% vs velocity's 61.73% —
>    within noise of each other and of the unmasked baseline's 63.82%.
> 4. **The compute-saving claim is arithmetically impossible.** The adapter is
>    0.437% of model parameters. Masking 25% of its layers skips ~0.1% of compute
>    while the frozen base still runs full forward+backward through all 32 layers.
>    Measured: velocity training was *slower* (198.1 s vs 182.6 s) because the
>    measurement hooks cost more than the skipped math saves. It cannot offset
>    DoRA's overhead, which was the entire reason it appeared in this plan.
>
> **What survives and is worth keeping** (all in `src/gnn_experiment/novel_peft.py`):
> the self-calibrating percentile gate (can't silently go inert the way a fixed
> threshold did), the warmup-complete-with-zero-observations warning, and the
> quiet-set churn metric that exposes static-vs-stochastic masking. These are
> useful diagnostics regardless of this technique being dead.

<details>
<summary>Original (disproven) text, kept for the reasoning trail</summary>

During training, **KronA + DoRA** defines _how_ parameters are factorized, while **Velocity-Masking** controls _when and where_ gradients are applied across model depth.

- **Regularization & Adherence Boost:** Skipping backward updates on low-velocity layers acts as a dynamic stochastic-depth regularizer (claimed: adherence boosted from ~59.95% up to 83.70%). That same regularization applies directly to the KronA directional matrices (A⊗B) and DoRA magnitude vectors (m).
- **Mitigating DoRA's Training Overhead:** DoRA adds ~20% wall-clock penalty due to dynamic norm calculations. Velocity-Masking completely bypasses the backward pass on quiet layers, eliminating gradient and norm calculations for those layers — clawing back that training time loss directly.

> **"Velocity-Masked SFT acts as the training regularizer that preserves adherence and cuts backward pass compute, while the fused KronA + DoRA HIP kernel compresses those updates into ~3 MB micro-adapters designed for zero-stall Billboard Impostor hot-swapping."**

</details>

The corrected training stack (velocity removed):

```text
       ┌─────────────────────────────────────────────────────────┐
       │                KRONA + DORA (A ⊗ B + m)                 │
       │   - High-rank Kronecker directional updates (A ⊗ B)     │
       │   - Decoupled DoRA magnitude scaling vector (m)         │
       │   - Step C/D below: PyTorch first, HIP kernel only if   │
       │     the quality-at-3MB result justifies it              │
       └────────────────────────┬────────────────────────────────┘
                                │
                                ▼
                    [~3 MB Micro-Adapters]
```

| Component | Layer in the Stack | Concrete Function |
| --- | --- | --- |
| ~~**Velocity-Masked SFT**~~ | 🔴 **REMOVED** | Falsified — see above. |
| **KronA + DoRA** | **Model Factorization & Execution** | Encodes high-rank, layer-independent updates into ultra-compact **~3 MB micro-adapters**. |
| **Billboard Impostor** | **Inference & Routing Engine** | Serves domain experts with **0 ms initial launch delay** and sub-15 ms PCIe hot-swapping. |

### 1.6 — Engineering vs. PyTorch Wrapper Trade-Off

| Metric | Eager PyTorch (KronA + DoRA) | **Custom Fused HIP Kernel (KronA + DoRA)** |
| --- | --- | --- |
| **Adapter File Size** | �� ~3 MB – 5 MB | 🟢 ~3 MB – 5 MB |
| **Adherence / Quality** | 🟢 Matches Full Fine-Tuning | 🟢 Matches Full Fine-Tuning |
| **Training Speed** | 🔴 +25% to 40% slower than LoRA | 🟢 **Matches standard LoRA speed (~3.5 min)** |
| **Training VRAM Spikes** | 🔴 High (Unrolled intermediate tensors) | �� **Zero extra VRAM (Computes in registers)** |
| **Engineering Effort** | 🟢 1–2 days (PyTorch wrapper) | 🔴 **2–3 weeks of low-level C++/HIP/ROCm dev** |

**Build the Custom Fused HIP Kernel IF:**
1. You plan to train dozens of adapters continuously — shaving ~30% off training time and eliminating VRAM spikes saves real GPU compute hours.
2. You want the systems engineering portfolio asset — a fused C++/HIP/ROCm kernel for RDNA3 handling Kronecker matrix algebra (A⊗B) and DoRA magnitude updates in registers is a top-tier systems accomplishment.

**Stick to a Clean PyTorch Wrapper IF:**
1. You only need 2–3 domain experts for testing — a standard PyTorch/PEFT wrapper gets the job done immediately without weeks of C++/HIP kernel development.

---

## Step 2: Dataset Pipeline & Domain Asset Generation

**Goal:** Generalize the training dataset pipeline to support any domain corpus and produce 2–3 distinct high-adherence domain adapters for testing.

### 2.1 — Dataset Pipeline (Already In Progress)

The dataset pipeline (`load_micro_dataset()` in `src/gnn_experiment/micro_probe/dataset.py`) has been updated:

- Use `tokenizer.apply_chat_template()` with `SFTTrainer(assistant_only_loss=True)` so response-only loss is active — LoRA gradients focus 100% on assistant output correctness.
- Persona sanitizer strips informal greeting/sign-off filler from assistant content before training (`sanitize_assistant_content()`).
- Fingerprint bumped to `astral_micro_docs_v2` to bust stale HF cache.
- Negative/rejection examples added (~15% of dataset) — teaches the adapter when NOT to use tools in the described way, preventing hallucination of invalid patterns.
- Default paths updated: `data/astral/raw` + `data/astral/training_data.jsonl`.

### 2.2 — Domain Asset Generation

Build 2–3 synthetic domain datasets to produce distinct, high-adherence test artifacts:

1. **SQL domain** — Strict SQL AST generation, schema validation, rejection of bad queries.
2. **YAML domain** — Structured YAML formatting with schema enforcement.
3. *(Optional)* **Rust/Python AST domain** — Code generation with syntax rules.

Each domain dataset should follow the same SFT format:
- Positive expert Q&A pairs (happy-path tool use)
- Negative/rejection pairs (~15% of total): "Should I do X?" → "No. Do Y instead because..."
- Consistent direct technical tone (no conversational filler)

---

## Step 3: Billboard Impostor Engine

**Goal:** Build the zero-stall multi-adapter routing system that serves domain experts with 0 ms first-token delay and sub-15 ms hot-swap latency.

> **Prerequisite:** Step 1 must be complete (need ~3 MB adapters). The Impostor system only makes physical sense with ultra-compact payloads. Standard 25 MB LoRA adapters would still stall.

### 3.1 — How On-the-Fly Switching Works

```text
Standard SOTA Engine (vLLM / SGLang):
[Request New Adapter] ──► [100-200ms PCIe Transfer Stall] ──► [GPU VRAM Allocation Spike] ──► [Token 1 Delayed]

Unified Architecture:
[Request New Adapter] ──► [Token 1 Generates Instantly via Impostor Proxy (0ms)]
                               │ (~10ms Background PCIe Stream of ~3MB Payload)
                               ▼
                          [In-Flight Slot Swap via Fused HIP Kernel] ──► [Token 2+ Full Quality]
```

### 3.2 — The Three Phases

**Phase 1 — Zero-Delay First Token (The Impostor Proxy)**

When an agent or user switches domains (e.g., jumping mid-chat from general conversation to a strict **SQL AST** or **Rust Clippy** task):
- The engine doesn't pause to wait for adapter weights to load.
- Token 1 generates immediately using the resident **100 KB Centroid Impostor Proxy**, incurring **0 ms of cold-start launch delay**.

**Phase 2 — Sub-15 ms In-Flight PCIe Background Streaming**

KronA + DoRA uses Kronecker factor expansion (A⊗B) with DoRA magnitude vectors (m) while preserving layer independence. The complete payload drops from **~25–30 MB down to ~2–4 MB**:
- Transferring a ~3 MB payload across the PCIe bus takes only **~10 ms to 15 ms** (vs. ~100–150 ms for standard LoRA).
- The true high-rank domain adapter streams into GPU VRAM in the background while the first token is being generated by the proxy.

**Phase 3 — Register-Level In-Flight Slot Swapping (No GPU Stalls)**

Once the ~3 MB payload lands in VRAM, the custom fused C++/HIP kernel executes an in-flight swap directly into a pre-allocated static slot inside the captured execution graph:
- **Zero Tensor Re-allocations:** The kernel handles Y = (A⊗B)X using register/LDS reshapes (B·X·A^T), avoiding PyTorch memory allocation spikes or VRAM fragmentation.
- **Seamless Transition:** The active model cleanly switches from Impostor Proxy to the true, full-rank domain adapter by Token 2 or 3 without dropping a single clock cycle.

### 3.3 — Streaming Latency vs. PCIe Bottleneck

The Impostor Routing algorithm relies on an initial low-latency phase (using lightweight centroid vectors) while streaming full A·B weights into the active slot in the background:

- **Standard LoRA (~25 MB):** Streaming 25 MB per adapter over PCIe consumes ~100–150 ms. If routing swaps adapters frequently across agent turns, this creates visible generation stutters.
- **KronA + DoRA Micro-Adapters (~2 MB – 4 MB):** Kronecker factorization drops the parameter payload by 80%+ while preserving high-rank expressiveness. Streaming a ~3 MB adapter takes **~10–15 ms**. The "background streaming" window becomes practically invisible.

### 3.4 — In-Memory Density for Fixed VRAM Slots

If pre-allocating a fixed-slot tensor engine in GPU VRAM (reserving VRAM for dynamic adapter slots):

- **Standard LoRA:** Storing 20 domain adapters resident in GPU VRAM: **~500 MB to 600 MB** overhead.
- **KronA + DoRA:** Storing 20 domain adapters: **under 60 MB total**. An entire library of domain experts stays permanently resident without fragmenting memory or starving the KV cache.

### 3.5 — Hot-Swap Metrics

| Metric / Experience | Standard Multi-LoRA Serving (vLLM / SGLang) | **Unified System** |
| --- | --- | --- |
| **First Token Launch (Un-cached)** | 🔴 **100 ms – 200 ms stall** (PCIe bottleneck) | �� **0 ms** (Instant launch via VRAM Centroid Impostor) |
| **Adapter Transfer Payload** | 🔴 **~20 MB – 60 MB** per adapter | 🟢 **~2 MB – 4 MB** (80%+ reduction via A⊗B) |
| **Adapter Hot-Swap Latency** | 🔴 ~100 ms – 150 ms transfer window | 🟢 **~10 ms – 15 ms** (Transfers invisibly mid-token) |
| **GPU Execution Allocation** | 🟡 High (Unrolls matrices in VRAM during forward pass) | 🟢 **Zero Allocation Spikes** (Fused register computation) |
| **VRAM Density (50 Adapters)** | 🔴 ~1 GB – 2.5 GB reserved VRAM | 🟢 **<150 MB** reserved VRAM |

---

## Step 4: Integration & Benchmarking

**Goal:** Wire all three components together, benchmark against standard LoRA and DoRA baselines (vLLM / SGLang), and validate the complete pipeline.

### 4.1 — Complete System Pipeline

```text
[Dataset / Prompts]
       │
       ▼ (Fast ~3.5 min SFT)
[KronA + DoRA HIP Kernel] ──► Produces ~3 MB Micro-Adapters
       │
       ▼
[Billboard Impostor Router] ──► Instant Centroid Proxy Execution
       │                       └──► ~10ms PCIe Background Stream
       ▼
[Fixed VRAM Tensor Slots] ──► Zero-Latency Domain-Expert Execution
```

### 4.2 — Benchmark Axes (vs. Standard LoRA and DoRA Baselines)

| Architectural Axis | Baseline (Standard LoRA) | Baseline (DoRA) | Target (This System) |
| --- | --- | --- | --- |
| **Adapter File Size** | ~21 MB – 30 MB | ~21 MB – 30 MB | **~2 MB – 4 MB** |
| **Adherence / Quality** | Moderate (~80%) | 🟢 High (~90%+) | 🟢 **High (~90%+)** |
| **Training Speed** | Baseline | Baseline –20% (unverified on this rig) | ⚪ **Unproven** — ~~Velocity-Masking recovers DoRA overhead~~ falsified; no mechanism currently identified to offset DoRA's cost |
| **Training VRAM Spikes** | Moderate | Moderate | 🟢 **Zero** (Fused HIP kernel in registers) |
| **Initial Token Launch (Un-cached)** | ~100–200 ms stall | ~100–200 ms stall | 🟢 **0 ms** (VRAM Centroid Impostor) |
| **PCIe Transfer Latency** | ~100–150 ms | ~100–150 ms | 🟢 **~10–15 ms** (Micro-payload) |
| **Multi-Tenant VRAM (50 Adapters)** | ~1–2.5 GB | ~1–2.5 GB | 🟢 **<150 MB** |

### 4.3 — What Was Previously Impossible

This system achieves simultaneously:

> **"You can run a fleet of 50+ specialized, high-adherence LLM domain experts on a single consumer GPU without paying for enterprise VRAM caching, without risking accuracy collapse, and without experiencing a single millisecond of hot-swapping latency."**

That is a complete, end-to-end systems engineering innovation spanning training regularizers, custom low-level C++/HIP kernels, and dynamic inference routing. You aren't just tweaking an existing tool; you are assembling a novel, end-to-end systems paradigm for local multi-tenant LLM execution.
