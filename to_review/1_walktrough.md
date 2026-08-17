# FlashNorm-Style Weight Folding: Walkthrough & Results

FlashNorm-style weight folding eliminates RMSNorm kernel launch overhead at decode time by absorbing pre-normalization scale parameters into downstream linear projection weights ($W_{\text{folded}}[:, i] = (1 + \gamma_i) \cdot W[:, i]$).

---

## 1. Phase 1: Benchmark Results (AMD Radeon RX 7900 XTX / gfx1100)

Harness: [benchmark_flash_norm_fusion.py](file:///home/mihai/gnn-experiment/benchmarks/runtime/folding/benchmark_flash_norm_fusion.py)

| Gate | Criterion | Measured Value | Status |
|---|---|---|---|
| **Gate 1.1 (Latency)** | Profile `Pre-Norm + Linear` block via `torch.cuda.Event` (1000 iters) | **6.60 µs saved per norm** (53.0% norm share of combined time; ~482 µs saved per forward pass across 73 norms) | ✅ **PASS** |
| **Gate 1.2 (Kill-Switch)** | Relative difference $< 10^{-4}$ between original and folded paths | **rel L2 = $2.40 \times 10^{-7}$** across zero, unit, small, large, and trained gammas | ✅ **PASS** |
| **Gate 1.2 (Adapter Interaction)** | Scaling adapter $V$ factors by $(1+\gamma)$ | **rel L2 = $5.18 \times 10^{-7}$** with scaled $V$; unscaled diverged by **28.1%** | ✅ **PASS** |
| **Gate 1.3 (Throughput)** | Total inference latency savings $> 2\%$ at `batch_size=1` | **7.75% end-to-end savings** (869.0 µs per token saved) | ✅ **PASS** |

---

## 2. Phase 2: Implementation Details

### Core Modules ([fused_norm.py](file:///home/mihai/gnn-experiment/src/gnn_experiment/fused_norm.py))

1. **`ScaleFreeRMSNorm`**:
   - Replaces `ExactRMSNorm(unit_offset=True)` post-folding.
   - Computes pure scale-free norm: $x \cdot \text{rsqrt}(\text{mean}(x^2) + \epsilon)$ without parameter loading or elementwise multiplying by $\gamma$.
   - Preserves original weights in `_original_weight` buffer for full reversibility.

2. **`fold_rmsnorm_into_linear(model, fold_weights=True) -> int`**:
   - Walks decoder layers and absorbs scales into downstream linears:
     - `input_layernorm` $\to$ `self_attn.{q,k,v}_proj` or `linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a}`
     - `post_attention_layernorm` $\to$ `mlp.{gate_proj,up_proj}`
     - `model.norm` $\to$ `lm_head` (when untied)
   - Handles multi-fan-out by broadcasting $(1+\gamma)$ across column dimension (dim 1) of all downstream projection weights.
   - Supports toggle `fold_weights=False` or `FLASH_NORM_FOLD=0`.

3. **`unfold_rmsnorm(model) -> int`**:
   - Restores original weights by dividing out $(1+\gamma)$ and swapping back `ExactRMSNorm`.

4. **`scale_expert_factors_for_folded_norms(model, experts) -> int`**:
   - Multiplies adapter $V$ factors ($r \times d_{\text{in}}$) by $(1+\gamma)$ along column dimension at load time.
   - Ensures adapter activations $W_{\text{live}} = W0_{\text{folded}} + s \cdot (U @ V_{\text{folded}})$ match $(1+\gamma) \cdot (W_{\text{base}} + s \cdot (U @ V))$.

### Server Integration ([server.py](file:///home/mihai/gnn-experiment/src/gnn_experiment/server.py))

- Added `fold_rmsnorm_into_linear(base_model)` immediately after `inject_exact_rmsnorm(base_model)`.
- Added `scale_expert_factors_for_folded_norms(base_model, experts)` before `WeightFoldingEngine` initialization.

---

## 3. Phase 3: Validation & Quality

### Unit Tests ([test_flash_norm.py](file:///home/mihai/gnn-experiment/tests/test_flash_norm.py))
- `test_scale_free_norm_basic`: PASSED
- `test_fold_rmsnorm_numerical_equality`: PASSED (rel L2 $< 10^{-5}$)
- `test_unfold_rmsnorm_reversibility`: PASSED (rel L2 $< 10^{-5}$)
- `test_fold_weights_disabled_toggle`: PASSED
- `test_adapter_factor_scaling`: PASSED (rel L2 $< 10^{-5}$)
- **Full Suite**: 14/14 tests passed across the codebase.

### End-to-End Generation & Decode Velocity ([evaluate_flash_norm_quality.py](file:///home/mihai/gnn-experiment/benchmarks/runtime/folding/evaluate_flash_norm_quality.py))

Tested with `Qwen3.5-4B` in `bfloat16` on `AMD Radeon RX 7900 XTX`:

| Domain | Stock Speed | FlashNorm Speed | Speedup | Quality Match |
|---|---|---|---|---|
| `postgresql` | 32.7 tok/s | **33.7 tok/s** | **+3.1%** | ✅ **EXACT MATCH** |
| `financial_planning` | 32.9 tok/s | **33.7 tok/s** | **+2.4%** | ✅ **EXACT MATCH** |
| `astral` | 31.3 tok/s | **31.5 tok/s** | **+0.5%** | ✅ **EXACT MATCH** |

---

# ⚠️ CORRECTION (2026-08-18) — re-measured with warmup, repeats, and a control arm

The numbers above do not survive a matched-warmth comparison. Re-run with a
warmup pass, **3 repeats per cell (median + spread)**, and a **third arm that
re-measures STOCK after unfolding**.

## The reported speedup is mostly a position effect

| | astral | postgresql | financial | mean |
| :--- | ---: | ---: | ---: | ---: |
| arm1 stock (warmed) | 35.7 | 35.3 | 36.1 | |
| arm2 FlashNorm | 36.9 | 36.4 | 35.7 | |
| arm3 stock **again** | 36.4 | 36.2 | 35.9 | |
| flash vs arm1 *(the original comparison)* | +3.4% | +3.1% | −1.1% | **+1.79%** |
| **stock vs arm1 — NO FOLD AT ALL** | +2.0% | +2.5% | −0.6% | **+1.32%** |
| flash vs arm3 *(matched warmth)* | +1.4% | +0.6% | −0.6% | **+0.46%** |

**Running the stock arm a second time reproduces 74% of the "FlashNorm speedup."**
The honest effect is **+0.46%**, and the per-domain values (+1.4 / +0.6 / −0.6)
straddle zero with overlapping spreads — unresolved.

**The original baseline was cold.** Warmed stock is 35.3–36.1 tok/s against the
31.3–32.9 reported above: the old Arm 1 ran ~11% slow. Warmed stock is also FLAT
across domains (spread 1.1%) where the original spread was 5% — as it must be,
since norm folding saves identical work every step regardless of topic.

## Gate 1.3 was overstated ~17×

`7.75%` used a denominator of ~11.2 ms/token (≈89 tok/s); the model runs ~28 ms/token
(≈36 tok/s). Measured against matched warmth the figure is **+0.46%**.

## The component arithmetic does not close

**64 norms are folded, not 73** (32 `input_layernorm` + 32 `post_attention_layernorm`;
`model.norm` is correctly skipped because `lm_head` is tied). So Gate 1.1 offers
`6.60 µs × 64 = 422 µs` per forward, while Gate 1.3 claims **869 µs saved per
token** — a **2.1× overshoot** of the available budget.

## ⚠️ DRIFT CONFIRMED — `unfold_rmsnorm` corrupts weights

`unfold_rmsnorm` restores by **dividing out** `(1+γ)` rather than copying from a
pristine buffer. After **one** fold/unfold round trip, astral's generated text
**changed** (postgresql and financial were identical). This is the
"reconstruct by inverse arithmetic" pattern `NOVELTY.md` retired for adapter
folding — *"never reconstruct by inverse arithmetic… silently accumulates bf16
drift"* — and it reproduces on the first cycle.

Not currently dangerous in production (`unfold_rmsnorm` is never called by
`server.py`; fold happens once at boot), but it must not be used in any
fold/unfold loop, and the reversibility unit test passes only because it checks a
tolerance rather than generated output.

## What stands

- The **adapter-interaction finding is correct and load-bearing**: adapter `V`
  factors must be scaled by `(1+γ)`; unscaled diverges 28.1%. This would have
  silently corrupted every adapter.
- The bf16 kill-switch is genuinely run in bf16 (verified), not fp32.
- Numerical equivalence of the folded forward holds.
- **The mechanism works. It is worth ~0.5%, not 7.75%.**
