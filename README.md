# ⚡ In-Place Folding OpenAI-Compatible Engine & REST Server

An OpenAI-compatible local REST server built on **in-place weight folding**: micro-expert adapters are merged into the base weights in VRAM so inference runs with no adapter wrappers in the execution path.

Exposes local micro-experts (`postgresql`, `astral`, `financial_planning`) through standard
`/v1/chat/completions` and `/v1/models` endpoints. A swap is an in-place mutation
$W_{\text{live}} \leftarrow W_0 + s \cdot U V$ measured at **18.8-19.3 ms** with **0 bytes**
transient VRAM churn (peak-above-baseline).

**Measured, on an RX 7900 XTX (gfx1100) / ROCm 7.2:**

| | |
| --- | --- |
| decode, adapter wrapped | 25.09 tok/s |
| decode, adapter folded | **30.38 tok/s** (+21.1%, = unadapted base speed) |
| expert swap (factors resident) | 18.8-19.3 ms; pays for itself after ~3 tokens |
| putting the swap behind an API boundary | +0.012 ms (a ~20 byte command) |
| CUDA graph replay on top of folding | **~1.01x** -- decode here is kernel-execution bound, not launch bound |
| AMD AITER | **not used**: its rmsnorm silently returns zeros at hidden=2560 on gfx1100, and the one correct kernel is 2-6x slower than ATen |


---

## 🚀 Key Performance Specs

All figures below are gated on a correctness check first: the folded model must
reproduce the wrapped adapter's tokens, and graph replay must reproduce eager
greedy decode token-for-token. Speed numbers taken without that gate passing are
not reported, because a graph that decodes against a stale mask is *faster*
precisely because it is doing the wrong thing.

| Optimization | Decode | Notes |
| :--- | :---: | :--- |
| adapter wrapped (`NovelLoraLinear`) | $25.09\text{ tok/s}$ | 256 extra kernel launches/token at batch 1 |
| **adapter folded into base weights** | **$30.38\text{ tok/s}$ ($+21.1\%$)** | reaches unadapted base speed; does not exceed it |
| folded + CUDA graph replay | $\approx 1.01\times$ over folded alone | decode here is kernel-execution bound, not launch bound |
| in-place expert swap | $18.8-19.3\text{ ms}$, $0$ bytes transient churn | break-even after ~3 generated tokens |

> **Retracted.** An earlier version of this table claimed $32.89\text{ tok/s}$
> ($1.21\times$) for "AITER + Folded CUDA Graph". That arm was confounded four
> ways: only it had an adapter folded, it used a different generation loop, its
> timing window excluded prefill, and the graph was decoding incorrectly so it
> never hit EOS and ran to the token cap. AITER was also never active -- see
> `src/gnn_experiment/fused_norm.py` for the measurements.

---

## 🔬 Measured Findings

All figures below come from runs on this machine (RX 7900 XTX / gfx1100, ROCm 7.2,
Qwen3.5-4B, bf16, greedy decode). Every result is gated on a correctness check
first — a folded model must reproduce the wrapped adapter's tokens, and graph
replay must reproduce eager decode token-for-token — because the failure mode
here is that a *wrong* implementation looks *fast*.

### 1. The bf16 merge-absorption law — replicated across 3 domains

Folding an adapter into bf16 base weights quantises the delta: `W0 + dW` is
rounded to 8 mantissa bits, so wherever `dW` is small relative to `W0` it is
partially or entirely absorbed. Swept over **three independent domains x five
alphas** (15 adapters; identical architecture `id_kron` r 8x8, 150 steps, batch
2, lr 2e-4 — alpha is the only variable):

| α | scaling | \|dW\|/\|W\| | merge err | absorbed |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.25 | ~0.023 | ~7.3% | ~8.0% |
| 32 | 0.50 | ~0.045 | ~3.7% | ~4.1% |
| 64 | 1.00 | ~0.090 | ~1.85% | ~2.0% |
| 128 | 2.00 | ~0.181 | ~0.93% | ~1.0% |
| 256 | 4.00 | ~0.358 | ~0.49% | ~0.51% |

The product `(|dW|/|W|) x merge_err` is constant to ~6% across a **15.7x range**
of the ratio, and **the same constant appears in all three domains**:

| domain | constant range | spread |
| --- | --- | ---: |
| financial_planning | 0.1664 – 0.1770 | 6.3% |
| astral | 0.1667 – 0.1769 | 6.0% |
| postgresql | 0.1664 – 0.1769 | 6.2% |

```
merge_rel_err ≈ 0.167 / (|dW|/|W|)
absorbed_frac ≈ 0.183 / (|dW|/|W|)
```

The `1/ratio` form follows from bf16 having fixed relative precision
(ULP ≈ |W|·2⁻⁸). The **constant is stable across domains** for adapters sharing
an architecture and training recipe. It is *not* universal: dense random deltas
give 0.234, and one adapter trained on degenerate data (940 copies of a single
templated prompt) gave 0.085. Calibrate per adapter family, not per domain.

### 2. The "don't fold low-scale adapters" rule — NOT SUPPORTED 🔴

The obvious corollary is a routing rule: fold high-scale adapters, keep
low-scale ones wrapped to avoid truncation. **Fifteen points across three
domains do not support it.**

| domain | corr(merge_err, folded−wrapped) | \|max Δ\| |
| --- | ---: | ---: |
| financial_planning | −0.113 | 2.50pp |
| astral | **+0.224** | 7.86pp |
| postgresql | **−0.909** | 5.00pp |
| **pooled (n=15)** | **−0.144**  (t=−0.53, df=13) | — |

The per-domain correlations run from **+0.224 to −0.909** — they do not even
agree on sign, which is what noise looks like at n=5. Pooled over all 15 points
the correlation is −0.144 with t=−0.53: nowhere near significant. Postgresql's
apparently strong −0.909 collapses to −0.376 when its single α=16 point is
removed, so it rests on one observation. And astral's *largest* penalty
(−7.86pp) occurs at its second-*lowest* merge error, directly against the rule.

Average cost of folding across all 15 adapters: **−0.231pp**.

**Verdict: fold by default.** The rule would trade a measured **+21% decode
speedup** for an effect that is smaller than this eval can resolve. Honest
bounds: folding *does* change outputs materially (deltas reach ±7.9pp), but
unpredictably and with a mean near zero — so it is not damage that merge error
lets you anticipate. At n=20 questions this rules out a large systematic
penalty; it does not prove the effect is zero.

### 3. Optimal α is domain-specific, and adapter quality is sharply non-monotonic

| domain | base | best α | best wrapped | gain over base |
| --- | ---: | ---: | ---: | ---: |
| astral | 12.20% | **64** | 60.20% | **+47.99pp** |
| postgresql | 49.67% | **64** | 74.67% | **+25.00pp** |
| financial_planning | 78.33% | **32** | 83.33% | **+5.00pp** |

All three domains have adapters that clearly beat base — **at the right α**.
The peak is *not* in the same place: α=64 for two domains, α=32 for the third,
so α=32 does not transfer (assuming it for astral would have cost ~13pp).

Training loss rose monotonically with α in **all three** domains
(astral 1.05→2.04, postgresql 1.34→2.12, financial 1.65→2.51) while quality
peaked in the middle. **Do not select adapters on loss** — it is anti-correlated
with adherence here.

This also invalidated an earlier conclusion in this repo. Sampling only α=16
and α=64 for financial — both in troughs — produced "this domain has too little
headroom for the adapter to demonstrate value." That was wrong. **Sweep α before
concluding an adapter does not work.**

### 4. Cross-task subspaces are orthogonal (with calibrated controls)

Whether adapters for different tasks share a low-rank subspace decides if
shared-basis compression (Tucker / LoKr / MasterBasis) can work at all.
**Read the ratio to chance, not the raw percentage**: projecting onto a
k-dimensional subspace of an (in·out)-dimensional matrix space captures
k/(in·out) by chance — ~0.0005% at k=32 — so a "retention < 1%" test can never
fail and carries no information.

| pair | retention | × chance | reading |
| --- | ---: | ---: | --- |
| any adapter vs itself | 12.5–19.6% | 26,569–41,447× | ceiling (control) |
| astral a256 vs astral a128 | 0.00338% | **7.15×** | same task → real shared structure |
| astral vs financial | 0.00061% | 1.28× | orthogonal |
| astral vs postgres | 0.00058% | 1.22× | orthogonal |
| financial vs postgres | 0.00052% | 1.10× | orthogonal |

**Every cross-task pair sits at chance.** Shared-basis compression cannot work
across these tasks — corroborating the 0.00% cross-task retention recorded
independently in `TODO.md`. The same-task pair at 7.15× shows the probe *can*
detect real structure, so the null is informative rather than a broken metric.

### 5. Eval-harness bug that inflated earlier base numbers

An un-fine-tuned model runs past its answer and fabricates a new
`### Question:` block, which then gets scored. Affected **18/20 astral, 16/20
postgresql and 13/20 financial** base answers; for financial that hallucinated
tail supplied **57%** of the base model's term hits. Fine-tuned adapters learned
to stop (0–4/20), so every adapter was being scored against an inflated base.
Fixed with `stop_strings` plus a regex backstop in `eval_suite.py`.

The size of the distortion is **not predictable from metric type** and must be
re-measured per domain. Measured before vs after the fix: postgresql base
57.57% → **49.67%** (−7.90pp, inflated), astral base 12.20% → **12.20%**
(unchanged), financial severely distorted (the hallucinated tail supplied 57%
of its hits). Astral's ratio happened to survive because its base is dominated
by *bad* hits and the tail carried a similar mix; postgresql's is dominated by
*good* hits and the tail skewed further that way. Treat any pre-fix adherence
figure as suspect until re-measured.

### Reproducing these

```bash
# alpha sweep (merge fidelity + wrapped vs folded) for any domain
uv run --env-file .env python scripts/benchmark_alpha_absorption_sweep.py --domain astral
uv run --env-file .env python scripts/benchmark_alpha_absorption_sweep.py --domain postgresql
uv run --env-file .env python scripts/benchmark_alpha_absorption_sweep.py --domain financial_planning

# cross-task subspace orthogonality map (CPU, seconds)
uv run scripts/build_orthogonality_map.py

# single pairwise subspace probe
uv run scripts/probe_subspace_overlap.py \
    --adapter-a results/adapters/astral_qwen3.5_micro_lora_a256 \
    --adapter-b results/adapters/financial_planning_standard_lora --k 32
```

---

## 💻 Quick Start: Launching the REST Server

To launch the production OpenAI-compatible REST server daemon on `http://127.0.0.1:8000`:

```bash
uv run --env-file .env python3 scripts/run_openai_api_server.py --host 127.0.0.1 --port 8000
```

The server initializes `Qwen/Qwen3.5-4B`, swaps in exact PyTorch RMSNorm stand-ins (CUDA-graph friendly; not a speedup), pre-loads the factor micro-experts, captures the CUDA/HIP graph once, and listens for HTTP requests.

---

## 🔌 Connecting Tools & Clients

### 1. Using OpenCode CLI

With `opencode.json` configured, you can select micro-experts directly using `opencode --model imb/<expert>`:

#### Bash / Linux / macOS
```bash
# Launch interactive session with Astral expert
opencode --model imb/astral

# One-off command execution with PostgreSQL expert
opencode run --model imb/postgresql "Design a PostgreSQL 18 schema for embeddings using pgvector"
```

#### Windows PowerShell
```powershell
# Launch interactive session with Astral expert
opencode --model imb/astral

# One-off command execution with PostgreSQL expert
opencode run --model imb/postgresql "Design a PostgreSQL 18 schema for embeddings using pgvector"
```

---

### 2. Using `curl` (HTTP REST Calls)

#### List Available Models (`GET /v1/models`)
```bash
curl http://127.0.0.1:8000/v1/models
```

#### Non-Streaming Chat Completion (`POST /v1/chat/completions`)
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "postgresql",
    "messages": [{"role": "user", "content": "Design a PostgreSQL schema for storing vector embeddings."}],
    "max_tokens": 64
  }'
```

#### SSE Token Streaming (`stream: true`)
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "astral",
    "messages": [{"role": "user", "content": "Write a FastAPI similarity search endpoint."}],
    "max_tokens": 64,
    "stream": true
  }'
```

---

### 3. Using Official OpenAI Python SDK

```python
from openai import OpenAI

# Point client to local IMB REST server
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")

# Instant zero-copy in-place expert swap to postgresql expert
response = client.chat.completions.create(
    model="postgresql",
    messages=[{"role": "user", "content": "Explain pgvector HNSW index creation."}],
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## 🎯 Pre-Loaded Micro-Experts

| Expert ID | Primary Domain & Target Specialization | In-Place Activation Latency |
| :--- | :--- | :---: |
| `postgresql` (or `postgres`) | Database schema design, `pgvector`, HNSW indexes, SQL queries | **$19.29\text{ ms}$** |
| `astral` | Python FastAPI, standard libraries, high-performance async APIs | **$19.29\text{ ms}$** |
| `financial_planning` (or `fin`) | Financial planning strategy, sequence-of-returns risk, wealth modeling | **$19.29\text{ ms}$** |
| `base` (or `qwen3.5`) | Standard Qwen 3.5 4B base model without expert weights | **$0.00\text{ ms}$** |

---

## 🧪 Testing & Verification Suite

Run the full GPU integration test suite:

```bash
# Full REST API server integration test suite
uv run --env-file .env python3 scripts/test_openai_api_server.py


# Zero-recapture expert swapping synergy benchmark
uv run --env-file .env python3 scripts/benchmark_zero_recapture_swapping.py
```

---

## 🛠️ Project Structure

```text
gnn-experiment/
├── src/gnn_experiment/
│   ├── server.py               # Production FastAPI OpenAI REST server & IMB gatekeeper
│   ├── fused_norm.py           # Exact PyTorch RMSNorm stand-ins for Qwen3.5
│   ├── cuda_graph.py           # FoldedCudaGraphDecoder locked graph replay engine
│   └── novel_peft.py           # WeightFoldingEngine & FoldableExpert mechanics
├── scripts/
│   ├── run_openai_api_server.py # Server CLI launcher daemon
│   ├── test_openai_api_server.py# Complete REST server integration test suite
│   └── benchmark_zero_recapture_swapping.py # Synergy benchmark
└── README.md                   # System documentation
```
