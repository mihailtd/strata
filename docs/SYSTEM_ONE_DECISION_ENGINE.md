# System One Decision Engine: Local Jev-Style Architecture & Implementation Plan

> **Status**: Approved Architectural Plan  
> **Target Hardware**: AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100, ROCm 6.2+)  
> **Integration Targets**: `apps/harness` (DSH agentic router & verification gates) & `apps/runtime-next` (native Rust/HIP engine)

---

## 1. Executive Summary & Paradigm Overview

### 1.1 The "Overqualified Checkbox" Problem in Autonomous Agents
In modern agentic workflows (e.g., DeepSeek Harness / DSH, SWE-bench loops), over 70% of model invocations are not open-ended creative code generation. They are mechanical administrative gates:
- *"Which specialist LoRA adapter should handle this task?"* (Categorical Choice)
- *"Did the compiler output fail due to a fatal syntax error?"* (Boolean Noul)
- *"Has the agent made measurable progress, or is it trapped in an infinite loop?"* (Readiness Score / Noul)
- *"Which tool should be invoked next: `read_file`, `grep_search`, or `run_command`?"* (Tool Choice)

Routing these mechanical checks through a 27B–70B autoregressive decoder incurs:
1. **Severe Latency Stalls**: Waiting 200–1,500 ms for sequential $O(N)$ token-by-token decoding.
2. **Context Bloat**: Serializing thousands of tokens of tool schemas into every prompt.
3. **Parser Brittleness**: Handling hallucinated schema boundaries, markdown backticks, and unescaped JSON quotes.
4. **Miscalibration**: Frontier LLMs are notoriously overconfident, assigning 99% probability to incorrect guesses.

### 1.2 What TypeSafe AI's Jev Proves
TypeSafe AI's **Jev** is a non-autoregressive "System One" decision model that discards token generation entirely. In a single parallel forward pass, it outputs typed, statistically calibrated decision primitives:
- **Choice**: Categorical selection over candidate options.
- **Noul**: Calibrated boolean probability ($P \in [0, 1]$).
- **Score**: Calibrated continuous rubric score ($S \in [0, 100]$).

### 1.3 Local Advantage over Cloud APIs
TypeSafe’s proprietary Jev API incurs a **70–500 ms cloud network round-trip** and costs external API credits. Running a local System One decision engine on consumer hardware (AMD RX 7900 XTX 24GB):
- **Latency**: Drops to **2–12 ms** (10x–50x faster than cloud API).
- **Zero VRAM Contention**: At 270M–800M parameters, the entire model occupies <1GB VRAM and stays permanently pinned in memory.
- **Zero Schema Errors**: 100% deterministic type safety (outputs are numerical matrix projections, never parsed strings).
- **Air-Gapped Privacy**: Sensitive agent execution traces, code diffs, and database schemas never leave the local machine.

---

## 2. Architectural Specification

```
                          [ Input Context + Query ]
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │      Selected Backbone (ModernBERT / Gemma /     │
             │                 Qwen 3.5 0.8B)                   │
             └────────────────────────┬─────────────────────────┘
                                      │
                         [ Pooled Representation h ]
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│   Choice Head    │        │    Noul Head     │        │    Score Head    │
│ (Dynamic Match)  │        │ (Calibrated Bool)│        │ (Bounded Rubric) │
└────────┬─────────┘        └────────┬─────────┘        └────────┬─────────┘
         │                           │                           │
  Softmax(h^T W e_i)          Sigmoid(w^T h + b)          S_max * Sigmoid(.)
         │                           │                           │
  P(choice_i) ∈ [0,1]         P(True) ∈ [0,1]             Score ∈ [0, 100]
```

### 2.1 The Three Decision Primitives

#### A. Choice Head (Dynamic Candidate Matching)
Rather than hardcoding static output classes, the Choice head uses **dynamic candidate matching**:
- Input: Context representation $h \in \mathbb{R}^D$ and $K$ candidate representations $e_1, \dots, e_K \in \mathbb{R}^{D_{\text{cand}}}$.
- Projection:
  $$\text{logits}_i = \frac{(W_c h)^T (W_e e_i)}{\sqrt{D_{\text{proj}}}}$$
  $$P(\text{choice}_i) = \frac{\exp(\text{logits}_i / \tau)}{\sum_{j=1}^K \exp(\text{logits}_j / \tau)}$$
- **VRAM Cache Optimization**: For fixed system classes (such as our 6 domain adapters: `postgresql`, `python_web`, `duckdb`, `astral`, `python_modern`, `financial_planning`), candidate embeddings $e_i$ are pre-computed once and pinned in VRAM. Routing over fixed classes runs in $<0.1\text{ ms}$ (identical to a static linear layer), while retaining full flexibility to score arbitrary agent options dynamically.

#### B. Noul Head (Calibrated Boolean Gate)
- Computes calibrated probability for binary readiness gates, verification checks, and if/else conditions:
  $$z = W_2 \cdot \text{GELU}(W_1 h) + b_2$$
  $$P(\text{True}) = \sigma(z / \tau_{\text{noul}})$$
- Output: Exact float probability $P \in [0.0, 1.0]$. When $P \ge 0.95$, the harness auto-executes; when $P < 0.80$, it flags for verification or fallback.

#### C. Score Head (Continuous Rubric Evaluation)
- Computes a bounded scalar evaluation over code diff quality, test pass likelihood, or progress metrics:
  $$S = S_{\max} \cdot \sigma(W_2 \cdot \text{GELU}(W_1 h) + b_2)$$
- Scaled to $[0, 100]$ (or $[0.0, 1.0]$).

---

## 3. The 3-Way Candidate Bake-Off

Before locking the production backbone, we implement an empirical 3-way evaluation bake-off across three distinct architectures:

| Dimension | Candidate A: ModernBERT-Large | Candidate B: FunctionGemma 270M | Candidate C: Qwen 3.5 0.8B |
| :--- | :--- | :--- | :--- |
| **Architecture** | Native Bidirectional Encoder | Causal Decoder (Tool-Aligned) | Hybrid GatedDeltaNet + Attention |
| **Parameter Count** | 395M | 270M | 800M |
| **Native Context** | 8,192 tokens | 8,192 tokens | 262,144 tokens |
| **B=1 GPU Latency** | **2 – 5 ms** (Fastest) | **4 – 8 ms** | **12 – 18 ms** |
| **VRAM Footprint** | ~800 MB (BF16) | **~550 MB** (BF16) / 200 MB (Q4) | ~1.6 GB (BF16) / 600 MB (W4A16) |
| **Code & JSON Reasoning** | Moderate (NLP pretraining) | High (Function-calling aligned) | **Exceptional** (Dense code pretraining) |
| **Runtime-Next Synergies** | Requires separate encoder runtime | Requires Gemma decoder pass | **Native**: Shares exact W4A16 kernels & tokenizer |

### Bake-Off Evaluation Criteria
1. **Decision Accuracy**: Top-1 Choice accuracy on DSH domain routing and BFCL tool selection.
2. **Probability Calibration**: Expected Calibration Error ($ECE < 0.04$) and Brier score on held-out validation.
3. **Execution Latency**: Single-pass batch=1 latency on AMD RX 7900 XTX (HIP stream).
4. **VRAM Footprint**: Static resident memory during active serving.

---

## 4. Training & Calibration Pipeline (Local Consumer GPU Recipe)

Training runs on the local AMD RX 7900 XTX (24GB VRAM) using PyTorch ROCm in `apps/factory/decision_engine/`.

### 4.1 Dataset Formulation (Comprehensive Agentic Mix)
We construct a 200,000-sample balanced dataset across four task categories:

1. **Agent Routing & Expert Selection (50,000 samples)**:
   - DSH domain routing queries mapped to specialist adapters (`postgresql`, `python_web`, `duckdb`, `astral`, `python_modern`, `financial_planning`).
   - Hard negatives: ambiguous multi-domain prompts with calibrated mixed probabilities.
2. **Tool Selection & Schema Matching (60,000 samples)**:
   - Berkeley Function Calling Leaderboard (BFCL) + Glaive Function Calling v2.
   - Dynamic Choice over 3 to 15 tool signatures given conversational and terminal context.
3. **Boolean Readiness & Verification (Noul) (50,000 samples)**:
   - SuperGLUE (BoolQ, MultiRC, RTE).
   - Compiler/test output triage: *"Did `cargo test` pass?"*, *"Is this error a VRAM OOM?"*, *"Has loop stalled?"*.
4. **Code Quality & Progress Rubrics (Score) (40,000 samples)**:
   - Git patch quality scoring (0–100), automated test coverage delta, heuristic diff cleanliness.

### 4.2 Loss Function & Calibration Objective (RLCD)
Standard cross-entropy creates extreme overconfidence. We train with a composite objective combining task loss and Brier score regularization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{brier}} \mathcal{L}_{\text{brier}} + \lambda_{\text{reg}} \mathcal{L}_{\text{smooth}}$$

1. **Choice Loss**: Cross-entropy over dynamic candidate inner products:
   $$\mathcal{L}_{\text{choice}} = - \sum_{i=1}^K y_i^* \log P(\text{choice}_i)$$
2. **Noul Loss & Brier Penalty**:
   $$\mathcal{L}_{\text{brier}} = \frac{1}{N} \sum_{i=1}^N \left( P(\text{bool}_i) - y_i^* \right)^2$$
   Directly penalizes the distance between predicted probability and true binary ground truth.
3. **Score Loss**: Smooth L1 / Huber regression loss on normalized targets:
   $$\mathcal{L}_{\text{score}} = \text{SmoothL1}(S_{\text{norm}}, y_{\text{norm}}^*)$$
4. **Post-Hoc Temperature / Platt Scaling**:
   After training, optimize validation temperatures $\tau_{\text{choice}}, \tau_{\text{noul}}$ to minimize Expected Calibration Error:
   $$ECE = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right| < 0.04$$

---

## 5. Phased Implementation Roadmap

### Phase 1: Dataset Generation & 3-Way Bake-Off Harness
- **Location**: `apps/factory/decision_engine/` & `benchmarks/decision_eval.py`
- **Deliverables**:
  1. `data_loader.py`: Ingestion for SuperGLUE, BFCL, and synthetic DSH routing traces.
  2. `models/`: PyTorch reference implementations for:
     - `modernbert_decision.py` (ModernBERT-Large backbone + heads)
     - `functiongemma_decision.py` (FunctionGemma-270M prefix-pooled + heads)
     - `qwen_decision.py` (Qwen3.5-0.8B prefix-pooled + LoRA + heads)
  3. `benchmarks/decision_eval.py`: Benchmark harness measuring accuracy, ECE calibration error, single-pass latency (ms), and VRAM (MB) on the RX 7900 XTX.

### Phase 2: Training & Calibration on RX 7900 XTX
- **Location**: `apps/factory/decision_engine/train.py`
- **Execution**:
  - Full-parameter fine-tuning for ModernBERT (395M, fits in ~11GB VRAM).
  - LoRA fine-tuning for Qwen 3.5 0.8B ($r=16, \alpha=32$, fits in ~8GB VRAM).
  - Run Brier regularization + temperature scaling calibration.
  - Save calibrated model checkpoints to `results/models/decision_engine_v1/`.

### Phase 3: DSH Harness Integration (Local Fast Microservice)
- **Location**: `apps/harness/router/`
- **Deliverables**:
  1. `decision_service.py`: Lightweight FastAPI / Uvicorn (or Unix Domain Socket) daemon exposing `/decide/choice`, `/decide/noul`, `/decide/score`.
  2. Latency: Serves decisions in $<5\text{ ms}$ over IPC.
  3. Wire into `router_cli.py`: Replace `DOMAIN_PATTERNS` regex matching with calibrated model routing.
  4. Wire into `task_progress`: Replace manual loop heuristic counters with calibrated Noul stalled-gate predictions.

### Phase 4: Native `runtime-next` In-Engine Integration
- **Location**: `apps/runtime-next/`
- **Deliverables**:
  1. If Qwen 3.5 0.8B wins: Native Rust/HIP prefix forward pass in `model.rs` that stops after the prompt prefill and projects the final hidden state through decision heads.
  2. Zero-IPC overhead: Decisions execute directly in the engine process with shared memory and zero PCIe/socket latency.
  3. Two-Tier Dispatcher: Single-pass decision head triggers tool selection in $<5\text{ ms}$, immediately followed by In-Place Weight Folded (IPWF) argument generation.

---

## 6. Verification Plan & Acceptance Gates

| Gate | Target Metric | Verification Method |
| :--- | :--- | :--- |
| **Calibration** | $ECE < 0.04$, Brier Score $< 0.08$ | Validation split evaluation across 10 probability bins |
| **Routing Accuracy** | $>96.5\%$ Top-1 on DSH domains | Evaluation against 1,000 real coding/engineering user requests |
| **Inference Latency** | $<8\text{ ms}$ on RX 7900 XTX (B=1) | Microbenchmark via HIP stream events (`benchmarks/decision_eval.py`) |
| **VRAM Footprint** | $<1,200\text{ MB}$ resident VRAM | Live `rocm-smi` compute VRAM telemetry during active serving |
| **Zero Regressions** | 70/70 existing `runtime-next` tests pass | `cargo test -- --test-threads=1` |
