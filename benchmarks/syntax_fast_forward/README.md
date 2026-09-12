# Deterministic AST & Syntax Fast-Forwarding (Jump Tokens)

## 1. Overview & Motivation

Speculative decoding relies on high-quality draft candidates to accelerate large language model inference. On structured code generation (Python, SQL, FastAPI, Pydantic, DuckDB), syntax structures follow deterministic syntactic patterns:
- Function/Class boilerplate: `def __init__(self, `, `class <Name>(BaseModel):`
- Asynchronous resource acquisition: `async with pool.acquire() as conn:\n        `
- Transaction scopes: `async with conn.transaction(readonly=True):\n            `
- Imports & Type Annotations: `from typing import List, Dict, Optional, Any, Tuple\n`
- Common guards: `if __name__ == "__main__":\n    `

### The N-Gram Drafter's "Cold Start" Blind Spot
Existing speculative engines use an in-context `NGramDrafter` that searches historical tokens for repeating n-grams.
- When an agent encounters boilerplate syntax for the first time in a turn, the context contains no prior match (`draft = []`).
- The system drops back to 1-token Neural MTP or single-token autoregression, experiencing a severe throughput slump.

### The Solution: Zero-VRAM `SyntaxTrieDrafter`
`SyntaxTrieDrafter` compiles canonical language and framework macros into an in-memory reverse-suffix Token Trie on the CPU host:
- Executes prefix lookup in **$<3\ \mu\text{s}$** (0 bytes VRAM).
- Proposes a linear candidate chain ($K=3..6$ tokens) at the very first occurrence of a syntax trigger.
- Emits candidate chains directly into our parallel verification kernel (`forward_verify`).
- Executes via pre-captured ROCm HIP Graphs (`verify_graphs_k4`) with sub-millisecond host launch latency on AMD Radeon RX 7900 XTX hardware.

---

## 2. Mathematical Formulation & Verification Invariant

Let $\mathbf{x}_{\le t}$ be the sequence of context tokens. The drafter evaluates candidate tokens via a 3-tier hierarchy:
$$\text{Draft}(\mathbf{x}_{\le t}) = \begin{cases} \text{TrieDrafter}(\mathbf{x}_{\le t}) & \text{if } \text{Trie}(\mathbf{x}_{\le t}) \ne \emptyset \\ \text{NGramDrafter}(\mathbf{x}_{\le t}) & \text{if } \text{NGram}(\mathbf{x}_{\le t}) \ne \emptyset \\ \text{NeuralMTP}(h_t, x_t) & \text{otherwise} \end{cases}$$

### Bit-Exact Verification Invariant
Candidates $[c_1, c_2, \dots, c_K]$ are evaluated in a single forward pass by the target model:
$$P(c_j \mid \mathbf{x}_{\le t}, c_1, \dots, c_{j-1})$$
A candidate $c_j$ is accepted if and only if:
$$\arg\max_{v} P(v \mid \mathbf{x}_{\le t}, c_1, \dots, c_{j-1}) = c_j$$
If the target model deviates at position $j^*$ (e.g. the user intended `class CustomList(UserList):` rather than `BaseModel`), token $c_{j^*}$ is replaced by the target model's argmax token, subsequent candidates are discarded, and the recurrent state is rolled back in $O(1)$ time. This guarantees **100% bit-exact equivalence with greedy decode**.

---

## 3. Empirical Synthetic Metrics (AMD Radeon RX 7900 XTX)

Benchmarking proposal accuracy, latency, and burst speedup across authentic code patterns:

| Syntax Domain | Trigger Pattern | Jump Tokens Emitted | Lookup Time | Local Burst Speedup vs Greedy |
| :--- | :--- | :--- | :--- | :--- |
| **Python Core** | `if __name__ == ` | 7 tokens | 2.1 µs | **5.8x Faster** |
| **Pydantic** | `from pydantic import ` | 8 tokens | 2.4 µs | **6.1x Faster** |
| **FastAPI** | `from fastapi import ` | 12 tokens | 2.6 µs | **6.4x Faster** |
| **PostgreSQL** | `async with pool.acquire() as ` | 4 tokens | 2.8 µs | **4.2x Faster** |
| **PostgreSQL** | `CREATE EXTENSION IF NOT EXISTS ` | 6 tokens | 2.2 µs | **5.2x Faster** |
| **DuckDB** | `QUALIFY ROW_NUMBER() OVER (` | 5 tokens | 2.5 µs | **4.9x Faster** |

---

## 4. Real-World End-to-End Dual A/B Evaluation (AMD Radeon RX 7900 XTX)

Tested on the live 27B model across 4 authentic software engineering tasks (FastAPI, PostgreSQL asyncpg, DuckDB Parquet OLAP, Python CLI entrypoint):
*Source: [`results/benchmarks/syntax_fast_forward_e2e_scorecard.json`](../../results/benchmarks/syntax_fast_forward_e2e_scorecard.json)*

| Benchmark Arm | Overall Throughput | Speedup vs Greedy | Speedup vs Previous | Memory Churn | Bit-Exact Parity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm 1: Pure Greedy Baseline ($K=1$)** | 10.4 tok/s | 1.00x | — | 0.0 MB | 100% (Reference) |
| **Arm 2: Previous Linear Speculation** | 19.2 tok/s | 1.85x | 1.00x | 0.0 MB | 100% Identical |
| **Arm 3: Our Engine (+ Syntax Fast-Forward)** | **19.2 tok/s** | **1.85x** | **1.00x** | **0.0 MB** | **100% Bit-Exact** |

### Why Were Arm 2 and Arm 3 Identical (19.2 tok/s vs 19.2 tok/s)?
1. **The Prefill Boundary Effect ("Prompt Swallowing")**: All task prompts placed triggers at the end of the input. Input prompts are ingested in a single prefill pass via `forward_prompt`. Speculative drafting only executes during autoregressive decode, so prompt triggers never triggered the Trie drafter.
2. **Macro Sparsity (0.4% Trigger Rate)**: Across 256 generated tokens, only 1 trigger appeared inside the decode loop (`from pydantic import `). For 255 of 256 tokens (99.6%), the Trie returned `[]` and fell back to Arm 2's pipeline (N-Gram + Neural MTP).
3. **Neural MTP Overlap**: At that single hit, the 27B model's Neural MTP head at block 64 *also* predicted `BaseModel`. Both arms drafted the exact same candidate.
4. **Wall-Clock Math**: Trie lookup added only $92\ \mu\text{s}$ CPU overhead across 256 tokens. On GPU, both arms ran identical verification steps (13.31s vs 13.32s).

---

## 5. The 1,024-Token Best-Case Empirical Benchmark

To evaluate Syntax Fast-Forwarding in its optimal scenario, we generated a full 1,024-token enterprise FastAPI service and async PyTest test suite with extensive repeated boilerplate (models, routers, error handlers, and test assertions).

Evaluated live on **AMD Radeon RX 7900 XTX (Navi 31, 24 GB VRAM)**:

| Benchmark Arm | Tokens | Wall-Clock (s) | Throughput (tok/s) | Speedup vs Greedy | Speedup vs Linear MTP | Memory Delta | Cross Parity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm 1: Pure Greedy Baseline ($K=1$)** | 1,024 | 89.08 s | 11.50 tok/s | 1.00x | — | 0.0 MB | Reference |
| **Arm 2: Linear MTP + N-Gram** | 1,024 | 47.64 s | **21.50 tok/s** | **1.87x** | **1.00x** | 0.0 MB | 100% Bit-Exact |
| **Arm 3: Syntax Fast-Forward + N-Gram + MTP** | 1,024 | 48.27 s | 21.21 tok/s | 1.84x | **0.99x** | 0.0 MB | 100% Bit-Exact |

### Granular Drafter Contribution Breakdown (1,024 Tokens)

The speculation engine tracked every proposed draft and accepted token by drafter source:

```
Arm 2 (Linear MTP + N-Gram) [Total Steps: 501]:
  - N-Gram Drafter       : 264 proposed, 152 accepted (57.6% acceptance rate)
  - Neural MTP Head      : 413 proposed, 370 accepted (89.6% acceptance rate)
  - Syntax Trie          : Disabled

Arm 3 (Syntax Fast-Forward + N-Gram + MTP) [Total Steps: 507]:
  - Syntax Trie Drafter  :  25 proposed,   4 accepted (16.0% acceptance rate)
  - N-Gram Drafter       : 264 proposed, 147 accepted (55.7% acceptance rate)
  - Neural MTP Head      : 410 proposed, 365 accepted (89.0% acceptance rate)
```

---

## 6. Empirical Root Cause: Why Static Syntax Macros Fail to Accelerate Code Generation

1. **Abysmal Acceptance Rate (16.0%) vs Neural MTP (89.6%)**:
   - Out of 25 static draft proposals from the Trie, only 4 were accepted by the 27B base model.
   - In contrast, the block-64 Neural MTP head achieved an astounding **89.6% acceptance rate** because it conditions on the full 27B hidden state $h_{\text{curr}}$.
2. **Verification Chunk Rollback Penalty**:
   - In `forward_verify`, proposing a 3-token static macro evaluates $K=4$ candidates.
   - When token 1 fails (84% of the time for static macros), all remaining tokens in the chunk are rejected. The engine takes a corrective rollback step.
   - Arm 3 required **507 decoding steps** vs **501 steps** for Arm 2, adding 0.63s of total wall-clock time.
3. **Natural Code Is Semantically Stochastic**:
   - Even in standard syntax patterns (e.g. `@pytest.mark.asyncio`, `raise HTTPException(...)`), the model often chooses variable names, status codes, or docstrings that differ from static templates.

---

## 7. Production Verdict: Keep Linear MTP + N-Gram as Default

| Technique | Hardware Throughput | Acceptance Rate | Verdict | Production Status |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Greedy ($K=1$)** | 11.5 tok/s | N/A | Baseline | Fallback only |
| **Linear MTP + N-Gram** | **21.5 tok/s** | **89.6% MTP / 57.6% N-Gram** | 🚀 **1.87x Speedup** | **DEFAULT PRODUCTION (100% RECOMMENDED)** |
| **Syntax Fast-Forwarding** | 21.2 tok/s | 16.0% Trie | ⚠️ **0.99x (Slight Drag)** | **OPT-IN ONLY (Constrained GBNF/JSON only)** |

### Conclusion
- **Unconstrained Code Generation**: Keep Syntax Fast-Forwarding **OFF** (or bypassed) in favor of **Linear MTP + N-Gram**, which delivers the maximum 21.5 tok/s throughput on AMD Radeon RX 7900 XTX.
- **Constrained Grammar Decoding**: Reserve Syntax Fast-Forwarding for formal grammars (GBNF, JSON schemas) where the grammar engine strictly guarantees 100% token transition determinism.

