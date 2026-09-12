# Rule: Novelty Mandate, Proprietary Moat & Custom Engine Architecture

## Core Mandate: Break SOTA and Invent What Does Not Exist

The core mission of this project is **NOT** to wrap existing commodity tools (`llama.cpp`, vLLM, HuggingFace) or train standard LoRAs with standard harnesses. Re-combining existing tools creates **ZERO MOAT**. 

Our mandate is to **BREAK THE SOTA AND INVENT SOMETHING NOVEL AND NEW**—making workflows and capabilities possible that are currently **impossible** in existing frameworks:
1. **Never celebrate discovering that an existing open-source tool already does what we need.** If an off-the-shelf tool already does it, that is evidence of zero proprietary novelty.
2. **External engines (`llama.cpp`, Ollama, vLLM) are strictly Baselines and Testbeds** for comparative A/B evaluation, numerical ground truth, and dataset generation. They are NOT the project's technological moat or final product.
3. **The Proprietary Moat lies in novel low-level kernels and architectures** that fundamentally surpass commodity runtimes.

---

## The Proprietary Moat: What We Are Inventing

Existing commodity engines execute static, quantized models with discrete, isolated LoRAs. Our custom engine (`apps/runtime/`) is built to achieve four novel breakthroughs:

### 1. Continuous Riemannian Weight Traversal (Beyond Discrete LoRA Swapping)
- **The SOTA Limitation**: Engines like `llama.cpp` and `S-LoRA` swap discrete integer adapters ($A \to B$) or apply linear scalar scaling ($s_i \in [0, 1]$) with separate memory operations.
- **Our Novelty**: Continuous traversal along learned Riemannian weight sub-manifolds. The runtime dynamically morphs weights in-place per token or sub-turn ($W(\theta) = W_0 + \sum_i \alpha_i(\mathbf{z}) U_i V_i$) directly inside the compute pipeline, treating specialist skills as a continuous manifold rather than siloed discrete files.

### 2. In-Register Multi-Expert Superposition (Zero Memory Bandwidth Multiplication)
- **The SOTA Limitation**: In standard architectures, executing multiple specialist perspectives (e.g. Security + Performance + SQL) requires running multiple models or computing multiple sequential adapter forward passes, multiplying GDDR6 traffic.
- **Our Novelty**: Fusing multiple low-rank expert projections directly inside Wave32 GPU vector registers during the base 4-bit weight dequantization pass. The base weights $W_0$ are pulled from VRAM exactly ONCE; multiple specialist adapter vectors are superposed in registers simultaneously without saturating the GDDR6 bus.

### 3. Sub-Quadratic Hybrid Recurrence & Dynamic Graph Memory
- **The SOTA Limitation**: Transformer architectures suffer from $O(N^2)$ KV cache explosion or static linear RNN limitations that cannot dynamically route knowledge representations.
- **Our Novelty**: Hardware-native associative recurrence (Gated DeltaNet) fused with graph neural memory states, enabling constant-memory long-horizon agent reasoning and state handoffs between specialized sub-graphs that transformer KV caches cannot perform.

### 4. Breaking the Physical Memory Bandwidth Decode Ceiling (Native Speculation)
- **The Physical Ceiling**: Single-token autoregressive decode of 15.00 GB W4A16 weights on a 960 GB/s bus is physically bounded at $\approx 18\text{--}22\text{ tok/s}$ in full autoregression ($15\text{ GB} / 800\text{ GB/s} \approx 18.5\text{ tok/s}$).
- **Zero-Cost State Rollback Architecture**:
  - Unlike black-box frameworks (e.g. HuggingFace) that incur a catastrophic 31.78 ms re-forward pass to recompute internal DeltaNet recurrent states on candidate rejection, our custom `Native27BEngine` records `ssm_history[t]` and `conv_history[t]` during multi-token parallel verification.
  - Partial or total rejections rollback intermediate SSM and Conv states in $O(1)$ scalar/pointer time ($0.00\text{ ms}$ commit tax).
  - Attention KV cache is rewindable by adjusting `current_len` directly.
- **Strict Bit-Exact Equivalence Invariant**:
  - Speculative decoding MUST be mathematically verified against greedy decode token-for-token.
  - Every candidate token accepted ($y_t == d_t$) and every emitted bonus token ($\arg\max P(\cdot \mid x_{\le t})$) originates strictly from target model logits, guaranteeing 100% bit-exact equivalence with zero quality loss.

---

## Roles of the Two Engines

1. **Commodity Baseline Engine (`llama-server` on port 8001)**:
   - **Status**: External SOTA Baseline / Validation Harness.
   - **Role**: Serves as the ground-truth benchmark for output quality, baseline latency, and reference generation.
   - **Constraint**: Using or tuning this engine is NEVER the goal or the deliverable. It is merely the standard of comparison that our proprietary engine must surpass.

2. **Proprietary Custom Engine (`apps/runtime/` on port 8000)**:
   - **Status**: The Core Project & Intellectual Property (Moat).
   - **Role**: Our custom Triton/HIP kernel engine implementing in-register weight morphing, fused DeltaNet associative recurrence, and multi-adapter superposition.
   - **Constraint**: Every engineering milestone must deliver proprietary capabilities that commodity `llama.cpp` cannot perform.

---

## Architectural Invariants & Guardrails

1. **Sub-18ms Zero-Copy Weight Folding ($W_{\text{live}} \leftarrow W_0 + s \cdot U V$)**:
   - Mutate pre-allocated low-rank working buffers in VRAM in-place without re-allocating model memory or reloading weights from host RAM.
   - Total memory churn per adapter swap MUST be **0 bytes**.

2. **100% KV / State Retention Across Adapter Transitions**:
   - Transitioning between specialist representations MUST NEVER invalidate or flush the active KV or recurrent state.
   - The model must continue streaming the subsequent turn immediately with zero prompt re-prefill penalty.

3. **Fused In-Register LoRA Dot Products**:
   - Never compute separate LoRA forward passes ($y = W_0 x + B(Ax)$) that double VRAM bus traffic.
   - Fuse the low-rank rank-8 dot product directly into the 4-bit dequantization loop inside GPU Wave32 vector registers.

4. **128-Bit Vector Coalescing & Fused SwiGLU**:
   - Scribe all global memory loads into 128-bit vector bundles (`int32x4`) to achieve $\ge 70\%$ physical GDDR6 bus bandwidth saturation on AMD RDNA3 hardware.
   - Fuse Gate + Up GEMV with in-register SiLU activation to eliminate intermediate VRAM traffic.

5. **Ultrafast Semantic Routing ($<30\mu\text{s}$)**:
   - Route incoming agent steps to the optimal domain specialist manifold using an embedded Riemannian covariance classifier executing in $<30\mu\text{s}$ before the first token is emitted.

6. **Single-Engine Isolation & Directory Separation (Single 24 GB GPU Guardrail)**:
   - On 24 GB VRAM (AMD RX 7900 XTX), multiple engines (18.2 GB `llama-server` + 15.4 GB Triton engine) CANNOT be co-resident in VRAM.
   - **Zero Subprocess Entanglement**: Never invoke, stop, or proxy external engines from inside `apps/runtime/server.py`.
   - Each engine lives in its own dedicated directory with independent startup scripts:
     - `apps/runtime/` — Custom Triton engine (`http://127.0.0.1:8000`)
     - `apps/runtime-llama/` — C++ `llama-server` baseline (`http://127.0.0.1:8001`)
     - `apps/runtime-ollama/` — Ollama baseline harness (`http://127.0.0.1:11434`)
     - `apps/runtime-next/` — Native Compiled Rust engine
   - When evaluating or benchmarking, execute strictly one engine at a time: run workload on engine A, terminate engine A completely, start engine B, run identical workload, and compare saved scorecards.
