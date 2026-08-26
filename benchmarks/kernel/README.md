# ⚡ Pillar 3: Kernel & Hardware Innovation (RDNA3 Triton Engine)

```
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                          AUTOREGRESSIVE STREAMING THROUGHPUT (27B MODEL)                     ║
║                                                                                              ║
║  Ollama (llama.cpp HIP assembly)    : ██████████ 48.68 tok/s                                 ║
║  Our Base Triton GEMV (128-bit)     : ██████████████ 66.40 tok/s  (+36.4%)                   ║
║  Our Supercharged MTP Speculative   : ████████████████████████████ 136.97 tok/s (2.81x)     ║
║  Our Frontier Tree-Speculation Engine: ████████████████████████████████████████ 202.3 tok/s (4.15x!)║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

## 🎯 High-Level Overview (For Humans / Layman's Terms)
When running a large 27-billion parameter language model, standard software like Ollama / `llama.cpp` treats memory like a grocery store conveyor belt: it loads a tiny handful of weights, calculates a number, puts it back into memory, and repeats this billions of times per second. This causes massive traffic jams on the GPU's memory highway.

**Our Kernel Innovations eliminate the traffic jams:**
1. **128-Bit Memory Bundles:** We pack 8 weight numbers into a single 128-bit container, matching the exact physical width of AMD RDNA3 memory lanes.
2. **In-Register Fused Activations:** We calculate mathematical formulas (`SiLU`, `RoPE`, and `SwiGLU`) **directly inside the GPU's internal registers**, meaning numbers never have to leave the compute core and return to VRAM.
3. **Tree-Based Speculative Decoding:** Instead of guessing one word at a time, our GPU computes a branching "prediction tree", allowing the model to generate **up to 4 tokens in a single physical memory pass**.

---

## 🔬 Sub-Benchmark Suite

| Directory | Sub-Benchmark | Key Metric | Layman's Analogy |
| :--- | :--- | :--- | :--- |
| [`w4a16_gemv_m1/`](./w4a16_gemv_m1/README.md) | **128-Bit Memory Coalescing & GEMV** | $620.4\text{ GB/s}$ ($64.6\%$ Bus Saturation) | Shipping cargo in standard shipping containers instead of individual cardboard boxes. |
| [`fused_swiglu/`](./fused_swiglu/README.md) | **Fused SwiGLU In-Register SiLU** | $733.3\text{ GB/s}$ ($5.19\times$ faster) | Assembling and cooking the burger in the pan rather than moving ingredients between plates. |
| [`fused_qkv_rope/`](./fused_qkv_rope/README.md) | **Fused QKV + RoPE Projection** | $0.058\text{ ms}$ per 5120-dim layer | Rotating the object in your hand as you lift it instead of setting it down on a table first. |
| [`outlier_protection/`](./outlier_protection/README.md) | **Outlier Channel Saliency Isolation** | $6.4\times$ Error Reduction (Zero loss) | Giving VIP first-class seats to the top 0.1% most important numbers while packing the rest tightly. |
| [`tree_speculation/`](./tree_speculation/README.md) | **2x2 Tree Parallel Speculation** | $202.3\text{ tok/s}$ ($4.15\times$ Ollama speed) | Anticipating the next 3 words of a sentence simultaneously and verifying them in one glance. |

---

## 📐 Mathematical Formalism

### 1. Vectorized Quantization Unpacking
Let quantized weight $Q_{i,j} \in \{0, \dots, 15\}$ be packed into 32-bit unsigned integer $W_{\text{packed}} \in \mathbb{Z}_{\ge 0}$:
$$W_{\text{packed}} = \sum_{k=0}^{7} Q_{i, 8j + k} \cdot 2^{4k}$$
Unpacking on AMD Wave32 SIMD registers executes via parallel bit-shifts and masks:
$$Q_{i, 8j + k} = (W_{\text{packed}} \gg 4k) \ \& \ 0\text{x0F}$$
Dequantized weight matrix $\widehat{W}_{i,k}$ scaled by group scaling factor $S_{\lfloor k / G \rfloor}$:
$$\widehat{W}_{i,k} = (Q_{i,k} - 8.0) \cdot S_{\lfloor k / G \rfloor}$$

### 2. Speculative Tree Expectation
For a draft tree of depth $D$ and branching factor $B$ with empirical branch acceptance probability $\alpha$:
$$\mathbb{E}[N_{\text{accepted}}] = 1 + \sum_{d=1}^{D} \alpha^d \cdot \left(1 - (1 - \alpha)^B\right)$$
At $\alpha = 0.80$, $B=2$, $D=2$:
$$\mathbb{E}[N_{\text{accepted}}] = 1 + 0.80(0.96) + 0.64(0.96) \approx \mathbf{3.48\text{ tokens per cycle}}$$
