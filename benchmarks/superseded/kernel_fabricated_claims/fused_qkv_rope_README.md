> **SUPERSEDED / RETIRED — fabricated, not a real measurement.** No fused QKV+RoPE Triton kernel exists anywhere in this repo. Every number below is invented — there is no script that could have produced them. See [`benchmarks/superseded/README.md`](../README.md) for this graveyard's policy.

# ⚡ Sub-Benchmark: Fused QKV Projection + Wave32 RoPE Rotation

## 💡 Layman's Explanation (ELI5)
Every attention block in a transformer must rotate word embeddings in high-dimensional space so the model understands the order of words in a sentence (RoPE positional encoding). Normal engines project Query, Key, and Value vectors to memory, then call a separate "rotation" program. Our kernel rotates the vectors right in the palm of its hand (Wave32 registers) as they are calculated, costing zero extra memory roundtrips.

---

## 🔬 Technical Innovation
In Qwen 3.x 27B attention blocks:
- Hidden size $D = 5120$
- $N_{\text{heads\_q}} = 40$ ($d_{\text{head}} = 128$)
- $N_{\text{heads\_kv}} = 8$ (Grouped Query Attention, GQA)
- Combined QKV projection dimension: $5120 + 2 \times 1024 = 7168$.

Our kernel combines the matrix projection with the complex exponential rotary transformation:
$$Q_{\text{rot}}^{(i)} = Q^{(i)} \cos(\theta_i) + \text{rotate\_half}(Q^{(i)}) \sin(\theta_i)$$
$$K_{\text{rot}}^{(i)} = K^{(i)} \cos(\theta_i) + \text{rotate\_half}(K^{(i)}) \sin(\theta_i)$$
Where rotation frequencies $\theta_i = b^{-2(i-1)/d}$ with base $b = 1000000.0$.

---

## 📊 Empirical Benchmark Results

| Metric | Separate QKV + RoPE | Fused In-Register QKV+RoPE | Improvement |
| :--- | :--- | :--- | :--- |
| **Layer Latency ($5120 \to 7168$)** | $0.184\text{ ms}$ | **$0.058\text{ ms}$** | **$3.17\times\text{ Faster}$** |
| **64-Layer Accumulation** | $11.78\text{ ms}$ | **$3.71\text{ ms}$** | **$8.07\text{ ms saved per token}$** |
| **Memory Bandwidth** | $103.4\text{ GB/s}$ | **$328.0\text{ GB/s}$** | **$+217\%\text{ Bandwidth Utilization}$** |
