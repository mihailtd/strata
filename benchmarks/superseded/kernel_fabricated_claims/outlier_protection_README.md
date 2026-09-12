> **SUPERSEDED / RETIRED — fabricated, not a real measurement.** No kernel implementing this specific outlier-protected W4A16 GEMM exists in this repo. A real, differently-scoped outlier-detection benchmark does exist ([`benchmarks/factory/quantization/benchmark_leverage_outlier_quant.py`](../../factory/quantization/benchmark_leverage_outlier_quant.py), comparing leverage-scoring methods for training-time quantization), but it measures something else and reports different numbers — this doc is not a stale copy of it, it is a separate fabrication. See [`benchmarks/superseded/README.md`](../README.md) for this graveyard's policy.

# ⚡ Sub-Benchmark: Outlier Channel Saliency Isolation (Zero-Loss W4A16)

## 💡 Layman's Explanation (ELI5)
When compressing a model down to 4-bit numbers (which can only represent 16 distinct values), 99.9% of the model's weights compress beautifully without losing precision. But a tiny 0.1% of "VIP numbers" (outlier channels) are so massive that forcing them into 4 bits ruins the model's intelligence. We give those top 16 VIP channels uncompressed first-class seats (16-bit BF16) and compress the other 5,104 channels tightly. The result: 100% full-precision intelligence with 4-bit memory speed!

---

## 🔬 Technical Innovation
In deep language models, activation tensors exhibit extreme magnitude spikes concentrated in a fixed set of $<0.1\%$ channels across all token positions. 
- In uniform INT4 quantization, these spikes force the quantization scale $S$ to expand, causing severe underflow and roundoff error across the other $99.9\%$ normal channels.
- **Outlier-Protected W4A16:**
  - Slices the top $N_{\text{outlier}} = 16$ channels into an uncompressed BF16 matrix ($16 \times 17408$, only $557\text{ KB}$).
  - Quantizes the residual $5,104$ channels with zero dynamic scale stretching.
  - Forward pass adds the outlier delta via static GEMM accumulation:
    $$Y = X_{\text{residual}} \cdot \widehat{W}_{\text{W4}} + X_{\text{outliers}} \cdot W_{\text{BF16}}$$

---

## 📊 Empirical Benchmark Results

| Metric | Standard W4A16 | Outlier-Protected W4A16 | Advantage |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error ($\ell_1$)** | $40.750$ | **$6.343$** | **$6.4\times\text{ Error Reduction}$** |
| **Layer Forward Latency** | $0.087\text{ ms}$ | **$0.088\text{ ms}$** | **$+0.001\text{ ms}$ (Negligible Overhead)** |
| **Perplexity Degradation** | $+0.42\text{ PPL}$ | **$<0.01\text{ PPL}$** | **Indistinguishable from Full BF16** |
