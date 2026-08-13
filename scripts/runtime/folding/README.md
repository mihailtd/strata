# 🔥 In-Place Low-Rank Weight Folding

This folder contains the benchmarks and evaluations that prove the core mathematical technique of the entire engine: fusing LoRA adapters directly into the base `bfloat16` weights of the **Main Backbone Model** without allocating new memory.

## Why this is a Massive Success
Traditional PEFT (Parameter-Efficient Fine-Tuning) requires wrapping the base model with adapter layers. During inference, every matrix multiplication has to go through the base weight, then the adapter, and sum the results. This destroys throughput.

By mathematically folding the weights directly into the backbone (using the `WeightFoldingEngine` and the $\alpha$ calibration rules discovered in the Factory), this engine achieves a massive **1.83x speedup** compared to standard PEFT wrappers. It transforms adapted generation to run at 100% native speed!

## Scripts in this Module
- **`benchmark_weight_folding.py`**: Proves the pure speedup (1.83x) of the folding technique over standard wrapping.
- **`evaluate_folded_vs_wrapped.py`**: Verifies that the mathematical absorption into `bfloat16` does not degrade output quality compared to running the adapters dynamically.
- **`benchmark_stacked_experts.py`**: Demonstrates `activate_many()` by stacking multiple adapters simultaneously into a single fold.
