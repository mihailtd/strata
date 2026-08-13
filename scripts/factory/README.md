# The Factory (Fine-Tuning)

This directory serves as the centralized engine for all training and fine-tuning operations. It is designed to replace scattered, one-off training scripts with a cohesive, unified pipeline.

## Core Concepts
- **Training Data Quality**: Data generation, formatting, and opinionated cleaning pipelines.
- **Loss & Backprop Math**: Implementation of advanced mathematical optimizations, such as Response-Only Completion Loss (masking prompts with `-100`) and Fused Triton Backprop Operators.
- **Parameter Geometry**: Pre-flight analysis, hyperparameter scaling (like the $\alpha$ sweep), and adapter geometry analysis.

By keeping these components centralized in "The Factory," any new domain (e.g., Financial, Astral, PostgreSQL) can be trained using state-of-the-art methodology by simply invoking the unified training script with a different dataset.
