# 🚀 Times-Above-Chance SVD Subspace Probe

This module contains tools for batch-processing and mapping the geometric orthogonality of multiple adapters using the **Times-Above-Chance** metric.

## The Metric: Why Raw Percentages Fail
If you project a small adapter onto a 32-dimensional subspace of a massive billion-parameter weight matrix, you will mathematically capture a tiny fraction of energy by pure chance (e.g., `0.0005%`). 

Because of this, standard overlap tests that look for "retention < 1%" are useless—they will always pass. If you test two identical adapters, two related adapters, and two completely different adapters, a raw percentage test will incorrectly classify all three pairs as "orthogonal".

**Times-Above-Chance** fixes this. Instead of a raw percentage, it divides the measured retention by the expected statistical floor:
- Same adapter vs itself: ~26,000x chance (Ceiling)
- Astral vs Astral: ~7.15x chance (Strong overlap)
- Astral vs Financial: ~1.28x chance (Statistically orthogonal)

## Scripts in this Module
- **`build_orthogonality_map.py`**: Runs this metric in a massive loop across every trained domain adapter in the repository to generate a cross-task subspace orthogonality heatmap. It acts as the bulk-analysis engine for the `Pre-Flight SVD Subspace Probe`.
