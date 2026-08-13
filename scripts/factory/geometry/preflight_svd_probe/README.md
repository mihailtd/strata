# ⭐ Pre-Flight SVD Subspace Probe

This tool is used as a fast, sub-second mathematical check to verify subspace orthogonality and cross-task basis transferability between two small, cheaply-trained LLM adapters.

## Workflow Integration
You run this probe **before** initiating expensive multi-task training (like Shared-Basis Tucker Factorization). 
If this probe proves that the two domains (e.g., Financial and Astral) are mathematically orthogonal (around `1.10x` chance), it confirms that their feature spaces don't overlap. In that case, you can safely skip complex shared-basis architectures and just train them as standard LoRAs that can be stacked natively without interfering with each other!

## Related Modules
- **[🚀 Times-Above-Chance SVD Subspace Probe](../times_above_chance/)**: The Pre-Flight Probe calculates a specific metric ("Times Above Chance") rather than a raw percentage. To see how this metric is used at scale to build a cross-task heatmap for the entire factory, see the `times_above_chance` module.
