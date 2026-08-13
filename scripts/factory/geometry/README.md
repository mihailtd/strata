# Parameter Geometry

This module handles the mathematical characteristics and geometric layout of the adapter weights in high-dimensional space. 

## Key Responsibilities
- **Hyperparameter Scale Calibration ($\alpha$ Sweep):** Determining the precise scaling multiplier required to ensure that low-rank adapters survive floating-point truncation when they are folded into native 16-bit base weights.
- **Pre-Flight Checks:** Analyzing domains mathematically *before* training to determine if advanced shared-basis architectures are viable, or if the domains are entirely orthogonal and should just be stacked independently.
