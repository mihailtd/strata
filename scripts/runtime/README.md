# The Runtime Engine

This directory contains the core inference machinery, speculative decoding mechanisms, and hardware-level optimizations that make the compiled models run extremely fast in production.

## Core Concepts
- **Speculative Engine**: Multi-token prediction (MTP) loop, state rollback mechanisms, and token acceptance ($\tau$) algorithms.
- **Weight Folding**: In-place VRAM mutation techniques (e.g. folding LoRA weights natively into base tensors for zero-cost inference).
- **Memory & CUDA**: Pointer-stable CUDA graph compatibility and pristine state buffering.
- **Performance**: High-level latency, throughput, and hardware benchmarking scripts.
