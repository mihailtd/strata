# Memory & Hardware State

Runtime VRAM residency, memory stability, and pointer integrity mechanisms: what stays on the device between requests, and how state is preserved or replayed.

| Submodule | Tier | Key Contribution |
| :--- | :---: | :--- |
| **[`factor_residency/`](factor_residency/)** | **⭐ Standard** / **🔥 Applied** | **Factor-Based VRAM Residency**: Keeps low-rank factor matrices ($U, V$) resident in standby memory (42.5 MB/expert), achieving a **200.6x memory reduction** and enabling **up to 216 concurrent domain experts** on a single 24GB GPU. |
| **[`pristine_state_buffer/`](pristine_state_buffer/)** | **🔥 Applied Practice** | **Zero-Drift Pristine Buffer ($W_0$)**: Transactional Memory State Checkpointing applied to LLM expert swapping, achieving bit-exact $L_\infty = 0.00$ drift across millions of hot-swaps. |
| **[`cuda_graph/`](cuda_graph/)** | **⭐ Standard** / **🔥 Applied** | **Pointer-Stable CUDA Graph Compatibility**: In-place low-rank weight mutation preserves static VRAM pointers (`data_ptr()`), allowing CUDA Graph replay across adapter swaps without re-capture penalty. |
| **[`zero_recapture_swapping/`](zero_recapture_swapping/)** | **🔥 Applied Practice** | **Zero-Recapture Swapping Synergy**: Proves single-capture CUDA Graph execution (`capture_count == 1`, 0.0 ms recapture overhead) across multi-turn expert swaps with 0 bytes transient VRAM churn. |
