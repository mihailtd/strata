# Speculative Decoding Engine

This module handles the logic, benchmarking, and architectural auditing for the Multi-Token Prediction (MTP) speculative decoding loop on hybrid recurrent architectures.

| Submodule | Tier | Key Contribution |
| :--- | :---: | :--- |
| **[`mtp_speculative/`](mtp_speculative/)** | **🔥 Applied Practice** | **Native MTP Speculative Engine (K-Sweep Frontier)**: Evaluates greedy speculative decoding with 52.5 MB recurrent state snapshot/restore controls, unlocking **2.20x net speedup at $K=6$** on full 256-token horizons. |
| **[`speculation_matrix/`](speculation_matrix/)** | **⭐ Standard** / **🔥 Applied** | **3x3 Speculation Matrix & Dynamic Router**: Rigorous 3x3 grid audit (180 runs, 3 interleaved repeats) measuring empirical acceptance ($\tau$) and net speedup across all domain adapters. |
| **[`mtp_head_folding/`](mtp_head_folding/)** | **Closed Boundary ❌** | **Draft Head Domain Adaptation Boundary**: Documents why adapting the MTP draft head (both via hard SFT and soft KL distillation) collapses draft acceptance, definitively proving that the pre-trained shipped MTP head must remain pristine. |
