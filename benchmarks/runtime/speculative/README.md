# Speculative Decoding Engine

This module handles the logic, benchmarking, and architectural auditing for the Multi-Token Prediction (MTP) speculative decoding loop on hybrid recurrent architectures.

| Submodule | Tier | Key Contribution |
| :--- | :---: | :--- |
| **[`mtp_speculative/`](mtp_speculative/)** | **🔥 Applied Practice** | **Native MTP Speculative Engine (K-Sweep Frontier)**: Evaluates greedy speculative decoding with 52.5 MB recurrent state snapshot/restore controls, unlocking **2.20x net speedup at $K=6$** on full 256-token horizons. |
| **[`speculation_matrix/`](speculation_matrix/)** | **⭐ Standard** / **🔥 Applied** | **3x3 Speculation Matrix & Dynamic Router**: Rigorous 3x3 grid audit (180 runs, 3 interleaved repeats) measuring empirical acceptance ($\tau$) and net speedup across all domain adapters. |
| **[`mtp_head_folding/`](mtp_head_folding/)** | **Closed Boundary ❌** | **Draft Head Domain Adaptation Boundary**: Documents why adapting the MTP draft head (both via hard SFT and soft KL distillation) collapses draft acceptance, definitively proving that the pre-trained shipped MTP head must remain pristine. |
| **[`speculation_quality/`](speculation_quality/)** | **⭐ Standard** (validity check) | **Does speculation cost quality? No.** Three arms on the canonical eval sets, n=100 paired triples: speculative − autoregressive = **+0.19pp, CI [−0.77, +1.17]**. A `forced_reject` control — identical machinery, every draft rejected — diverges from plain decode on **28%** of prompts against speculation's **33%**, so the chunked kernel accounts for nearly all divergence; 30 of 33 diverging prompts scored *identically*. Closes the last open doubt on the 2.20×. Also documents a **retracted run** whose EOS-overrun bug manufactured a "+5.71pp significant *improvement*" — read it before writing another arm-vs-arm quality benchmark. |
