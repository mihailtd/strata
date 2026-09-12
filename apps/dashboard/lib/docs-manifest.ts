export interface DocItem {
  id: string;
  title: string;
  category:
    | "1. Factory & Geometry"
    | "2. Runtime & Speculative"
    | "3. Kernel & Hardware (RDNA3)"
    | "Core Architecture & Decisions";
  relativePath: string; // relative to repo root
  description?: string;
}

export const DOCS_MANIFEST: DocItem[] = [
  // --------------------------------------------------------------------------
  // Pillar 3: Kernel & Hardware Innovation (RDNA3)
  // --------------------------------------------------------------------------
  {
    id: "swe-bench-27b",
    title: "Autonomous SWE-Bench 27B Benchmark (Whitepaper)",
    category: "3. Kernel & Hardware (RDNA3)",
    relativePath: "docs/SWE_BENCH_AUTONOMOUS_27B_EVALUATION.md",
    description:
      "Multi-file coding evaluation proving Specialist LoRA + Speculation is 8.67x faster and +50% smarter (Pass@1 100%).",
  },
  {
    id: "bench-swe-bench",
    title: "Autonomous SWE-Bench Benchmark (Interactive)",
    category: "3. Kernel & Hardware (RDNA3)",
    relativePath: "benchmarks/swe_bench/README.md",
    description:
      "Interactive head-to-head comparison across 6 software engineering tasks with isolated pytest sandboxes.",
  },
  {
    id: "adaptive-tree-27b-native",
    title: "Native 64-Layer 27B & Adaptive Tree Engine (Whitepaper)",
    category: "3. Kernel & Hardware (RDNA3)",
    relativePath: "docs/ADAPTIVE_TREE_27B_NATIVE_ENGINE.md",
    description:
      "Pure native 64-layer Triton 27B architecture with Shannon entropy dynamic speculation (>220 tok/s).",
  },
  {
    id: "bench-kernel-adaptive-tree",
    title: "Entropy-Adaptive Dynamic Tree Speculation (Interactive)",
    category: "3. Kernel & Hardware (RDNA3)",
    relativePath: "benchmarks/kernel/adaptive_tree_speculation/README.md",
    description:
      "Dynamic topology modulation (Deep Burst vs 2x2 Tree vs Guard) scaling up to 221 tok/s.",
  },
  {
    id: "w4a16-beating-ollama",
    title: "Beating Ollama by 4x on RDNA3 GPU (Whitepaper)",
    category: "3. Kernel & Hardware (RDNA3)",
    relativePath: "docs/W4A16_RDNA3_BEATING_OLLAMA.md",
    description:
      "Full engineering whitepaper documenting 128-bit memory coalescing, fused SwiGLU, and tree speculation on RX 7900 XTX.",
  },
  {
    id: "bench-kernel-gemv-m1",
    title: "128-Bit Coalesced W4A16 GEMV Kernel (Interactive)",
    category: "3. Kernel & Hardware (RDNA3)",
    relativePath: "benchmarks/kernel/w4a16_gemv_m1/README.md",
    description:
      "620.4 GB/s GDDR6 bus saturation via 128-bit int32x4 vector transactions and Wave32 VGPR register bit-shifts.",
  },
  {
    id: "bench-kernel-fused-swiglu",
    title: "Fused SwiGLU In-Register GEMV Kernel (Interactive)",
    category: "3. Kernel & Hardware (RDNA3)",
    relativePath: "benchmarks/kernel/fused_swiglu/README.md",
    description:
      "733.3 GB/s (76.4% bus saturation) in-register SiLU activation eliminating 4 intermediate VRAM roundtrips (5.19x speedup).",
  },
  {
    id: "bench-kernel-tree-speculation",
    title: "Tree-Based Speculative Decoder 2x2 (Interactive)",
    category: "3. Kernel & Hardware (RDNA3)",
    relativePath: "benchmarks/kernel/tree_speculation/README.md",
    description:
      "2x2 branching speculative tree generating 3.48 accepted tokens per cycle, achieving 202.3 tok/s (4.15x Ollama speed).",
  },
  {
    id: "bench-kernel-fused-qkv-rope",
    title: "Fused QKV + RoPE Wave32 Projection",
    category: "3. Kernel & Hardware (RDNA3)",
    relativePath: "benchmarks/kernel/fused_qkv_rope/README.md",
    description: "0.058 ms per layer in-register complex exponential rotary position embedding.",
  },
  {
    id: "bench-kernel-outlier-protection",
    title: "Outlier-Protected W4A16 (Zero Precision Loss)",
    category: "3. Kernel & Hardware (RDNA3)",
    relativePath: "benchmarks/kernel/outlier_protection/README.md",
    description:
      "Top-16 BF16 channel slice isolation reducing INT4 quantization error by 6.4x with zero latency penalty.",
  },
  {
    id: "bench-kernel-root",
    title: "Kernel & Hardware Pillar Overview",
    category: "3. Kernel & Hardware (RDNA3)",
    relativePath: "benchmarks/kernel/README.md",
    description:
      "Comprehensive overview of physical hardware saturation, micro-kernels, and memory layout on AMD RDNA3.",
  },

  // --------------------------------------------------------------------------
  // Pillar 2: Runtime Engine & Speculative Dynamics
  // --------------------------------------------------------------------------
  {
    id: "the-runtime",
    title: "THE RUNTIME (Surgical Stacking & Dynamics)",
    category: "2. Runtime & Speculative",
    relativePath: "docs/THE_RUNTIME_SURGICAL_STACKING_AND_TEMPORAL_DYNAMICS.md",
    description:
      "In-place weight-folding, transactional pristine state buffer, and low-latency serving dynamics.",
  },
  {
    id: "surgical-stacking",
    title: "Surgical Multi-Expert Stacking (Interactive)",
    category: "2. Runtime & Speculative",
    relativePath: "docs/SURGICAL_MULTI_EXPERT_STACKING.md",
    description:
      "Interactive infographic for LV-GLasso conflict routing + POET channel notching in activate_many().",
  },
  {
    id: "bench-weibull-hazard",
    title: "Weibull Hazard & Bollinger Gating (Interactive)",
    category: "2. Runtime & Speculative",
    relativePath: "benchmarks/runtime/speculative/weibull_hazard_gating/README.md",
    description:
      "Discrete Weibull wear-out hazard & rolling Bollinger Bands volatility gating for speculative draft truncation.",
  },
  {
    id: "bench-cut-set-reliability",
    title: "Minimal Cut-Set DAG Reliability (Interactive)",
    category: "2. Runtime & Speculative",
    relativePath: "benchmarks/agentic/CUT_SET_README.md",
    description: "Chapter 6 Minimal Cut Sets & k-out-of-n speculative hedging for agent tool DAGs.",
  },
  {
    id: "thinking-supervisor",
    title: "Dynamic Thinking Supervisor & Renko Exit",
    category: "2. Runtime & Speculative",
    relativePath: "docs/THINKING_SUPERVISOR.md",
    description:
      "Entropy-bounded early exit saving 70-90% token overhead when reasoning converges.",
  },
  {
    id: "bench-speculative-root",
    title: "Runtime Engine Overview",
    category: "2. Runtime & Speculative",
    relativePath: "benchmarks/runtime/README.md",
    description:
      "Full overview of speculative graph decoding, S_t state handoff, and VRAM residency.",
  },
  {
    id: "bench-runtime-memory",
    title: "Runtime Memory & Factor Residency",
    category: "2. Runtime & Speculative",
    relativePath: "benchmarks/runtime/memory/README.md",
    description:
      "Factor-based standby residency achieving 200.6x memory reduction (200+ resident LoRAs).",
  },

  // --------------------------------------------------------------------------
  // Pillar 1: Factory & Geometric Fine-Tuning
  // --------------------------------------------------------------------------
  {
    id: "the-factory",
    title: "THE FACTORY (Fine-Tuning & Geometry)",
    category: "1. Factory & Geometry",
    relativePath: "docs/THE_FACTORY_FINE_TUNING_AND_GEOMETRY.md",
    description: "Adapter factory training, geometric stopping, and Ledoit-Wolf precision.",
  },
  {
    id: "ledoit-wolf-routing",
    title: "Ledoit-Wolf & Riemannian Routing (Interactive)",
    category: "1. Factory & Geometry",
    relativePath: "docs/LEDOIT_WOLF_RIEMANNIAN_ROUTING.md",
    description:
      "Interactive infographic & theorems for Ledoit-Wolf shrinkage and Riemannian manifold expert routing.",
  },
  {
    id: "bench-ssi-quant",
    title: "SSI Quantization Calibration (Interactive)",
    category: "1. Factory & Geometry",
    relativePath: "benchmarks/factory/quantization/SSI_README.md",
    description:
      "Closed-form Stress-Strength Interference optimal clipping for low-bit quantization scaling.",
  },
  {
    id: "bench-riemannian-metric",
    title: "Riemannian AIRM Metric Benchmark",
    category: "1. Factory & Geometry",
    relativePath: "benchmarks/factory/geometry/riemannian_metric/README.md",
    description: "6x6 geodesic distance manifold and Ledoit-Wolf shrinkage.",
  },
  {
    id: "bench-alpha-calibration",
    title: "Dynamic Alpha Calibration Benchmark",
    category: "1. Factory & Geometry",
    relativePath: "benchmarks/factory/geometry/dynamic_alpha_calibration/README.md",
    description: "IEEE 754 precision floor and perturbation Goldilocks band (alpha=128).",
  },
  {
    id: "bench-poet-causal-graph",
    title: "POET Tool Causal DAG Benchmark",
    category: "1. Factory & Geometry",
    relativePath: "benchmarks/factory/agentic/poet_tool_causal_graph/README.md",
    description: "NOTEARS continuous acyclicity and tool-to-expert transitions.",
  },
  {
    id: "bench-factory-root",
    title: "Factory Benchmarks Overview",
    category: "1. Factory & Geometry",
    relativePath: "benchmarks/factory/README.md",
    description: "Suite overview of training, calibration, and geometric evaluation.",
  },

  // --------------------------------------------------------------------------
  // Core Architecture & Decisions
  // --------------------------------------------------------------------------
  {
    id: "decisions",
    title: "DECISIONS.md (Architecture & Retirals)",
    category: "Core Architecture & Decisions",
    relativePath: "docs/DECISIONS.md",
    description:
      "Measurement-backed architecture decisions, retirements, and speculative replay benchmarks.",
  },
  {
    id: "research-roadmap",
    title: "RESEARCH_ROADMAP.md",
    category: "Core Architecture & Decisions",
    relativePath: "docs/RESEARCH_ROADMAP.md",
    description: "Comprehensive research roadmap and milestone deliverables.",
  },
  {
    id: "why-experts",
    title: "WHY_EXPERTS.md",
    category: "Core Architecture & Decisions",
    relativePath: "docs/WHY_EXPERTS.md",
    description: "Theoretical rationale for micro-expert dynamic morphing.",
  },
  {
    id: "corpus-design",
    title: "CORPUS_DESIGN.md",
    category: "Core Architecture & Decisions",
    relativePath: "docs/CORPUS_DESIGN.md",
    description: "Domain dataset curation and synthetic generation methodology.",
  },
];
