export interface DocItem {
  id: string;
  title: string;
  category: "Core Architecture & Decisions" | "Benchmark Reports" | "Runtime & Speculative" | "Factory & Geometry";
  relativePath: string; // relative to repo root
  description?: string;
}

export const DOCS_MANIFEST: DocItem[] = [
  // Core Architecture
  {
    id: "ledoit-wolf-routing",
    title: "Ledoit-Wolf & Riemannian Routing (Interactive)",
    category: "Core Architecture & Decisions",
    relativePath: "docs/LEDOIT_WOLF_RIEMANNIAN_ROUTING.md",
    description: "Interactive infographic & theorems for Ledoit-Wolf shrinkage and Riemannian manifold expert routing.",
  },
  {
    id: "surgical-stacking",
    title: "Surgical Multi-Expert Stacking (Interactive)",
    category: "Core Architecture & Decisions",
    relativePath: "docs/SURGICAL_MULTI_EXPERT_STACKING.md",
    description: "Interactive infographic for LV-GLasso conflict routing + POET channel notching in activate_many().",
  },
  {
    id: "decisions",
    title: "DECISIONS.md (Architecture & Retirals)",
    category: "Core Architecture & Decisions",
    relativePath: "docs/DECISIONS.md",
    description: "Measurement-backed architecture decisions, retirements, and speculative replay benchmarks.",
  },
  {
    id: "the-runtime",
    title: "THE RUNTIME (Surgical Stacking & Dynamics)",
    category: "Core Architecture & Decisions",
    relativePath: "docs/THE_RUNTIME_SURGICAL_STACKING_AND_TEMPORAL_DYNAMICS.md",
    description: "In-place weight-folding and low-latency serving dynamics.",
  },
  {
    id: "the-factory",
    title: "THE FACTORY (Fine-Tuning & Geometry)",
    category: "Core Architecture & Decisions",
    relativePath: "docs/THE_FACTORY_FINE_TUNING_AND_GEOMETRY.md",
    description: "Adapter factory training, geometric stopping, and Ledoit-Wolf precision.",
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

  // Runtime & Speculative
  {
    id: "bench-speculative-root",
    title: "Speculative Engine Suite Overview",
    category: "Runtime & Speculative",
    relativePath: "benchmarks/runtime/speculative/README.md",
    description: "Full overview of speculative graph decoding and ring buffers.",
  },
  {
    id: "bench-poet-temporal",
    title: "POET Temporal Compression Benchmark",
    category: "Runtime & Speculative",
    relativePath: "benchmarks/runtime/speculative/poet_temporal_compression/README.md",
    description: "Dynamic factor compression on recurrent state trajectories.",
  },
  {
    id: "bench-live-speculative",
    title: "Live Speculative Engine Benchmark",
    category: "Runtime & Speculative",
    relativePath: "benchmarks/runtime/speculative/live_speculative_engine/README.md",
    description: "End-to-end token verification and speedup benchmarks.",
  },
  {
    id: "bench-cache-counter",
    title: "Cache Length Counter Pinning Benchmark",
    category: "Runtime & Speculative",
    relativePath: "benchmarks/runtime/speculative/cache_length_counter/README.md",
    description: "StaticCache cumulative_length graph-safety attribution.",
  },
  {
    id: "bench-graphed-draft",
    title: "Graphed Draft Head Benchmark",
    category: "Runtime & Speculative",
    relativePath: "benchmarks/runtime/speculative/graphed_draft_head/README.md",
    description: "MTP draft head latency and CUDA graph capture.",
  },
  {
    id: "bench-range-gate",
    title: "Range Speculative Gate Benchmark",
    category: "Runtime & Speculative",
    relativePath: "benchmarks/runtime/speculative/range_statistic_gating/README.md",
    description: "Single-pass O(1) register range statistics for speculative draft early-exit.",
  },
  {
    id: "bench-weibull-hazard",
    title: "Weibull Hazard & Bollinger Gating (Ch. 3 & 5)",
    category: "Runtime & Speculative",
    relativePath: "benchmarks/runtime/speculative/weibull_hazard_gating/README.md",
    description: "Discrete Weibull wear-out hazard & rolling Bollinger Bands volatility gating for speculative draft truncation.",
  },
  {
    id: "bench-runtime-memory",
    title: "Runtime Memory & Residency",
    category: "Runtime & Speculative",
    relativePath: "benchmarks/runtime/memory/README.md",
    description: "Zero-recapture VRAM residency and state management.",
  },

  // Factory & Geometry
  {
    id: "bench-factory-root",
    title: "Factory Benchmarks Overview",
    category: "Factory & Geometry",
    relativePath: "benchmarks/factory/README.md",
    description: "Suite overview of training, calibration, and evaluation.",
  },
  {
    id: "bench-pissa-assessment",
    title: "PiSSA & SVD Probe Assessment",
    category: "Factory & Geometry",
    relativePath: "benchmarks/factory/geometry/preflight_svd_probe/PISSA_ASSESSMENT.md",
    description: "Empirical refutation of PiSSA initialisation hypothesis.",
  },
  {
    id: "bench-alpha-calibration",
    title: "Dynamic Alpha Calibration Benchmark",
    category: "Factory & Geometry",
    relativePath: "benchmarks/factory/geometry/dynamic_alpha_calibration/README.md",
    description: "IEEE 754 precision floor and perturbation Goldilocks band.",
  },
  {
    id: "bench-poet-causal-graph",
    title: "POET Tool Causal DAG Benchmark",
    category: "Factory & Geometry",
    relativePath: "benchmarks/factory/agentic/poet_tool_causal_graph/README.md",
    description: "NOTEARS continuous acyclicity and tool-to-expert transitions.",
  },
  {
    id: "bench-riemannian-metric",
    title: "Riemannian AIRM Metric Benchmark",
    category: "Factory & Geometry",
    relativePath: "benchmarks/factory/geometry/riemannian_metric/README.md",
    description: "6x6 geodesic distance manifold and Ledoit-Wolf shrinkage.",
  },
  {
    id: "bench-poet-decomposition",
    title: "POET Decomposition & Pruning",
    category: "Factory & Geometry",
    relativePath: "benchmarks/factory/agentic/poet_decomposition/README.md",
    description: "Surgical neuron pruning and collision resolution.",
  },
  {
    id: "bench-leverage-quant",
    title: "Leverage Outlier Quantization (Ch. 21)",
    category: "Factory & Geometry",
    relativePath: "benchmarks/factory/quantization/README.md",
    description: "Single-pass diagnostic leverage projection for W4A16 outlier channel protection without grid search.",
  },
  {
    id: "bench-ssi-quant",
    title: "SSI Quantization Calibration (Ch. 8.5)",
    category: "Factory & Geometry",
    relativePath: "benchmarks/factory/quantization/SSI_README.md",
    description: "Closed-form Stress-Strength Interference optimal clipping for low-bit quantization scaling.",
  },
  {
    id: "bench-cut-set-reliability",
    title: "Minimal Cut-Set DAG Reliability (Ch. 6)",
    category: "Runtime & Speculative",
    relativePath: "benchmarks/agentic/CUT_SET_README.md",
    description: "Chapter 6 Minimal Cut Sets & k-out-of-n speculative hedging for agent tool DAGs.",
  },
];
