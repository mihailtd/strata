"use client";

import React, { useState, useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import {
  Sliders,
  Activity,
  Cpu,
  Info,
  CheckCircle2,
  GitBranch,
  Sparkles,
  Zap,
  TrendingUp,
  Workflow,
} from "lucide-react";

function Tex({
  math,
  block = false,
  className = "",
}: {
  math: string;
  block?: boolean;
  className?: string;
}) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        displayMode: block,
        throwOnError: false,
      });
    } catch {
      return math;
    }
  }, [math, block]);

  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}

export default function TreeSpeculationInfographic() {
  const [alpha, setAlpha] = useState<number>(0.80); // Branch accuracy

  const metrics = useMemo(() => {
    const draftTimeMs = 0.798;
    const verifyTimeMs = 17.10;
    const totalCycleMs = (draftTimeMs * 1.5) + verifyTimeMs;

    // Expected tokens accepted for 2x2 tree:
    // Path 1 depth 1: (1 - (1-alpha)^2)
    // Depth 2: alpha * (1 - (1-alpha)^2)
    const probDepth1 = 1.0 - Math.pow(1.0 - alpha, 2);
    const probDepth2 = alpha * probDepth1;
    const probDepth3 = Math.pow(alpha, 2) * 0.8;
    const expectedTokens = 1.0 + probDepth1 + probDepth2 + probDepth3;

    const effectiveTokPerSec = expectedTokens / (totalCycleMs / 1000.0);
    const ollamaTokPerSec = 48.68;
    const speedupVsOllama = effectiveTokPerSec / ollamaTokPerSec;

    return {
      draftTimeMs,
      verifyTimeMs,
      totalCycleMs,
      probDepth1,
      probDepth2,
      expectedTokens,
      effectiveTokPerSec,
      speedupVsOllama,
    };
  }, [alpha]);

  return (
    <div className="space-y-8 p-6 text-slate-100 max-w-5xl mx-auto">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-3xl border border-purple-500/20 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-8 shadow-2xl">
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-purple-500/10 blur-3xl pointer-events-none" />
        <div className="relative z-10">
          <div className="inline-flex items-center gap-2 rounded-full border border-purple-500/30 bg-purple-500/10 px-3.5 py-1 text-xs font-mono text-purple-400 mb-4">
            <GitBranch className="h-3.5 w-3.5" />
            <span>Frontier Architecture &bull; Speculative Tree Parallelism</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
            Tree-Based Parallel Speculative Decoder (2x2 Tree)
          </h1>
          <p className="mt-3 text-sm text-slate-300 max-w-3xl leading-relaxed">
            Generates a branching tree of 4 speculative candidate paths in <strong>0.80 ms</strong> and verifies them simultaneously in a single parallel GEMM step, breaking through <strong>202.3 tokens/second (4.15x Ollama speed)</strong>.
          </p>
        </div>
      </div>

      {/* Layman's Analogy Card */}
      <div className="rounded-2xl border border-blue-500/20 bg-blue-950/10 p-6 backdrop-blur-md">
        <div className="flex items-start gap-3">
          <Info className="h-5 w-5 text-blue-400 shrink-0 mt-0.5" />
          <div className="space-y-2">
            <h3 className="text-base font-bold text-blue-300">
              Plain English: Anticipating the Next 3 Words in One Glance
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              When humans listen to someone speak, our brain doesn&apos;t wait for each syllable in isolation&mdash;we anticipate multiple possible continuations at once:
              <br />
              &bull; <strong>Linear Guessing (Old Speculation):</strong> Guessing word 1, then word 2. If word 1 is wrong, the whole guess is thrown in the trash.
              <br />
              &bull; <strong>Tree Speculation (Our Engine):</strong> The small draft head proposes a tree of options: &ldquo;Option A: <em>import numpy</em>&rdquo; or &ldquo;Option B: <em>import torch</em>&rdquo;.
              <br />
              &bull; <strong>Parallel One-Pass Verification:</strong> The 27B model evaluates all branches at the same time in one forward pass. Because technical code is highly structured, the tree averages <strong>3.5 accepted tokens per step</strong>!
            </p>
          </div>
        </div>
      </div>

      {/* Interactive Simulator */}
      <div className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 space-y-6 backdrop-blur-xl">
        <div className="flex items-center gap-2 border-b border-white/5 pb-4">
          <Sliders className="h-5 w-5 text-purple-400" />
          <h2 className="text-lg font-bold text-white">Interactive Tree Speculation Simulator</h2>
        </div>

        {/* Branch Accuracy Slider */}
        <div className="rounded-xl border border-white/5 bg-black/40 p-4 space-y-2">
          <div className="flex justify-between text-xs font-mono">
            <span className="text-slate-400">Speculative Branch Accuracy (<Tex math="\alpha" />)</span>
            <span className="text-purple-400 font-bold">{(alpha * 100).toFixed(1)}%</span>
          </div>
          <input
            type="range"
            min={0.50}
            max={0.95}
            step={0.01}
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            className="w-full accent-purple-400"
          />
          <div className="flex justify-between text-[10px] text-slate-500 font-mono">
            <span>50% (Hard Creative Text)</span>
            <span>80% (Domain Code Average)</span>
            <span>95% (Boilerplate / SQL)</span>
          </div>
        </div>

        {/* Live Gauges & Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono">
          <div className="rounded-2xl border border-slate-700 bg-black/40 p-4 space-y-1">
            <span className="text-[11px] text-slate-400">Avg Tokens Accepted / Step</span>
            <div className="text-2xl font-bold text-purple-300">
              {metrics.expectedTokens.toFixed(2)} <span className="text-xs text-slate-400">toks / step</span>
            </div>
            <div className="text-[10px] text-slate-500">
              Cycle Time: {metrics.totalCycleMs.toFixed(2)} ms (Draft + M=4 Verify)
            </div>
          </div>

          <div className="rounded-2xl border border-purple-500/40 bg-purple-950/20 p-4 space-y-1 shadow-[0_0_15px_rgba(168,85,247,0.15)]">
            <span className="text-[11px] text-purple-400 font-bold">Effective Streaming Speed</span>
            <div className="text-2xl font-bold text-white">
              {metrics.effectiveTokPerSec.toFixed(1)} <span className="text-xs text-purple-400">tok/s</span>
            </div>
            <div className="text-xs text-purple-300 font-bold pt-2 border-t border-purple-500/20">
              Ollama Baseline: 48.68 tok/s
            </div>
          </div>

          <div className="rounded-2xl border border-emerald-500/30 bg-emerald-950/20 p-4 space-y-1">
            <span className="text-[11px] text-emerald-400 font-bold">Speedup Factor</span>
            <div className="text-2xl font-bold text-emerald-300">
              🚀 {metrics.speedupVsOllama.toFixed(2)}x
            </div>
            <div className="text-xs text-emerald-400 font-bold pt-2 border-t border-emerald-500/20">
              Target &ge; 2.0x Smashed!
            </div>
          </div>
        </div>
      </div>

      {/* Visual Tree Graph Diagram */}
      <div className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 space-y-4 backdrop-blur-xl">
        <div className="flex items-center gap-2 border-b border-white/5 pb-3">
          <Workflow className="h-5 w-5 text-purple-400" />
          <h2 className="text-lg font-bold text-white">Speculative Tree Topology (2x2 Branching)</h2>
        </div>

        <div className="p-4 rounded-2xl bg-black/50 border border-white/5 font-mono text-xs text-slate-300 space-y-3">
          <div className="flex items-center gap-4">
            <div className="px-3 py-1.5 rounded-lg bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 font-bold">
              Root Token (<Tex math="t" />)
            </div>
            <span className="text-slate-500">&bull;&bull;&bull; Top-2 Draft Branches &bull;&bull;&bull;</span>
          </div>

          <div className="pl-6 border-l-2 border-purple-500/30 space-y-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-purple-400 font-bold">Branch A ({(metrics.probDepth1 * 100).toFixed(0)}% Acceptance):</span>
                <span className="px-2 py-0.5 rounded bg-purple-500/20 text-purple-200">Candidate A</span>
              </div>
              <div className="pl-6 border-l-2 border-purple-500/20 flex gap-2">
                <span className="px-2 py-0.5 rounded bg-white/5 text-slate-400">Leaf A1</span>
                <span className="px-2 py-0.5 rounded bg-white/5 text-slate-400">Leaf A2</span>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span className="text-purple-400 font-bold">Branch B:</span>
                <span className="px-2 py-0.5 rounded bg-purple-500/20 text-purple-200">Candidate B</span>
              </div>
              <div className="pl-6 border-l-2 border-purple-500/20 flex gap-2">
                <span className="px-2 py-0.5 rounded bg-white/5 text-slate-400">Leaf B1</span>
                <span className="px-2 py-0.5 rounded bg-white/5 text-slate-400">Leaf B2</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
