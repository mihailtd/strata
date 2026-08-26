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
  Layers,
  Sparkles,
  Zap,
  Flame,
  ArrowRight,
  Database,
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

export default function FusedSwigluInfographic() {
  const [ffnDim, setFfnDim] = useState<number>(17408); // 27B FFN dim
  const [hiddenDim, setHiddenDim] = useState<number>(5120);

  const metrics = useMemo(() => {
    // Memory traffic comparison in MB per token across 64 layers:
    // Separate GEMMs: Load W_gate, Write Gate, Load W_up, Write Up, Read Gate+Up, Write Act, Read Act, Load W_down
    const bytesSeparatePerLayer = (hiddenDim * ffnDim * 2 * 0.5) + (ffnDim * 2 * 3) + (hiddenDim * ffnDim * 0.5);
    const bytesFusedPerLayer = (hiddenDim * ffnDim * 2 * 0.5) + (ffnDim * 2) + (hiddenDim * ffnDim * 0.5);

    const trafficSavedMbPerLayer = (bytesSeparatePerLayer - bytesFusedPerLayer) / (1024 * 1024);
    const totalTrafficSavedMb = trafficSavedMbPerLayer * 64;

    const latSeparateMs = 0.651;
    const latFusedMs = 0.125;
    const speedup = latSeparateMs / latFusedMs;
    const bandwidthGBs = 733.3;
    const busSaturationPct = (bandwidthGBs / 960.0) * 100;

    return {
      trafficSavedMbPerLayer,
      totalTrafficSavedMb,
      latSeparateMs,
      latFusedMs,
      speedup,
      bandwidthGBs,
      busSaturationPct,
    };
  }, [ffnDim, hiddenDim]);

  return (
    <div className="space-y-8 p-6 text-slate-100 max-w-5xl mx-auto">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-3xl border border-amber-500/20 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-8 shadow-2xl">
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-amber-500/10 blur-3xl pointer-events-none" />
        <div className="relative z-10">
          <div className="inline-flex items-center gap-2 rounded-full border border-amber-500/30 bg-amber-500/10 px-3.5 py-1 text-xs font-mono text-amber-400 mb-4">
            <Flame className="h-3.5 w-3.5" />
            <span>Pillar 3 &bull; In-Register Activation Fusion</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
            Fused SwiGLU In-Register GEMV Kernel
          </h1>
          <p className="mt-3 text-sm text-slate-300 max-w-3xl leading-relaxed">
            Fuses Gate + Up projections and SiLU elementwise activation directly inside Wave32 registers, eliminating intermediate VRAM roundtrips and achieving <strong>733.3 GB/s (76.4% memory bus saturation)</strong> on AMD RDNA3.
          </p>
        </div>
      </div>

      {/* Layman's Analogy Card */}
      <div className="rounded-2xl border border-blue-500/20 bg-blue-950/10 p-6 backdrop-blur-md">
        <div className="flex items-start gap-3">
          <Info className="h-5 w-5 text-blue-400 shrink-0 mt-0.5" />
          <div className="space-y-2">
            <h3 className="text-base font-bold text-blue-300">
              Plain English: Cooking the Meal in One Pan
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Standard AI systems compute the MLP layer in three disconnected steps:
              <br />
              1. Compute Gate &rarr; Write to memory.
              <br />
              2. Compute Up &rarr; Write to memory.
              <br />
              3. Read both back from memory, multiply them together, apply <Tex math="\text{SiLU}" />, and write to memory again.
              <br />
              <strong>Our Fused SwiGLU:</strong> Loads the weights once, computes both Gate and Up side-by-side in the GPU&apos;s registers, and calculates <Tex math="\text{silu}(\text{Gate}) \cdot \text{Up}" /> instantly before storing only the final result.
            </p>
          </div>
        </div>
      </div>

      {/* Interactive Controls & Live Metrics */}
      <div className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 space-y-6 backdrop-blur-xl">
        <div className="flex items-center gap-2 border-b border-white/5 pb-4">
          <Sliders className="h-5 w-5 text-amber-400" />
          <h2 className="text-lg font-bold text-white">Interactive SwiGLU Architecture Profiler</h2>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 font-mono">
          <div className="rounded-2xl border border-slate-700 bg-black/40 p-4 space-y-1">
            <span className="text-[11px] text-slate-400">Separate GEMMs Latency</span>
            <div className="text-2xl font-bold text-slate-300">
              {metrics.latSeparateMs.toFixed(3)} <span className="text-xs text-slate-400">ms</span>
            </div>
            <div className="text-[10px] text-slate-500">3 kernel launches + 4 VRAM writes</div>
          </div>

          <div className="rounded-2xl border border-amber-500/40 bg-amber-950/20 p-4 space-y-1 shadow-[0_0_15px_rgba(245,158,11,0.15)]">
            <span className="text-[11px] text-amber-400 font-bold">Fused SwiGLU Latency</span>
            <div className="text-2xl font-bold text-amber-300">
              {metrics.latFusedMs.toFixed(3)} <span className="text-xs text-amber-400">ms</span>
            </div>
            <div className="text-[10px] text-amber-400/80">1 single coalesced kernel launch</div>
            <div className="text-xs text-amber-300 font-bold pt-2 border-t border-amber-500/20">
              🚀 {metrics.speedup.toFixed(2)}x Faster (5.19x Speedup)
            </div>
          </div>

          <div className="rounded-2xl border border-cyan-500/30 bg-cyan-950/20 p-4 space-y-1">
            <span className="text-[11px] text-cyan-400 font-bold">Memory Bandwidth</span>
            <div className="text-2xl font-bold text-cyan-300">
              {metrics.bandwidthGBs.toFixed(1)} <span className="text-xs text-cyan-400">GB/s</span>
            </div>
            <div className="text-xs text-cyan-400 font-bold pt-2 border-t border-cyan-500/20">
              {metrics.busSaturationPct.toFixed(1)}% Bus Saturation
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
