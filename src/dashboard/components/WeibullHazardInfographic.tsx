"use client";

import React, { useState, useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import {
  Zap,
  Activity,
  Sliders,
  ShieldCheck,
  TrendingDown,
  Info,
  CheckCircle2,
  Sparkles,
  Layers,
  ArrowRight,
  Cpu,
  Flame,
  Clock,
  Play,
  RotateCcw,
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

export default function WeibullHazardInfographic() {
  // Simulator State
  const [beta, setBeta] = useState<number>(2.2);
  const [eta, setEta] = useState<number>(4.0);
  const [gamma, setGamma] = useState<number>(0.6);
  const [tau0, setTau0] = useState<number>(3.5);

  // Dynamic Token Draft Simulator
  const [sampleTokens, setSampleTokens] = useState<number[]>([12.5, 9.8, 5.2, 4.3, 3.8, 2.1, 1.4, 0.9]);

  // Compute step-by-step wear-out values
  const stepMetrics = useMemo(() => {
    return Array.from({ length: 8 }, (_, i) => {
      const step = i + 1;
      const hazard = (beta / eta) * Math.pow(step / eta, beta - 1);
      const tauEff = tau0 * (1.0 + gamma * hazard);
      const sampleRange = sampleTokens[i] || 0;
      const willAbort = sampleRange < tauEff && i > 0;
      return {
        step,
        hazard,
        tauEff,
        sampleRange,
        willAbort,
      };
    });
  }, [beta, eta, gamma, tau0, sampleTokens]);

  const earlyExitStep = useMemo(() => {
    for (let i = 1; i < stepMetrics.length; i++) {
      if (stepMetrics[i].willAbort) return i + 1;
    }
    return 8;
  }, [stepMetrics]);

  const randomizeTokens = () => {
    // Generate realistic decaying confidence stream
    const newRanges = [
      12.0 + Math.random() * 4.0,
      8.0 + Math.random() * 3.0,
      4.5 + Math.random() * 2.0,
      3.8 + Math.random() * 2.0,
      2.5 + Math.random() * 2.0,
      1.8 + Math.random() * 1.5,
      1.0 + Math.random() * 1.2,
      0.5 + Math.random() * 1.0,
    ];
    setSampleTokens(newRanges);
  };

  return (
    <div className="space-y-8 p-6 text-slate-100 max-w-5xl mx-auto">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-3xl border border-cyan-500/20 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-8 shadow-2xl">
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl pointer-events-none" />
        <div className="relative z-10">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3.5 py-1 text-xs font-mono text-cyan-400 mb-4">
            <Flame className="h-3.5 w-3.5" />
            <span>Reliability Engineering &bull; Chapter 3</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
            Weibull Hazard Spatio-Temporal Speculative Gating
          </h1>
          <p className="mt-3 text-sm text-slate-300 max-w-3xl leading-relaxed">
            Eliminates speculative &ldquo;over-drafting collapse&rdquo; by parameterizing sequential autoregressive token acceptance as a discrete Weibull wear-out hazard process. Dynamically tightens logit margin thresholds as draft depth increases.
          </p>
        </div>
      </div>

      {/* Layman's Analogy Card */}
      <div className="rounded-2xl border border-amber-500/20 bg-amber-950/10 p-6 backdrop-blur-md">
        <div className="flex items-start gap-3">
          <Info className="h-5 w-5 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-2">
            <h3 className="text-base font-bold text-amber-300">
              Plain English: The &ldquo;Speculative Emergency Brake&rdquo;
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              Imagine driving down a road that gets progressively foggier with every mile. 
              <strong> Blind Speculation</strong> hits the accelerator and guesses 8 miles ahead regardless of fog, usually crashing around mile 4 and wasting enormous fuel recovering. 
              <strong> Weibull Hazard Gating</strong> monitors the fog: as elapsed miles accumulate, it raises the safety threshold. The instant conviction drops on token #4, it brakes immediately&mdash;saving GPU time and boosting generation speed from <strong>54.9 to 74.6 tok/s (+35.8%)</strong>.
            </p>
          </div>
        </div>
      </div>

      {/* Interactive Simulator Section */}
      <div className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 space-y-6 backdrop-blur-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/5 pb-4">
          <div className="flex items-center gap-2">
            <Sliders className="h-5 w-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white">Interactive Weibull Hazard Simulator</h2>
          </div>
          <button
            onClick={randomizeTokens}
            className="flex items-center gap-1.5 rounded-xl border border-cyan-500/30 bg-cyan-500/10 px-3.5 py-1.5 text-xs font-mono text-cyan-300 hover:bg-cyan-500/20 transition-all shadow-sm"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>Generate New Token Stream</span>
          </button>
        </div>

        {/* Sliders Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="rounded-xl border border-white/5 bg-black/40 p-4 space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">Shape Parameter (<Tex math="\beta" />)</span>
              <span className="text-cyan-400 font-bold">{beta.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={1.0}
              max={4.0}
              step={0.1}
              value={beta}
              onChange={(e) => setBeta(parseFloat(e.target.value))}
              className="w-full accent-cyan-400"
            />
            <p className="text-[10px] text-slate-500">&beta; &gt; 1 = Wear-out acceleration</p>
          </div>

          <div className="rounded-xl border border-white/5 bg-black/40 p-4 space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">Scale Horizon (<Tex math="\eta" />)</span>
              <span className="text-cyan-400 font-bold">{eta.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={1.5}
              max={8.0}
              step={0.5}
              value={eta}
              onChange={(e) => setEta(parseFloat(e.target.value))}
              className="w-full accent-cyan-400"
            />
            <p className="text-[10px] text-slate-500">Characteristic token lifetime</p>
          </div>

          <div className="rounded-xl border border-white/5 bg-black/40 p-4 space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">Hazard Weight (<Tex math="\gamma" />)</span>
              <span className="text-cyan-400 font-bold">{gamma.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={2.0}
              step={0.1}
              value={gamma}
              onChange={(e) => setGamma(parseFloat(e.target.value))}
              className="w-full accent-cyan-400"
            />
            <p className="text-[10px] text-slate-500">Dynamic penalty multiplier</p>
          </div>

          <div className="rounded-xl border border-white/5 bg-black/40 p-4 space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">Base Cutoff (<Tex math="\tau_0" />)</span>
              <span className="text-cyan-400 font-bold">{tau0.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={1.0}
              max={6.0}
              step={0.2}
              value={tau0}
              onChange={(e) => setTau0(parseFloat(e.target.value))}
              className="w-full accent-cyan-400"
            />
            <p className="text-[10px] text-slate-500">Baseline spatial logit spread</p>
          </div>
        </div>

        {/* Dynamic Horizon Bar Progression */}
        <div className="space-y-3 pt-2">
          <div className="flex justify-between items-center text-xs font-mono text-slate-400">
            <span>Draft Step Horizon Progression (<Tex math="k = 1 \dots 8" />)</span>
            <span className="text-emerald-400 font-bold">
              Early-Exit Triggered at Token #{earlyExitStep}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
            {stepMetrics.map((m) => {
              const isPruned = m.step >= earlyExitStep && earlyExitStep < 8;
              return (
                <div
                  key={m.step}
                  className={`rounded-2xl border p-3 flex flex-col justify-between transition-all ${
                    isPruned
                      ? "border-red-500/30 bg-red-950/20 opacity-60"
                      : "border-cyan-500/30 bg-cyan-950/20 shadow-[0_0_12px_rgba(0,242,255,0.1)]"
                  }`}
                >
                  <div className="flex justify-between items-center font-mono text-[11px]">
                    <span className="text-slate-400">Token #{m.step}</span>
                    {isPruned ? (
                      <span className="text-red-400 font-bold text-[10px]">PRUNED</span>
                    ) : (
                      <span className="text-emerald-400 font-bold text-[10px]">DRAFT</span>
                    )}
                  </div>

                  {/* Visual threshold gauge */}
                  <div className="my-2 space-y-1">
                    <div className="flex justify-between text-[10px] font-mono">
                      <span className="text-slate-500">Range <Tex math="R_8" /></span>
                      <span className={m.sampleRange >= m.tauEff ? "text-emerald-400" : "text-red-400"}>
                        {m.sampleRange.toFixed(1)}
                      </span>
                    </div>
                    <div className="w-full bg-black/60 rounded-full h-1.5 overflow-hidden">
                      <div
                        className={`h-full rounded-full ${
                          m.sampleRange >= m.tauEff ? "bg-emerald-400" : "bg-red-400"
                        }`}
                        style={{ width: `${Math.min(100, (m.sampleRange / 15.0) * 100)}%` }}
                      />
                    </div>
                  </div>

                  <div className="pt-2 border-t border-white/5 space-y-0.5 text-[10px] font-mono">
                    <div className="flex justify-between text-slate-400">
                      <span><Tex math="\tau_{\text{eff}}" /></span>
                      <span className="text-cyan-300 font-bold">{m.tauEff.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between text-slate-500">
                      <span><Tex math="h(k)" /></span>
                      <span>{m.hazard.toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* GPU Benchmark Results Table */}
      <div className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 space-y-4 backdrop-blur-xl">
        <div className="flex items-center gap-2 border-b border-white/5 pb-3">
          <Activity className="h-5 w-5 text-emerald-400" />
          <h2 className="text-lg font-bold text-white">Empirical GPU Telemetry (AMD Radeon RX 7900 XTX)</h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-white/10 text-slate-400">
                <th className="pb-3 px-3">Horizon (<Tex math="K" />)</th>
                <th className="pb-3 px-3">Arm A: Blind Fixed</th>
                <th className="pb-3 px-3">Arm B: Static Range</th>
                <th className="pb-3 px-3 text-cyan-300">Arm C: Weibull Hazard Gate</th>
                <th className="pb-3 px-3 text-emerald-400">Pruning %</th>
                <th className="pb-3 px-3 text-emerald-400">Net Speedup</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-slate-200">
              <tr>
                <td className="py-3 px-3 font-bold text-white"><Tex math="K = 4" /></td>
                <td className="py-3 px-3 text-slate-400">72.70 tok/s</td>
                <td className="py-3 px-3 text-slate-400">72.70 tok/s</td>
                <td className="py-3 px-3 text-cyan-300 font-bold">75.87 tok/s</td>
                <td className="py-3 px-3 text-emerald-400">12.9%</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">+4.4% ⚡</td>
              </tr>
              <tr>
                <td className="py-3 px-3 font-bold text-white"><Tex math="K = 6" /></td>
                <td className="py-3 px-3 text-slate-400">62.07 tok/s</td>
                <td className="py-3 px-3 text-slate-400">62.07 tok/s</td>
                <td className="py-3 px-3 text-cyan-300 font-bold">74.53 tok/s</td>
                <td className="py-3 px-3 text-emerald-400">39.9%</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">+20.1% ⚡</td>
              </tr>
              <tr className="bg-cyan-500/5">
                <td className="py-3 px-3 font-bold text-white"><Tex math="K = 8" /></td>
                <td className="py-3 px-3 text-slate-400">54.93 tok/s</td>
                <td className="py-3 px-3 text-slate-400">54.93 tok/s</td>
                <td className="py-3 px-3 text-cyan-300 font-bold">74.60 tok/s</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">53.8%</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">+35.8% ⚡</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Mathematical Rigor Card */}
      <div className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 space-y-4 backdrop-blur-xl">
        <div className="flex items-center gap-2 border-b border-white/5 pb-3">
          <Cpu className="h-5 w-5 text-purple-400" />
          <h2 className="text-lg font-bold text-white">Mathematical Formulation</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
          <div className="rounded-2xl border border-white/5 bg-black/40 p-4 space-y-2">
            <h4 className="text-purple-300 font-bold">1. Weibull Wear-Out Hazard Function</h4>
            <p className="text-slate-400">
              For step <Tex math="k \in [0, K-1]" /> (1-indexed <Tex math="t = k+1" />):
            </p>
            <div className="py-2 text-center text-white">
              <Tex math="h(k; \beta, \eta) = \frac{\beta}{\eta} \left(\frac{k + 1}{\eta}\right)^{\beta - 1}, \quad \beta > 1" block />
            </div>
            <p className="text-slate-500 text-[11px]">
              When <Tex math="\beta > 1" />, the instantaneous hazard rate grows monotonically with draft depth.
            </p>
          </div>

          <div className="rounded-2xl border border-white/5 bg-black/40 p-4 space-y-2">
            <h4 className="text-cyan-300 font-bold">2. Dynamic Spatio-Temporal Boundary</h4>
            <p className="text-slate-400">
              Combines extreme logit range spread with hazard wear-out:
            </p>
            <div className="py-2 text-center text-white">
              <Tex math="\tau_{\text{eff}}(k) = \tau_0 \cdot \left[1 + \gamma \cdot h(k)\right]" block />
            </div>
            <p className="text-slate-500 text-[11px]">
              Drafting continues if <Tex math="R_8(k) \ge \tau_{\text{eff}}(k)" />; aborts immediately otherwise.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
