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
  ShieldAlert,
  AlertTriangle,
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
  // Weibull Parameters
  const [beta, setBeta] = useState<number>(2.2);
  const [eta, setEta] = useState<number>(4.0);
  const [gammaW, setGammaW] = useState<number>(0.6);
  const [tau0, setTau0] = useState<number>(3.5);

  // Bollinger Parameters
  const [bollingerK, setBollingerK] = useState<number>(2.0);
  const [gammaB, setGammaB] = useState<number>(0.5);

  // Scenario Mode
  const [scenario, setScenario] = useState<"decay" | "vol_crash" | "high_conf">("vol_crash");

  // Sample Tokens (Range R8 and Margin Spread Delta L)
  const sampleData = useMemo(() => {
    if (scenario === "vol_crash") {
      // Normal tokens then sudden volatility collapse at step 3
      return [
        { r8: 12.5, spread: 4.8 },
        { r8: 9.8, spread: 3.9 },
        { r8: 4.2, spread: 0.3 }, // Sudden collapse!
        { r8: 3.8, spread: 0.4 },
        { r8: 2.5, spread: 0.2 },
        { r8: 1.8, spread: 0.1 },
        { r8: 1.2, spread: 0.1 },
        { r8: 0.8, spread: 0.05 },
      ];
    } else if (scenario === "decay") {
      // Standard gradual wear-out
      return [
        { r8: 12.0, spread: 4.5 },
        { r8: 8.5, spread: 3.2 },
        { r8: 5.5, spread: 2.1 },
        { r8: 4.1, spread: 1.5 },
        { r8: 3.2, spread: 1.1 },
        { r8: 2.4, spread: 0.8 },
        { r8: 1.5, spread: 0.5 },
        { r8: 0.9, spread: 0.3 },
      ];
    } else {
      // High confidence streak
      return [
        { r8: 14.5, spread: 6.2 },
        { r8: 13.8, spread: 5.9 },
        { r8: 12.2, spread: 5.1 },
        { r8: 11.5, spread: 4.8 },
        { r8: 10.8, spread: 4.5 },
        { r8: 9.9, spread: 4.1 },
        { r8: 8.8, spread: 3.7 },
        { r8: 7.9, spread: 3.2 },
      ];
    }
  }, [scenario]);

  // Compute step-by-step wear-out & Bollinger values
  const stepMetrics = useMemo(() => {
    let meanSpread = 3.5;
    let varSpread = 0.5;
    let atr = 0.5;
    let lastSpread = 3.5;

    return sampleData.map((d, i) => {
      const step = i + 1;
      const hazard = (beta / eta) * Math.pow(step / eta, beta - 1);

      const std = Math.sqrt(varSpread);
      const lowerBand = meanSpread - bollingerK * std;
      const upperBand = meanSpread + bollingerK * std;

      let volPenalty = 0.0;
      if (d.spread < lowerBand) {
        volPenalty = (lowerBand - d.spread) / Math.max(1e-4, atr);
      }

      // Dynamic Threshold
      const tauEff = tau0 * (1.0 + gammaW * hazard + gammaB * volPenalty);

      // Early exit checks
      const hardVolBreakout = d.spread < lowerBand - 1.5 * atr;
      const thresholdBreached = d.r8 < tauEff && i > 0;
      const willAbort = hardVolBreakout || thresholdBreached;

      let reason = "PASSED";
      if (hardVolBreakout) {
        reason = "BOLLINGER_VOLATILITY_COLLAPSE";
      } else if (thresholdBreached) {
        reason = "WEIBULL_WEAROUT_ABORT";
      }

      // Update state for next step
      const alpha = 0.25;
      const diff = d.spread - meanSpread;
      meanSpread += alpha * diff;
      varSpread = (1.0 - alpha) * varSpread + alpha * (diff * diff);
      const tr = Math.max(Math.abs(d.spread - lastSpread), 1e-4);
      atr = 0.8 * atr + 0.2 * tr;
      lastSpread = d.spread;

      return {
        step,
        r8: d.r8,
        spread: d.spread,
        hazard,
        lowerBand,
        upperBand,
        volPenalty,
        tauEff,
        willAbort,
        reason,
      };
    });
  }, [beta, eta, gammaW, tau0, bollingerK, gammaB, sampleData]);

  const earlyExitStep = useMemo(() => {
    for (let i = 0; i < stepMetrics.length; i++) {
      if (stepMetrics[i].willAbort) return i + 1;
    }
    return 8;
  }, [stepMetrics]);

  return (
    <div className="space-y-8 p-6 text-slate-100 max-w-5xl mx-auto">
      {/* Hero Header */}
      <div className="relative overflow-hidden rounded-3xl border border-cyan-500/20 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 p-8 shadow-2xl">
        <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-cyan-500/10 blur-3xl pointer-events-none" />
        <div className="relative z-10">
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3.5 py-1 text-xs font-mono text-cyan-400 mb-4">
            <Flame className="h-3.5 w-3.5" />
            <span>Reliability &bull; Chapter 3 &amp; Chapter 5</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
            Weibull Hazard &amp; Bollinger Band Volatility Gating
          </h1>
          <p className="mt-3 text-sm text-slate-300 max-w-3xl leading-relaxed">
            Unifies temporal wear-out hazard modeling (<Tex math="h(k) \propto k^{\beta-1}" />) with
            rolling logit volatility bands (<Tex math="\mu \pm 2\sigma" />
            ). Dynamically tightens speculative acceptance thresholds as draft depth grows and
            intercepts unexpected reasoning collapses before doomed tokens ever reach the backbone
            verifier.
          </p>
        </div>
      </div>

      {/* Layman's Analogy Card */}
      <div className="rounded-2xl border border-cyan-500/20 bg-cyan-950/10 p-6 backdrop-blur-md">
        <div className="flex items-start gap-3">
          <Info className="h-5 w-5 text-cyan-400 shrink-0 mt-0.5" />
          <div className="space-y-2">
            <h3 className="text-base font-bold text-cyan-300">
              Plain English: The Fatigued Runner + Sudden Earthquake
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              &bull; <strong>Weibull Wear-Out (Time Hazard):</strong> Just like a sprinter gets
              tired after 400 meters, speculative draft heads get less reliable as draft depth grows
              (<Tex math="k=1 \to 8" />
              ). Weibull naturally raises the required confidence bar for tokens 4, 5, and 6.
              <br />
              &bull; <strong>Bollinger Bands (Volatility Breakdown):</strong> Even at token 2 or 3,
              if the model hits a tricky reasoning step, its top-1 vs runner-up margin suddenly
              collapses. Bollinger Bands catch this sudden volatility crash instantly, halting
              speculation before wasting GPU compute on doomed tokens.
            </p>
          </div>
        </div>
      </div>

      {/* Interactive Simulator Deck */}
      <div className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 space-y-6 backdrop-blur-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/5 pb-4">
          <div className="flex items-center gap-3">
            <Sliders className="h-5 w-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white">Spatio-Temporal Volatility Simulator</h2>
          </div>

          <div className="flex rounded-xl bg-black/50 p-1 border border-white/10 text-xs font-mono">
            <button
              onClick={() => setScenario("vol_crash")}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                scenario === "vol_crash"
                  ? "bg-cyan-600 text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Sudden Volatility Crash
            </button>
            <button
              onClick={() => setScenario("decay")}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                scenario === "decay"
                  ? "bg-cyan-600 text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              Gradual Tail Wear-Out
            </button>
            <button
              onClick={() => setScenario("high_conf")}
              className={`px-3 py-1.5 rounded-lg transition-all ${
                scenario === "high_conf"
                  ? "bg-cyan-600 text-white font-bold"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              High Confidence Streak
            </button>
          </div>
        </div>

        {/* Sliders Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="rounded-xl border border-white/5 bg-black/40 p-4 space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">
                Weibull Shape (<Tex math="\beta" />)
              </span>
              <span className="text-cyan-400 font-bold">{beta.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={1.0}
              max={3.5}
              step={0.1}
              value={beta}
              onChange={(e) => setBeta(parseFloat(e.target.value))}
              className="w-full accent-cyan-400"
            />
            <p className="text-[10px] text-slate-500 font-mono">
              <Tex math="\beta > 1" />: Accelerating wear-out risk
            </p>
          </div>

          <div className="rounded-xl border border-white/5 bg-black/40 p-4 space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">
                Weibull Weight (<Tex math="\gamma_w" />)
              </span>
              <span className="text-cyan-400 font-bold">{gammaW.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={1.5}
              step={0.05}
              value={gammaW}
              onChange={(e) => setGammaW(parseFloat(e.target.value))}
              className="w-full accent-cyan-400"
            />
            <p className="text-[10px] text-slate-500 font-mono">Temporal penalty multiplier</p>
          </div>

          <div className="rounded-xl border border-white/5 bg-black/40 p-4 space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">
                Bollinger Multiplier (<Tex math="k_{\text{bb}}" />)
              </span>
              <span className="text-purple-400 font-bold">
                {bollingerK.toFixed(1)}
                <Tex math="\sigma" />
              </span>
            </div>
            <input
              type="range"
              min={1.0}
              max={3.0}
              step={0.1}
              value={bollingerK}
              onChange={(e) => setBollingerK(parseFloat(e.target.value))}
              className="w-full accent-purple-400"
            />
            <p className="text-[10px] text-slate-500 font-mono">
              Confidence band standard deviations
            </p>
          </div>

          <div className="rounded-xl border border-white/5 bg-black/40 p-4 space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-slate-400">
                Bollinger Weight (<Tex math="\gamma_b" />)
              </span>
              <span className="text-purple-400 font-bold">{gammaB.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.0}
              max={1.5}
              step={0.05}
              value={gammaB}
              onChange={(e) => setGammaB(parseFloat(e.target.value))}
              className="w-full accent-purple-400"
            />
            <p className="text-[10px] text-slate-500 font-mono">
              Volatility breakout penalty weight
            </p>
          </div>
        </div>

        {/* Step-by-Step Dynamic Decision Matrix */}
        <div className="space-y-3">
          <div className="flex justify-between items-center text-xs font-mono text-slate-400">
            <span>
              Intra-Round Speculative Draft Sequence (<Tex math="K=8" /> Horizon)
            </span>
            <span className="text-cyan-400 font-bold">
              {earlyExitStep < 8
                ? `Early Exit Triggered at Step K=${earlyExitStep}`
                : "Full Draft Horizon (K=8) Passed"}
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2 font-mono text-xs">
            {stepMetrics.map((m, idx) => {
              const isAborted = idx + 1 >= earlyExitStep;
              const isFirstAbort = idx + 1 === earlyExitStep;
              return (
                <div
                  key={m.step}
                  className={`rounded-2xl border p-3 flex flex-col justify-between transition-all ${
                    isFirstAbort
                      ? "border-red-500/80 bg-red-950/40 shadow-[0_0_15px_rgba(239,68,68,0.3)] ring-2 ring-red-400"
                      : isAborted
                        ? "border-slate-800 bg-slate-950/30 opacity-40"
                        : "border-cyan-500/30 bg-slate-900/80"
                  }`}
                >
                  <div>
                    <div className="flex justify-between items-center text-[10px] text-slate-400">
                      <span>Step {m.step}</span>
                      {isFirstAbort && <span className="text-red-400 font-bold">ABORT</span>}
                    </div>

                    <div className="mt-2 text-sm font-bold text-white">
                      R<sub>8</sub>: {m.r8.toFixed(1)}
                    </div>
                    <div className="text-[11px] text-purple-300">
                      <Tex math="\Delta l" />: {m.spread.toFixed(2)}
                    </div>
                  </div>

                  <div className="mt-3 pt-2 border-t border-white/5 space-y-1 text-[10px]">
                    <div className="flex justify-between text-slate-400">
                      <span>
                        <Tex math="\tau_{\text{eff}}" />
                      </span>
                      <span className="font-bold text-cyan-300">{m.tauEff.toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between text-slate-400">
                      <span>Band</span>
                      <span className="text-purple-300">{m.lowerBand.toFixed(1)}</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Live Result Alert */}
        <div className="rounded-2xl border border-cyan-500/30 bg-cyan-950/20 p-4 flex flex-wrap items-center justify-between gap-4 font-mono text-xs">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-cyan-400" />
            <span>
              {earlyExitStep < 8
                ? `Pruned ${8 - earlyExitStep + 1} doomed draft tokens! Saved ${((8 - earlyExitStep + 1) * 3.5).toFixed(1)}ms of useless verification compute.`
                : "All 8 tokens satisfied both temporal hazard and volatility bounds."}
            </span>
          </div>
          <span className="text-cyan-300 font-bold">DECISION LATENCY: 0.099 ms</span>
        </div>
      </div>

      {/* GPU Benchmark Results Table */}
      <div className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 space-y-4 backdrop-blur-xl">
        <div className="flex items-center gap-2 border-b border-white/5 pb-3">
          <Activity className="h-5 w-5 text-emerald-400" />
          <h2 className="text-lg font-bold text-white">
            AMD RX 7900 XTX Measured GPU Telemetry (500 Rounds)
          </h2>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-white/10 text-slate-400">
                <th className="pb-3 px-3">Horizon Mode</th>
                <th className="pb-3 px-3">Blind (Fixed K)</th>
                <th className="pb-3 px-3">Static Gate (R8)</th>
                <th className="pb-3 px-3">Weibull Gate (Ch.3)</th>
                <th className="pb-3 px-3 text-cyan-400">Weibull + Bollinger (Ch.3+5)</th>
                <th className="pb-3 px-3 text-emerald-400">Speedup vs Blind</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-slate-200">
              <tr>
                <td className="py-3 px-3 font-bold text-white">Horizon K = 4</td>
                <td className="py-3 px-3 text-slate-400">88.28 tok/s</td>
                <td className="py-3 px-3 text-slate-300">98.17 tok/s</td>
                <td className="py-3 px-3 text-slate-300">98.17 tok/s</td>
                <td className="py-3 px-3 text-cyan-300 font-bold">97.87 tok/s</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">+10.9%</td>
              </tr>
              <tr>
                <td className="py-3 px-3 font-bold text-white">Horizon K = 6</td>
                <td className="py-3 px-3 text-slate-400">75.77 tok/s</td>
                <td className="py-3 px-3 text-slate-300">97.70 tok/s</td>
                <td className="py-3 px-3 text-slate-300">97.70 tok/s</td>
                <td className="py-3 px-3 text-cyan-300 font-bold">97.42 tok/s</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">+28.6%</td>
              </tr>
              <tr className="bg-cyan-500/5">
                <td className="py-3 px-3 font-bold text-white">Horizon K = 8</td>
                <td className="py-3 px-3 text-red-400 font-bold">65.20 tok/s</td>
                <td className="py-3 px-3 text-slate-300">97.58 tok/s</td>
                <td className="py-3 px-3 text-slate-300">97.58 tok/s</td>
                <td className="py-3 px-3 text-cyan-300 font-bold">97.31 tok/s ⚡</td>
                <td className="py-3 px-3 text-emerald-400 font-bold">+49.3% (60.3% Pruned) 🔥</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Mathematical Rigor Card */}
      <div className="rounded-3xl border border-white/10 bg-slate-900/60 p-6 space-y-4 backdrop-blur-xl">
        <div className="flex items-center gap-2 border-b border-white/5 pb-3">
          <Cpu className="h-5 w-5 text-cyan-400" />
          <h2 className="text-lg font-bold text-white">
            Mathematical Rigor: Composite Spatio-Temporal Volatility Formulation
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
          <div className="rounded-2xl border border-white/5 bg-black/40 p-4 space-y-2">
            <h4 className="text-cyan-300 font-bold">1. Weibull Wear-Out Hazard (Chapter 3)</h4>
            <p className="text-slate-400">
              For shape <Tex math="\beta > 1" /> and characteristic horizon <Tex math="\eta" />:
            </p>
            <div className="py-2 text-center text-white">
              <Tex
                math="h(k) = \frac{\beta}{\eta} \left(\frac{k+1}{\eta}\right)^{\beta - 1}"
                block
              />
            </div>
            <p className="text-slate-500 text-[11px]">
              Models monotonic decline of draft token acceptance across sequential tokens.
            </p>
          </div>

          <div className="rounded-2xl border border-white/5 bg-black/40 p-4 space-y-2">
            <h4 className="text-purple-300 font-bold">
              2. Bollinger Bands on Logit Spread (Chapter 5)
            </h4>
            <p className="text-slate-400">
              Tracks margin spread <Tex math="\Delta l_t = z_{(1)} - z_{(2)}" /> with rolling
              variance:
            </p>
            <div className="py-2 text-center text-white">
              <Tex math="\text{Lower Band}_t = \mu_t - k_{\text{bb}} \cdot \sigma_t" block />
            </div>
            <p className="text-slate-500 text-[11px]">
              Catches local uncertainty collapses and spikes dynamic penalty{" "}
              <Tex math="\text{VolPenalty}" />.
            </p>
          </div>
        </div>

        <div className="rounded-2xl border border-white/5 bg-black/40 p-4 text-center text-xs font-mono">
          <span className="text-emerald-300 font-bold block mb-2">
            Composite Tri-Modal Gating Threshold Formula
          </span>
          <Tex
            math="\tau_{\text{eff}}(k, \Delta l_t) = \tau_0 \cdot \left[ 1 + \gamma_w \cdot h(k) + \gamma_b \cdot \max\left(0, \frac{\text{Lower Band}_t - \Delta l_t}{\text{ATR}_t + 10^{-6}}\right) \right]"
            block
          />
        </div>
      </div>
    </div>
  );
}
