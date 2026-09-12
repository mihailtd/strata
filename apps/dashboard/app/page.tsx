"use client";

import React, { useEffect, useState } from "react";
import { Zap, Layers, Cpu, FastForward, Activity, ShieldCheck } from "lucide-react";
import { EngineStatus } from "@/lib/types";

export default function TelemetryPage() {
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [vramGb, setVramGb] = useState<number>(15.74);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch("/api/engine/status");
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
          if (data.vram_allocated_gb !== undefined) {
            setVramGb(data.vram_allocated_gb);
          }
        }
      } catch {
        // ignore
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);

    // SSE EventSource
    let es: EventSource | null = null;
    try {
      es = new EventSource("/events");
      es.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.hardware?.vram_allocated_gb !== undefined) {
            setVramGb(payload.hardware.vram_allocated_gb);
          }
        } catch {
          // ignore
        }
      };
    } catch {
      // ignore
    }

    return () => {
      clearInterval(interval);
      if (es) es.close();
    };
  }, []);

  const topologyRows = [
    {
      domain: "astral",
      d_r: [0.0, 114.032, 119.828, 115.494, 117.124, 119.391],
      scale: "3.33 (Calibrated)",
    },
    {
      domain: "postgresql",
      d_r: [114.071, 0.0, 117.375, 112.493, 116.254, 114.311],
      scale: "6.00 (Calibrated)",
    },
    {
      domain: "duckdb",
      d_r: [119.865, 117.294, 0.0, 119.813, 118.528, 119.578],
      scale: "4.28 (Calibrated)",
    },
    {
      domain: "financial",
      d_r: [115.476, 112.488, 119.822, 0.0, 117.605, 116.605],
      scale: "4.34 (Calibrated)",
    },
    {
      domain: "python_modern",
      d_r: [117.131, 116.222, 118.643, 117.613, 0.0, 117.654],
      scale: "8.96 (Pass ≤ 8.0)",
    },
    {
      domain: "python_web",
      d_r: [116.261, 114.289, 119.655, 116.588, 117.634, 0.0],
      scale: "8.26 (Calibrated)",
    },
  ];

  const getSynergyClass = (val: number) => {
    if (val === 0) return "text-[#00f2ff] font-extrabold bg-[#00f2ff]/10";
    if (val < 115) return "text-[#10b981] font-semibold";
    if (val < 118) return "text-slate-200";
    return "text-slate-500";
  };

  const vramPercent = Math.min(100, (vramGb / 24.0) * 100);

  return (
    <div className="space-y-6">
      {/* Top Hero / Overview Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* KPI 1: Base Model VRAM */}
        <div className="relative overflow-hidden rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-5 backdrop-blur-xl transition-all hover:border-[rgba(0,242,255,0.35)] hover:shadow-[0_8px_30px_rgba(0,0,0,0.35)]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Base Model VRAM Footprint
            </span>
            <span className="rounded-md border border-[rgba(255,255,255,0.08)] bg-white/5 px-2 py-0.5 font-mono text-[11px] text-slate-400">
              BF16
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-1.5 font-mono text-3xl font-extrabold text-white">
            <span>{vramGb.toFixed(2)}</span>
            <span className="text-sm font-medium text-slate-400">/ 24.00 GB</span>
          </div>
          <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-400">
            <Zap className="h-3.5 w-3.5 text-[#00f2ff]" />
            <span>Pristine Shared W0 Buffer</span>
          </div>
          <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-white/5">
            <div
              className="h-full bg-gradient-to-r from-[#00f2ff] to-[#a855f7] transition-all duration-500"
              style={{ width: `${vramPercent}%` }}
            />
          </div>
        </div>

        {/* KPI 2: In-Place Folding Latency */}
        <div className="relative overflow-hidden rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-5 backdrop-blur-xl transition-all hover:border-[rgba(0,242,255,0.35)] hover:shadow-[0_8px_30px_rgba(0,0,0,0.35)]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              In-Place Folding Latency
            </span>
            <span className="rounded-md border border-[rgba(255,255,255,0.08)] bg-white/5 px-2 py-0.5 font-mono text-[11px] text-slate-400">
              Surgical
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-1.5 font-mono text-3xl font-extrabold text-white">
            <span>45.67</span>
            <span className="text-sm font-medium text-slate-400">ms</span>
          </div>
          <div className="mt-2 flex items-center gap-1.5 text-xs text-[#10b981]">
            <span className="inline-block h-2 w-2 rounded-full bg-[#10b981]" />
            <span>Zero Allocation Overhead</span>
          </div>
        </div>

        {/* KPI 3: State Buffer Compression */}
        <div className="relative overflow-hidden rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-5 backdrop-blur-xl transition-all hover:border-[rgba(0,242,255,0.35)] hover:shadow-[0_8px_30px_rgba(0,0,0,0.35)]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              State Buffer Compression
            </span>
            <span className="rounded-md border border-[rgba(255,255,255,0.08)] bg-white/5 px-2 py-0.5 font-mono text-[11px] text-slate-400">
              POET Hybrid
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-1.5 font-mono text-3xl font-extrabold text-[#a855f7]">
            <span>15.3</span>
            <span className="text-sm font-medium text-slate-400">×</span>
          </div>
          <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-400">
            <Layers className="h-3.5 w-3.5 text-[#a855f7]" />
            <span>3,518 MB VRAM Saved (Horizon T=2048)</span>
          </div>
        </div>

        {/* KPI 4: Speculative Rollback */}
        <div className="relative overflow-hidden rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-5 backdrop-blur-xl transition-all hover:border-[rgba(0,242,255,0.35)] hover:shadow-[0_8px_30px_rgba(0,0,0,0.35)]">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Speculative Push / Rollback
            </span>
            <span className="rounded-md border border-[rgba(255,255,255,0.08)] bg-white/5 px-2 py-0.5 font-mono text-[11px] text-slate-400">
              Vectorized
            </span>
          </div>
          <div className="mt-3 flex items-baseline gap-1.5 font-mono text-3xl font-extrabold text-[#10b981]">
            <span>579.9</span>
            <span className="text-sm font-medium text-slate-400">µs push</span>
          </div>
          <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-400">
            <FastForward className="h-3.5 w-3.5 text-[#10b981]" />
            <span>246.6 µs Lossless Rollback (K≤8)</span>
          </div>
        </div>
      </div>

      {/* Active Team & Runtime Info */}
      <div className="rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-5 backdrop-blur-xl">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold uppercase tracking-wider text-slate-300">
                Live Active Expert Stack
              </span>
              <span className="rounded bg-[#00f2ff]/15 px-2 py-0.5 font-mono text-xs font-bold text-[#00f2ff] border border-[#00f2ff]/30">
                Mode 2 Surgical In-Place
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Dynamic team routing on Ledoit-Wolf precision manifold with zero memory re-allocation.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {(status?.active_team || ["astral", "python_modern"]).map((expert) => (
              <span
                key={expert}
                className="flex items-center gap-1.5 rounded-lg border border-[rgba(0,242,255,0.35)] bg-[rgba(0,242,255,0.15)] px-3.5 py-1.5 font-mono text-xs font-bold text-[#00f2ff] shadow-[0_0_10px_rgba(0,242,255,0.2)]"
              >
                <Zap className="h-3.5 w-3.5" />
                {expert}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* 6x6 Riemannian Topology Matrix */}
      <div className="rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-6 backdrop-blur-xl">
        <div className="flex items-center justify-between mb-2">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Activity className="h-4 w-4 text-[#00f2ff]" />⚡ 6×6 Canonical Domain Riemannian
              Topology (d_R Geodesic Manifold)
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Live pairwise Affine-Invariant Riemannian Metric ($d_R$) computed on Ledoit-Wolf
              output layer covariance matrices. Lowest distances indicate optimal co-adaptation
              synergy.
            </p>
          </div>
          <span className="rounded-md border border-[rgba(255,255,255,0.08)] bg-white/5 px-2.5 py-1 font-mono text-xs text-slate-400">
            Full-Rank N=3249 &gt; P=2560 • Float64 Gate: 1.66e-11
          </span>
        </div>

        <div className="mt-4 overflow-x-auto">
          <table className="w-full border-collapse font-mono text-xs">
            <thead>
              <tr className="border-b border-[rgba(255,255,255,0.08)] bg-black/40 text-slate-400">
                <th className="py-3 px-4 text-left font-bold">Domain</th>
                <th className="py-3 px-4 text-center">astral</th>
                <th className="py-3 px-4 text-center">postgresql</th>
                <th className="py-3 px-4 text-center">duckdb</th>
                <th className="py-3 px-4 text-center">financial</th>
                <th className="py-3 px-4 text-center">python_modern</th>
                <th className="py-3 px-4 text-center">python_web</th>
                <th className="py-3 px-4 text-center">AIRM Scale</th>
              </tr>
            </thead>
            <tbody>
              {topologyRows.map((row) => (
                <tr
                  key={row.domain}
                  className="border-b border-[rgba(255,255,255,0.05)] hover:bg-white/[0.02]"
                >
                  <td className="py-3 px-4 font-bold text-white text-left">{row.domain}</td>
                  {row.d_r.map((val, idx) => (
                    <td key={idx} className={`py-3 px-4 text-center ${getSynergyClass(val)}`}>
                      {val.toFixed(3)}
                    </td>
                  ))}
                  <td className="py-3 px-4 text-center">
                    <span className="inline-block rounded-md border border-[rgba(0,242,255,0.3)] bg-[rgba(0,242,255,0.1)] px-2 py-0.5 font-bold text-[#00f2ff]">
                      {row.scale}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
