"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";
import dynamic from "next/dynamic";
import {
  Factory,
  Play,
  Square,
  RefreshCw,
  Zap,
  CheckCircle,
  XCircle,
  Sliders,
  Activity,
  ChevronDown,
  ChevronUp,
  Search,
  ArrowUpDown,
  Sparkles,
  Database,
  BarChart3,
} from "lucide-react";
import { TrainingRun, TrainingStatus, CalibrateAlphaResponse } from "@/lib/types";
import type { EChartsCoreOption } from "echarts/core";

// Dynamically import AdaptiveChart to isolate canvas SSR
const AdaptiveChart = dynamic(() => import("@/components/AdaptiveChart"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-[360px] flex flex-col items-center justify-center bg-black/40 rounded-xl border border-dashed border-white/10 animate-pulse">
      <Sparkles className="h-6 w-6 text-[#00f2ff] animate-spin mb-2" />
      <span className="text-xs font-mono text-slate-400">Loading Goldilocks Zone ECharts Canvas...</span>
    </div>
  ),
});

export default function TrainingPage() {
  const [domain, setDomain] = useState("postgresql");
  const [rank, setRank] = useState(8);
  const [alpha, setAlpha] = useState(128);
  const [targetDw, setTargetDw] = useState(0.071);
  const [maxSteps, setMaxSteps] = useState(150);
  const [completionOnly, setCompletionOnly] = useState(true);
  const [liger, setLiger] = useState(true);
  const [geometricStop, setGeometricStop] = useState(true);

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus>({ active: false });
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [isCalibrating, setIsCalibrating] = useState<string | null>(null);

  // Calibration Results Cache keyed by run_id or adapter_name
  const [calibCache, setCalibCache] = useState<Record<string, CalibrateAlphaResponse>>({});

  // Expandable Row State
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);

  // Filter & Sort State
  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState<keyof TrainingRun | "optimal_alpha">("created_at");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");

  const consoleEndRef = useRef<HTMLDivElement>(null);

  const fetchRuns = async () => {
    try {
      const res = await fetch("/api/training/runs");
      if (res.ok) {
        const data = await res.json();
        setRuns(data.runs || []);
      }
    } catch {
      // ignore
    }
  };

  const fetchCalibrations = async () => {
    try {
      const res = await fetch("/api/factory/calibrations");
      if (res.ok) {
        const data = await res.json();
        if (data.calibrations && Array.isArray(data.calibrations)) {
          const map: Record<string, CalibrateAlphaResponse> = {};
          data.calibrations.forEach((c: CalibrateAlphaResponse) => {
            if (c.run_id) map[c.run_id] = c;
            if (c.adapter_name) map[c.adapter_name] = c;
            if (c.domain) map[c.domain] = c;
          });
          setCalibCache((prev) => ({ ...prev, ...map }));
        }
      }
    } catch {
      // ignore
    }
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch("/api/training/status");
      if (res.ok) {
        const data: TrainingStatus = await res.json();
        setTrainingStatus(data);
      }
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    fetchRuns();
    fetchCalibrations();
    fetchStatus();
    const interval = setInterval(() => {
      fetchStatus();
      fetchRuns();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (trainingStatus.active) {
      consoleEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [trainingStatus]);

  const handleStart = async () => {
    try {
      const resp = await fetch("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          domain,
          rank,
          alpha,
          target_dw_w: targetDw,
          max_steps: maxSteps,
          completion_only: completionOnly,
          liger,
          geometric_stop: geometricStop,
        }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        alert(data.error || "Failed to start training");
      } else {
        await fetchStatus();
      }
    } catch (e) {
      alert("Error starting training: " + e);
    }
  };

  const handleCancel = async () => {
    try {
      await fetch("/api/training/cancel", { method: "POST" });
      await fetchStatus();
    } catch {
      // ignore
    }
  };

  const handleCalibrate = async (dom: string, runId: string, adapterName?: string) => {
    setIsCalibrating(runId);
    try {
      const resp = await fetch("/api/factory/calibrate_alpha", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ domain: dom, run_id: runId, adapter_name: adapterName, apply: true }),
      });
      const data = await resp.json();
      if (resp.ok) {
        setCalibCache((prev) => ({
          ...prev,
          [runId]: data,
          [data.adapter_name]: data,
          [dom]: data,
        }));
        setExpandedRunId(runId);
        await fetchRuns();
      } else {
        alert(data.error || "Calibration failed");
      }
    } catch (e) {
      alert("Error during calibration: " + e);
    } finally {
      setIsCalibrating(null);
    }
  };

  const handleCalibrateAll = async () => {
    const domains = ["astral", "postgresql", "duckdb", "financial_planning", "python_modern", "python_web"];
    for (const d of domains) {
      await handleCalibrate(d, `bulk_${d}`);
    }
  };

  const toggleExpand = async (run: TrainingRun) => {
    if (expandedRunId === run.run_id) {
      setExpandedRunId(null);
    } else {
      setExpandedRunId(run.run_id);
      if (!calibCache[run.run_id] && !calibCache[run.domain]) {
        // Auto fetch or calculate calibration on row expand
        handleCalibrate(run.domain, run.run_id);
      }
    }
  };

  // Sorting & Filtering Logic
  const filteredAndSortedRuns = useMemo(() => {
    return runs
      .filter((r) => {
        const query = searchTerm.toLowerCase();
        return (
          r.domain.toLowerCase().includes(query) ||
          (r.run_id && r.run_id.toLowerCase().includes(query)) ||
          (r.adapter_version && r.adapter_version.toLowerCase().includes(query))
        );
      })
      .sort((a, b) => {
        let valA: any = a[sortField as keyof TrainingRun];
        let valB: any = b[sortField as keyof TrainingRun];

        if (sortField === "optimal_alpha") {
          valA = calibCache[a.run_id]?.alpha_opt || a.alpha || 0;
          valB = calibCache[b.run_id]?.alpha_opt || b.alpha || 0;
        }

        if (valA === undefined || valA === null) valA = "";
        if (valB === undefined || valB === null) valB = "";

        if (valA < valB) return sortOrder === "asc" ? -1 : 1;
        if (valA > valB) return sortOrder === "asc" ? 1 : -1;
        return 0;
      });
  }, [runs, searchTerm, sortField, sortOrder, calibCache]);

  const handleSort = (field: keyof TrainingRun | "optimal_alpha") => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
  };

  // Build ECharts Option for Goldilocks Zone
  const getEchartsOption = (calib: CalibrateAlphaResponse): EChartsCoreOption => {
    const alphas = calib.curve.map((c) => c.alpha);
    const dwValues = calib.curve.map((c) => c.dw_over_w);
    const mergeErrValues = calib.curve.map((c) => c.merge_err_pct);

    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(16, 22, 34, 0.95)",
        borderColor: "rgba(0, 242, 255, 0.4)",
        textStyle: { color: "#f8fafc", fontFamily: "monospace", fontSize: 11 },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return "";
          const p = params[0];
          const idx = p.dataIndex;
          const item = calib.curve[idx];
          return `
            <div style="font-weight:bold;color:#00f2ff;margin-bottom:4px;">Alpha α = ${item.alpha} (${item.scaling.toFixed(1)}x scaling)</div>
            <div>||ΔW|| / ||W||: <b style="color:#00f2ff">${item.dw_over_w.toFixed(4)}</b></div>
            <div>BF16 Mantissa Merge Error: <b style="color:#a855f7">${item.merge_err_pct.toFixed(2)}%</b></div>
            <div>In Goldilocks Band: <b style="color:${item.in_target_band ? "#10b981" : "#f43f5e"}">${item.in_target_band ? "YES (Optimal)" : "NO"}</b></div>
            <div>Below Precision Floor: <b style="color:${item.below_floor ? "#f59e0b" : "#10b981"}">${item.below_floor ? "YES (Unsafe)" : "SAFE"}</b></div>
          `;
        },
      },
      legend: {
        data: ["||ΔW|| / ||W|| Perturbation", "BF16 Merge Error (%)"],
        textStyle: { color: "#94a3b8", fontSize: 11, fontFamily: "monospace" },
        top: 5,
      },
      grid: {
        left: "4%",
        right: "5%",
        bottom: "10%",
        top: "16%",
        containLabel: true,
      },
      xAxis: {
        type: "category",
        data: alphas.map((a) => `α=${a}`),
        axisLine: { lineStyle: { color: "rgba(255,255,255,0.15)" } },
        axisLabel: { color: "#94a3b8", fontFamily: "monospace" },
      },
      yAxis: [
        {
          type: "value",
          name: "||ΔW|| / ||W||",
          nameTextStyle: { color: "#00f2ff", fontFamily: "monospace" },
          axisLine: { lineStyle: { color: "rgba(255,255,255,0.15)" } },
          splitLine: { lineStyle: { color: "rgba(255,255,255,0.05)" } },
          axisLabel: { color: "#00f2ff", fontFamily: "monospace" },
        },
        {
          type: "value",
          name: "Merge Err (%)",
          nameTextStyle: { color: "#a855f7", fontFamily: "monospace" },
          axisLine: { lineStyle: { color: "rgba(255,255,255,0.15)" } },
          splitLine: { show: false },
          axisLabel: { color: "#a855f7", fontFamily: "monospace" },
        },
      ],
      series: [
        {
          name: "||ΔW|| / ||W|| Perturbation",
          type: "line",
          data: dwValues,
          smooth: true,
          symbol: "circle",
          symbolSize: 8,
          itemStyle: { color: "#00f2ff" },
          lineStyle: { width: 3, color: "#00f2ff", shadowColor: "rgba(0,242,255,0.4)", shadowBlur: 10 },
          markArea: {
            itemStyle: { color: "rgba(16, 185, 129, 0.12)" },
            data: [
              [
                { name: "Goldilocks Zone (0.040 - 0.085)", yAxis: 0.04 },
                { yAxis: 0.085 },
              ],
            ],
            label: { color: "#10b981", fontSize: 10, fontFamily: "monospace" },
          },
          markPoint: {
            data: [
              {
                name: "Optimal α",
                coord: [
                  `α=${calib.alpha_opt}`,
                  calib.optimal_point.dw_over_w,
                ],
                value: `🏆 Optimal α=${calib.alpha_opt}`,
                itemStyle: { color: "#10b981" },
              },
            ],
          },
        },
        {
          name: "BF16 Merge Error (%)",
          type: "line",
          yAxisIndex: 1,
          data: mergeErrValues,
          smooth: true,
          symbol: "rect",
          symbolSize: 7,
          itemStyle: { color: "#a855f7" },
          lineStyle: { width: 2, type: "dashed", color: "#a855f7" },
          markLine: {
            lineStyle: { color: "#f59e0b", type: "dotted", width: 2 },
            data: [{ name: "5% Precision Floor", yAxis: 5.0 }],
            label: { formatter: "5.0% Mantissa Limit", color: "#f59e0b", fontFamily: "monospace", fontSize: 10 },
          },
        },
      ],
    };
  };

  return (
    <div className="space-y-6">
      {/* Top Split: Training Controls & Live Progress Console */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Configuration Form */}
        <div className="lg:col-span-5 rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-5 backdrop-blur-xl shadow-xl">
          <div className="flex items-center justify-between mb-4">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Factory className="h-4 w-4 text-[#00f2ff]" /> Train Domain Expert
            </span>
            <span
              className={`rounded-md border px-2.5 py-0.5 font-mono text-[11px] font-bold ${
                trainingStatus.active
                  ? "border-[#00f2ff]/40 bg-[#00f2ff]/15 text-[#00f2ff] animate-pulse"
                  : "border-white/10 bg-white/5 text-slate-400"
              }`}
            >
              {trainingStatus.active ? `TRAINING [${trainingStatus.domain}]` : "IDLE"}
            </span>
          </div>

          <div className="space-y-3.5">
            <div>
              <label className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                Domain Target
              </label>
              <select
                value={domain}
                onChange={(e) => setDomain(e.target.value)}
                disabled={trainingStatus.active}
                className="mt-1 w-full rounded-xl border border-white/10 bg-black/50 p-2.5 text-xs text-white focus:border-[#00f2ff] focus:outline-none"
              >
                <option value="postgresql">🐘 PostgreSQL (v6: asyncpg + pgvector)</option>
                <option value="duckdb">🦆 DuckDB (v6: Parquet Analytics &amp; Window)</option>
                <option value="astral">⚡ Astral (v6: Separated uv/ruff/ty)</option>
                <option value="python_modern">🐍 Python Modern (v6: 1,418 Deduplicated)</option>
                <option value="python_web">🌐 Python Web (v6: FastAPI / Pydantic)</option>
                <option value="financial">📈 Financial Planning (v6)</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[11px] font-bold text-slate-400">Rank (r)</label>
                <input
                  type="number"
                  value={rank}
                  onChange={(e) => setRank(parseInt(e.target.value, 10))}
                  disabled={trainingStatus.active}
                  className="mt-1 w-full rounded-xl border border-white/10 bg-black/50 p-2 text-xs text-white focus:border-[#00f2ff] focus:outline-none"
                />
              </div>
              <div>
                <label className="text-[11px] font-bold text-slate-400">Alpha (&alpha;)</label>
                <input
                  type="number"
                  value={alpha}
                  onChange={(e) => setAlpha(parseInt(e.target.value, 10))}
                  disabled={trainingStatus.active}
                  className="mt-1 w-full rounded-xl border border-white/10 bg-black/50 p-2 text-xs text-white focus:border-[#00f2ff] focus:outline-none"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-[11px] font-bold text-slate-400">Stop ||&Delta;W||/||W||</label>
                <input
                  type="number"
                  step="0.005"
                  value={targetDw}
                  onChange={(e) => setTargetDw(parseFloat(e.target.value))}
                  disabled={trainingStatus.active}
                  className="mt-1 w-full rounded-xl border border-white/10 bg-black/50 p-2 text-xs text-white focus:border-[#00f2ff] focus:outline-none"
                />
              </div>
              <div>
                <label className="text-[11px] font-bold text-slate-400">Max Steps</label>
                <input
                  type="number"
                  value={maxSteps}
                  onChange={(e) => setMaxSteps(parseInt(e.target.value, 10))}
                  disabled={trainingStatus.active}
                  className="mt-1 w-full rounded-xl border border-white/10 bg-black/50 p-2 text-xs text-white focus:border-[#00f2ff] focus:outline-none"
                />
              </div>
            </div>

            {/* Checkbox Toggles */}
            <div className="space-y-2 rounded-xl border border-white/5 bg-black/30 p-3 text-xs">
              <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                <input
                  type="checkbox"
                  checked={completionOnly}
                  onChange={(e) => setCompletionOnly(e.target.checked)}
                  disabled={trainingStatus.active}
                  className="rounded border-white/20 bg-black/40 text-[#00f2ff]"
                />
                <span>Completion-Only Loss (Instruction Masking)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                <input
                  type="checkbox"
                  checked={liger}
                  onChange={(e) => setLiger(e.target.checked)}
                  disabled={trainingStatus.active}
                  className="rounded border-white/20 bg-black/40 text-[#00f2ff]"
                />
                <span>Fused Liger Kernels (RMSNorm + Cross-Entropy)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                <input
                  type="checkbox"
                  checked={geometricStop}
                  onChange={(e) => setGeometricStop(e.target.checked)}
                  disabled={trainingStatus.active}
                  className="rounded border-white/20 bg-black/40 text-[#00f2ff]"
                />
                <span>Geometric Early Stopping ($||\Delta W||/||W|| \ge 0.071$)</span>
              </label>
            </div>

            <div className="pt-2">
              {trainingStatus.active ? (
                <button
                  onClick={handleCancel}
                  className="w-full flex items-center justify-center gap-2 rounded-xl border border-rose-500/40 bg-rose-500/15 py-3 text-xs font-bold text-rose-400 hover:bg-rose-500/25 transition-all"
                >
                  <Square className="h-4 w-4 fill-current" /> Abort Training Job
                </button>
              ) : (
                <button
                  onClick={handleStart}
                  className="w-full flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-[#00f2ff] to-[#a855f7] py-3 text-xs font-bold text-slate-950 shadow-[0_0_20px_rgba(0,242,255,0.3)] transition-all hover:scale-[1.01]"
                >
                  <Play className="h-4 w-4 fill-current" /> Launch Autonomous Training
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right: Live Training Terminal Output */}
        <div className="lg:col-span-7 rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(12,16,24,0.9)] p-5 backdrop-blur-xl flex flex-col shadow-xl">
          <div className="flex items-center justify-between pb-3 border-b border-white/5">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Activity className="h-4 w-4 text-[#a855f7]" /> Live Subprocess Stream
            </span>
            <span className="font-mono text-[11px] text-slate-500">factory.db sync active</span>
          </div>

          <div className="flex-1 overflow-y-auto font-mono text-xs text-slate-300 p-3 space-y-1 bg-black/60 rounded-xl mt-3 max-h-[340px]">
            {trainingStatus.recent_logs && trainingStatus.recent_logs.length > 0 ? (
              trainingStatus.recent_logs.map((log, idx) => (
                <div key={idx} className="leading-relaxed">
                  <span className="text-slate-600 mr-2">&gt;</span>
                  <span
                    className={
                      log.includes("AIRM") || log.includes("optimal")
                        ? "text-[#10b981] font-bold"
                        : log.includes("Error") || log.includes("FAILED")
                        ? "text-rose-400 font-bold"
                        : log.includes("step")
                        ? "text-cyan-300"
                        : "text-slate-300"
                    }
                  >
                    {log}
                  </span>
                </div>
              ))
            ) : (
              <div className="text-slate-600 italic py-16 text-center">
                Ready for next training run. Press &quot;Launch Autonomous Training&quot; to begin.
              </div>
            )}
            <div ref={consoleEndRef} />
          </div>
        </div>
      </div>

      {/* Historical Runs Ledger Table with Sorting, Search and Expandable ECharts Rows */}
      <div className="rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-5 backdrop-blur-xl shadow-2xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <Database className="h-4 w-4 text-[#00f2ff]" />
              Adapter Training History &amp; AIRM Acceptance Audit Ledger (results/factory.db)
            </h2>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Click any row to expand the interactive <b>Apache ECharts Goldilocks Zone</b> perturbation &amp; IEEE 754 precision curve.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Search Input */}
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-500" />
              <input
                type="text"
                placeholder="Search domain, run ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-8 pr-3 py-1.5 rounded-lg border border-white/10 bg-black/40 font-mono text-xs text-white focus:border-[#00f2ff] focus:outline-none"
              />
            </div>

            <button
              onClick={handleCalibrateAll}
              className="flex items-center gap-1.5 rounded-lg border border-[#00f2ff]/40 bg-[#00f2ff]/15 px-3 py-1.5 font-mono text-xs font-bold text-[#00f2ff] hover:bg-[#00f2ff]/25 transition-all shadow-[0_0_10px_rgba(0,242,255,0.2)]"
            >
              <Zap className="h-3 w-3" /> Auto-Calibrate All &alpha;
            </button>
            <button
              onClick={fetchRuns}
              className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 font-mono text-xs text-slate-300 hover:bg-white/10 transition-all"
            >
              <RefreshCw className="h-3 w-3" /> Refresh
            </button>
          </div>
        </div>

        <div className="overflow-x-auto rounded-xl border border-white/5 bg-black/40">
          <table className="w-full border-collapse font-mono text-xs text-left">
            <thead>
              <tr className="border-b border-white/10 bg-white/[0.03] text-slate-400 select-none">
                <th className="w-8 py-3 px-3"></th>
                <th
                  onClick={() => handleSort("domain")}
                  className="py-3 px-3 cursor-pointer hover:text-[#00f2ff] transition-colors"
                >
                  <div className="flex items-center gap-1">
                    <span>Domain</span>
                    <ArrowUpDown className="h-3 w-3 text-slate-600" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("adapter_version")}
                  className="py-3 px-3 cursor-pointer hover:text-[#00f2ff] transition-colors"
                >
                  <div className="flex items-center gap-1">
                    <span>Version</span>
                    <ArrowUpDown className="h-3 w-3 text-slate-600" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("optimal_alpha")}
                  className="py-3 px-3 text-center cursor-pointer hover:text-[#10b981] transition-colors"
                >
                  <div className="flex items-center justify-center gap-1 text-[#10b981] font-bold">
                    <span>Optimal &alpha; (&alpha;_opt)</span>
                    <ArrowUpDown className="h-3 w-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("final_dw_w")}
                  className="py-3 px-3 text-center cursor-pointer hover:text-[#00f2ff] transition-colors"
                >
                  <div className="flex items-center justify-center gap-1">
                    <span>||&Delta;W||/||W||</span>
                    <ArrowUpDown className="h-3 w-3 text-slate-600" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("predicted_merge_err")}
                  className="py-3 px-3 text-center cursor-pointer hover:text-[#a855f7] transition-colors"
                >
                  <div className="flex items-center justify-center gap-1">
                    <span>Merge Err</span>
                    <ArrowUpDown className="h-3 w-3 text-slate-600" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("adapter_size_mb")}
                  className="py-3 px-3 text-center cursor-pointer hover:text-white transition-colors"
                >
                  <div className="flex items-center justify-center gap-1">
                    <span>VRAM Size</span>
                    <ArrowUpDown className="h-3 w-3 text-slate-600" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("stopped_at_step")}
                  className="py-3 px-3 text-center cursor-pointer hover:text-white transition-colors"
                >
                  <div className="flex items-center justify-center gap-1">
                    <span>Steps</span>
                    <ArrowUpDown className="h-3 w-3 text-slate-600" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort("gate_passed")}
                  className="py-3 px-3 text-center cursor-pointer hover:text-white transition-colors"
                >
                  <div className="flex items-center justify-center gap-1">
                    <span>AIRM Gate</span>
                    <ArrowUpDown className="h-3 w-3 text-slate-600" />
                  </div>
                </th>
                <th className="py-3 px-3 text-center">Action &amp; Inspect</th>
              </tr>
            </thead>
            <tbody>
              {filteredAndSortedRuns.length === 0 ? (
                <tr>
                  <td colSpan={10} className="py-12 text-center text-slate-500 italic">
                    No runs found matching query.
                  </td>
                </tr>
              ) : (
                filteredAndSortedRuns.map((r) => {
                  const isExpanded = expandedRunId === r.run_id;
                  const calib = calibCache[r.run_id] || calibCache[r.domain];
                  const optAlpha = calib?.alpha_opt || r.alpha;

                  return (
                    <React.Fragment key={r.run_id}>
                      <tr
                        onClick={() => toggleExpand(r)}
                        className={`border-b border-white/5 cursor-pointer transition-colors ${
                          isExpanded ? "bg-white/[0.06]" : "hover:bg-white/[0.02]"
                        }`}
                      >
                        <td className="py-3 px-3 text-center">
                          {isExpanded ? (
                            <ChevronUp className="h-4 w-4 text-[#00f2ff]" />
                          ) : (
                            <ChevronDown className="h-4 w-4 text-slate-500" />
                          )}
                        </td>
                        <td className="py-3 px-3 font-bold text-white flex items-center gap-2">
                          <span>{r.domain}</span>
                          <span className="text-[10px] text-slate-500 font-normal">({r.run_id})</span>
                        </td>
                        <td className="py-3 px-3 text-slate-400">
                          <span className="rounded bg-white/5 px-2 py-0.5 text-[11px]">
                            {r.adapter_version || "v6"}
                          </span>
                        </td>
                        <td className="py-3 px-3 text-center">
                          <span className="rounded-lg bg-emerald-500/15 border border-emerald-500/30 px-2.5 py-1 text-xs font-bold text-[#10b981] shadow-[0_0_10px_rgba(16,185,129,0.2)]">
                            &alpha; = {optAlpha}
                          </span>
                        </td>
                        <td className="py-3 px-3 text-center font-bold text-[#00f2ff]">
                          {r.final_dw_w !== undefined && r.final_dw_w !== null
                            ? r.final_dw_w.toFixed(4)
                            : "-"}
                        </td>
                        <td className="py-3 px-3 text-center text-[#a855f7]">
                          {r.predicted_merge_err !== undefined && r.predicted_merge_err !== null
                            ? r.predicted_merge_err.toFixed(2) + "%"
                            : "-"}
                        </td>
                        <td className="py-3 px-3 text-center text-slate-300">
                          {(r.adapter_size_mb || 40.53).toFixed(2)} MB
                        </td>
                        <td className="py-3 px-3 text-center text-slate-300">
                          <b>{r.stopped_at_step || "-"}</b> / {r.max_steps || 150}
                        </td>
                        <td className="py-3 px-3 text-center">
                          {r.gate_passed ? (
                            <span className="inline-flex items-center gap-1 rounded bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 font-bold text-[#10b981]">
                              <CheckCircle className="h-3 w-3" /> PASS
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 rounded bg-rose-500/15 border border-rose-500/30 px-2 py-0.5 font-bold text-rose-400">
                              <XCircle className="h-3 w-3" /> FAIL
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-3 text-center" onClick={(e) => e.stopPropagation()}>
                          <button
                            disabled={isCalibrating === r.run_id}
                            onClick={() => handleCalibrate(r.domain, r.run_id)}
                            className="rounded-lg border border-[#00f2ff]/40 bg-[#00f2ff]/10 px-3 py-1 text-xs font-bold text-[#00f2ff] hover:bg-[#00f2ff]/20 disabled:opacity-50 transition-all shadow-[0_0_10px_rgba(0,242,255,0.15)]"
                          >
                            {isCalibrating === r.run_id ? "Optimizing..." : "⚡ Calibrate &amp; Save"}
                          </button>
                        </td>
                      </tr>

                      {/* Expandable Chart Row */}
                      {isExpanded && (
                        <tr className="bg-black/60 border-b border-white/10">
                          <td colSpan={10} className="p-6">
                            <div className="rounded-2xl border border-[rgba(0,242,255,0.3)] bg-[rgba(12,16,24,0.95)] p-6 shadow-2xl backdrop-blur-2xl space-y-4">
                              <div className="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-white/10">
                                <div>
                                  <div className="flex items-center gap-2">
                                    <span className="rounded bg-[#00f2ff]/20 border border-[#00f2ff]/40 px-2 py-0.5 font-mono text-xs font-bold text-[#00f2ff]">
                                      {r.domain}
                                    </span>
                                    <span className="font-mono text-xs text-slate-400">
                                      In-Place Alpha Calibration &amp; SQLite Ledger
                                    </span>
                                  </div>
                                  <h3 className="text-base font-bold text-white mt-1 flex items-center gap-2">
                                    <BarChart3 className="h-4 w-4 text-[#10b981]" />
                                    Goldilocks Zone (0.040 &le; ||&Delta;W||/||W|| &le; 0.085) &amp; IEEE 754 Precision Floor
                                  </h3>
                                </div>

                                {calib && (
                                  <div className="flex items-center gap-3">
                                    <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-center">
                                      <div className="text-[10px] text-slate-400">Optimal Alpha</div>
                                      <div className="font-mono text-sm font-bold text-[#10b981]">
                                        &alpha; = {calib.alpha_opt} ({calib.optimal_point.scaling.toFixed(1)}x)
                                      </div>
                                    </div>
                                    <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-center">
                                      <div className="text-[10px] text-slate-400">Precision Floor</div>
                                      <div className="font-mono text-sm font-bold text-[#f59e0b]">
                                        &alpha;_min = {calib.alpha_min}
                                      </div>
                                    </div>
                                    <div className="rounded-xl border border-white/10 bg-white/5 px-3 py-1.5 text-center">
                                      <div className="text-[10px] text-slate-400">BF16 Merge Error</div>
                                      <div className="font-mono text-sm font-bold text-[#a855f7]">
                                        {calib.optimal_point.merge_err_pct.toFixed(2)}%
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>

                              {/* Interactive Apache ECharts Canvas */}
                              <div className="h-[360px] w-full pt-2">
                                {calib ? (
                                  <AdaptiveChart options={getEchartsOption(calib)} theme="dark" />
                                ) : (
                                  <div className="h-full flex items-center justify-center text-slate-400 font-mono text-xs">
                                    Calculating calibration curve from adapter tensors...
                                  </div>
                                )}
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
