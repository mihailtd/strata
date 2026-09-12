"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Zap,
  Activity,
  MessageSquare,
  Factory,
  Network,
  BookOpen,
  Cpu,
  ChevronDown,
  Layers,
} from "lucide-react";
import { useEngineStatus } from "@/lib/useEngineStatus";

export const AVAILABLE_BASE_MODELS = [
  {
    id: "Qwen/Qwen3.5-4B",
    name: "Qwen 3.5 4B (Dense BF16)",
    vram: "~10.5 GB",
    adapters_count: 6,
    available: true,
  },
  {
    id: "Qwen/Qwen3.5-9B",
    name: "Qwen 3.5 9B (Dense BF16)",
    vram: "~18.2 GB",
    adapters_count: 6,
    available: true,
  },
  {
    id: "qwen3.8:27b",
    name: "Qwen 3.8 27B (Native W4A16 + LoRA)",
    vram: "~16.8 GB",
    adapters_count: 6,
    available: true,
  },
  {
    id: "Qwen/Qwen3.5-2B",
    name: "Qwen 3.5 2B (Dense BF16)",
    vram: "~4.8 GB",
    adapters_count: 0,
    available: false,
  },
  {
    id: "Qwen/Qwen3.5-0.8B",
    name: "Qwen 3.5 0.8B (Draft Head)",
    vram: "~2.2 GB",
    adapters_count: 0,
    available: false,
  },
];

export default function Header() {
  const pathname = usePathname();
  const { status, refetch } = useEngineStatus();
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("Qwen/Qwen3.5-4B");
  const [showModelDropdown, setShowModelDropdown] = useState(false);

  // Sync selectedModel with live loaded model
  React.useEffect(() => {
    if (status?.model_id && status?.loaded) {
      setSelectedModel(status.model_id);
    }
  }, [status?.model_id, status?.loaded]);

  const activeModelId = status?.loaded ? status.model_id || selectedModel : selectedModel;
  const activeModelMeta =
    AVAILABLE_BASE_MODELS.find((m) => m.id === activeModelId) || AVAILABLE_BASE_MODELS[0];

  const handleSelectModel = async (modelId: string) => {
    setSelectedModel(modelId);
    setShowModelDropdown(false);

    if (status?.loaded && status?.model_id !== modelId) {
      // Direct model switch
      setIsLoading(true);
      try {
        await fetch("/api/engine/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_id: modelId }),
        });
        await refetch();
      } catch (e) {
        alert("Failed to switch model: " + e);
      } finally {
        setIsLoading(false);
      }
    }
  };

  const toggleEngine = async () => {
    setIsLoading(true);
    try {
      if (status?.loaded) {
        await fetch("/api/engine/unload", { method: "POST" });
      } else {
        await fetch("/api/engine/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_id: selectedModel }),
        });
      }
      await refetch();
    } catch (e) {
      alert("Failed to toggle engine: " + e);
    } finally {
      setIsLoading(false);
    }
  };

  const resetVram = async () => {
    setIsLoading(true);
    try {
      await fetch("/api/engine/force_reset_vram", { method: "POST" });
      await refetch();
    } catch (e) {
      alert("Failed to reset VRAM: " + e);
    } finally {
      setIsLoading(false);
    }
  };

  const navItems = [
    { href: "/", label: "Dynamic Agent Matrix", icon: Activity },
    { href: "/chat", label: "Morphing Studio", icon: MessageSquare },
    { href: "/pipeline", label: "Multi-Agent Studio", icon: Layers },
    { href: "/dag", label: "NOTEARS Causal Graph", icon: Network },
    { href: "/training", label: "Training Factory & Audit", icon: Factory },
    { href: "/docs", label: "Research Docs", icon: BookOpen },
  ];

  return (
    <header className="sticky top-0 z-50 flex flex-wrap items-center justify-between gap-4 border-b border-[rgba(255,255,255,0.08)] bg-[rgba(12,16,24,0.85)] px-6 py-3.5 backdrop-blur-xl">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-[#00f2ff] to-[#a855f7] text-xl shadow-[0_0_20px_rgba(0,242,255,0.4)]">
          <Zap className="h-5 w-5 text-slate-950 fill-current" />
        </div>
        <div>
          <div className="bg-gradient-to-r from-white via-slate-100 to-[#00f2ff] bg-clip-text text-lg font-extrabold tracking-tight text-transparent">
            Autonomous Runtime Engine
          </div>
          <div className="font-mono text-xs text-slate-400">
            ROCm 7.2 • In-Place Weight-Folding &amp; Riemannian Topology
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav className="flex items-center gap-1 rounded-xl border border-[rgba(255,255,255,0.08)] bg-black/40 p-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-bold transition-all ${
                isActive
                  ? "border border-[rgba(0,242,255,0.4)] bg-gradient-to-r from-[rgba(0,242,255,0.2)] to-[rgba(168,85,247,0.2)] text-white shadow-[0_0_15px_rgba(0,242,255,0.2)]"
                  : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Engine Controls, Model Selector & Speculative K */}
      <div className="flex items-center gap-3">
        {/* Model Selector Dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowModelDropdown(!showModelDropdown)}
            disabled={isLoading}
            className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-mono transition-all ${
              status?.loaded
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300 hover:border-emerald-400"
                : "border-[rgba(255,255,255,0.1)] bg-black/40 text-slate-300 hover:border-[#00f2ff]/40"
            } disabled:opacity-60`}
          >
            <Cpu
              className={`h-3.5 w-3.5 ${status?.loaded ? "text-emerald-400" : "text-[#00f2ff]"}`}
            />
            <span className="font-bold">{activeModelMeta.name.split(" (")[0]}</span>
            {status?.loaded && (
              <span className="rounded bg-emerald-500/20 px-1.5 py-0.2 text-[10px] text-emerald-300 font-semibold">
                LIVE
              </span>
            )}
            <ChevronDown className="h-3 w-3 text-slate-400" />
          </button>

          {showModelDropdown && (
            <div className="absolute left-0 top-full mt-2 w-72 rounded-xl border border-[rgba(255,255,255,0.1)] bg-[rgba(12,16,24,0.98)] p-2 shadow-2xl backdrop-blur-2xl z-50">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 px-2 py-1 flex justify-between">
                <span>Select Architecture</span>
                {status?.loaded && (
                  <span className="text-emerald-400 font-mono">
                    Live: {status.model_id?.split("/")[1]}
                  </span>
                )}
              </div>
              {AVAILABLE_BASE_MODELS.map((m) => {
                const isCurrentLive = status?.loaded && status.model_id === m.id;
                const isSelected = selectedModel === m.id;
                return (
                  <button
                    key={m.id}
                    disabled={!m.available || isLoading}
                    onClick={() => handleSelectModel(m.id)}
                    className={`w-full text-left rounded-lg p-2 text-xs transition-all flex flex-col gap-0.5 mt-1 ${
                      isCurrentLive
                        ? "bg-emerald-500/15 border border-emerald-500/40 text-white font-bold"
                        : isSelected
                          ? "bg-[#00f2ff]/15 border border-[#00f2ff]/30 text-white font-bold"
                          : m.available
                            ? "hover:bg-white/5 text-slate-300"
                            : "opacity-40 cursor-not-allowed text-slate-500"
                    }`}
                  >
                    <div className="flex justify-between items-center">
                      <span className="flex items-center gap-1.5">
                        {isCurrentLive && (
                          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                        )}
                        {m.name}
                      </span>
                      <span className="font-mono text-[10px] text-[#00f2ff]">{m.vram}</span>
                    </div>
                    <div className="flex justify-between items-center text-[10px] text-slate-400">
                      <span>
                        {m.adapters_count > 0 ? `6 Domain Adapters Ready` : "No Adapters"}
                      </span>
                      {!m.available && <span className="text-amber-400">Coming soon</span>}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Load/Unload & VRAM Reset Controls */}
        <div className="flex items-center gap-2">
          {status?.has_residual_vram && !status?.loaded && (
            <button
              disabled={isLoading}
              onClick={resetVram}
              title="Force clear residual GPU VRAM and reset PyTorch/HIP memory allocator"
              className="flex items-center gap-1.5 rounded-lg border border-amber-500/40 bg-amber-500/15 px-3 py-1.5 text-xs font-bold text-amber-300 hover:bg-amber-500/25 transition-all shadow-[0_0_10px_rgba(245,158,11,0.2)] disabled:opacity-50"
            >
              <Zap className="h-3.5 w-3.5" />
              {isLoading
                ? "Purging..."
                : `Purge Residual VRAM (${status.total_vram_used_gb || status.residual_vram_gb || 0} GB)`}
            </button>
          )}

          <button
            disabled={isLoading}
            onClick={toggleEngine}
            className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-bold transition-all border ${
              status?.loaded
                ? "border-rose-500/40 bg-rose-500/15 text-rose-400 hover:bg-rose-500/25"
                : "border-[rgba(0,242,255,0.4)] bg-[rgba(0,242,255,0.15)] text-[#00f2ff] hover:bg-[rgba(0,242,255,0.25)] shadow-[0_0_12px_rgba(0,242,255,0.2)]"
            } disabled:opacity-50`}
          >
            <Zap className="h-3.5 w-3.5" />
            {isLoading
              ? "Working..."
              : status?.loaded
                ? `Unload VRAM (${status.vram_allocated_gb} GB)`
                : "Load Engine to VRAM"}
          </button>
        </div>

        {/* Status Pill */}
        <div className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 font-mono text-xs font-semibold text-emerald-400">
          <div
            className={`h-2 w-2 rounded-full ${
              status?.loaded
                ? "bg-emerald-400 shadow-[0_0_8px_#10b981] animate-pulse"
                : status?.has_residual_vram
                  ? "bg-amber-400 animate-ping"
                  : "bg-slate-400"
            }`}
          />
          <span>
            {status?.loaded
              ? `ACTIVE (${status.vram_allocated_gb} GB)`
              : status?.has_residual_vram
                ? `RESIDUAL (${status.total_vram_used_gb || status.residual_vram_gb || 0} GB)`
                : "STANDBY (0 GB)"}
          </span>
        </div>
      </div>
    </header>
  );
}
