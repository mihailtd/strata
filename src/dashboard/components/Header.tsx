"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Zap, Activity, MessageSquare, Factory, Network, BookOpen, Cpu, ChevronDown } from "lucide-react";
import { useEngineStatus } from "@/lib/useEngineStatus";

export const AVAILABLE_BASE_MODELS = [
  { id: "Qwen/Qwen3.5-4B", name: "Qwen 3.5 4B (Dense BF16)", vram: "~10.5 GB", adapters_count: 6, available: true },
  { id: "Qwen/Qwen3.5-2B", name: "Qwen 3.5 2B (Dense BF16)", vram: "~4.8 GB", adapters_count: 0, available: true },
  { id: "Qwen/Qwen3.5-0.8B", name: "Qwen 3.5 0.8B (Draft Head)", vram: "~2.2 GB", adapters_count: 0, available: true },
  { id: "Qwen/Qwen3.5-9B", name: "Qwen 3.5 9B (Dense BF16)", vram: "~18.2 GB", adapters_count: 0, available: true },
];

export default function Header() {
  const pathname = usePathname();
  // Live status: one SSE subscription shared across every component via the hook
  // (see lib/useEngineStatus.ts). Header used to poll GET /api/engine/status on its
  // own 3s setInterval, independently of whatever chat/page.tsx was doing -- the two
  // could show different snapshots for up to 3s after any toggle. Both now read the
  // same push-on-change stream, so nothing here calls fetchStatus() after an action
  // anymore: the SSE frame for the new state arrives on its own, typically within a
  // second of the backend committing it.
  const { status, refetch } = useEngineStatus();
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("Qwen/Qwen3.5-4B");
  const [showModelDropdown, setShowModelDropdown] = useState(false);

  const toggleEngine = async () => {
    // No `if (!status) return` guard: status can legitimately stay null for a
    // moment while the SSE connection (lib/useEngineStatus.ts) is still
    // establishing -- the old one-shot fetch always resolved eventually, this
    // doesn't block the same way. Blocking here on a still-null status is exactly
    // what made the button do nothing after the SSE switch. status?.loaded is
    // already null-safe everywhere else below.
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
      // load/unload's response isn't a typed EngineStatus subset the way the
      // sidebar toggles' responses are (see page.tsx's patchStatus calls) -- it's
      // its own ad-hoc shape. A full refetch is simpler and correct here, and this
      // is not a hot path: load/unload are infrequent, multi-second operations
      // already, so one extra GET is free.
      await refetch();
    } catch (e) {
      alert("Failed to toggle engine: " + e);
    } finally {
      setIsLoading(false);
    }
  };

  const navItems = [
    { href: "/", label: "Dynamic Agent Matrix", icon: Activity },
    { href: "/chat", label: "Morphing Studio", icon: MessageSquare },
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
            disabled={status?.loaded || isLoading}
            className="flex items-center gap-1.5 rounded-lg border border-[rgba(255,255,255,0.1)] bg-black/40 px-3 py-1.5 text-xs font-mono text-slate-300 hover:border-[#00f2ff]/40 transition-colors disabled:opacity-60"
          >
            <Cpu className="h-3.5 w-3.5 text-[#00f2ff]" />
            <span className="font-bold">{AVAILABLE_BASE_MODELS.find((m) => m.id === selectedModel)?.name.split(" ")[0]}</span>
            <ChevronDown className="h-3 w-3 text-slate-400" />
          </button>

          {showModelDropdown && !status?.loaded && (
            <div className="absolute left-0 top-full mt-2 w-64 rounded-xl border border-[rgba(255,255,255,0.1)] bg-[rgba(12,16,24,0.95)] p-2 shadow-2xl backdrop-blur-2xl z-50">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 px-2 py-1">
                Base Architecture
              </div>
              {AVAILABLE_BASE_MODELS.map((m) => (
                <button
                  key={m.id}
                  disabled={!m.available}
                  onClick={() => {
                    setSelectedModel(m.id);
                    setShowModelDropdown(false);
                  }}
                  className={`w-full text-left rounded-lg p-2 text-xs transition-all flex flex-col gap-0.5 ${
                    selectedModel === m.id
                      ? "bg-[#00f2ff]/15 border border-[#00f2ff]/30 text-white font-bold"
                      : m.available
                      ? "hover:bg-white/5 text-slate-300"
                      : "opacity-40 cursor-not-allowed text-slate-500"
                  }`}
                >
                  <div className="flex justify-between items-center">
                    <span>{m.name}</span>
                    <span className="font-mono text-[10px] text-[#00f2ff]">{m.vram}</span>
                  </div>
                  {!m.available && <span className="text-[10px] text-amber-400">Coming soon</span>}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Load/Unload Button */}
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

        {/* Status Pill */}
        <div className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 font-mono text-xs font-semibold text-emerald-400">
          <div
            className={`h-2 w-2 rounded-full ${
              status?.loaded ? "bg-emerald-400 shadow-[0_0_8px_#10b981] animate-pulse" : "bg-amber-400"
            }`}
          />
          <span>
            {status?.loaded
              ? `ACTIVE (${status.vram_allocated_gb} GB)`
              : "STANDBY (0 GB)"}
          </span>
        </div>
      </div>
    </header>
  );
}
