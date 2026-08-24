"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Bot,
  User,
  Zap,
  Sparkles,
  ChevronRight,
  Sliders,
  Check,
  Brain,
  Hash,
  RotateCcw,
  Square,
} from "lucide-react";
import { useEngineStatus } from "@/lib/useEngineStatus";
import { usePersistentState } from "@/lib/usePersistentState";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  meta?: string;
  team?: string[];
}

const AVAILABLE_ADAPTERS = [
  { id: "astral", name: "Astral (uv / ruff / packaging)", color: "#00f2ff", icon: "⚡" },
  { id: "postgresql", name: "PostgreSQL (pgvector / SQL)", color: "#60a5fa", icon: "🐘" },
  { id: "duckdb", name: "DuckDB (OLAP / Parquet)", color: "#f59e0b", icon: "🦆" },
  { id: "financial", name: "Financial Planning", color: "#a855f7", icon: "📈" },
  { id: "python_modern", name: "Python Modern (Clean Core)", color: "#10b981", icon: "🐍" },
  { id: "python_web", name: "Python Web (FastAPI / Pydantic)", color: "#ec4899", icon: "🌐" },
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "init",
      role: "assistant",
      content:
        "Hello! I am your Qwen3.5-4B Agentic Assistant running live on AMD Radeon RX 7900 XTX.\n\nYou have full flexible control over adapter morphing: choose Base Model (0 adapters), Single Adapter, Manual Multi-Adapter Stacking, or Dynamic Auto-Routing. You can also tune the Thinking Effort level and Max Tokens budget dynamically!",
      team: ["astral", "python_modern"],
      meta: "Active: astral + python_modern • 37.7 tok/s",
    },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  // Live status: shared SSE subscription (see lib/useEngineStatus.ts) instead of a
  // one-shot fetch on mount plus manual refreshStatus() calls after every toggle.
  // The backend pushes a new frame within ~0.5s of any /api/engine/set_* call
  // changing state, so nothing here needs to ask for a refresh -- it just arrives.
  const { status, patchStatus } = useEngineStatus();

  // Which model/adapters you're talking to, thinking effort, and max tokens all
  // persist across a refresh now (localStorage, see lib/usePersistentState.ts) --
  // these were plain useState before, so a reload silently dropped back to
  // "Auto (d_R)" / astral+python_modern / medium / 4096 regardless of what was
  // actually selected.

  // Stacking Mode: "auto" | "manual" | "base" | "single"
  const [routingMode, setRoutingMode] = usePersistentState<"auto" | "manual" | "base" | "single">(
    "gnn_chat_routingMode", "auto"
  );
  const [selectedAdapters, setSelectedAdapters] = usePersistentState<string[]>(
    "gnn_chat_selectedAdapters", ["astral", "python_modern"]
  );
  const [singleAdapter, setSingleAdapter] = usePersistentState<string>(
    "gnn_chat_singleAdapter", "postgresql"
  );
  const [minAdapters, setMinAdapters] = usePersistentState<number>("gnn_chat_minAdapters", 1);
  const [maxAdapters, setMaxAdapters] = usePersistentState<number>("gnn_chat_maxAdapters", 2);

  // Thinking Effort & Max Tokens Controls
  const [thinkingEffort, setThinkingEffort] = usePersistentState<"off" | "low" | "medium" | "high">(
    "gnn_chat_thinkingEffort", "medium"
  );
  const [maxTokens, setMaxTokens] = usePersistentState<number>("gnn_chat_maxTokens", 4096);

  // Telemetry & Prefold
  const [activeTeam, setActiveTeam] = useState<string[]>(["astral", "python_modern"]);
  const [intraDr, setIntraDr] = useState<number>(117.12);
  const [prefoldExpert, setPrefoldExpert] = useState("postgresql");
  const [prefoldConf, setPrefoldConf] = useState("95.0");
  const [morphAlert, setMorphAlert] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

  // active_team used to be pushed in from the one-shot status fetch on mount; now
  // it tracks the live SSE status whenever the engine reports a team, so a page
  // reload (or another client's action) is reflected here too, not just this tab's
  // own morph calls.
  useEffect(() => {
    if (status?.active_team && status.active_team.length > 0) {
      setActiveTeam(status.active_team);
    }
  }, [status?.active_team]);

  const [togglesBusy, setTogglesBusy] = useState<string | null>(null);
  const [toggleError, setToggleError] = useState<string | null>(null);

  const postEngine = async (path: string, body: object) => {
    const resp = await fetch(`/api/engine/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `${path} failed`);
    return data;
  };

  const toggleSpeculativeDecode = async () => {
    if (togglesBusy) return;
    setTogglesBusy("spec_decode");
    setToggleError(null);
    try {
      const data = await postEngine("set_speculative_decode", { enabled: !status?.spec_decode_enabled });
      // Reconcile from the response body immediately -- it's the authoritative new
      // state the backend just confirmed, not a guess. SSE will arrive too (this
      // patch is idempotent against it), but that could be up to ~0.5s behind; this
      // is what makes the toggle feel instant instead of "worked, but wait for it".
      patchStatus({
        spec_decode_enabled: data.spec_decode_enabled,
        spec_k: data.spec_k ?? status?.spec_k,
        ring_buffer_mode: data.ring_buffer_mode ?? status?.ring_buffer_mode,
      });
    } catch (e) {
      setToggleError(e instanceof Error ? e.message : String(e));
    } finally {
      setTogglesBusy(null);
    }
  };

  const setSpeculativeK = async (k: number) => {
    if (togglesBusy || k === status?.spec_k) return;
    setTogglesBusy("spec_k");
    setToggleError(null);
    try {
      const data = await postEngine("set_speculative_k", { k });
      patchStatus({ spec_k: data.spec_k, ring_buffer_mode: data.ring_buffer_mode ?? status?.ring_buffer_mode });
    } catch (e) {
      setToggleError(e instanceof Error ? e.message : String(e));
    } finally {
      setTogglesBusy(null);
    }
  };

  const setRingBufferMode = async (mode: string) => {
    if (togglesBusy || mode === status?.ring_buffer_mode) return;
    setTogglesBusy("ring_mode");
    setToggleError(null);
    try {
      const data = await postEngine("set_ring_buffer_mode", { mode });
      patchStatus({ ring_buffer_mode: data.ring_buffer_mode });
    } catch (e) {
      setToggleError(e instanceof Error ? e.message : String(e));
    } finally {
      setTogglesBusy(null);
    }
  };

  const setScaleMode = async (mode: string) => {
    if (togglesBusy || mode === status?.scale_mode) return;
    setTogglesBusy("scale_mode");
    setToggleError(null);
    try {
      const data = await postEngine("set_scale_mode", { mode });
      patchStatus({ scale_mode: data.scale_mode });
    } catch (e) {
      setToggleError(e instanceof Error ? e.message : String(e));
    } finally {
      setTogglesBusy(null);
    }
  };

  const togglePrefold = async () => {
    if (togglesBusy) return;
    setTogglesBusy("prefold");
    setToggleError(null);
    try {
      const data = await postEngine("set_prefold_enabled", { enabled: !status?.prefold_enabled });
      patchStatus({ prefold_enabled: data.prefold_enabled });
    } catch (e) {
      setToggleError(e instanceof Error ? e.message : String(e));
    } finally {
      setTogglesBusy(null);
    }
  };

  const toggleAdapterSelection = (adapterId: string) => {
    if (selectedAdapters.includes(adapterId)) {
      setSelectedAdapters(selectedAdapters.filter((id) => id !== adapterId));
    } else {
      if (selectedAdapters.length >= 4) {
        alert("Maximum 4 adapters can be stacked concurrently.");
        return;
      }
      setSelectedAdapters([...selectedAdapters, adapterId]);
    }
  };

  const presets = [
    {
      tag: "Turn 1: Tooling & Packaging",
      text: "Add ruff and ty as dev dependencies, then format and lint the whole codebase.",
    },
    {
      tag: "Turn 2: PostgreSQL & Vector Search",
      text: 'We store product descriptions in Postgres and want "find me similar products" without standing up new infrastructure.',
    },
    {
      tag: "Turn 3: Type-Safe Web API",
      text: "Write an async FastAPI endpoint with Pydantic request and response models and dependency injection.",
    },
    {
      tag: "Turn 4: Analytical / OLAP",
      text: "Aggregate a directory of parquet files and return the top 3 rows per group.",
    },
  ];

  const [stopping, setStopping] = useState(false);

  const stopGeneration = async () => {
    if (!isStreaming || stopping) return;
    setStopping(true);
    try {
      await fetch("/api/engine/stop_generation", { method: "POST" });
      // Not setIsStreaming(false) here -- the SSE read loop in handleSend still owns
      // that transition, once the now-terminated stream actually closes. Whatever
      // text already streamed into the bubble stays; this only stops more arriving.
    } catch (e) {
      console.error("stop_generation failed:", e);
    } finally {
      setStopping(false);
    }
  };

  const resetChat = () => {
    if (isStreaming) return;
    setMessages([
      {
        id: "init",
        role: "assistant",
        content:
          "Hello! I am your Qwen3.5-4B Agentic Assistant running live on AMD Radeon RX 7900 XTX.\n\nYou have full flexible control over adapter morphing: choose Base Model (0 adapters), Single Adapter, Manual Multi-Adapter Stacking, or Dynamic Auto-Routing. You can also tune the Thinking Effort level and Max Tokens budget dynamically!",
        team: ["astral", "python_modern"],
        meta: "Active: astral + python_modern • 37.7 tok/s",
      },
    ]);
  };

  const handleSend = async (promptText?: string) => {
    const textToSend = promptText || input;
    if (!textToSend.trim() || isStreaming) return;

    setInput("");
    const userMsgId = "u_" + Date.now();
    const asstMsgId = "a_" + Date.now();

    // Determine target model and team based on user selection mode
    let targetModel = "dynamic";
    let displayedTeam: string[] = [];

    if (routingMode === "base") {
      targetModel = "qwen3.5-4b-base";
      displayedTeam = ["Base (Pristine W0)"];
    } else if (routingMode === "single") {
      targetModel = singleAdapter;
      displayedTeam = [singleAdapter];
    } else if (routingMode === "manual") {
      if (selectedAdapters.length === 0) {
        targetModel = "qwen3.5-4b-base";
        displayedTeam = ["Base (Pristine W0)"];
      } else {
        targetModel = selectedAdapters[0];
        displayedTeam = selectedAdapters;
      }
    } else {
      // Automatic dynamic routing
      targetModel = "dynamic";
      const pLower = textToSend.toLowerCase();
      if (pLower.includes("postgres") || pLower.includes("asyncpg") || pLower.includes("vector") || pLower.includes("sql")) {
        displayedTeam = ["postgresql", "python_modern"];
      } else if (pLower.includes("fastapi") || pLower.includes("pydantic") || pLower.includes("route") || pLower.includes("endpoint")) {
        displayedTeam = ["python_web", "python_modern"];
      } else if (pLower.includes("duckdb") || pLower.includes("parquet") || pLower.includes("olap")) {
        displayedTeam = ["duckdb", "postgresql"];
      } else {
        displayedTeam = ["astral", "python_modern"];
      }
    }

    if (displayedTeam.length > 0 && JSON.stringify(displayedTeam) !== JSON.stringify(activeTeam)) {
      setMorphAlert(displayedTeam.join(" + "));
      setTimeout(() => {
        setActiveTeam(displayedTeam);
        setMorphAlert(null);
      }, 500);
    }

    setMessages((prev) => [
      ...prev,
      { id: userMsgId, role: "user", content: textToSend },
      {
        id: asstMsgId,
        role: "assistant",
        content: status?.loaded
          ? `⚡ [Morphing weights to ${displayedTeam.join(" + ")}...]`
          : `⚡ [Auto-loading model & adapters into VRAM...]`,
        team: displayedTeam,
        meta: `Active: ${displayedTeam.join(" + ")} • Connecting stream...`,
      },
    ]);

    setIsStreaming(true);

    // Full history, not just this turn. `messages` here is still the PRE-update
    // closure value (setMessages above hasn't committed yet), so it already excludes
    // the user/assistant pair just appended for this turn -- append textToSend once,
    // explicitly, rather than relying on state that hasn't landed.
    //
    // Two things filtered out deliberately:
    //   - id === "init": the synthetic welcome message. It's UI decoration ("running
    //     live on AMD Radeon RX 7900 XTX" etc), not a real turn -- sending it back as
    //     assistant-authored context would bias every conversation with boilerplate
    //     the model never actually said in response to anything.
    //   - m.reasoning: `content` already excludes it (extract_thinking_and_content
    //     splits them server-side, streamed separately). Sending reasoning back as
    //     part of an assistant turn's content would replay chain-of-thought into
    //     context that was never meant to persist.
    const history = messages
      .filter((m) => m.id !== "init" && m.content && !m.content.startsWith("⚡ ["))
      .map((m) => ({ role: m.role, content: m.content }));
    const fullMessages = [...history, { role: "user" as const, content: textToSend }];

    try {
      // Use direct streaming route with unbuffered SSE
      const resp = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: targetModel,
          messages: fullMessages,
          stream: true,
          max_tokens: maxTokens || 4096,
          thinking_effort: thinkingEffort,
        }),
      });

      if (!resp.ok) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === asstMsgId ? { ...m, content: `Error: ${resp.status} ${resp.statusText}` } : m
          )
        );
        setIsStreaming(false);
        return;
      }

      const reader = resp.body?.getReader();
      const decoder = new TextDecoder("utf-8");
      let accumulatedContent = "";
      let accumulatedReasoning = "";
      let sseBuffer = "";
      let displaySpeed = "37.5";

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        sseBuffer += decoder.decode(value, { stream: true });
        const lines = sseBuffer.split("\n");
        sseBuffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ") && !trimmed.includes("[DONE]")) {
            try {
              const data = JSON.parse(trimmed.slice(6));
              const delta = data.choices?.[0]?.delta || {};

              if (delta.reasoning_content) {
                accumulatedReasoning += delta.reasoning_content;
              }
              if (delta.content) {
                accumulatedContent += delta.content;
              }

              if (data.usage?.tokens_per_second) {
                displaySpeed = data.usage.tokens_per_second.toFixed(1);
              }

              let extraMeta = `Active: ${displayedTeam.join(" + ")} • ${displaySpeed} tok/s`;
              if (data.usage?.predicted_next_expert) {
                const nextExp = data.usage.predicted_next_expert;
                const conf = ((data.usage.predicted_confidence || 0.85) * 100).toFixed(1);
                setPrefoldExpert(nextExp);
                setPrefoldConf(conf);
                extraMeta += ` • 🕸️ Pre-folded: ${nextExp} (${conf}%)`;
              }

              // Update message text token by token immediately in UI
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === asstMsgId
                    ? {
                        ...m,
                        content: accumulatedContent,
                        reasoning: accumulatedReasoning,
                        meta: extraMeta,
                      }
                    : m
                )
              );
            } catch {
              // ignore parse errors
            }
          }
        }
      }
    } catch (e) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === asstMsgId ? { ...m, content: `Error streaming response: ${e}` } : m
        )
      );
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 h-[calc(100vh-140px)]">
      {/* Left Sidebar: Controls & Adapter Stacking Panel */}
      <div className="lg:col-span-4 flex flex-col gap-4 overflow-y-auto pr-1">
        {/* Adapter Stacking Configuration Card */}
        <div className="rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.85)] p-5 backdrop-blur-xl shadow-xl">
          <div className="flex items-center justify-between pb-3 border-b border-white/5">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Sliders className="h-4 w-4 text-[#00f2ff]" /> Adapter Stacking &amp; Routing
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                Target: {status?.model_id || "Qwen/Qwen3.5-4B"}
              </span>
            </div>
            <span className="rounded bg-white/5 px-2 py-0.5 font-mono text-[11px] text-slate-400">
              {(!status?.model_id || status.model_id.includes("4B")) ? routingMode.toUpperCase() : "BASE (0)"}
            </span>
          </div>

          {/* If not a 4B model, show architecture mismatch notice and lock to Base mode */}
          {status?.model_id && !status.model_id.includes("4B") ? (
            <div className="mt-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200/90 font-mono">
              <div className="font-bold flex items-center gap-1.5 text-amber-300 mb-1">
                <Sliders className="h-3.5 w-3.5" /> Architecture-Specific Adapter Boundary
              </div>
              Active model is <b>{status.model_id}</b>. Existing adapters are trained specifically for <b>Qwen 3.5 4B</b> ($d_m=2560$).
              <div className="mt-1.5 text-[10px] text-amber-300/80">
                0 adapters available for this size. Operating in pristine Base $W_0$ mode.
              </div>
            </div>
          ) : (
            <>
              {/* Mode Tabs */}
              <div className="grid grid-cols-4 gap-1 rounded-xl bg-black/40 p-1 border border-white/5 mt-3">
                {[
                  { id: "auto", label: "Auto (d_R)" },
                  { id: "manual", label: "Manual" },
                  { id: "single", label: "Single" },
                  { id: "base", label: "Base (0)" },
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setRoutingMode(tab.id as any)}
                    className={`rounded-lg py-1.5 text-[11px] font-bold font-mono transition-all ${
                      routingMode === tab.id
                        ? "bg-[#00f2ff] text-slate-950 shadow-[0_0_10px_rgba(0,242,255,0.3)]"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            </>
          )}

          {/* Context Options depending on Routing Mode */}
          {routingMode === "auto" && (
            <div className="mt-3 space-y-2 rounded-xl bg-white/[0.02] border border-white/5 p-3">
              <div className="text-[11px] font-bold text-slate-300">
                Riemannian Automatic Co-Routing
              </div>
              <p className="text-[11px] text-slate-400 leading-normal">
                Classifies prompt intent across 6 domain manifolds and stacks optimal adapters with minimum geodesic distance ($d_R$).
              </p>
              <div className="grid grid-cols-2 gap-2 pt-1 font-mono text-xs">
                <div>
                  <span className="text-[10px] text-slate-500">Min Stack</span>
                  <input
                    type="number"
                    min={1}
                    max={2}
                    value={minAdapters}
                    onChange={(e) => setMinAdapters(parseInt(e.target.value, 10))}
                    className="w-full rounded border border-white/10 bg-black/40 p-1 text-white text-xs mt-0.5"
                  />
                </div>
                <div>
                  <span className="text-[10px] text-slate-500">Max Stack</span>
                  <input
                    type="number"
                    min={2}
                    max={4}
                    value={maxAdapters}
                    onChange={(e) => setMaxAdapters(parseInt(e.target.value, 10))}
                    className="w-full rounded border border-white/10 bg-black/40 p-1 text-white text-xs mt-0.5"
                  />
                </div>
              </div>
            </div>
          )}

          {routingMode === "manual" && (
            <div className="mt-3 space-y-2">
              <div className="text-[11px] font-bold text-slate-300 flex justify-between items-center">
                <span>Select Custom Adapter Stack</span>
                <span className="font-mono text-[10px] text-[#00f2ff]">{selectedAdapters.length} Selected</span>
              </div>
              <div className="grid grid-cols-1 gap-1.5">
                {AVAILABLE_ADAPTERS.map((adapter) => {
                  const isChecked = selectedAdapters.includes(adapter.id);
                  return (
                    <button
                      key={adapter.id}
                      onClick={() => toggleAdapterSelection(adapter.id)}
                      className={`flex items-center justify-between rounded-xl p-2.5 text-xs font-mono transition-all border ${
                        isChecked
                          ? "border-[#00f2ff]/40 bg-[rgba(0,242,255,0.12)] text-white font-bold"
                          : "border-white/5 bg-black/30 text-slate-400 hover:border-white/10"
                      }`}
                    >
                      <span className="flex items-center gap-2">
                        <span>{adapter.icon}</span>
                        <span>{adapter.name}</span>
                      </span>
                      {isChecked && <Check className="h-3.5 w-3.5 text-[#00f2ff]" />}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {routingMode === "single" && (
            <div className="mt-3 space-y-2">
              <div className="text-[11px] font-bold text-slate-300">
                Single Adapter Solo Mode
              </div>
              <select
                value={singleAdapter}
                onChange={(e) => setSingleAdapter(e.target.value)}
                className="w-full rounded-xl border border-white/10 bg-black/50 p-2.5 text-xs font-mono text-white focus:border-[#00f2ff] focus:outline-none"
              >
                {AVAILABLE_ADAPTERS.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.icon} {a.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {routingMode === "base" && (
            <div className="mt-3 rounded-xl bg-white/[0.02] border border-white/5 p-3 text-xs text-slate-400">
              <b className="text-white">Pristine Base Model W0</b>
              <p className="mt-1 text-[11px]">
                No low-rank adapters folded. Queries run directly against un-enhanced base weights.
              </p>
            </div>
          )}
        </div>

        {/* Live Active Stack Card with Morph Animation & Intra-Team Geodesic */}
        <div className="rounded-2xl border border-[rgba(0,242,255,0.35)] bg-black/60 p-4 shadow-[0_0_25px_rgba(0,242,255,0.1)] backdrop-blur-xl">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[#00f2ff] flex items-center gap-1.5">
              <Zap className="h-3.5 w-3.5" /> Live Active Stack
            </span>
            <span className="rounded-md bg-white/5 border border-white/10 px-2 py-0.5 font-mono text-[11px] text-slate-300">
              45.67 ms
            </span>
          </div>

          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {activeTeam.map((exp) => (
              <span
                key={exp}
                className="flex items-center gap-1.5 rounded-lg border border-[rgba(0,242,255,0.4)] bg-[rgba(0,242,255,0.15)] px-2.5 py-1 font-mono text-xs font-bold text-[#00f2ff] shadow-[0_0_8px_rgba(0,242,255,0.2)]"
              >
                <Zap className="h-3 w-3" /> {exp}
              </span>
            ))}
          </div>

          <div className="mt-2.5 font-mono text-[11px] text-slate-400">
            Intra-Team Geodesic: <b className="text-[#00f2ff]">d_R = {intraDr.toFixed(2)}</b> (Calibrated Synergy)
          </div>

          {morphAlert && (
            <div className="mt-2.5 flex items-center gap-2 rounded-lg border border-[#00f2ff]/40 bg-[#00f2ff]/10 p-2 font-mono text-xs text-[#00f2ff] animate-pulse">
              <Zap className="h-3.5 w-3.5" />
              <span>Swapping Stack to: <b>{morphAlert}</b> (45.6 ms)</span>
            </div>
          )}
        </div>

        {/* NOTEARS Predictive Pre-Folder Card (Next in Line Prediction) */}
        <div className="rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-4 backdrop-blur-xl">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[#10b981] flex items-center gap-1.5">
              <Sparkles className="h-3.5 w-3.5" /> NOTEARS Pre-Folder
            </span>
            <span className="rounded-md bg-emerald-500/15 border border-emerald-500/30 px-2 py-0.5 font-mono text-[11px] text-[#10b981]">
              0.0ms Proactive
            </span>
          </div>
          <div className="mt-2 text-[11px] text-slate-400">Next Predicted Turn Expert:</div>
          <div className="mt-1 flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.02] p-2">
            <b className="font-mono text-xs text-[#00f2ff] flex items-center gap-1">
              <Zap className="h-3 w-3 text-[#00f2ff]" /> {prefoldExpert}
            </b>
            <span className="rounded bg-emerald-500/20 px-2 py-0.5 font-mono text-[10px] font-bold text-[#10b981]">
              {prefoldConf}% Conf
            </span>
          </div>
          <div className="mt-2 flex items-center gap-1.5 text-[10px] text-slate-400">
            <span className="h-1.5 w-1.5 rounded-full bg-[#10b981] animate-ping" />
            <span>Proactive in-VRAM weight-folding for next turn</span>
          </div>
        </div>

        {/* Engine Controls -- runtime toggles. None of these need a server restart or
            model reload: speculative decode / ring buffer mode rebuild a small graph
            or object in place (seconds or less); scale mode and pre-folding are pure
            per-request config reads. See /api/engine/set_* docstrings for exactly
            what each rebuild does and does not touch. */}
        <div className="rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-4 backdrop-blur-xl">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2.5 flex items-center gap-1.5">
            <Sliders className="h-3.5 w-3.5 text-[#00f2ff]" /> Engine Controls
          </span>

          {toggleError && (
            <div className="mt-2 rounded-lg border border-rose-400/30 bg-rose-500/10 px-2.5 py-1.5 text-[10px] text-rose-300 font-mono break-words">
              {toggleError}
            </div>
          )}

          <div className="mt-2.5 space-y-2.5">
            {/* Speculative Decode ON/OFF */}
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">Speculative Decode</span>
              <button
                onClick={toggleSpeculativeDecode}
                disabled={!status?.loaded || togglesBusy !== null}
                className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[10px] font-bold uppercase transition-all disabled:cursor-not-allowed disabled:opacity-40 ${
                  status?.spec_decode_enabled
                    ? "border-emerald-400/40 bg-emerald-500/15 text-emerald-300"
                    : "border-white/10 bg-black/40 text-slate-400"
                }`}
              >
                {togglesBusy === "spec_decode" ? "..." : status?.spec_decode_enabled ? "ON" : "OFF"}
              </button>
            </div>

            {/* Draft Depth (K) -- moved here from the header top bar, where it sat
                alone next to the model selector instead of with the rest of the
                engine options it's coupled to. */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[11px] text-slate-400">Draft Depth (K)</span>
                {!status?.spec_decode_enabled && (
                  <span className="text-[9px] text-amber-400/80">needs spec decode ON</span>
                )}
              </div>
              <div className="flex rounded-lg bg-black/40 p-0.5 border border-white/10">
                {(status?.spec_k_options || [2, 4, 8]).map((k) => (
                  <button
                    key={k}
                    onClick={() => setSpeculativeK(k)}
                    disabled={!status?.spec_decode_enabled || togglesBusy !== null}
                    className={`flex-1 rounded px-1.5 py-1 text-[10px] font-bold uppercase transition-all disabled:cursor-not-allowed disabled:opacity-30 ${
                      status?.spec_k === k
                        ? "bg-[#00f2ff] text-black shadow-[0_0_8px_rgba(0,242,255,0.5)]"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {togglesBusy === "spec_k" && status?.spec_k !== k ? "" : k}
                  </button>
                ))}
              </div>
            </div>

            {/* Predictive Pre-folding ON/OFF */}
            <div className="flex items-center justify-between">
              <span className="text-[11px] text-slate-400">Predictive Pre-folding</span>
              <button
                onClick={togglePrefold}
                disabled={!status?.loaded || togglesBusy !== null}
                className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[10px] font-bold uppercase transition-all disabled:cursor-not-allowed disabled:opacity-40 ${
                  status?.prefold_enabled
                    ? "border-emerald-400/40 bg-emerald-500/15 text-emerald-300"
                    : "border-white/10 bg-black/40 text-slate-400"
                }`}
              >
                {togglesBusy === "prefold" ? "..." : status?.prefold_enabled ? "ON" : "OFF"}
              </button>
            </div>

            {/* Ring Buffer Mode -- only meaningful while speculative decode is on */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[11px] text-slate-400">Ring Buffer Mode</span>
                {!status?.ring_buffer_wired && (
                  <span className="text-[9px] text-amber-400/80">needs spec decode ON</span>
                )}
              </div>
              <div className="flex rounded-lg bg-black/40 p-0.5 border border-white/10">
                {(status?.ring_buffer_options || ["dense", "poet", "selective_hybrid", "pointer"]).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setRingBufferMode(mode)}
                    disabled={!status?.ring_buffer_wired || togglesBusy !== null}
                    title={mode}
                    className={`flex-1 rounded px-1 py-1 text-[9px] font-bold uppercase transition-all disabled:cursor-not-allowed disabled:opacity-30 ${
                      status?.ring_buffer_mode === mode
                        ? "bg-[#a855f7] text-white shadow-[0_0_8px_rgba(168,85,247,0.5)]"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {mode === "selective_hybrid" ? "hybrid" : mode}
                  </button>
                ))}
              </div>
            </div>

            {/* Surgical Stacking vs plain additive */}
            <div>
              <span className="text-[11px] text-slate-400 block mb-1">Multi-Expert Stacking</span>
              <div className="flex rounded-lg bg-black/40 p-0.5 border border-white/10">
                {(["surgical", "none"] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setScaleMode(mode)}
                    disabled={!status?.loaded || togglesBusy !== null}
                    className={`flex-1 rounded px-1.5 py-1 text-[10px] font-bold uppercase transition-all disabled:cursor-not-allowed disabled:opacity-30 ${
                      status?.scale_mode === mode
                        ? "bg-[#a855f7] text-white shadow-[0_0_8px_rgba(168,85,247,0.5)]"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {mode === "surgical" ? "Surgical" : "Plain Additive"}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Preset Prompt Matrix */}
        <div className="flex-1 rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-4 backdrop-blur-xl flex flex-col">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2.5">
            Prompt-to-Prompt Test Matrix
          </span>
          <div className="space-y-1.5 flex-1 overflow-y-auto">
            {presets.map((preset, idx) => (
              <button
                key={idx}
                onClick={() => handleSend(preset.text)}
                className="w-full text-left rounded-xl border border-white/5 bg-black/40 p-2.5 text-xs transition-all hover:border-[#00f2ff]/40 hover:bg-[#00f2ff]/5 group"
              >
                <div className="font-mono font-bold text-[#00f2ff] text-[10px] mb-0.5 flex items-center justify-between">
                  <span>{preset.tag}</span>
                  <ChevronRight className="h-3 w-3 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
                <div className="text-slate-300 line-clamp-2 text-[11px]">&quot;{preset.text}&quot;</div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Right: Main Chat Stream Console */}
      <div className="lg:col-span-8 flex flex-col rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] backdrop-blur-xl overflow-hidden shadow-2xl">
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`flex gap-3 max-w-[85%] ${
                m.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"
              }`}
            >
              {/* Avatar */}
              <div
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-sm font-bold ${
                  m.role === "user"
                    ? "bg-white/10 border border-white/10 text-white"
                    : "bg-gradient-to-br from-[#00f2ff] to-[#a855f7] text-slate-950 shadow-[0_0_15px_rgba(0,242,255,0.4)]"
                }`}
              >
                {m.role === "user" ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4 fill-current" />}
              </div>

              {/* Message Bubble */}
              <div
                className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                  m.role === "user"
                    ? "border border-[#00f2ff]/40 bg-[#00f2ff]/15 text-white"
                    : "border border-white/10 bg-[rgba(12,16,24,0.85)] text-slate-100"
                }`}
              >
                {m.reasoning && (
                  <div className="mb-2.5 rounded-lg border-l-2 border-[#a855f7] bg-black/40 p-2.5 text-xs italic text-slate-400">
                    <div className="font-mono text-[10px] font-bold text-[#a855f7] mb-1">Reasoning Chain</div>
                    {m.reasoning}
                  </div>
                )}
                <div className="whitespace-pre-wrap">{m.content}</div>
                {m.meta && (
                  <div className="mt-2.5 flex items-center gap-2 font-mono text-[11px] text-slate-400 border-t border-white/5 pt-2">
                    <span>{m.meta}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar & Controls Header */}
        <div className="border-t border-[rgba(255,255,255,0.08)] bg-[rgba(12,16,24,0.95)] p-4 space-y-3">
          {/* Thinking Effort Pills and Max Tokens Input */}
          <div className="flex flex-wrap items-center justify-between gap-3 font-mono text-xs">
            {/* Thinking Effort Selector */}
            <div className="flex items-center gap-2">
              <span className="text-slate-400 flex items-center gap-1">
                <Brain className="h-3.5 w-3.5 text-[#a855f7]" /> Thinking Effort:
              </span>
              <div className="flex rounded-lg bg-black/40 p-0.5 border border-white/10">
                {(["off", "low", "medium", "high"] as const).map((level) => (
                  <button
                    key={level}
                    onClick={() => setThinkingEffort(level)}
                    className={`rounded px-2.5 py-1 text-[11px] font-bold uppercase transition-all ${
                      thinkingEffort === level
                        ? "bg-[#a855f7] text-white shadow-[0_0_8px_rgba(168,85,247,0.5)]"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {level}
                  </button>
                ))}
              </div>
            </div>

            {/* Max Tokens Number Input */}
            <div className="flex items-center gap-2">
              <span className="text-slate-400 flex items-center gap-1">
                <Hash className="h-3.5 w-3.5 text-[#00f2ff]" /> Max Tokens:
              </span>
              <input
                type="number"
                min={64}
                max={32768}
                step={256}
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value, 10) || 4096)}
                className="w-24 rounded-lg border border-white/10 bg-black/50 px-2.5 py-1 text-xs text-white font-mono focus:border-[#00f2ff] focus:outline-none"
              />
            </div>

            {/* Reset Chat -- clears displayed history AND the context sent to the
                server on the next turn, since both now come from the same `messages`
                state. */}
            <button
              onClick={resetChat}
              disabled={isStreaming}
              title="Clear conversation history and start fresh"
              className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-black/40 px-2.5 py-1 text-[11px] font-bold uppercase text-slate-400 transition-all hover:border-rose-400/40 hover:text-rose-300 disabled:cursor-not-allowed disabled:opacity-40"
            >
              <RotateCcw className="h-3.5 w-3.5" /> Reset Chat
            </button>
          </div>

          {/* Text Input and Send Button */}
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Ask a question (e.g. uv packaging, PostgreSQL vector search, FastAPI, DuckDB)..."
              className="flex-1 rounded-xl border border-white/10 bg-black/50 px-4 py-3 text-sm text-white placeholder-slate-500 focus:border-[#00f2ff] focus:outline-none focus:ring-1 focus:ring-[#00f2ff]"
            />
            {isStreaming ? (
              <button
                disabled={stopping}
                onClick={stopGeneration}
                title="Stop the current response -- does not unload the model"
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-rose-500 to-rose-600 px-6 py-3 text-sm font-bold text-white shadow-[0_0_20px_rgba(244,63,94,0.35)] transition-all hover:scale-[1.02] disabled:opacity-50"
              >
                <Square className="h-4 w-4 fill-current" />
                <span>{stopping ? "Stopping..." : "Stop"}</span>
              </button>
            ) : (
              <button
                disabled={isStreaming}
                onClick={() => handleSend()}
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-[#00f2ff] to-[#a855f7] px-6 py-3 text-sm font-bold text-slate-950 shadow-[0_0_20px_rgba(0,242,255,0.3)] transition-all hover:scale-[1.02] disabled:opacity-50"
              >
                <Send className="h-4 w-4 fill-current" />
                <span>Send &amp; Morph ⚡</span>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
