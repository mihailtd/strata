"use client";

import React, { useState, useRef, useEffect, useMemo } from "react";
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
  Cpu,
  Layers,
  FlaskConical,
  Activity,
  AlertTriangle,
  Play,
} from "lucide-react";
import { useEngineStatus } from "@/lib/useEngineStatus";
import { usePersistentState } from "@/lib/usePersistentState";
import { LaymanTooltip } from "@/components/LaymanTooltip";

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
        "Hello! I am your Qwen3.5 Agentic Assistant running live on AMD Radeon RX 7900 XTX.\n\nYou have full flexible control over adapter morphing: choose Base Model (0 adapters), Single Adapter, Manual Multi-Adapter Stacking, or Dynamic Auto-Routing. You can also tune the Thinking Effort level and Max Tokens budget dynamically!",
      team: ["astral", "python_modern"],
      meta: "Active: astral + python_modern • 19.8 tok/s",
    },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const { status, patchStatus } = useEngineStatus();

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
    "gnn_chat_thinkingEffort", "off"
  );
  const [maxTokens, setMaxTokens] = usePersistentState<number>("gnn_chat_maxTokens", 400);

  // Telemetry & Prefold
  const [activeTeam, setActiveTeam] = useState<string[]>(["astral", "python_modern"]);
  const [intraDr, setIntraDr] = useState<number>(117.12);
  const [prefoldExpert, setPrefoldExpert] = useState("postgresql");
  const [prefoldConf, setPrefoldConf] = useState("95.0");
  const [morphAlert, setMorphAlert] = useState<string | null>(null);

  // Right section prompt category tab
  const [promptTab, setPromptTab] = useState<"multiturn" | "hybrid" | "probes">("multiturn");

  // Reactive derived stack based on current routing mode and selections
  const currentActiveStack = useMemo(() => {
    if (routingMode === "base") return ["Base (Pristine W0)"];
    if (routingMode === "single") return [singleAdapter];
    if (routingMode === "manual") return selectedAdapters.length > 0 ? selectedAdapters : ["Base (Pristine W0)"];
    return activeTeam.length > 0 ? activeTeam : ["Base (Pristine W0)"];
  }, [routingMode, singleAdapter, selectedAdapters, activeTeam]);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isStreaming]);

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

  const toggleRangeGate = async () => {
    if (togglesBusy) return;
    setTogglesBusy("range_gate");
    setToggleError(null);
    try {
      const data = await postEngine("set_speculative_range_gate", {
        enabled: !status?.spec_range_gate_enabled,
      });
      patchStatus({
        spec_range_gate_enabled: data.spec_range_gate_enabled,
        spec_range_threshold: data.spec_range_threshold,
      });
    } catch (e) {
      setToggleError(e instanceof Error ? e.message : String(e));
    } finally {
      setTogglesBusy(null);
    }
  };

  const toggleCircuitBreaker = async () => {
    if (togglesBusy) return;
    setTogglesBusy("circuit_breaker");
    setToggleError(null);
    try {
      const data = await postEngine("set_spec_circuit_breaker", {
        enabled: status?.spec_circuit_breaker_enabled === false,
      });
      patchStatus({
        spec_circuit_breaker_enabled: data.spec_circuit_breaker_enabled,
        macd_disengage_threshold: data.disengage_threshold,
        macd_reengage_threshold: data.reengage_threshold,
      });
    } catch (e) {
      setToggleError(e instanceof Error ? e.message : String(e));
    } finally {
      setTogglesBusy(null);
    }
  };

  const toggleRenkoSmoothing = async () => {
    if (togglesBusy) return;
    setTogglesBusy("renko");
    setToggleError(null);
    try {
      const data = await postEngine("set_renko_smoothing", {
        enabled: !status?.renko_smoothing_enabled,
        epsilon: status?.renko_epsilon ?? 5.0,
      });
      patchStatus({
        renko_smoothing_enabled: data.renko_smoothing_enabled,
        renko_epsilon: data.renko_epsilon,
      });
    } catch (e) {
      setToggleError(e instanceof Error ? e.message : String(e));
    } finally {
      setTogglesBusy(null);
    }
  };

  const toggleThinkingSupervisor = async () => {
    if (togglesBusy) return;
    setTogglesBusy("thinking_sup");
    setToggleError(null);
    try {
      const data = await postEngine("set_thinking_supervisor", {
        enabled: status?.thinking_supervisor_enabled === false,
      });
      patchStatus({
        thinking_supervisor_enabled: data.thinking_supervisor_enabled,
      });
    } catch (e) {
      setToggleError(e instanceof Error ? e.message : String(e));
    } finally {
      setTogglesBusy(null);
    }
  };

  const setRenkoEpsilon = async (epsilon: number) => {
    if (togglesBusy || epsilon === status?.renko_epsilon) return;
    setTogglesBusy("renko_eps");
    setToggleError(null);
    try {
      const data = await postEngine("set_renko_smoothing", {
        enabled: status?.renko_smoothing_enabled ?? true,
        epsilon,
      });
      patchStatus({
        renko_smoothing_enabled: data.renko_smoothing_enabled,
        renko_epsilon: data.renko_epsilon,
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

  const toggleStateHandoff = async () => {
    if (togglesBusy) return;
    setTogglesBusy("state_handoff");
    setToggleError(null);
    try {
      const data = await postEngine("set_state_handoff", { enabled: !status?.state_handoff_enabled });
      patchStatus({
        state_handoff_enabled: data.state_handoff_enabled,
        state_handoff_mb: data.state_handoff_mb,
      });
    } catch (e) {
      setToggleError(e instanceof Error ? e.message : String(e));
    } finally {
      setTogglesBusy(null);
    }
  };

  const toggleW4A16 = async () => {
    if (togglesBusy) return;
    setTogglesBusy("w4a16");
    setToggleError(null);
    try {
      const data = await postEngine("set_w4a16", { enabled: !status?.w4a16_enabled });
      patchStatus({
        w4a16_enabled: data.w4a16_enabled,
      });
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
      setSelectedAdapters([...selectedAdapters, adapterId]);
    }
  };

  // Predefined prompts organized into tabs
  const promptTabs = {
    multiturn: [
      {
        tag: "Turn 1: Astral Tooling",
        expert: "astral",
        text: "Add ruff and ty as dev dependencies, then format and lint the whole codebase.",
        desc: "uv add --dev, ruff check, ruff format",
      },
      {
        tag: "Turn 2: PostgreSQL pgvector",
        expert: "postgresql",
        text: "We store product descriptions in Postgres and want 'find me similar products' without standing up new infrastructure.",
        desc: "pgvector cosine hnsw indexing",
      },
      {
        tag: "Turn 3: FastAPI Web API",
        expert: "python_web",
        text: "Write an async FastAPI endpoint with Pydantic request and response models and dependency injection.",
        desc: "FastAPI router, pydantic BaseModel, Depends",
      },
      {
        tag: "Turn 4: DuckDB OLAP",
        expert: "duckdb",
        text: "Aggregate a directory of parquet files and return the top 3 rows per group.",
        desc: "DuckDB parquet glob, ROW_NUMBER window query",
      },
    ],
    hybrid: [
      {
        tag: "Hybrid: FastAPI + DuckDB",
        expert: "python_web + duckdb",
        text: "Build an async FastAPI streaming endpoint that queries a local DuckDB parquet lakehouse using arrow streaming and returns ndjson.",
        desc: "Simultaneous Python web routing + DuckDB in-process analytics",
      },
      {
        tag: "Hybrid: Postgres + Modern Python",
        expert: "postgresql + python_modern",
        text: "Write a Python asyncpg connection pool helper with robust retry backoff and a pgvector cosine similarity search function.",
        desc: "Modern Python 3.12+ async pool + pgvector query builder",
      },
      {
        tag: "Hybrid: uv Workspace + Python 3.12",
        expert: "astral + python_modern",
        text: "Create a uv workspace configuration with multiple Python packages, custom ruff lint rules for modern Python 3.12+ syntax, and ty typechecking.",
        desc: "Astral workspace packaging + modern type system",
      },
    ],
    probes: [
      {
        tag: "Specialist: Advanced uv compile",
        expert: "astral",
        text: "How do I use uv to compile locked dependencies for multiple platforms into standard requirements files?",
        desc: "uv pip compile multi-platform lockfile generation",
      },
      {
        tag: "Specialist: Postgres GIN & Partitioning",
        expert: "postgresql",
        text: "Design a declarative time-partitioned table in PostgreSQL 16 with a GIN index on a JSONB metadata payload and an automated cleanup retention policy.",
        desc: "Table partitioning, GIN index on jsonb_path_ops",
      },
      {
        tag: "Specialist: DuckDB QUALIFY & ASOF",
        expert: "duckdb",
        text: "Write a DuckDB SQL query using QUALIFY, ASOF JOIN, and window functions to compute rolling 7-day user churn over partitioned iceberg files.",
        desc: "QUALIFY window filter, temporal ASOF joins",
      },
    ],
  };

  const [stopping, setStopping] = useState(false);

  const stopGeneration = async () => {
    if (!isStreaming || stopping) return;
    setStopping(true);
    try {
      await fetch("/api/engine/stop_generation", { method: "POST" });
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
          "Hello! I am your Qwen3.5 Agentic Assistant running live on AMD Radeon RX 7900 XTX.\n\nYou have full flexible control over adapter morphing: choose Base Model (0 adapters), Single Adapter, Manual Multi-Adapter Stacking, or Dynamic Auto-Routing. You can also tune the Thinking Effort level and Max Tokens budget dynamically!",
        team: ["astral", "python_modern"],
        meta: "Active: astral + python_modern • 19.8 tok/s",
      },
    ]);
  };

  const handleSend = async (promptText?: string) => {
    const textToSend = promptText || input;
    if (!textToSend.trim() || isStreaming) return;

    setInput("");
    const userMsgId = "u_" + Date.now();
    const asstMsgId = "a_" + Date.now();

    let targetModel = "dynamic";
    let displayedTeam: string[] = [];

    const is9b = status?.model_id?.includes("9B") || status?.model_id?.includes("9b");
    const baseAlias = is9b ? "qwen3.5-9b-base" : "qwen3.5-4b-base";
    const expertPrefix = is9b ? "qwen3.5-9b" : "qwen3.5-4b";

    if (routingMode === "base") {
      targetModel = baseAlias;
      displayedTeam = ["Base (Pristine W0)"];
    } else if (routingMode === "single") {
      targetModel = `${expertPrefix}-${singleAdapter}`;
      displayedTeam = [singleAdapter];
    } else if (routingMode === "manual") {
      if (selectedAdapters.length === 0) {
        targetModel = baseAlias;
        displayedTeam = ["Base (Pristine W0)"];
      } else {
        targetModel = `${expertPrefix}-${selectedAdapters[0]}`;
        displayedTeam = selectedAdapters;
      }
    } else {
      targetModel = "dynamic";
      const pLower = textToSend.toLowerCase();
      const matched: string[] = [];
      if (pLower.includes("postgres") || pLower.includes("asyncpg") || pLower.includes("vector") || pLower.includes("sql") || pLower.includes("database")) {
        matched.push("postgresql");
      }
      if (pLower.includes("fastapi") || pLower.includes("pydantic") || pLower.includes("route") || pLower.includes("endpoint") || pLower.includes("http") || pLower.includes("web")) {
        matched.push("python_web");
      }
      if (pLower.includes("duckdb") || pLower.includes("parquet") || pLower.includes("olap") || pLower.includes("analytics") || pLower.includes("arrow")) {
        matched.push("duckdb");
      }
      if (pLower.includes("finance") || pLower.includes("stock") || pLower.includes("portfolio") || pLower.includes("interest") || pLower.includes("valuation") || pLower.includes("dividend")) {
        matched.push("financial");
      }
      if (pLower.includes("uv") || pLower.includes("ruff") || pLower.includes("ty") || pLower.includes("pip") || pLower.includes("package") || pLower.includes("astral")) {
        matched.push("astral");
      }
      if (pLower.includes("python") || pLower.includes("def ") || pLower.includes("class ") || pLower.includes("async") || pLower.includes("type")) {
        if (!matched.includes("python_modern")) matched.push("python_modern");
      }

      if (matched.length === 0) {
        if (minAdapters === 0) {
          targetModel = baseAlias;
          displayedTeam = ["Base (Pristine W0)"];
        } else {
          displayedTeam = ["astral", "python_modern"].slice(0, Math.max(minAdapters, 1));
        }
      } else {
        // Enforce max 1 expert if speculative decoding is active
        const upperCap = status?.spec_decode_enabled ? 1 : maxAdapters;
        const targetCount = Math.max(minAdapters, Math.min(matched.length, upperCap));
        displayedTeam = matched.slice(0, targetCount);
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

    const history = messages
      .filter((m) => m.id !== "init" && m.content && !m.content.startsWith("⚡ ["))
      .map((m) => ({ role: m.role, content: m.content }));
    const fullMessages = [...history, { role: "user" as const, content: textToSend }];

    try {
      const resp = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: targetModel,
          messages: fullMessages,
          stream: true,
          max_tokens: maxTokens || 400,
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
      let displaySpeed = "...";
      let tokenCount = 0;
      let firstTokenTime: number | null = null;

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

              const newText = (delta.content || "") + (delta.reasoning_content || "");
              if (newText) {
                const now = performance.now();
                if (!firstTokenTime) firstTokenTime = now;
                const chunkTokens = Math.max(1, Math.round(newText.length / 3.8));
                tokenCount += chunkTokens;

                const elapsedSec = (now - firstTokenTime) / 1000;
                if (elapsedSec > 0.08) {
                  displaySpeed = (tokenCount / elapsedSec).toFixed(1);
                }
              }

              if (data.usage?.tokens_per_second) {
                displaySpeed = data.usage.tokens_per_second.toFixed(1);
              }
              if (data.usage?.completion_tokens) {
                tokenCount = data.usage.completion_tokens;
              }

              let extraMeta = `Active: ${displayedTeam.join(" + ")} • ${displaySpeed} tok/s (${tokenCount} tokens)`;
              if (data.usage?.predicted_next_expert) {
                const nextExp = data.usage.predicted_next_expert;
                const conf = ((data.usage.predicted_confidence || 0.85) * 100).toFixed(1);
                setPrefoldExpert(nextExp);
                setPrefoldConf(conf);
                extraMeta += ` • 🕸️ Pre-folded: ${nextExp} (${conf}%)`;
              }

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
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 h-[calc(100vh-130px)]">
      {/* ========================================================================= */}
      {/* SECTION 1 (LEFT, col-span-3): Controls, Toggles, Explanations & Rules    */}
      {/* ========================================================================= */}
      <div className="xl:col-span-3 lg:col-span-4 flex flex-col gap-3 overflow-y-auto pr-1">
        {/* Adapter Stacking Configuration Card */}
        <div className="rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.85)] p-4 backdrop-blur-xl shadow-xl">
          <div className="flex items-center justify-between pb-2.5 border-b border-white/5">
            <div>
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Sliders className="h-3.5 w-3.5 text-[#00f2ff]" /> Stacking &amp; Routing
                <LaymanTooltip
                  title="Dynamic LoRA Expert Routing"
                  simpleExplanation="Selects specialist mini-models (adapters) on the fly based on what you ask. You can let the router pick automatically, choose manually, pick a single expert, or run the base model without adapters."
                  technicalDetails="Computes manifold geodesic distance d_R across weight subspaces to assemble synergistic multi-expert teams."
                  chapter="Ch. 3 & 4"
                />
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                Target: {status?.model_id || "Qwen/Qwen3.5-4B"}
              </span>
            </div>
            <span className="rounded bg-white/5 px-2 py-0.5 font-mono text-[10px] text-slate-400 font-bold">
              {routingMode.toUpperCase()}
            </span>
          </div>

          {/* Mode Tabs */}
          <div className="grid grid-cols-4 gap-1 rounded-xl bg-black/40 p-1 border border-white/5 mt-2.5">
            {[
              { id: "auto", label: "Auto" },
              { id: "manual", label: "Manual" },
              { id: "single", label: "Single" },
              { id: "base", label: "Base" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setRoutingMode(tab.id as any)}
                className={`rounded-lg py-1 text-[10px] font-bold font-mono transition-all ${
                  routingMode === tab.id
                    ? "bg-[#00f2ff] text-slate-950 shadow-[0_0_10px_rgba(0,242,255,0.3)]"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {/* Context Options depending on Routing Mode */}
          {routingMode === "auto" && (
            <div className="mt-2.5 space-y-1.5 rounded-xl bg-white/[0.02] border border-white/5 p-2.5">
              <div className="text-[10px] font-bold text-slate-300 flex justify-between items-center">
                <span>Riemannian Intent Co-Routing</span>
                <span className="font-mono text-[9px] text-emerald-400">0..{maxAdapters} Experts</span>
              </div>
              <div className="grid grid-cols-2 gap-2 pt-1 font-mono text-xs">
                <div>
                  <span className="text-[9px] text-slate-500">Min Stack</span>
                  <input
                    type="number"
                    min={0}
                    max={6}
                    value={minAdapters}
                    onChange={(e) => setMinAdapters(Math.max(0, Math.min(6, parseInt(e.target.value, 10) || 0)))}
                    className="w-full rounded border border-white/10 bg-black/40 p-1 text-white text-xs mt-0.5"
                  />
                </div>
                <div>
                  <span className="text-[9px] text-slate-500">Max Stack</span>
                  <input
                    type="number"
                    min={1}
                    max={6}
                    value={maxAdapters}
                    onChange={(e) => setMaxAdapters(Math.max(1, Math.min(6, parseInt(e.target.value, 10) || 1)))}
                    className="w-full rounded border border-white/10 bg-black/40 p-1 text-white text-xs mt-0.5"
                  />
                </div>
              </div>
            </div>
          )}

          {routingMode === "manual" && (
            <div className="mt-2.5 space-y-1">
              <div className="grid grid-cols-1 gap-1">
                {AVAILABLE_ADAPTERS.map((adapter) => {
                  const isChecked = selectedAdapters.includes(adapter.id);
                  return (
                    <button
                      key={adapter.id}
                      onClick={() => toggleAdapterSelection(adapter.id)}
                      className={`flex items-center justify-between rounded-lg px-2 py-1.5 text-[11px] font-mono transition-all border ${
                        isChecked
                          ? "border-[#00f2ff]/40 bg-[rgba(0,242,255,0.12)] text-white font-bold"
                          : "border-white/5 bg-black/30 text-slate-400 hover:border-white/10"
                      }`}
                    >
                      <span className="flex items-center gap-1.5">
                        <span>{adapter.icon}</span>
                        <span className="truncate">{adapter.name.split(" (")[0]}</span>
                      </span>
                      {isChecked && <Check className="h-3 w-3 text-[#00f2ff]" />}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {routingMode === "single" && (
            <div className="mt-2.5 space-y-1">
              <select
                value={singleAdapter}
                onChange={(e) => setSingleAdapter(e.target.value)}
                className="w-full rounded-lg border border-white/10 bg-black/50 p-2 text-xs font-mono text-white focus:border-[#00f2ff] focus:outline-none"
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
            <div className="mt-2 rounded-lg bg-white/[0.02] border border-white/5 p-2 text-[11px] text-slate-400">
              <b className="text-white">Pristine Base Model ($W_0$)</b>
              <p className="mt-0.5 text-[10px]">
                0 adapters folded. Runs raw weights with zero domain specialization.
              </p>
            </div>
          )}
        </div>

        {/* Live Active Stack Badge */}
        <div className="rounded-2xl border border-[rgba(0,242,255,0.35)] bg-black/60 p-3.5 shadow-[0_0_20px_rgba(0,242,255,0.08)] backdrop-blur-xl">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider text-[#00f2ff] flex items-center gap-1.5">
              <Zap className="h-3 w-3" /> Live Active Stack
            </span>
            <span className="rounded bg-white/5 border border-white/10 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
              {routingMode === "base" || currentActiveStack[0] === "Base (Pristine W0)" ? "Base $W_0$" : "In-VRAM Folded"}
            </span>
          </div>

          <div className="mt-2 flex flex-wrap gap-1">
            {currentActiveStack.map((exp) => (
              <span
                key={exp}
                className={`flex items-center gap-1 rounded-md border px-2 py-0.5 font-mono text-[10px] font-bold ${
                  exp === "Base (Pristine W0)"
                    ? "border-slate-500/40 bg-slate-500/15 text-slate-300"
                    : "border-[rgba(0,242,255,0.4)] bg-[rgba(0,242,255,0.15)] text-[#00f2ff]"
                }`}
              >
                <Zap className="h-2.5 w-2.5" /> {exp}
              </span>
            ))}
          </div>

          {morphAlert && (
            <div className="mt-2 flex items-center gap-1.5 rounded-lg border border-[#00f2ff]/40 bg-[#00f2ff]/10 p-1.5 font-mono text-[10px] text-[#00f2ff] animate-pulse">
              <Zap className="h-3 w-3" />
              <span>Morphing stack to: <b>{morphAlert}</b></span>
            </div>
          )}
        </div>

        {/* Engine Controls Panel with Layman Tooltips & Incompatibility Rules */}
        <div className="rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-4 backdrop-blur-xl">
          <div className="flex items-center justify-between pb-2 border-b border-white/5">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
              <Sliders className="h-3.5 w-3.5 text-[#00f2ff]" /> Runtime Controls
            </span>
            <span className="text-[9px] font-mono text-slate-500">Live Hardware Toggles</span>
          </div>

          {toggleError && (
            <div className="mt-2 rounded-lg border border-rose-400/30 bg-rose-500/10 px-2 py-1 text-[10px] text-rose-300 font-mono">
              {toggleError}
            </div>
          )}

          <div className="mt-2.5 space-y-2.5">
            {/* 1. Speculative Decode */}
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <span className="text-[11px] text-slate-300 font-medium">Speculative Decode</span>
                <LaymanTooltip
                  title="Speculative Decoding"
                  simpleExplanation="Runs a small, ultra-fast 'draft brain' to guess ahead several words at a time. The main model verifies all guesses in one single GPU step. If guesses are accurate, speeds double!"
                  technicalDetails="K-step autoregressive draft generation with MTP head verified via batch CUDA Graph pass."
                  chapter="Ch. 5"
                  incompatibleWith={["Multi-Expert Stacking > 1 (CUDA Graph Safety)"]}
                />
              </div>
              <button
                onClick={toggleSpeculativeDecode}
                disabled={!status?.loaded || togglesBusy !== null}
                className={`flex items-center gap-1 rounded-lg border px-2 py-0.5 text-[10px] font-bold uppercase transition-all disabled:opacity-40 ${
                  status?.spec_decode_enabled
                    ? "border-emerald-400/40 bg-emerald-500/15 text-emerald-300 shadow-[0_0_8px_rgba(16,185,129,0.2)]"
                    : "border-white/10 bg-black/40 text-slate-400"
                }`}
              >
                {togglesBusy === "spec_decode" ? "..." : status?.spec_decode_enabled ? "ON" : "OFF"}
              </button>
            </div>

            {/* Incompatibility Badge if Spec Decode is ON */}
            {status?.spec_decode_enabled && (
              <div className="flex items-center gap-1 rounded-lg bg-amber-500/10 border border-amber-400/20 px-2 py-1 text-[9px] font-mono text-amber-300">
                <AlertTriangle className="h-3 w-3 shrink-0" />
                <span>Locked to 1 expert max (CUDA Graph Weight Lock)</span>
              </div>
            )}

            {/* 2. Draft Depth (K) */}
            <div className={!status?.spec_decode_enabled ? "opacity-40" : ""}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center">
                  <span className="text-[11px] text-slate-400">Draft Depth ($K$)</span>
                  <LaymanTooltip
                    title="Draft Depth (K)"
                    simpleExplanation="How many tokens ahead the draft brain tries to guess before checking in with the main model. Higher K gives massive speedups on easy repetitive text, but wastes time if guesses are wrong."
                    chapter="Ch. 5"
                    requires="Speculative Decode ON"
                  />
                </div>
                {!status?.spec_decode_enabled && (
                  <span className="text-[9px] text-amber-400/80 font-mono">needs spec ON</span>
                )}
              </div>
              <div className="flex rounded-lg bg-black/40 p-0.5 border border-white/10">
                {(status?.spec_k_options || [2, 4, 8]).map((k) => (
                  <button
                    key={k}
                    onClick={() => setSpeculativeK(k)}
                    disabled={!status?.spec_decode_enabled || togglesBusy !== null}
                    className={`flex-1 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase transition-all disabled:cursor-not-allowed ${
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

            {/* 3. Range Spec Gate */}
            <div className={`flex items-center justify-between ${!status?.spec_decode_enabled ? "opacity-40" : ""}`}>
              <div className="flex items-center">
                <span className="text-[11px] text-slate-300">⚡ Range Spec Gate</span>
                <LaymanTooltip
                  title="Range Speculative Gate"
                  simpleExplanation="An intelligent mathematical brake. If the draft brain is hesitant or uncertain about the next token, this gate instantly aborts speculation early so GPU compute is not wasted."
                  technicalDetails="O(1) logit spread check + Weibull hazard probability filter."
                  chapter="Ch. 8"
                  requires="Speculative Decode ON"
                />
              </div>
              <button
                onClick={toggleRangeGate}
                disabled={!status?.loaded || !status?.spec_decode_enabled || togglesBusy !== null}
                className={`flex items-center gap-1 rounded-lg border px-2 py-0.5 text-[10px] font-bold uppercase transition-all ${
                  status?.spec_range_gate_enabled
                    ? "border-cyan-400/40 bg-cyan-500/15 text-cyan-300 shadow-[0_0_8px_rgba(0,242,255,0.2)]"
                    : "border-white/10 bg-black/40 text-slate-400"
                }`}
              >
                {togglesBusy === "range_gate" ? "..." : status?.spec_range_gate_enabled ? "ON" : "OFF"}
              </button>
            </div>

            {/* 3b. Dual-EMA / MACD Circuit-Breaker */}
            <div className={`flex items-center justify-between ${!status?.spec_decode_enabled ? "opacity-40" : ""}`}>
              <div className="flex items-center">
                <span className="text-[11px] text-slate-300 font-medium">⚡ MACD Circuit-Breaker</span>
                <LaymanTooltip
                  title="Dual-EMA / MACD Speculation Circuit-Breaker"
                  simpleExplanation="An automatic thermostat. When acceptance rate tau drops below 1.8 (hard reasoning/unfamiliar code), speculation is temporarily disengaged to avoid the verification penalty, guaranteeing raw CUDA Graph baseline speed. Once confidence resumes (>2.2), speculation automatically re-engages!"
                  technicalDetails="Dual Exponential Moving Averages (alpha_fast=0.25, alpha_slow=0.08) with MACD crossover thresholding."
                  chapter="Ch. 5 & 8"
                  requires="Speculative Decode ON"
                />
              </div>
              <button
                onClick={toggleCircuitBreaker}
                disabled={!status?.loaded || !status?.spec_decode_enabled || togglesBusy !== null}
                className={`flex items-center gap-1 rounded-lg border px-2 py-0.5 text-[10px] font-bold uppercase transition-all ${
                  status?.spec_circuit_breaker_enabled !== false
                    ? "border-emerald-400/40 bg-emerald-500/15 text-emerald-300 shadow-[0_0_8px_rgba(16,185,129,0.2)]"
                    : "border-white/10 bg-black/40 text-slate-400"
                }`}
              >
                {togglesBusy === "circuit_breaker" ? "..." : status?.spec_circuit_breaker_enabled !== false ? "ON" : "OFF"}
              </button>
            </div>

            {/* 4. Renko Brick Smoothing (Ch. 4) */}
            <div className="border-t border-white/5 pt-2">
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center">
                  <span className="text-[11px] text-slate-300 font-medium">⚡ Renko Smoothing</span>
                  <LaymanTooltip
                    title="Renko Brick Smoothing"
                    simpleExplanation="A noise filter for dynamic adapter routing. Ordinary punctuation (commas, periods) or words like 'the' cause false adapter swap requests. Renko filters out this jitter and only swaps experts when a true topic change happens!"
                    technicalDetails="Accumulates Riemannian latent displacement D_t. Only emits re-classification event when ΔD >= ε_box."
                    chapter="Ch. 4"
                  />
                </div>
                <button
                  onClick={toggleRenkoSmoothing}
                  disabled={!status?.loaded || togglesBusy !== null}
                  className={`flex items-center gap-1 rounded-lg border px-2 py-0.5 text-[10px] font-bold uppercase transition-all ${
                    status?.renko_smoothing_enabled !== false
                      ? "border-purple-400/40 bg-purple-500/15 text-purple-300 shadow-[0_0_8px_rgba(168,85,247,0.2)]"
                      : "border-white/10 bg-black/40 text-slate-400"
                  }`}
                >
                  {togglesBusy === "renko" ? "..." : status?.renko_smoothing_enabled !== false ? "ON" : "OFF"}
                </button>
              </div>

              {/* Epsilon Box Selector */}
              {status?.renko_smoothing_enabled !== false && (
                <div className="flex items-center justify-between gap-1 mt-1 bg-black/40 p-1 rounded-lg border border-white/5">
                  <span className="text-[9px] font-mono text-slate-400 pl-1">Threshold ($\epsilon$):</span>
                  <div className="flex gap-1">
                    {(status?.renko_epsilon_options || [3.0, 5.0, 8.0]).map((eps) => (
                      <button
                        key={eps}
                        onClick={() => setRenkoEpsilon(eps)}
                        disabled={togglesBusy !== null}
                        className={`rounded px-1.5 py-0.5 text-[9px] font-mono font-bold transition-all ${
                          (status?.renko_epsilon ?? 5.0) === eps
                            ? "bg-[#a855f7] text-white shadow-[0_0_6px_rgba(168,85,247,0.4)]"
                            : "text-slate-400 hover:text-white"
                        }`}
                      >
                        {eps.toFixed(1)}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* 4b. Thinking Supervisor (Chapters 3, 4, 5 & 8) */}
            <div className="flex items-center justify-between border-t border-white/5 pt-2">
              <div className="flex items-center">
                <span className="text-[11px] text-slate-300 font-medium">🧠 Thinking Supervisor</span>
                <LaymanTooltip
                  title="Runtime Thinking Supervisor"
                  simpleExplanation="Eliminates infinite reasoning loops and enforces strict thinking budgets (Low/Med/High) at the GPU sampler level. Uses Renko latent attractor detection to break repetitive loops and Shannon entropy stoploss to exit the instant the answer is found."
                  technicalDetails="Logit masking floor, sigmoid boost, Renko latent cosine recurrence (Sim > 0.96), and structured transition delimiter (\n</think>\n\n) injection."
                  chapter="Ch. 3, 4 & 5"
                />
              </div>
              <button
                onClick={toggleThinkingSupervisor}
                disabled={!status?.loaded || togglesBusy !== null}
                className={`flex items-center gap-1 rounded-lg border px-2 py-0.5 text-[10px] font-bold uppercase transition-all ${
                  status?.thinking_supervisor_enabled !== false
                    ? "border-amber-400/40 bg-amber-500/15 text-amber-300 shadow-[0_0_8px_rgba(245,158,11,0.2)]"
                    : "border-white/10 bg-black/40 text-slate-400"
                }`}
              >
                {togglesBusy === "thinking_sup" ? "..." : status?.thinking_supervisor_enabled !== false ? "ON" : "OFF"}
              </button>
            </div>

            {/* 5. Multi-Expert Stacking */}
            <div>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center">
                  <span className="text-[11px] text-slate-400">Multi-Expert Stacking</span>
                  <LaymanTooltip
                    title="Surgical LoRA Stacking"
                    simpleExplanation="Blends the weights of multiple expert models (e.g. Python + Postgres) directly into the base model on-the-fly using singular value scaling, without memory duplication."
                    chapter="Ch. 3"
                  />
                </div>
              </div>
              <div className="flex rounded-lg bg-black/40 p-0.5 border border-white/10">
                {(["surgical", "none"] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setScaleMode(mode)}
                    disabled={!status?.loaded || togglesBusy !== null}
                    className={`flex-1 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase transition-all ${
                      status?.scale_mode === mode
                        ? "bg-[#a855f7] text-white shadow-[0_0_8px_rgba(168,85,247,0.5)]"
                        : "text-slate-400 hover:text-white"
                    }`}
                  >
                    {mode === "surgical" ? "Surgical" : "Additive"}
                  </button>
                ))}
              </div>
            </div>

            {/* 6. Predictive Pre-folding */}
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <span className="text-[11px] text-slate-300">Predictive Pre-folding</span>
                <LaymanTooltip
                  title="NOTEARS Predictive Pre-folding"
                  simpleExplanation="Anticipates what tool or domain you will ask about next turn, preparing the right expert adapter in GPU memory while you are still typing or reading."
                  chapter="Ch. 7"
                />
              </div>
              <button
                onClick={togglePrefold}
                disabled={!status?.loaded || togglesBusy !== null}
                className={`flex items-center gap-1 rounded-lg border px-2 py-0.5 text-[10px] font-bold uppercase transition-all ${
                  status?.prefold_enabled
                    ? "border-emerald-400/40 bg-emerald-500/15 text-emerald-300"
                    : "border-white/10 bg-black/40 text-slate-400"
                }`}
              >
                {togglesBusy === "prefold" ? "..." : status?.prefold_enabled ? "ON" : "OFF"}
              </button>
            </div>

            {/* 7. Ring Buffer Mode */}
            <div className={!status?.ring_buffer_wired ? "opacity-40" : ""}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center">
                  <span className="text-[11px] text-slate-400">Ring Buffer Rollback</span>
                  <LaymanTooltip
                    title="State Ring Buffer Replay"
                    simpleExplanation="An ultra-fast instant rollback mechanism for speculative decoding. Instantly rewinds the memory state in 0ms when the draft model makes an incorrect guess."
                    chapter="Ch. 5"
                    requires="Speculative Decode ON"
                  />
                </div>
                {!status?.ring_buffer_wired && (
                  <span className="text-[9px] text-amber-400/80 font-mono">needs spec ON</span>
                )}
              </div>
              <div className="flex rounded-lg bg-black/40 p-0.5 border border-white/10">
                {(status?.ring_buffer_options || ["dense", "poet", "selective_hybrid", "pointer"]).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setRingBufferMode(mode)}
                    disabled={!status?.ring_buffer_wired || togglesBusy !== null}
                    title={mode}
                    className={`flex-1 rounded px-1 py-0.5 text-[9px] font-bold uppercase transition-all ${
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

            {/* 8. Tensor State Handoff ($S_t$) */}
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <span className="text-[11px] text-slate-300">Tensor Handoff ($S_t$)</span>
                <LaymanTooltip
                  title="Tensor State Handoff"
                  simpleExplanation="Passes the raw mathematical memory of previous turns across requests, eliminating the need to re-read earlier chat messages from scratch."
                  chapter="Ch. 2"
                />
              </div>
              <button
                onClick={toggleStateHandoff}
                disabled={!status?.loaded || togglesBusy !== null}
                className={`flex items-center gap-1 rounded-lg border px-2 py-0.5 text-[10px] font-bold uppercase transition-all ${
                  status?.state_handoff_enabled !== false
                    ? "border-[#00f2ff]/40 bg-[#00f2ff]/15 text-[#00f2ff]"
                    : "border-white/10 bg-black/40 text-slate-400"
                }`}
              >
                {togglesBusy === "state_handoff" ? "..." : status?.state_handoff_enabled !== false ? "ON" : "OFF"}
              </button>
            </div>

            {/* 9. Fused W4A16 */}
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <span className="text-[11px] text-slate-300">Fused W4A16</span>
                <LaymanTooltip
                  title="Fused W4A16 Quantization"
                  simpleExplanation="Compresses weights into 4-bit memory while executing high-speed 16-bit math on AMD RDNA3 matrix cores, saving 75% GPU memory."
                  chapter="Ch. 1"
                />
              </div>
              <button
                onClick={toggleW4A16}
                disabled={!status?.loaded || togglesBusy !== null}
                className={`flex items-center gap-1 rounded-lg border px-2 py-0.5 text-[10px] font-bold uppercase transition-all ${
                  status?.w4a16_enabled
                    ? "border-[#10b981]/40 bg-[#10b981]/15 text-[#10b981]"
                    : "border-white/10 bg-black/40 text-slate-400"
                }`}
              >
                {togglesBusy === "w4a16" ? "..." : status?.w4a16_enabled ? "ON" : "OFF"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 2 (MIDDLE, col-span-6): Live Chat Console & Streaming Stream      */}
      {/* ========================================================================= */}
      <div className="xl:col-span-6 lg:col-span-8 flex flex-col rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] backdrop-blur-xl overflow-hidden shadow-2xl">
        {/* Chat Messages Stream */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3.5">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`flex gap-3 max-w-[90%] ${
                m.role === "user" ? "ml-auto flex-row-reverse" : "mr-auto"
              }`}
            >
              {/* Avatar */}
              <div
                className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-xl text-xs font-bold ${
                  m.role === "user"
                    ? "bg-white/10 border border-white/10 text-white"
                    : "bg-gradient-to-br from-[#00f2ff] to-[#a855f7] text-slate-950 shadow-[0_0_12px_rgba(0,242,255,0.3)]"
                }`}
              >
                {m.role === "user" ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5 fill-current" />}
              </div>

              {/* Message Bubble */}
              <div
                className={`rounded-2xl px-4 py-3 text-xs leading-relaxed ${
                  m.role === "user"
                    ? "border border-[#00f2ff]/40 bg-[#00f2ff]/15 text-white"
                    : "border border-white/10 bg-[rgba(12,16,24,0.85)] text-slate-100"
                }`}
              >
                {m.reasoning && (
                  <div className="mb-2 rounded-lg border-l-2 border-[#a855f7] bg-black/40 p-2 text-[11px] italic text-slate-400">
                    <div className="font-mono text-[9px] font-bold text-[#a855f7] mb-0.5">Reasoning Chain</div>
                    {m.reasoning}
                  </div>
                )}
                <div className="whitespace-pre-wrap font-sans">{m.content}</div>
                {m.meta && (
                  <div className="mt-2 flex items-center gap-2 font-mono text-[10px] text-slate-400 border-t border-white/5 pt-1.5">
                    <span>{m.meta}</span>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar & Thinking Effort Bar */}
        <div className="border-t border-[rgba(255,255,255,0.08)] bg-[rgba(12,16,24,0.95)] p-3 space-y-2.5">
          <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-xs">
            {/* Thinking Effort Selector */}
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400 flex items-center gap-1 text-[11px]">
                <Brain className="h-3 w-3 text-[#a855f7]" /> Thinking:
              </span>
              <div className="flex rounded-lg bg-black/40 p-0.5 border border-white/10">
                {(["off", "low", "medium", "high"] as const).map((level) => (
                  <button
                    key={level}
                    onClick={() => setThinkingEffort(level)}
                    className={`rounded px-2 py-0.5 text-[10px] font-bold uppercase transition-all ${
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

            {/* Max Tokens */}
            <div className="flex items-center gap-1.5">
              <span className="text-slate-400 flex items-center gap-1 text-[11px]">
                <Hash className="h-3 w-3 text-[#00f2ff]" /> Max Toks:
              </span>
              <input
                type="number"
                min={64}
                max={4096}
                step={64}
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value, 10) || 400)}
                className="w-16 rounded-lg border border-white/10 bg-black/50 px-2 py-0.5 text-[11px] text-white font-mono focus:border-[#00f2ff] focus:outline-none"
              />
            </div>

            {/* Reset Chat */}
            <button
              onClick={resetChat}
              disabled={isStreaming}
              title="Clear conversation history and start fresh"
              className="flex items-center gap-1 rounded-lg border border-white/10 bg-black/40 px-2 py-0.5 text-[10px] font-bold uppercase text-slate-400 transition-all hover:border-rose-400/40 hover:text-rose-300"
            >
              <RotateCcw className="h-3 w-3" /> Reset
            </button>
          </div>

          {/* Text Input & Send */}
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Ask a question (e.g. uv packaging, PostgreSQL vector search, FastAPI, DuckDB)..."
              className="flex-1 rounded-xl border border-white/10 bg-black/50 px-3.5 py-2.5 text-xs text-white placeholder-slate-500 focus:border-[#00f2ff] focus:outline-none focus:ring-1 focus:ring-[#00f2ff]"
            />
            {isStreaming ? (
              <button
                disabled={stopping}
                onClick={stopGeneration}
                className="flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-rose-500 to-rose-600 px-4 py-2 text-xs font-bold text-white shadow-[0_0_15px_rgba(244,63,94,0.35)] transition-all hover:scale-[1.02]"
              >
                <Square className="h-3.5 w-3.5 fill-current" />
                <span>{stopping ? "..." : "Stop"}</span>
              </button>
            ) : (
              <button
                disabled={isStreaming}
                onClick={() => handleSend()}
                className="flex items-center gap-1.5 rounded-xl bg-gradient-to-r from-[#00f2ff] to-[#a855f7] px-4 py-2 text-xs font-bold text-slate-950 shadow-[0_0_15px_rgba(0,242,255,0.3)] transition-all hover:scale-[1.02]"
              >
                <Send className="h-3.5 w-3.5 fill-current" />
                <span>Send ⚡</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* SECTION 3 (RIGHT, col-span-3): Example Prompts & Benchmark Test Matrix   */}
      {/* ========================================================================= */}
      <div className="xl:col-span-3 lg:col-span-12 flex flex-col rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-4 backdrop-blur-xl overflow-y-auto">
        <div className="flex items-center justify-between pb-2 border-b border-white/5">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
            <FlaskConical className="h-3.5 w-3.5 text-[#a855f7]" /> Test Matrix &amp; Evals
          </span>
          <span className="text-[9px] font-mono text-slate-500">1-Click Probes</span>
        </div>

        {/* Category Tabs */}
        <div className="grid grid-cols-3 gap-1 rounded-xl bg-black/40 p-1 border border-white/5 mt-2.5">
          {[
            { id: "multiturn", label: "Multi-Turn" },
            { id: "hybrid", label: "Hybrid" },
            { id: "probes", label: "Probes" },
          ].map((t) => (
            <button
              key={t.id}
              onClick={() => setPromptTab(t.id as any)}
              className={`rounded-lg py-1 text-[10px] font-bold font-mono transition-all ${
                promptTab === t.id
                  ? "bg-[#a855f7] text-white shadow-[0_0_8px_rgba(168,85,247,0.4)]"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab Description */}
        <div className="mt-2 text-[10px] text-slate-400 italic">
          {promptTab === "multiturn" && "Sequential 4-turn chat sequence in shared context window."}
          {promptTab === "hybrid" && "Multi-domain challenges to trigger dynamic expert stacking."}
          {promptTab === "probes" && "Deep single-domain probes testing specialized syntax."}
        </div>

        {/* Prompt List */}
        <div className="mt-2 space-y-2 flex-1 overflow-y-auto pr-0.5">
          {promptTabs[promptTab].map((p, idx) => (
            <div
              key={idx}
              className="rounded-xl border border-white/5 bg-black/40 p-2.5 text-xs transition-all hover:border-[#00f2ff]/40 hover:bg-[#00f2ff]/5 group"
            >
              <div className="flex items-center justify-between mb-1 font-mono">
                <span className="font-bold text-[#00f2ff] text-[10px] flex items-center gap-1">
                  <Zap className="h-2.5 w-2.5" /> {p.tag}
                </span>
                <button
                  onClick={() => handleSend(p.text)}
                  disabled={isStreaming}
                  className="flex items-center gap-1 rounded bg-[#00f2ff]/10 hover:bg-[#00f2ff]/20 text-[#00f2ff] px-2 py-0.5 text-[9px] font-bold uppercase transition-all"
                >
                  <Play className="h-2.5 w-2.5 fill-current" /> Run
                </button>
              </div>

              <p className="text-slate-300 text-[11px] leading-relaxed line-clamp-3 mb-1.5">
                &quot;{p.text}&quot;
              </p>

              <div className="flex items-center justify-between text-[9px] font-mono text-slate-500 border-t border-white/5 pt-1">
                <span>{p.desc}</span>
                <span className="text-slate-400">{p.expert}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
