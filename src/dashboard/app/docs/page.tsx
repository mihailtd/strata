"use client";

import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import {
  BookOpen,
  FileText,
  ChevronRight,
  Sparkles,
  Layers,
  Cpu,
  Compass,
  Atom,
  Flame,
  Zap,
  GitBranch,
} from "lucide-react";
import { DOCS_MANIFEST, DocItem } from "@/lib/docs-manifest";
import { useMDXComponents } from "@/mdx-components";
import LedoitWolfInfographic from "@/components/LedoitWolfInfographic";
import WeibullHazardInfographic from "@/components/WeibullHazardInfographic";
import SSIQuantizationInfographic from "@/components/SSIQuantizationInfographic";
import CutSetReliabilityInfographic from "@/components/CutSetReliabilityInfographic";
import KernelGemvInfographic from "@/components/KernelGemvInfographic";
import TreeSpeculationInfographic from "@/components/TreeSpeculationInfographic";
import FusedSwigluInfographic from "@/components/FusedSwigluInfographic";
import SurgicalStackingInfographic from "@/components/SurgicalStackingInfographic";

const INTERACTIVE_DOC_IDS = [
  "w4a16-beating-ollama",
  "bench-kernel-gemv-m1",
  "bench-kernel-tree-speculation",
  "bench-kernel-fused-swiglu",
  "ledoit-wolf-routing",
  "surgical-stacking",
  "bench-weibull-hazard",
  "bench-ssi-quant",
  "bench-cut-set-reliability",
];

export default function DocsPage() {
  const [selectedDocId, setSelectedDocId] = useState<string>("w4a16-beating-ollama");
  const [docMetadata, setDocMetadata] = useState<DocItem | null>(null);
  const [content, setContent] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [filterCategory, setFilterCategory] = useState<string>("All");

  const [activeTab, setActiveTab] = useState<"interactive" | "markdown">("interactive");

  const components = useMDXComponents({});

  const loadDoc = async (id: string) => {
    setIsLoading(true);
    setSelectedDocId(id);
    if (INTERACTIVE_DOC_IDS.includes(id)) {
      setActiveTab("interactive");
    }
    try {
      const resp = await fetch(`/api/docs?id=${id}`);
      if (!resp.ok) {
        throw new Error("Failed to load document");
      }
      const data = await resp.json();
      setDocMetadata({
        id: data.id,
        title: data.title,
        category: data.category,
        relativePath: data.relativePath,
        description: data.description,
      });
      setContent(data.content || "");
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadDoc("bench-kernel-tree-speculation");
  }, []);

  const categories = ["All", ...Array.from(new Set(DOCS_MANIFEST.map((d) => d.category)))];

  const filteredDocs =
    filterCategory === "All"
      ? DOCS_MANIFEST
      : DOCS_MANIFEST.filter((d) => d.category === filterCategory);

  const getCategoryIcon = (cat: string) => {
    switch (cat) {
      case "3. Kernel & Hardware (RDNA3)":
        return Zap;
      case "2. Runtime & Speculative":
        return Cpu;
      case "1. Factory & Geometry":
        return Layers;
      case "Core Architecture & Decisions":
        return Compass;
      default:
        return FileText;
    }
  };

  const getCategoryBadgeClass = (cat: string) => {
    switch (cat) {
      case "3. Kernel & Hardware (RDNA3)":
        return "bg-cyan-500/20 border-cyan-500/40 text-cyan-300";
      case "2. Runtime & Speculative":
        return "bg-purple-500/20 border-purple-500/40 text-purple-300";
      case "1. Factory & Geometry":
        return "bg-emerald-500/20 border-emerald-500/40 text-emerald-300";
      default:
        return "bg-amber-500/20 border-amber-500/40 text-amber-300";
    }
  };

  const hasInteractive = INTERACTIVE_DOC_IDS.includes(selectedDocId);

  const renderInteractiveComponent = () => {
    switch (selectedDocId) {
      case "w4a16-beating-ollama":
      case "bench-kernel-gemv-m1":
        return <KernelGemvInfographic />;
      case "bench-kernel-tree-speculation":
        return <TreeSpeculationInfographic />;
      case "bench-kernel-fused-swiglu":
        return <FusedSwigluInfographic />;
      case "ledoit-wolf-routing":
        return <LedoitWolfInfographic />;
      case "surgical-stacking":
        return <SurgicalStackingInfographic />;
      case "bench-weibull-hazard":
        return <WeibullHazardInfographic />;
      case "bench-ssi-quant":
        return <SSIQuantizationInfographic />;
      case "bench-cut-set-reliability":
        return <CutSetReliabilityInfographic />;
      default:
        return null;
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 min-h-[calc(100vh-140px)]">
      {/* Left Sidebar: Document Menu */}
      <div className="lg:col-span-4 flex flex-col gap-4 overflow-hidden rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.75)] p-5 backdrop-blur-xl max-h-[calc(100vh-140px)]">
        <div className="flex items-center justify-between pb-3 border-b border-white/5">
          <div className="flex items-center gap-2 font-bold text-sm text-white">
            <BookOpen className="h-4 w-4 text-[#00f2ff]" />
            <span>Research &amp; Benchmark Docs</span>
          </div>
          <span className="rounded bg-white/5 px-2 py-0.5 font-mono text-[11px] text-slate-400">
            {DOCS_MANIFEST.length} Files
          </span>
        </div>

        {/* Category Filter Pills */}
        <div className="flex flex-wrap gap-1.5 pb-2">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setFilterCategory(cat)}
              className={`rounded-lg px-2.5 py-1 text-[11px] font-mono font-medium transition-all ${
                filterCategory === cat
                  ? "bg-[#00f2ff] text-slate-950 font-bold shadow-[0_0_10px_rgba(0,242,255,0.4)]"
                  : "bg-white/5 text-slate-400 hover:bg-white/10 hover:text-white"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Document List */}
        <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
          {filteredDocs.map((doc) => {
            const isSelected = selectedDocId === doc.id;
            const Icon = getCategoryIcon(doc.category);
            const isInteractive = INTERACTIVE_DOC_IDS.includes(doc.id);

            return (
              <button
                key={doc.id}
                onClick={() => loadDoc(doc.id)}
                className={`w-full text-left rounded-xl p-3 transition-all border group ${
                  isSelected
                    ? "bg-[#00f2ff]/10 border-[#00f2ff]/30 shadow-[0_0_15px_rgba(0,242,255,0.15)]"
                    : "border-transparent hover:bg-white/5 hover:border-white/10"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2 font-bold text-slate-100 group-hover:text-[#00f2ff] transition-colors">
                    <Icon
                      className={`h-3.5 w-3.5 shrink-0 ${
                        isSelected ? "text-[#00f2ff]" : "text-slate-500"
                      }`}
                    />
                    <span className="line-clamp-1">{doc.title}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    {isInteractive && (
                      <span className="rounded bg-cyan-500/20 border border-cyan-500/30 px-1.5 py-0.5 text-[9px] font-mono text-cyan-300">
                        LIVE
                      </span>
                    )}
                    <ChevronRight
                      className={`h-3.5 w-3.5 shrink-0 transition-transform ${
                        isSelected
                          ? "text-[#00f2ff] translate-x-0.5"
                          : "text-slate-600 opacity-0 group-hover:opacity-100"
                      }`}
                    />
                  </div>
                </div>
                <div className="mt-1 font-mono text-[10px] text-slate-500 line-clamp-1">
                  {doc.relativePath}
                </div>
                {doc.description && (
                  <p className="mt-1.5 text-[11px] text-slate-400 line-clamp-2 leading-normal font-normal">
                    {doc.description}
                  </p>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Right Main Panel: Prose Render View & Interactive Infographic */}
      <div className="lg:col-span-8 flex flex-col rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.85)] backdrop-blur-xl shadow-2xl overflow-hidden max-h-[calc(100vh-140px)]">
        {/* Document Header Bar */}
        {docMetadata && (
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 bg-black/50 px-6 py-4">
            <div>
              <div className="flex items-center gap-2">
                <span
                  className={`rounded border px-2 py-0.5 font-mono text-[11px] font-bold ${getCategoryBadgeClass(
                    docMetadata.category
                  )}`}
                >
                  {docMetadata.category}
                </span>
                <span className="font-mono text-xs text-slate-400">
                  {docMetadata.relativePath}
                </span>
              </div>
              <h1 className="text-lg font-bold text-white mt-1 flex items-center gap-2">
                <FileText className="h-4 w-4 text-[#00f2ff]" />
                {docMetadata.title}
              </h1>
            </div>

            {/* Tab Switcher if doc has interactive visualizer */}
            {hasInteractive ? (
              <div className="flex rounded-xl bg-black/50 p-1 border border-white/10">
                <button
                  onClick={() => setActiveTab("interactive")}
                  className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-bold transition-all ${
                    activeTab === "interactive"
                      ? "bg-[#00f2ff] text-slate-950 shadow-[0_0_10px_rgba(0,242,255,0.3)]"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  <Atom className="h-3.5 w-3.5" />
                  <span>Interactive Infographic</span>
                </button>
                <button
                  onClick={() => setActiveTab("markdown")}
                  className={`flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-bold transition-all ${
                    activeTab === "markdown"
                      ? "bg-purple-600 text-white shadow-[0_0_10px_rgba(168,85,247,0.3)]"
                      : "text-slate-400 hover:text-white"
                  }`}
                >
                  <FileText className="h-3.5 w-3.5" />
                  <span>Full Paper (Markdown)</span>
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <span className="rounded-md border border-white/10 bg-white/5 px-2.5 py-1 font-mono text-xs text-slate-400">
                  Markdown Render
                </span>
              </div>
            )}
          </div>
        )}

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center h-64 space-y-3">
              <Sparkles className="h-8 w-8 text-[#00f2ff] animate-spin" />
              <span className="font-mono text-xs text-slate-400">
                Loading document content...
              </span>
            </div>
          ) : hasInteractive && activeTab === "interactive" ? (
            renderInteractiveComponent()
          ) : (
            <div className="prose prose-invert max-w-none prose-pre:p-0 prose-pre:bg-transparent">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeSlug, rehypeHighlight, rehypeKatex]}
                components={components as any}
              >
                {content}
              </ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
