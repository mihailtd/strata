"use client";

import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeSlug from "rehype-slug";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { BookOpen, FileText, ChevronRight, Sparkles, Layers, Cpu, Compass, Atom } from "lucide-react";
import { DOCS_MANIFEST, DocItem } from "@/lib/docs-manifest";
import { useMDXComponents } from "@/mdx-components";
import LedoitWolfInfographic from "@/components/LedoitWolfInfographic";

export default function DocsPage() {
  const [selectedDocId, setSelectedDocId] = useState<string>("decisions");
  const [docMetadata, setDocMetadata] = useState<DocItem | null>(null);
  const [content, setContent] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [filterCategory, setFilterCategory] = useState<string>("All");

  const components = useMDXComponents({});

  const loadDoc = async (id: string) => {
    setIsLoading(true);
    setSelectedDocId(id);
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
    loadDoc("decisions");
  }, []);

  const categories = ["All", ...Array.from(new Set(DOCS_MANIFEST.map((d) => d.category)))];

  const filteredDocs =
    filterCategory === "All"
      ? DOCS_MANIFEST
      : DOCS_MANIFEST.filter((d) => d.category === filterCategory);

  const getCategoryIcon = (cat: string) => {
    switch (cat) {
      case "Core Architecture & Decisions":
        return Compass;
      case "Runtime & Speculative":
        return Cpu;
      case "Factory & Geometry":
        return Layers;
      default:
        return FileText;
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
              {cat === "Core Architecture & Decisions"
                ? "Core"
                : cat === "Runtime & Speculative"
                ? "Runtime"
                : cat === "Factory & Geometry"
                ? "Factory"
                : cat}
            </button>
          ))}
        </div>

        {/* Document List */}
        <div className="flex-1 overflow-y-auto space-y-2 pr-1">
          {filteredDocs.map((doc) => {
            const isSelected = selectedDocId === doc.id;
            const Icon = getCategoryIcon(doc.category);
            return (
              <button
                key={doc.id}
                onClick={() => loadDoc(doc.id)}
                className={`w-full text-left rounded-xl p-3 text-xs transition-all border group ${
                  isSelected
                    ? "border-[#00f2ff]/50 bg-gradient-to-r from-[rgba(0,242,255,0.15)] to-[rgba(168,85,247,0.15)] text-white shadow-[0_0_15px_rgba(0,242,255,0.2)]"
                    : "border-white/5 bg-black/40 text-slate-300 hover:border-white/20 hover:bg-white/[0.04]"
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
                  <ChevronRight
                    className={`h-3.5 w-3.5 shrink-0 transition-transform ${
                      isSelected
                        ? "text-[#00f2ff] translate-x-0.5"
                        : "text-slate-600 opacity-0 group-hover:opacity-100"
                    }`}
                  />
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

      {/* Right Main Panel: Prose Render View */}
      <div className="lg:col-span-8 flex flex-col rounded-2xl border border-[rgba(255,255,255,0.08)] bg-[rgba(16,22,34,0.85)] backdrop-blur-xl shadow-2xl overflow-hidden max-h-[calc(100vh-140px)]">
        {/* Document Header Bar */}
        {docMetadata && (
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 bg-black/50 px-6 py-4">
            <div>
              <div className="flex items-center gap-2">
                <span className="rounded bg-[#a855f7]/20 border border-[#a855f7]/40 px-2 py-0.5 font-mono text-[11px] font-bold text-[#a855f7]">
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

            <div className="flex items-center gap-2">
              <span className="rounded-md border border-white/10 bg-white/5 px-2.5 py-1 font-mono text-xs text-slate-400">
                Render Active
              </span>
            </div>
          </div>
        )}

        {/* Markdown Content Area with Tailwind Typography Prose */}
        <div className="flex-1 overflow-y-auto p-8 prose prose-invert max-w-none prose-pre:p-0 prose-pre:bg-transparent">
          {isLoading ? (
            <div className="flex flex-col items-center justify-center h-64 space-y-3">
              <Sparkles className="h-8 w-8 text-[#00f2ff] animate-spin" />
              <span className="font-mono text-xs text-slate-400">
                Loading document content...
              </span>
            </div>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeSlug, rehypeHighlight, rehypeKatex]}
              components={components as any}
            >
              {content}
            </ReactMarkdown>
          )}
        </div>
      </div>
    </div>
  );
}
