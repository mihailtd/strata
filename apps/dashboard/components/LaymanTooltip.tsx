"use client";

import React, { useState } from "react";
import { HelpCircle, AlertTriangle, CheckCircle2 } from "lucide-react";

interface LaymanTooltipProps {
  title: string;
  simpleExplanation: string;
  technicalDetails?: string;
  chapter?: string;
  incompatibleWith?: string[];
  requires?: string;
  side?: "right" | "left" | "top" | "bottom";
}

export function LaymanTooltip({
  title,
  simpleExplanation,
  technicalDetails,
  chapter,
  incompatibleWith,
  requires,
  side = "right",
}: LaymanTooltipProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div
      className="relative inline-flex items-center ml-1 group"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className="text-slate-400 hover:text-[#00f2ff] transition-colors focus:outline-none"
        aria-label={`Info about ${title}`}
      >
        <HelpCircle className="h-3 w-3 inline" />
      </button>

      {isOpen && (
        <div
          className={`absolute z-50 w-72 p-3 text-left rounded-xl border border-[rgba(0,242,255,0.3)] bg-[rgba(10,14,23,0.96)] shadow-[0_0_25px_rgba(0,242,255,0.15)] backdrop-blur-xl transition-all pointer-events-none ${
            side === "right"
              ? "left-5 top-1/2 -translate-y-1/2"
              : side === "left"
                ? "right-5 top-1/2 -translate-y-1/2"
                : side === "top"
                  ? "bottom-5 left-1/2 -translate-x-1/2"
                  : "top-5 left-1/2 -translate-x-1/2"
          }`}
        >
          {/* Header with Title & Chapter */}
          <div className="flex items-center justify-between border-b border-white/10 pb-1.5 mb-2">
            <span className="font-bold text-xs text-white flex items-center gap-1">{title}</span>
            {chapter && (
              <span className="px-1.5 py-0.5 rounded text-[9px] font-mono font-bold bg-[#00f2ff]/15 text-[#00f2ff] border border-[#00f2ff]/30">
                {chapter}
              </span>
            )}
          </div>

          {/* Simple Layman Explanation */}
          <div className="space-y-1.5 text-[11px] leading-relaxed">
            <div className="text-slate-200">
              <span className="font-bold text-[#00f2ff] block text-[10px] uppercase tracking-wider mb-0.5">
                💡 In Plain English:
              </span>
              {simpleExplanation}
            </div>

            {/* Technical Detail (optional) */}
            {technicalDetails && (
              <div className="text-slate-400 text-[10px] pt-1 border-t border-white/5 font-mono">
                <span className="text-slate-300 font-bold block">⚙️ Under the Hood:</span>
                {technicalDetails}
              </div>
            )}

            {/* Dependency Requirement */}
            {requires && (
              <div className="flex items-center gap-1 text-[10px] font-mono text-cyan-300 bg-cyan-500/10 p-1.5 rounded-lg border border-cyan-400/20 mt-1">
                <CheckCircle2 className="h-3 w-3 shrink-0" />
                <span>
                  Requires: <b>{requires}</b>
                </span>
              </div>
            )}

            {/* Incompatibility Rule */}
            {incompatibleWith && incompatibleWith.length > 0 && (
              <div className="flex items-start gap-1 text-[10px] font-mono text-amber-300 bg-amber-500/10 p-1.5 rounded-lg border border-amber-400/20 mt-1">
                <AlertTriangle className="h-3 w-3 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold">⚠️ Incompatible with:</span>
                  <div className="text-[9px] text-amber-200/80">{incompatibleWith.join(", ")}</div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
