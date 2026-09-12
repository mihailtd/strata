import type { MDXComponents } from "mdx/types";
import React from "react";

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    h1: ({ children }) => (
      <h1 className="text-3xl font-extrabold tracking-tight text-white mb-6 mt-8 border-b border-white/10 pb-3 flex items-center gap-2">
        <span className="text-[#00f2ff]">#</span> {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2 className="text-2xl font-bold tracking-tight text-slate-100 mb-4 mt-8 border-b border-white/5 pb-2 flex items-center gap-2">
        <span className="text-[#a855f7]">##</span> {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className="text-lg font-bold text-slate-200 mb-3 mt-6">
        <span className="text-[#10b981]">###</span> {children}
      </h3>
    ),
    h4: ({ children }) => (
      <h4 className="text-base font-bold text-slate-300 mb-2 mt-4">{children}</h4>
    ),
    p: ({ children }) => (
      <p className="text-slate-300 leading-relaxed mb-4 text-sm font-normal">{children}</p>
    ),
    ul: ({ children }) => (
      <ul className="list-disc list-inside space-y-2 mb-4 text-sm text-slate-300 ml-2">
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol className="list-decimal list-inside space-y-2 mb-4 text-sm text-slate-300 ml-2">
        {children}
      </ol>
    ),
    li: ({ children }) => <li className="leading-relaxed">{children}</li>,
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-[#00f2ff] bg-[#00f2ff]/5 p-4 rounded-r-xl my-4 text-sm italic text-slate-300">
        {children}
      </blockquote>
    ),
    table: ({ children }) => (
      <div className="overflow-x-auto my-6 rounded-xl border border-white/10 bg-black/40">
        <table className="w-full text-xs font-mono border-collapse text-left">{children}</table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className="bg-white/5 border-b border-white/10 text-slate-300">{children}</thead>
    ),
    tbody: ({ children }) => <tbody className="divide-y divide-white/5">{children}</tbody>,
    tr: ({ children }) => <tr className="hover:bg-white/[0.02] transition-colors">{children}</tr>,
    th: ({ children }) => <th className="p-3 font-bold text-slate-200">{children}</th>,
    td: ({ children }) => <td className="p-3 text-slate-300">{children}</td>,
    pre: ({ children }) => (
      <pre className="rounded-xl border border-white/10 bg-black/70 p-4 font-mono text-xs text-cyan-200 overflow-x-auto my-4 shadow-xl">
        {children}
      </pre>
    ),
    code: ({ children }) => (
      <code className="rounded bg-white/10 px-1.5 py-0.5 font-mono text-xs text-[#00f2ff] border border-white/10">
        {children}
      </code>
    ),
    hr: () => <hr className="my-8 border-white/10" />,
    a: ({ href, children }) => (
      <a
        href={href}
        className="text-[#00f2ff] underline underline-offset-4 hover:text-[#a855f7] transition-colors font-medium"
      >
        {children}
      </a>
    ),
    ...components,
  };
}
