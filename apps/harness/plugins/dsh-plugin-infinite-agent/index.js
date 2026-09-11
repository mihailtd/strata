/**
 * dsh-plugin-infinite-agent
 *
 * Turns DSH into a self-managing infinite-horizon agent harness.
 * Three tools the agent uses as part of its normal workflow:
 *
 *   route_task           — activate the right LoRA expert team at task start
 *   task_progress        — self-check for loops + context budget (call when stuck)
 *   summarize_and_delegate — produce a handoff brief before spawning a subagent
 *
 * Per-session state is kept in-memory (Map) for the lifetime of the DSH process.
 * The AGENTS.md file in the repo root tells the agent WHEN to call these tools.
 */

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { defineTool } from "@deepseek-ai/dsh-tools";

// ── Paths ──────────────────────────────────────────────────────────────────
const PLUGIN_DIR = new URL(".", import.meta.url).pathname;
const REPO_ROOT = resolve(PLUGIN_DIR, "../../../..");
const ROUTER_CLI = resolve(REPO_ROOT, "apps/harness/router/router_cli.py");

// ── Per-session state ──────────────────────────────────────────────────────
// Keyed by session_id string. Each entry tracks:
//   recentHashes   - circular buffer of last 20 (tool+args+result) hashes
//   tokenEstimate  - running estimate of tokens consumed this session
//   turnCount      - number of task_progress calls (= agent self-checks)
//   loopCount      - times a loop was detected and flagged
//   contextWindow  - target context limit (pulled from task config or default)
const _sessions = new Map();

function getSession(id) {
  if (!_sessions.has(id)) {
    _sessions.set(id, {
      recentHashes: [],
      tokenEstimate: 0,
      turnCount: 0,
      loopCount: 0,
      contextWindow: 16384, // conservative default; agent can pass their model's value
    });
  }
  return _sessions.get(id);
}

function sha1short(s) {
  return createHash("sha1").update(s).digest("hex").slice(0, 8);
}

// ── Loop detector ──────────────────────────────────────────────────────────
// A "loop" = the same (tool, result) pair appearing ≥3 times in the last 8 entries.
// Using result hash (not just call hash) so legitimate retries with different
// outcomes (e.g. test passes on 3rd try) are not flagged.
function detectLoop(session, toolName, resultSummary) {
  const h = sha1short(toolName + "||" + resultSummary);
  session.recentHashes.push(h);
  if (session.recentHashes.length > 20) session.recentHashes.shift();

  const window = session.recentHashes.slice(-8);
  const freq = {};
  for (const entry of window) freq[entry] = (freq[entry] || 0) + 1;
  const max = Math.max(...Object.values(freq));
  return { isLoop: max >= 3, repeats: max, hash: h };
}

// ── Router CLI ─────────────────────────────────────────────────────────────
function invokeRouter(taskDescription) {
  try {
    const raw = execFileSync("uv", ["run", "python", ROUTER_CLI, taskDescription], {
      cwd: REPO_ROOT,
      encoding: "utf-8",
      timeout: 8000,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const result = JSON.parse(raw.trim());
    if (result.error) throw new Error(result.error);
    return result;
  } catch (err) {
    return {
      error: err.message,
      is_multi_expert: false,
      experts: { general: 1.0 },
      recommended_model_id: "qwen3.8:27b",
      active_domains: [],
      system_prompt_prefix:
        "You are an expert autonomous software engineer writing clean modern Python code.",
      rationale: "Router unavailable — using general engine.",
      routing_latency_ms: 0,
    };
  }
}

// ── Render helpers ─────────────────────────────────────────────────────────
function renderRouteTask(_args, v) {
  if (!v) return [{ type: "text", text: "[Router returned no result]" }];
  const domains = Object.entries(v.experts || {})
    .sort(([, a], [, b]) => b - a)
    .map(([d, w]) => `${d} (γ=${w.toFixed(2)})`)
    .join(" · ");
  const label = v.is_multi_expert ? "MULTI-EXPERT FUSION" : "SINGLE EXPERT";
  return [
    {
      type: "text",
      text: [
        `[Expert Team — ${label}] ${v.routing_latency_ms}ms`,
        `Model endpoint: ${v.recommended_model_id}`,
        domains ? `Domains: ${domains}` : "",
        ``,
        v.system_prompt_prefix,
      ]
        .filter(Boolean)
        .join("\n"),
    },
  ];
}

function renderProgress(_args, v) {
  const statusIcon = v.status === "STAGNATED" ? "🔴" : "🟢";
  const actionIcon =
    { CONTINUE: "▶️", REPLAN: "🔄", COMPACT: "🗜️", DELEGATE: "📤" }[v.action] || "▶️";
  return [
    {
      type: "text",
      text: [
        `[Task Progress] ${statusIcon} ${v.status}`,
        `Context: ${v.budget_pct}% of ${v.context_window.toLocaleString()} tokens`,
        `Loops detected: ${v.loop_count} | Self-checks: ${v.turn_count}`,
        ``,
        `${actionIcon} Recommended action: ${v.action}`,
        v.reasoning,
      ].join("\n"),
    },
  ];
}

function renderHandoff(_args, v) {
  return [
    {
      type: "text",
      text: [
        `[Handoff Brief Ready]`,
        `Completed: ${v.completed_count} items  |  Pending: ${v.pending_count} items  |  Blockers: ${v.blocker_count}`,
        ``,
        `Pass the brief below as the opening message to a fresh subagent:`,
        `─────────────────────────────────────────────────────────`,
        v.handoff_brief,
      ].join("\n"),
    },
  ];
}

// ── Plugin entry point ─────────────────────────────────────────────────────
export const name = "infinite-agent";
export const inject = ["tools"];

export function apply(ctx) {
  // ── Tool 1: route_task ─────────────────────────────────────────────────
  ctx.tools.register(
    defineTool({
      name: "route_task",
      description: [
        "Activate the right LoRA expert team for your current task.",
        "Call this at the START of any coding, analysis, or engineering task.",
        "Returns: (1) the optimal model endpoint ID, (2) expert mixture weights γₖ,",
        "and (3) a system prompt prefix to guide your approach with domain precision.",
        "Available experts: postgresql, python_web (FastAPI), duckdb, astral (uv/ruff),",
        "python_modern (PEP 695), financial_planning.",
        "Multi-domain tasks automatically get a fused adapter stack.",
      ].join(" "),
      parameters: {
        task_description: {
          type: "string",
          description:
            "Describe the task you are about to work on. Be specific — " +
            "mention frameworks, databases, or patterns involved. " +
            "Example: 'Build a FastAPI endpoint that queries pgvector with HNSW search'.",
        },
      },
      output: {
        schema: { type: "object", additionalProperties: true },
        render: renderRouteTask,
      },
      async execute({ task_description }) {
        return invokeRouter(task_description || "general software engineering task");
      },
    })
  );

  // ── Tool 2: task_progress ──────────────────────────────────────────────
  ctx.tools.register(
    defineTool({
      name: "task_progress",
      description: [
        "Self-check tool. Call when you have tried the same approach 2+ times",
        "without progress, or when you want to assess context budget.",
        "Returns stagnation analysis and a recommended action:",
        "CONTINUE (keep going), REPLAN (stuck — change strategy fundamentally),",
        "COMPACT (context getting full — compress history before continuing),",
        "DELEGATE (context almost full — handoff to a subagent).",
        "You MUST follow the recommended action, not ignore it.",
      ].join(" "),
      parameters: {
        session_id: {
          type: "string",
          description: "Use the current conversation or session ID. If unknown, use 'main'.",
        },
        tool_just_called: {
          type: "string",
          description:
            "Name of the tool you have been repeatedly calling without progress " +
            "(e.g. 'verify_project', 'bash', 'str_replace_editor').",
        },
        result_summary: {
          type: "string",
          description:
            "One-line summary of the result of your last 1-3 attempts. " +
            "Be specific: include error messages or test failure counts.",
        },
        tokens_used_estimate: {
          type: "number",
          description:
            "Rough token estimate for this session. Divide total characters in this " +
            "conversation by 4 to get a reasonable approximation.",
        },
        context_window_size: {
          type: "number",
          description:
            "The context window of your active model. " +
            "16384 for Qwen 3.8 27B models, 32768 for Ornith 1.5 35B models.",
        },
      },
      output: {
        schema: { type: "object", additionalProperties: true },
        render: renderProgress,
      },
      async execute({
        session_id,
        tool_just_called,
        result_summary,
        tokens_used_estimate,
        context_window_size,
      }) {
        const sid = (session_id || "main").trim();
        const sess = getSession(sid);
        sess.turnCount++;

        // Update context window if provided
        if (context_window_size && context_window_size > 0) {
          sess.contextWindow = context_window_size;
        }
        // Update token estimate (take max of what we know vs what agent reports)
        if (tokens_used_estimate && tokens_used_estimate > 0) {
          sess.tokenEstimate = Math.max(sess.tokenEstimate, tokens_used_estimate);
        }

        const budgetPct = Math.round(
          (sess.tokenEstimate / sess.contextWindow) * 100
        );

        // Loop detection
        const { isLoop, repeats } = detectLoop(
          sess,
          tool_just_called || "unknown",
          result_summary || ""
        );
        if (isLoop) sess.loopCount++;

        // Determine action
        let action = "CONTINUE";
        let reasoning = "No stagnation detected. Current approach is making progress.";

        if (isLoop) {
          action = "REPLAN";
          reasoning =
            `'${tool_just_called}' has produced the same result ${repeats}× in a row. ` +
            `You are in a loop. STOP this approach entirely. ` +
            `Ask yourself: wrong file? wrong abstraction? missing import? wrong assumption? ` +
            `Try a completely different approach — don't just retry with minor tweaks.`;
        } else if (budgetPct >= 88) {
          action = "DELEGATE";
          reasoning =
            `Context is ${budgetPct}% full (${sess.tokenEstimate.toLocaleString()} / ` +
            `${sess.contextWindow.toLocaleString()} tokens). ` +
            `Call summarize_and_delegate now to create a handoff brief, ` +
            `then spawn a subagent with that brief. Do not continue in this context.`;
        } else if (budgetPct >= 72) {
          action = "COMPACT";
          reasoning =
            `Context is ${budgetPct}% full. Run the /compact command to compress ` +
            `historical tool outputs before continuing. This will recover headroom ` +
            `without losing the task thread.`;
        }

        return {
          status: isLoop ? "STAGNATED" : "PROGRESSING",
          loop_count: sess.loopCount,
          turn_count: sess.turnCount,
          budget_pct: budgetPct,
          tokens_estimate: sess.tokenEstimate,
          context_window: sess.contextWindow,
          action,
          reasoning,
        };
      },
    })
  );

  // ── Tool 3: summarize_and_delegate ─────────────────────────────────────
  ctx.tools.register(
    defineTool({
      name: "summarize_and_delegate",
      description: [
        "Generate a structured handoff brief for subagent delegation.",
        "Use when: (a) task_progress returns DELEGATE action, or (b) you are ",
        "fundamentally stuck and want a fresh context to continue the task.",
        "After calling this, use the tool-subagent tool to spawn a new agent and",
        "pass the handoff_brief as its opening instruction. The subagent will",
        "continue seamlessly from where you left off.",
      ].join(" "),
      parameters: {
        goal: {
          type: "string",
          description: "The overall task goal in 1-2 sentences.",
        },
        completed: {
          type: "array",
          items: { type: "string" },
          description: "Subtasks or files that are fully DONE and verified (tests pass).",
        },
        pending: {
          type: "array",
          items: { type: "string" },
          description: "Subtasks or files that still need to be done.",
        },
        blockers: {
          type: "array",
          items: { type: "string" },
          description:
            "Known errors, failing tests, or approach dead-ends the subagent should avoid.",
        },
        key_decisions: {
          type: "array",
          items: { type: "string" },
          description:
            "Architectural or design decisions already made that the subagent must respect.",
        },
        working_dir: {
          type: "string",
          description: "Absolute path of the project directory.",
        },
      },
      output: {
        schema: { type: "object", additionalProperties: true },
        render: renderHandoff,
      },
      async execute({ goal, completed = [], pending = [], blockers = [], key_decisions = [], working_dir = "" }) {
        const brief = [
          `# Task Continuation Brief`,
          ``,
          `## Goal`,
          goal,
          ``,
          working_dir ? `## Working Directory\n\`${working_dir}\`` : "",
          ``,
          `## Completed ✅ (do NOT redo these)`,
          completed.length
            ? completed.map((x) => `- ${x}`).join("\n")
            : "- Nothing completed yet.",
          ``,
          `## Pending 🔲 (start here)`,
          pending.length
            ? pending.map((x) => `- ${x}`).join("\n")
            : "- No specific pending items listed — assess the goal state and continue.",
          ``,
          `## Known Blockers & Dead Ends ⚠️`,
          blockers.length
            ? blockers.map((x) => `- ${x}`).join("\n")
            : "- None identified.",
          ``,
          `## Architectural Decisions Already Made`,
          key_decisions.length
            ? key_decisions.map((x) => `- ${x}`).join("\n")
            : "- None specified.",
          ``,
          `## How to Continue`,
          `1. Call \`route_task\` with the first pending item to activate the right expert team.`,
          `2. Work through the Pending list in order.`,
          `3. Call \`verify_project\` after completing each significant file.`,
          `4. Call \`task_progress\` if you get stuck or repeat the same fix 2+ times.`,
          `5. You are done when all pending items are complete and all tests pass.`,
        ]
          .filter((line) => line !== undefined)
          .join("\n");

        return {
          handoff_brief: brief,
          completed_count: completed.length,
          pending_count: pending.length,
          blocker_count: blockers.length,
        };
      },
    })
  );
}

apply.inject = ["tools"];
export default apply;
