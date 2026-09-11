/**
 * dsh-plugin-moa-router-tool
 *
 * Registers a `route_prompt` tool inside DeepSeek Harness that calls the
 * DynamicMoARouter via a lightweight Python CLI bridge (no GPU/torch deps).
 *
 * The agent uses this to:
 *   1. Understand which specialist LoRA team is optimal for the current task.
 *   2. Receive the recommended model endpoint ID (already registered in settings.yaml).
 *   3. Get a pre-built system prompt prefix it can inject for max domain fidelity.
 *
 * The agent can then steer itself: reference the recommended_model_id when
 * delegating to sub-agents, or prepend system_prompt_prefix to its own reasoning.
 */

import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { defineTool } from "@deepseek-ai/dsh-tools";

export const name = "tool-moa-router";
export const inject = ["tools"];
export const description =
  "Dynamic Mixture-of-Adapters (MoA) Router. Given a task description, " +
  "analyzes which specialist LoRA domains are relevant (postgresql, python_web, " +
  "duckdb, astral, python_modern, financial_planning) and returns: the computed " +
  "expert mixture weights (γₖ), the recommended model endpoint ID (pre-registered " +
  "in the harness), and a domain-specific system prompt prefix you can inject into " +
  "your reasoning or into sub-agent instructions for maximum domain fidelity. " +
  "Call this at the start of complex multi-domain tasks to ensure the right expert " +
  "team is activated.";

// Resolve the CLI bridge path relative to this plugin file.
// Works regardless of where dsh is launched from.
const PLUGIN_DIR = new URL(".", import.meta.url).pathname;
const REPO_ROOT = resolve(PLUGIN_DIR, "../../../.."); // gnn-experiment/
const CLI_PATH = resolve(REPO_ROOT, "apps/harness/router/router_cli.py");

function invokeRouterCli(prompt) {
  if (!existsSync(CLI_PATH)) {
    throw new Error(
      `MoA router CLI not found at: ${CLI_PATH}. ` +
        "Ensure apps/harness/router/router_cli.py exists."
    );
  }

  const raw = execFileSync("uv", ["run", "python", CLI_PATH, prompt], {
    cwd: REPO_ROOT,
    encoding: "utf-8",
    timeout: 10_000, // routing is pure regex+math, should be <50ms; 10s is generous
    stdio: ["pipe", "pipe", "pipe"],
  });

  const result = JSON.parse(raw.trim());
  if (result.error) {
    throw new Error(`Router error: ${result.error}`);
  }
  return result;
}

function renderRouteResult(_args, value) {
  const expertLines = Object.entries(value.experts)
    .sort(([, a], [, b]) => b - a)
    .map(([domain, weight]) => `  • ${domain.padEnd(20)} γ = ${weight.toFixed(3)}`)
    .join("\n");

  const multiLabel = value.is_multi_expert ? "MULTI-EXPERT FUSION" : "SINGLE EXPERT";
  const latency = value.routing_latency_ms;

  return [
    {
      type: "text",
      text: [
        `[MoA Router — ${multiLabel}] (${latency}ms)`,
        ``,
        `📌 Recommended model: ${value.recommended_model_id}`,
        ``,
        `Expert weights:`,
        expertLines,
        ``,
        `Rationale: ${value.rationale}`,
        ``,
        `System prompt prefix:`,
        value.system_prompt_prefix,
      ].join("\n"),
    },
  ];
}

export function apply(ctx) {
  ctx.tools.register(
    defineTool({
      name: "route_prompt",
      description,
      parameters: {
        task: {
          type: "string",
          description:
            "The task or prompt text to analyze. Can be your full user request, " +
            "a planning description, or a sub-task you're about to delegate. " +
            "The more specific, the better the routing signal.",
        },
      },
      output: {
        schema: {
          type: "object",
          additionalProperties: true,
          properties: {
            is_multi_expert: { type: "boolean" },
            experts: {
              type: "object",
              additionalProperties: false,
              description: "Domain name to mixture weight map.",
            },
            recommended_model_id: { type: "string" },
            active_domains: { type: "array", items: { type: "string" } },
            system_prompt_prefix: { type: "string" },
            rationale: { type: "string" },
            routing_latency_ms: { type: "number" },
          },
        },
        render: renderRouteResult,
      },
      async execute(args) {
        const task = (args.task || "").trim();
        if (!task) {
          return {
            is_multi_expert: false,
            experts: { general: 1.0 },
            recommended_model_id: "qwen3.8:27b",
            active_domains: [],
            system_prompt_prefix:
              "You are an expert autonomous software engineer writing clean modern Python code.",
            rationale: "Empty task description — defaulting to general engine.",
            routing_latency_ms: 0,
          };
        }
        return invokeRouterCli(task);
      },
    })
  );
}

apply.inject = ["tools"];

export default apply;
