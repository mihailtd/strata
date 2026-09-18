/**
 * dsh-plugin-moa-router-tool (System One Decision Sidecar Plugin)
 *
 * Exposes the System One Decision Engine as first-class tools inside DeepSeek Harness:
 *   1. `route_task` / `route_prompt`: Activates optimal specialist LoRA team (sub-10ms).
 *   2. `system_one_health`: Hello World & diagnostics tool reporting device, VRAM, and discovered domains.
 *   3. `decision_noul`: Calibrated boolean verification gate (loop detection, test pass/fail check).
 *   4. `decision_score`: Continuous 0-100 rubric score evaluation.
 *
 * Fast path: Direct asynchronous HTTP probe to local sidecar daemon (http://127.0.0.1:8100).
 * Fallback path: Lightweight Python CLI bridge via AdapterRegistry (zero GPU/torch deps).
 */

import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { defineTool } from "@deepseek-ai/dsh-tools";

export const name = "tool-moa-router";
export const inject = ["tools"];
export const description =
  "System One Decision Engine & Dynamic Mixture-of-Adapters (MoA) Router. " +
  "Analyzes tasks to determine optimal specialist adapter teams, boolean verification, " +
  "and rubric scoring via sub-10ms non-autoregressive decision primitives.";

const SIDECAR_URL = process.env.DECISION_SERVICE_URL || "http://127.0.0.1:8100";

// Resolve the CLI bridge path relative to this plugin file.
const PLUGIN_DIR = new URL(".", import.meta.url).pathname;
const REPO_ROOT = resolve(PLUGIN_DIR, "../../../.."); // gnn-experiment/
const CLI_PATH = resolve(REPO_ROOT, "apps/harness/router/router_cli.py");

/**
 * Fast path: Async HTTP fetch with strict timeout.
 */
async function fetchSidecar(endpoint, body = null, timeoutMs = 150) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const opts = {
      method: body ? "POST" : "GET",
      signal: controller.signal,
      headers: { "Content-Type": "application/json" },
    };
    if (body) {
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch(`${SIDECAR_URL}${endpoint}`, opts);
    clearTimeout(timer);
    if (resp.ok) {
      return await resp.json();
    }
  } catch (_err) {
    clearTimeout(timer);
  }
  return null;
}

/**
 * Fallback path: CLI execution.
 */
function invokeRouterCli(prompt) {
  if (!existsSync(CLI_PATH)) {
    throw new Error(
      `Decision router CLI not found at: ${CLI_PATH}. ` +
        "Ensure apps/harness/router/router_cli.py exists."
    );
  }

  const raw = execFileSync("uv", ["run", "python", CLI_PATH, prompt], {
    cwd: REPO_ROOT,
    encoding: "utf-8",
    timeout: 10_000,
    stdio: ["pipe", "pipe", "pipe"],
  });

  const result = JSON.parse(raw.trim());
  if (result.error) {
    throw new Error(`Router error: ${result.error}`);
  }
  return result;
}

function renderRouteResult(_args, value) {
  const expertLines = Object.entries(value.experts || {})
    .sort(([, a], [, b]) => b - a)
    .map(([domain, weight]) => `  • ${domain.padEnd(20)} γ = ${weight.toFixed(3)}`)
    .join("\n");

  const multiLabel = value.is_multi_expert ? "MULTI-EXPERT FUSION" : "SINGLE EXPERT";
  const latency = value.routing_latency_ms;
  const engineStr = value.engine || "neural_system_one_sidecar";

  return [
    {
      type: "text",
      text: [
        `[System One Router — ${multiLabel}] (${latency}ms, engine: ${engineStr})`,
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

async function executeRoute(args) {
  const task = (args.task || "").trim();
  if (!task) {
    return {
      is_multi_expert: false,
      experts: { general: 1.0 },
      recommended_model_id: "qwen3.8:27b",
      active_domains: [],
      system_prompt_prefix:
        "You are an expert autonomous software engineer writing clean modern code.",
      rationale: "Empty task description — defaulting to general base engine.",
      routing_latency_ms: 0,
      engine: "default",
    };
  }

  // 1. Try sidecar daemon
  const sidecarRes = await fetchSidecar("/route", { prompt: task }, 200);
  if (sidecarRes) {
    sidecarRes.engine = "neural_system_one_sidecar";
    return sidecarRes;
  }

  // 2. Fall back to offline CLI bridge
  return invokeRouterCli(task);
}

export function apply(ctx) {
  // 1. Register route_prompt tool
  const routeToolConfig = {
    description:
      "Universal System One Decision Router. Analyzes task descriptions to assemble " +
      "the optimal specialist LoRA team with dynamic weights, recommended model ID, " +
      "and system prompt prefix. Call at task start for maximum domain fidelity.",
    parameters: {
      task: {
        type: "string",
        description: "The task or prompt text to analyze.",
      },
    },
    output: {
      schema: {
        type: "object",
        additionalProperties: true,
        properties: {
          is_multi_expert: { type: "boolean" },
          experts: { type: "object", additionalProperties: false },
          recommended_model_id: { type: "string" },
          active_domains: { type: "array", items: { type: "string" } },
          system_prompt_prefix: { type: "string" },
          rationale: { type: "string" },
          routing_latency_ms: { type: "number" },
          engine: { type: "string" },
        },
      },
      render: renderRouteResult,
    },
    execute: executeRoute,
  };

  ctx.tools.register(defineTool({ ...routeToolConfig, name: "route_prompt" }));
  ctx.tools.register(defineTool({ ...routeToolConfig, name: "route_task" }));


  // 2. Register system_one_health diagnostic tool
  ctx.tools.register(
    defineTool({
      name: "system_one_health",
      description:
        "Hello-world diagnostic tool for the System One Decision Service. " +
        "Reports active device (CPU, CUDA, MPS), resident VRAM, model architecture, " +
        "and registered adapter domains.",
      parameters: {},
      output: {
        schema: { type: "object", additionalProperties: true },
        render: (_args, value) => [
          {
            type: "text",
            text: [
              `[System One Decision Service Status]`,
              `• Status:              ${value.status}`,
              `• Device:              ${value.device}`,
              `• VRAM Allocation:     ${value.vram_mb} MB`,
              `• Model:              ${value.model_name || "N/A"}`,
              `• Registered Domains: (${(value.registered_domains || []).length}) ${ (value.registered_domains || []).join(", ") }`,
              `• Service URL:         ${SIDECAR_URL}`,
            ].join("\n"),
          },
        ],
      },
      async execute() {
        const health = await fetchSidecar("/health", null, 300);
        if (health) {
          return health;
        }
        return {
          status: "sidecar_offline",
          device: "offline (CLI fallback ready)",
          vram_mb: 0,
          model_name: "ModernBERT-base (offline)",
          registered_domains: [
            "general",
            "postgresql",
            "python_web",
            "duckdb",
            "astral",
            "python_modern",
            "financial_planning",
          ],
        };
      },
    })
  );

  // 3. Register decision_noul boolean verification gate
  ctx.tools.register(
    defineTool({
      name: "decision_noul",
      description:
        "Calibrated non-autoregressive boolean verification gate. Answers binary questions " +
        "(e.g. 'Has the task stalled in a loop?', 'Did the code satisfy the acceptance test?') " +
        "in sub-10ms with calibrated probabilities (ECE < 0.04).",
      parameters: {
        prompt: {
          type: "string",
          description: "The verification prompt or condition to evaluate.",
        },
        threshold: {
          type: "number",
          description: "Probability threshold for True decision (default 0.50).",
        },
      },
      output: {
        schema: { type: "object", additionalProperties: true },
        render: (_args, value) => [
          {
            type: "text",
            text: `[System One Noul Gate] Decision: ${value.decision ? "TRUE" : "FALSE"} (Probability: ${value.probability}, Confidence: ${value.confidence}, Latency: ${value.latency_ms}ms)`,
          },
        ],
      },
      async execute(args) {
        const res = await fetchSidecar(
          "/decide/noul",
          { prompt: args.prompt, threshold: args.threshold || 0.5 },
          300
        );
        if (res) return res;
        return {
          decision: false,
          probability: 0.5,
          confidence: 0.5,
          latency_ms: 0,
          error: "Sidecar daemon offline",
        };
      },
    })
  );

  // 4. Register decision_score rubric scoring tool
  ctx.tools.register(
    defineTool({
      name: "decision_score",
      description:
        "Continuous 0-100 rubric scoring head. Evaluates code diffs, answers, or agent actions " +
        "against an explicit rubric in a single non-autoregressive forward pass.",
      parameters: {
        prompt: {
          type: "string",
          description: "Rubric and content to score.",
        },
      },
      output: {
        schema: { type: "object", additionalProperties: true },
        render: (_args, value) => [
          {
            type: "text",
            text: `[System One Score Head] Score: ${value.score}/100.0 (Latency: ${value.latency_ms}ms)`,
          },
        ],
      },
      async execute(args) {
        const res = await fetchSidecar("/decide/score", { prompt: args.prompt }, 300);
        if (res) return res;
        return {
          score: 50.0,
          latency_ms: 0,
          error: "Sidecar daemon offline",
        };
      },
    })
  );
}

apply.inject = ["tools"];
