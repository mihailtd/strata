import { execSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve, join } from "node:path";
import { defineTool } from "@deepseek-ai/dsh-tools";

const name = "tool-code-verify";
const inject = ["tools"];
const description =
  "Language-agnostic project verification tool. Automatically detects the project stack (Python pytest/ruff, Rust cargo test, Node npm/pnpm test, Go test), executes the test/lint suite, and returns structured diagnostics so you can autonomously diagnose and fix broken code.";

function runCommand(cmd, cwd) {
  try {
    const stdout = execSync(cmd, {
      cwd,
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
      timeout: 60000,
    });
    return { exitCode: 0, output: stdout, error: null };
  } catch (err) {
    const stdout = err.stdout ? err.stdout.toString() : "";
    const stderr = err.stderr ? err.stderr.toString() : "";
    return {
      exitCode: err.status ?? 1,
      output: stdout + "\n" + stderr,
      error: err.message,
    };
  }
}

function detectAndVerify(projectDir, mode = "all") {
  const absDir = resolve(projectDir);
  if (!existsSync(absDir)) {
    return {
      status: "ERROR",
      all_passed: false,
      framework: "unknown",
      summary: `Target path does not exist: ${absDir}`,
      details: "",
    };
  }

  // 1. Python Project Detection (pyproject.toml, pytest.ini, setup.py, requirements.txt)
  if (
    existsSync(join(absDir, "pyproject.toml")) ||
    existsSync(join(absDir, "pytest.ini")) ||
    existsSync(join(absDir, "setup.py"))
  ) {
    const results = [];
    let allPassed = true;

    // A. Linter (Ruff)
    if (mode === "all" || mode === "lint_only") {
      const lintRes = runCommand("uvx ruff check .", absDir);
      if (lintRes.exitCode !== 0) {
        allPassed = false;
        results.push(`[LINTER - RUFF]: FAILED\n${lintRes.output.trim()}`);
      } else {
        results.push("[LINTER - RUFF]: PASSED (Clean)");
      }
    }

    // B. Test Runner (Pytest)
    if (mode === "all" || mode === "tests_only") {
      let testRes = runCommand("uv run pytest -v", absDir);
      if (testRes.exitCode !== 0 && testRes.output.includes("command not found")) {
        testRes = runCommand("pytest -v", absDir);
      }
      if (testRes.exitCode !== 0) {
        allPassed = false;
        results.push(`[TEST RUNNER - PYTEST]: FAILED\n${testRes.output.trim()}`);
      } else {
        results.push(`[TEST RUNNER - PYTEST]: PASSED\n${testRes.output.trim()}`);
      }
    }

    return {
      status: allPassed ? "PASSED" : "FAILED",
      all_passed: allPassed,
      framework: "python",
      summary: allPassed
        ? "All Python tests and lint checks passed successfully."
        : "Python verification detected errors.",
      details: results.join("\n\n"),
    };
  }

  // 2. Rust Project Detection (Cargo.toml)
  if (existsSync(join(absDir, "Cargo.toml"))) {
    const testRes = runCommand("cargo test", absDir);
    const passed = testRes.exitCode === 0;
    return {
      status: passed ? "PASSED" : "FAILED",
      all_passed: passed,
      framework: "rust",
      summary: passed ? "All Cargo tests passed." : "Cargo test failed.",
      details: testRes.output.trim(),
    };
  }

  // 3. Node.js / TypeScript Project Detection (package.json)
  if (existsSync(join(absDir, "package.json"))) {
    const testCmd = existsSync(join(absDir, "pnpm-lock.yaml"))
      ? "pnpm test"
      : "npm test";
    const testRes = runCommand(testCmd, absDir);
    const passed = testRes.exitCode === 0;
    return {
      status: passed ? "PASSED" : "FAILED",
      all_passed: passed,
      framework: "nodejs",
      summary: passed ? "All Node.js tests passed." : "Node.js test failed.",
      details: testRes.output.trim(),
    };
  }

  // 4. Go Project Detection (go.mod)
  if (existsSync(join(absDir, "go.mod"))) {
    const testRes = runCommand("go test ./...", absDir);
    const passed = testRes.exitCode === 0;
    return {
      status: passed ? "PASSED" : "FAILED",
      all_passed: passed,
      framework: "golang",
      summary: passed ? "All Go tests passed." : "Go test failed.",
      details: testRes.output.trim(),
    };
  }

  return {
    status: "UNKNOWN",
    all_passed: false,
    framework: "none",
    summary: `No recognized project manifest (pyproject.toml, Cargo.toml, package.json, go.mod) found in ${absDir}`,
    details: "",
  };
}

function apply(ctx) {
  ctx.tools.register(
    defineTool({
      name: "verify_project",
      description,
      parameters: {
        path: {
          type: "string",
          description:
            "Directory of the project to verify (defaults to current working directory '.').",
        },
        mode: {
          type: "string",
          description:
            "Verification mode: 'all' (run tests and linter), 'tests_only', or 'lint_only'. Defaults to 'all'.",
        },
      },
      output: {
        schema: {
          type: "object",
          additionalProperties: true,
          properties: {
            status: { type: "string" },
            all_passed: { type: "boolean" },
            framework: { type: "string" },
            summary: { type: "string" },
            details: { type: "string" },
          },
        },
        render: (_args, value) => [
          {
            type: "text",
            text: `[Project Verification: ${value.status}]\nFramework: ${value.framework}\n${value.summary}\n\n${value.details}`,
          },
        ],
      },
      async execute(args) {
        const targetPath = args.path || ".";
        const mode = args.mode || "all";
        return detectAndVerify(targetPath, mode);
      },
    })
  );
}

apply.inject = ["tools"];

export { apply, description, inject, name, apply as default };

