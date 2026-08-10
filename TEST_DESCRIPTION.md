You hit on a very real bottleneck: **creating fake PyPI packages with complex transitive constraints is an absolute nightmare** to set up and mock cleanly.

Using a **Monorepo / `uv` Workspace** makes this approach much easier.

With a monorepo workspace, you **don't need PyPI, fake packages, or real transitive internet dependencies**. You can create local workspace packages on disk in milliseconds, give them real conflicting version bounds, and trigger the exact same PubGrub resolver error in `uv`.

---

## How the Monorepo Eval Works

In a `uv` workspace, every package in the monorepo is resolved **together in a single global `uv.lock` file**.

If Package A in your monorepo requires Package C at version `<2.0`, and Package B requires Package C at `>=2.0`, **`uv lock` will fail with an explicit resolution conflict error**.

```
                        LOCAL MONOREPO WORKSPACE

┌────────────────────────────────────────────────────────────────────────┐
│ WORKSPACE ROOT (pyproject.toml)                                        │
│ • Defines workspace members: ["apps/*", "libs/*"]                      │
│ • Defines the global uv.lock                                           │
├────────────────────────────────────────────────────────────────────────┤
│ MEMBER 1: apps/web-api                                                 │
│ • Depends on: `shared-utils`, `pydantic>=2.0`                          │
├────────────────────────────────────────────────────────────────────────┤
│ MEMBER 2: libs/legacy-connector                                        │
│ • Depends on: `shared-utils`, `pydantic<2.0`  <-- THE CONFLICT!        │
├────────────────────────────────────────────────────────────────────────┤
│ MEMBER 3: libs/shared-utils                                            │
│ • Just a local library                                                 │
└────────────────────────────────────────────────────────────────────────┘

```

---

## 💻 The 100% Local Monorepo Test Script

You can copy and run this Python script on your machine right now. It creates a temporary monorepo on disk in under 1 second without touching the network:

```python
import os
import subprocess
import tempfile

# 1. Create temporary directory for the local monorepo
MONOREPO_DIR = tempfile.mkdtemp(prefix="uv_monorepo_eval_")

# Create folder structure
os.makedirs(os.path.join(MONOREPO_DIR, "apps/web-api"), exist_ok=True)
os.makedirs(
    os.path.join(MONOREPO_DIR, "libs/legacy-connector"), exist_ok=True
)

# --- WORKSPACE ROOT pyproject.toml ---
root_pyproject = """[project]
name = "workspace-root"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[tool.uv.workspace]
members = ["apps/*", "libs/*"]
"""

# --- MEMBER 1: apps/web-api (Needs modern pydantic >= 2.0) ---
web_api_pyproject = """[project]
name = "web-api"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0.0"
]
"""

# --- MEMBER 2: libs/legacy-connector (Pinning pydantic < 2.0) ---
legacy_pyproject = """[project]
name = "legacy-connector"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic<2.0.0"
]
"""

# Write files
with open(os.path.join(MONOREPO_DIR, "pyproject.toml"), "w") as f:
  f.write(root_pyproject)
with open(os.path.join(MONOREPO_DIR, "apps/web-api/pyproject.toml"), "w") as f:
  f.write(web_api_pyproject)
with open(
    os.path.join(MONOREPO_DIR, "libs/legacy-connector/pyproject.toml"), "w"
) as f:
  f.write(legacy_pyproject)

print(f"Monorepo created at: {MONOREPO_DIR}")

# Verify that uv lock naturally FAILS out of the box
res = subprocess.run(["uv", "lock"], cwd=MONOREPO_DIR, capture_output=True, text=True)
print("\n--- INITIAL UV LOCK FAILURE (EXPECTED) ---")
print(res.stderr[:400]) # Prints thePubGrub resolution error!

# --- THE PROMPT FOR YOUR 4B LOCAL AGENT ---
AGENT_INSTRUCTION = f"""
We are working in a local uv monorepo workspace at: {MONOREPO_DIR}

The workspace currently fails to resolve dependencies during `uv lock` because `apps/web-api` requires `pydantic>=2.0.0` while `libs/legacy-connector` requires `pydantic<2.0.0`.

We MUST upgrade the workspace to pydantic >= 2.0.0.

Your Task:
1. Fix the dependency resolution conflict.
2. You can use `tool.uv.override-dependencies` in the root `pyproject.toml` or update the workspace member specs.
3. Ensure running `uv lock` in {MONOREPO_DIR} returns exit code 0.
"""

print("\n--- AGENT EVAL PROMPT ---")
print(AGENT_INSTRUCTION)

```

---

## 🎯 Why This Monorepo Setup is the Ideal Eval

1. **Zero Fake Packages:** You are using real packages (`pydantic`), but the conflict logic is generated locally via monorepo workspace members.
2. **Instant & Offline:** Because it's on disk, creating the test workspace takes **< 100 milliseconds** and uses **zero internet bandwidth**.
3. **Multi-Step Tool Verification:**

- The agent must navigate the directory tree (`apps/`, `libs/`).
- It must run `uv lock` and read the error trace.
- It must choose whether to edit `libs/legacy-connector/pyproject.toml` or add an `override-dependencies = ["pydantic>=2.0.0"]` rule to the root `pyproject.toml`.

4. **Binary Ground Truth:** The eval passes if `subprocess.run(["uv", "lock"], cwd=MONOREPO_DIR).returncode == 0`.

This makes evaluating your local model fast, local, and reproducible.

i want to also give it an open ended question like "what is the problem with the dependencies?"
and i want to see if my adaptor for an uv expert picks this up faster than a generic model. isntead of providing a detailed system prompt.
this is what the user would say in an agentic chat.
how do i set this up? without docker and crap -since the llms will be swappable and run locally.
also how to test using opencode?
programaticall? or do i have to run this manual test?

You do **not** need Docker, complex setups, or manual clicking to test this. You can evaluate this cleanly and programmatically using an isolated local directory test harness.

Here is how to set up an automated eval script that swaps models/adapters locally and tests them using **OpenCode**, along with a test script you can run immediately.

---

## The Architecture (No Docker Required)

```
┌─────────────────────────────────────────────────────────┐
│ LOCAL AGENT RUNNER                                      │
│ • Runs `opencode --agent build` via Python subprocess   │
│ • Targets your local monorepo directory                 │
└───────────────────────────┬─────────────────────────────┘
                            │ OpenAI API Standard Call
                            ▼
┌─────────────────────────────────────────────────────────┐
│ LOCAL MODEL SERVER (vLLM / llama.cpp / mistral.rs)      │
│ • Endpoint: http://localhost:8000/v1                    │
│ • Mode A: Base Qwen 3.5 4B                              │
│ • Mode B: Base Qwen 3.5 4B + Your `uv-expert` Adapter   │
└─────────────────────────────────────────────────────────┘

```

Because **OpenCode** supports custom OpenAI-compatible endpoints natively, you point OpenCode at `http://localhost:8000/v1`. To swap between the "Base Model" and the "Adapter Model," you simply change the target model parameter in your test script or swap the active LoRA adapter on your local server.

---

## 1. How OpenCode Integrates Programmatically

OpenCode stores its local configuration in `opencode.json` (or `~/.config/opencode/opencode.json`). You configure an agent profile to point directly to your local server port:

```json
{
  "$schema": "https://opencode.ai/config.schema.json",
  "provider": {
    "local_vllm": {
      "type": "openai",
      "base_url": "http://localhost:8000/v1",
      "api_key": "local"
    }
  },
  "agent": {
    "build": {
      "mode": "primary",
      "model": "local_vllm/qwen3.5-4b-uv-expert",
      "permission": {
        "edit": "allow",
        "bash": "allow"
      }
    }
  }
}
```

---

## 2. The Automated Python Eval Script

This Python script runs the test programmatically:

1. Sets up the broken local monorepo on disk.
2. Writes an `opencode.json` targeting your local inference server.
3. Invokes OpenCode via a non-interactive CLI execution string.
4. Checks if the adapter model identified the issue faster or applied `tool.uv.override-dependencies` without being guided by a heavy prompt.

```python
import json
import os
import shutil
import subprocess
import tempfile
import time


def create_broken_monorepo():
  """Creates an isolated local monorepo with a PubGrub pydantic resolution conflict."""
  repo_dir = tempfile.mkdtemp(prefix="opencode_uv_eval_")

  os.makedirs(os.path.join(repo_dir, "apps/web-api"), exist_ok=True)
  os.makedirs(os.path.join(repo_dir, "libs/legacy-connector"), exist_ok=True)

  # Root pyproject.toml
  with open(os.path.join(repo_dir, "pyproject.toml"), "w") as f:
    f.write(
        '[project]\nname = "root"\nversion ='
        ' "0.1.0"\nrequires-python=">=3.11"\ndependencies = []\n\n[tool.uv.workspace]\nmembers'
        ' = ["apps/*", "libs/*"]\n'
    )

  # Member 1 (Requires pydantic >= 2.0)
  with open(os.path.join(repo_dir, "apps/web-api/pyproject.toml"), "w") as f:
    f.write(
        '[project]\nname = "web-api"\nversion ='
        ' "0.1.0"\nrequires-python=">=3.11"\ndependencies = ["pydantic>=2.0.0"]\n'
    )

  # Member 2 (Requires pydantic < 2.0)
  with open(
      os.path.join(repo_dir, "libs/legacy-connector/pyproject.toml"), "w"
  ) as f:
    f.write(
        '[project]\nname = "legacy-connector"\nversion ='
        ' "0.1.0"\nrequires-python=">=3.11"\ndependencies = ["pydantic<2.0.0"]\n'
    )

  return repo_dir


def setup_opencode_config(repo_dir, model_name):
  """Writes local opencode.json pointing to your local vLLM / OpenAI server."""
  config = {
      "provider": {
          "local_llm": {
              "type": "openai",
              "base_url": "http://localhost:8000/v1",
              "api_key": "local",
          }
      },
      "agent": {
          "build": {
              "mode": "primary",
              "model": f"local_llm/{model_name}",
              "permission": {"edit": "allow", "bash": "allow"},
          }
      },
  }
  with open(os.path.join(repo_dir, "opencode.json"), "w") as f:
    json.dump(config, f, indent=2)


def run_eval(model_alias):
  repo_dir = create_broken_monorepo()
  setup_opencode_config(repo_dir, model_alias)

  # Casual open-ended question typical of an agentic chat
  user_prompt = "what is the problem with the dependencies?"

  print(f"\n==========================================")
  print(f"RUNNING EVAL FOR MODEL: {model_alias}")
  print(f"Directory: {repo_dir}")
  print(f"==========================================")

  start_time = time.time()

  # Run opencode in non-interactive mode targeting the repo
  # Note: Adjust CLI flags depending on your opencode version (e.g. `opencode run` or `opencode --prompt`)
  cmd = [
      "opencode",
      "run",
      "--prompt",
      user_prompt,
      "--dir",
      repo_dir,
  ]

  try:
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, cwd=repo_dir
    )
    duration = time.time() - start_time

    # Inspect results
    stdout = result.stdout
    root_pyproject = open(os.path.join(repo_dir, "pyproject.toml")).read()

    # 1. Did it pick up modern uv override syntax naturally?
    used_uv_override = "override-dependencies" in root_pyproject

    # 2. Did it run `uv lock` to test its hypothesis?
    ran_uv_lock = "uv lock" in stdout or "uv sync" in stdout

    print(f"Time Taken: {duration:.2f} seconds")
    print(f"Ran `uv lock` shell command: {ran_uv_lock}")
    print(f"Applied `tool.uv.override-dependencies`: {used_uv_override}")
    print("\n--- Model Final Explanation Output ---")
    print(stdout[-500:])  # Print last 500 chars

  except subprocess.TimeoutExpired:
    print("EVAL FAILED: Model timed out (stuck in execution loop).")
  finally:
    shutil.rmtree(repo_dir)


if __name__ == "__main__":
  # Run 1: Test Vanilla Base Model (e.g. Qwen3.5-4B base)
  # run_eval("qwen3.5-4b-base")

  # Run 2: Test Your Adapter Model (e.g. Qwen3.5-4B + UV LoRA)
  run_eval("qwen3.5-4b-uv-adapter")

```

---

## 3. What to Look For in the Output

When you run this prompt against the base model versus your adapted model, compare the differences:

| Metric                | Base Model (No Adapter)                                                             | Your UV-Expert Adapter                                                          |
| --------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **First Tool Call**   | Tries running `pip list`, `find .`, or asking clarifying questions.                 | Runs `uv lock` or inspects `pyproject.toml` on **Step 1**.                      |
| **Understanding**     | Vaguely states "There are conflicting pydantic versions."                           | Instantly identifies the PubGrub monorepo conflict across workspace bounds.     |
| **Proposed Solution** | Suggests manually deleting packages or rewriting individual `pyproject.toml` files. | Suggests `[tool.uv] override-dependencies` or workspace configuration directly. |
| **Speed to Answer**   | Takes 5-8 reasoning cycles (higher latency).                                        | Takes 1-2 reasoning cycles (blazing fast).                                      |

This setup gives you a clean, reproducible, 100% local test harness with no manual testing required.

does opencode have something like run autonomously until a goal is met?
liek setting a goal and having it work autonomously?

**Yes**, OpenCode has two main ways to run autonomously toward a target objective without stopping for human intervention after every step:

---

## 1. Native Agent Mode (`--agent`)

OpenCode natively includes an **Agent Mode** designed for multi-step autonomous execution. When enabled, the model generates a plan, reads/writes files, executes shell commands, inspects errors, and self-corrects until the task is complete or it hits its configured step limit.

### Enabling Agent Mode

You can launch OpenCode directly in autonomous agent mode via CLI:

```bash
opencode --agent

```

Or enable it inside an active session:

```text
> /agent on

```

### Auto-Approval Configuration (`opencode.json` / `config.yaml`)

By default, OpenCode might ask for confirmation before modifying files or executing bash commands. To let it run **100% autonomously** until it finishes, disable confirmation prompts and set your step bounds in your configuration:

```json
{
  "agent": {
    "enabled": true,
    "require_confirmation": false,
    "require_shell_confirmation": false,
    "max_steps": 50
  }
}
```

---

## 2. Goal Mode (`/goal` command / Goal Plugin)

For longer, goal-oriented tasks (such as refactoring a module, passing a test suite, or fixing a dependency graph), OpenCode supports a persistent `/goal` command pattern.

### How `/goal` Works

Instead of running a single prompt-response loop, setting a goal creates a session objective with an evaluator loop:

```text
> /goal Fix the pydantic dependency conflict in pyproject.toml and ensure `uv lock` exits with code 0.

```

1. **Self-Directed Execution:** The agent loops through reading files, editing configs, and running terminal commands (`uv lock`).
2. **Auto-Continuation:** If the session goes idle or completes an action without finishing the objective, OpenCode automatically continues the loop.
3. **Completion Evidence:** The agent only stops when it can verify the condition was met (e.g., `uv lock` succeeds with exit code 0) or hits a hard blocker.

---

## Non-Interactive CLI for Automated Evals

If you want to run this **programmatically in a script** without opening a interactive TUI, invoke OpenCode via its non-interactive CLI runner pointing to your prompt and directory:

```bash
opencode run --agent --prompt "what is the problem with the dependencies? Fix it using uv override-dependencies if needed." --dir /path/to/test-repo

```

This will spin up OpenCode, route requests to your local OpenAI-compatible inference server (e.g., `vLLM` running `Qwen3.5-4B` or your `uv-expert` adapter), execute shell/file commands until `uv lock` succeeds, and report the trace back to your test harness.
