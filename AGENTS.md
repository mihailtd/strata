# Agent Instructions — Infinite-Horizon Task Harness

## Expert Routing (MANDATORY at task start)

At the beginning of **every coding, engineering, or analysis task**, call `route_task`
with a description of what you are about to do. This activates the right specialist
LoRA adapter team transparently. The user does not need to know this happened.

```
Available expert domains: postgresql, python_web (FastAPI/Pydantic),
duckdb (OLAP/Parquet), astral (uv/ruff toolchain), python_modern (PEP 695),
financial_planning (VaR/CVaR).
```

Do this silently — do not announce "I am calling route_task". Just call it and proceed
with the returned `system_prompt_prefix` guiding your approach.

---

## Loop Detection & Progress Monitoring

You MUST call `task_progress` when ANY of the following are true:

- You have tried the **same fix** (same file, same change) **2 or more times** without
  the test result changing.
- You have called `verify_project` (or any verification tool) **3+ times** in a row
  and the error output is identical or nearly identical each time.
- You feel uncertain whether the current approach is working.
- You have not made measurable progress (new file created, test count improved) in
  the last 4 turns.

**You MUST follow the action returned by `task_progress`**:

| Action | What to do |
|--------|-----------|
| `CONTINUE` | Keep going with current approach |
| `REPLAN` | Stop. Think from first principles. Try a fundamentally different approach. |
| `COMPACT` | Run `/compact` command, then continue |
| `DELEGATE` | Call `summarize_and_delegate`, then spawn a subagent with the brief |

---

## Context Management

Estimate your token usage as: `(total characters in conversation) / 4`.

When `task_progress` returns `COMPACT` or `DELEGATE`, act on it immediately.
Do not try to squeeze "one more fix" before compacting — the recommendation is
already conservative (triggered at 72% and 88% of context window respectively).

---

## Delegation Protocol

When spawning a subagent to continue a task:

1. Call `summarize_and_delegate` to produce a structured handoff brief.
2. Use `tool-subagent` to spawn a new agent.
3. Pass the `handoff_brief` verbatim as the subagent's opening instruction.
4. The subagent will call `route_task` and `task_progress` autonomously from there.

The subagent does **not** need to re-read the conversation history.
The handoff brief contains everything it needs.

---

## Task Completion Signal

A task is complete when:
- `verify_project` returns `all_passed: true`, OR
- All deliverables specified by the user exist and are verified.

Once done, report concisely: what was built, where it lives, how to run it.
Do not over-explain what you did step by step — the user wants the result.

---

## Engineering Integrity & Zero-Mock Invariant (MANDATORY)

Never simulate, mock, fake, or disguise functionality:
1. **Zero Mocks & Dummy Artifacts**: Never create dummy weights (`torch.randn`), fake training passes, simulated inference, or stub pipelines. Every implementation must be genuine, complete, and mathematically functional.
2. **No Deceptive Facades**: Never brand an external commodity tool (e.g., Ollama, standard llama.cpp) as a custom engine or proprietary fleet. If an external service is called, label it plainly as an external proxy.
3. **No Fabricated Numbers**: All throughput (tok/s), latency, memory, and benchmark scores must be measured directly from live hardware telemetry. Never hardcode estimated or fake performance metrics.
4. **Radical Transparency**: If a feature or model cannot be implemented on current hardware, declare it plainly. Never create a workaround that can be misinterpreted as a real implementation.

---

## Runtime Details

- Engine: AMD ROCm RX 7900 XTX, native W4A16 Triton + MTP speculation
- Models available: see `apps/harness/settings.yaml` for full model list
- LoRA adapters live in `results/adapters/` — do not modify them directly
- Verification tool: `verify_project` (runs ruff + pytest automatically)

---

## Sequential Single-Engine Execution & A/B Testing Invariant

1. **Only One Engine Active At A Time**:
   - On single-GPU systems (24 GB VRAM), NEVER run multiple inference runtimes simultaneously.
   - Running two engines concurrently causes VRAM contention, driver context thrashing, and memory spilling to PCIe host memory.
2. **Complete Separation of Runtimes**:
   - `apps/runtime/` (+ `apps/runtime-triton/`): Custom Python / Triton Engine (Port 8000)
   - `apps/runtime-llama/`: Standalone C++ `llama-server` Baseline (Port 8001)
   - `apps/runtime-ollama/`: Standalone Ollama Baseline (Port 11434)
   - `apps/runtime-vllm/`: Standalone vLLM Baseline (Port 8004)
   - `apps/runtime-ipwf/`: In-place weight-folding engine, 3B/9B models (Port 8002)
   - `apps/runtime-next/`: Native Compiled Rust Engine (skeleton only, not yet functional)
   - Never proxy, embed, or manage one runtime from inside another runtime's process.
3. **Mandatory Sequential A/B Testing Protocol**:
   To benchmark and compare two runtimes (e.g., custom Triton vs llama.cpp or Ollama):
   - **Step 1**: Start Runtime A.
   - **Step 2**: Connect DSH harness, run end-to-end evaluation, and save `results/benchmarks/scorecard_<runtime_a>.json`.
   - **Step 3**: Fully terminate Runtime A and verify VRAM is released.
   - **Step 4**: Start Runtime B.
   - **Step 5**: Connect DSH harness, run the exact same evaluation, and save `results/benchmarks/scorecard_<runtime_b>.json`.
   - **Step 6**: Fully terminate Runtime B.
   - **Step 7**: Perform comparative analysis strictly using the two saved scorecards.

