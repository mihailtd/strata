# training

Isolated uv project for [Unsloth](https://unsloth.ai) — fast LoRA/QLoRA/DoRA fine-tuning
on AMD ROCm. Kept separate from the root `gnn-experiment` project because unsloth pins
`torch<2.12.0`, incompatible with root's `torch==2.13.0+rocm7.2` (same reasoning as
`serving/` being separate).

## Setup summary

- `torch==2.11.0+rocm7.2` / `torchvision==0.26.0+rocm7.2` / `triton-rocm==3.6.0`, all from
  the same `pytorch-rocm` index used elsewhere in this repo, pinned in `pyproject.toml`.
- `bitsandbytes==0.50.0` resolved automatically — above the `<=0.49.2` version with a known
  NaN bug on AMD (per Unsloth's AMD docs), so no special preview wheel needed.
- GPU confirmed working: `torch.cuda.is_available()` → `True`, `AMD Radeon RX 7900 XTX`.

## Commands

On native CachyOS Linux, the AMD Radeon RX 7900 XTX is accessed directly via native `/dev/kfd` with no `LD_PRELOAD` or wrapper flags needed.

**Verify GPU access:**

```bash
uv run python -c "import torch, unsloth; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Train / infer / chat / export via the unsloth CLI** (from this project's venv):

```bash
uv run unsloth train ...
uv run unsloth chat ...
uv run unsloth export ...
```

## Unsloth Studio (local web UI)

Studio backend runs directly against native ROCm 7.2:

```bash
~/.unsloth/studio/unsloth_studio/bin/unsloth studio -p 8888
```

Confirms real GPU use via the startup log line: `Hardware detected: ROCm (HIP ...) -- AMD Radeon RX 7900 XTX`
(vs `Hardware detected: CPU training backend` if the fix above wasn't applied).

Open **<http://127.0.0.1:8888>** in a browser. Bound to localhost only by default — pass
`-H 0.0.0.0` to expose on the LAN, or `--cloudflare` for a public HTTPS tunnel (off by
default, don't enable without knowing what you're exposing).

**Other useful Studio flags**: `--disable-tools` (turn off server-side web search/code
execution, e.g. if you want opencode/an agent to own execution instead), `--parallel N`
(llama-server decode slots, default 4), `--password <pw>` (set initial admin password
non-interactively).

**Stop Studio**: Ctrl+C in its terminal, or `unsloth studio stop`.

## Benchmark & Metric Logging

All benchmark, datagen, and evaluation runs log structured metrics directly to zero-overhead JSON Lines (`.jsonl`) files in `results/` (e.g., `results/micro_probe_runs.jsonl`, `results/eval_runs.jsonl`).

Historical MLflow runs have been exported to `results/mlflow_export/` (`runs_summary.jsonl`, `metrics_full.jsonl`, `experiments.json`), and the legacy `mlruns.db` database is preserved on disk as a fallback archive.
