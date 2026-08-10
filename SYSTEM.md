# System Details (WSL2 Dev Environment)

Snapshot taken 2026-08-10. Kept here as a reference for environment setup, especially since ML tooling (PyTorch, ROCm, ONNX Runtime) is version- and vendor-sensitive.

---

## Host / WSL

- **Windows**: Windows 11, version 25H2, build 26200 (registry reports `ProductName: Windows 10 Pro`, but build 26200 is a Windows 11 branch — likely an in-place-upgrade licensing artifact, not actually Windows 10).
- **WSL**: version 2.7.11.0, kernel 6.18.33.2-microsoft-standard-WSL2
- **Distro**: Ubuntu 22.04.5 LTS (jammy)
- **WSLg**: 1.0.73.2, Direct3D 1.611.1, DXCore 10.0.26100.1

### `.wslconfig` resource limits (`C:\Users\farca\.wslconfig`)
```
[wsl2]
processors=4
memory=24GB
swap=2GB
```
Host actually has **24 logical processors** and **~61.6 GB RAM** total — WSL is currently capped to only 4 cores / 24 GB. Worth raising `processors` (and possibly `memory`) in `.wslconfig` before running CPU-heavy data loading or multi-process training; requires `wsl --shutdown` + restart to take effect.

## CPU / Memory / Disk (as seen inside WSL)

- **CPU**: AMD Ryzen 9 9900X (12-core / 24-thread host part; only 4 threads exposed to WSL per config above)
- **RAM**: 23 GiB visible in WSL (per `.wslconfig` cap), 2 GiB swap
- **Disk**: `/` on `/dev/sdd`, 1007 GB total, 641 GB available (33% used)

## GPU

- **Card**: AMD Radeon RX 7900 XTX — **24 GB VRAM** (matches user-stated spec)
- Also present: integrated AMD Radeon Graphics (2 GB shared), and a "Meta Virtual Monitor" (Oculus/Quest Link virtual display, unrelated to compute)
- **No NVIDIA GPU in this machine** — `nvidia-smi`, CUDA, etc. do not apply here.

### WSL GPU passthrough status
- `/dev/dxg` present (`crw-rw-rw- ... 10, 258`) — the GPU paravirtualization device node WSL uses for all vendors is active.
- `/usr/lib/wsl/lib/` has `libd3d12.so`, `libd3d12core.so`, `libdxcore.so` — the DirectX/DXCore translation layer is installed and working.
- **Conclusion**: the GPU is correctly exposed to WSL at the OS level. This is the AMD equivalent of what `nvidia-smi` would confirm for an NVIDIA card — but it only proves passthrough, not that an ML compute stack is installed yet.
- **Not installed yet**: `rocminfo`, `rocm-smi`, `clinfo` — no ROCm or OpenCL runtime is present in WSL currently, so no compute framework has actually exercised the GPU.

### What's needed to actually run ML workloads on this GPU
The 7900 XTX (RDNA3, `gfx1100`) has official AMD ROCm-for-WSL support (ROCm ≥ 6.1). Two practical paths:
1. **ROCm + PyTorch (rocm build)** — install AMD's WSL-targeted ROCm packages (`amdgpu-install --usecase=wsl,rocm`) then `pip install torch --index-url https://download.pytorch.org/whl/rocm6.x`. Best performance, most compatible with the PEFT/MoE-LoRA stack (bitsandbytes' ROCm support is spottier — check before relying on 4-bit/8-bit quantization).
2. **DirectML** (`torch-directml`) — works on any DX12 GPU with no ROCm install, easier setup, but slower and less feature-complete than native ROCm for training.
Given this repo's roadmap (QLoRA, quantized adapters, custom CUDA/HIP-adjacent kernels), ROCm is the better long-term investment; DirectML is fine for quick smoke tests.

Windows GPU driver: 32.0.31035.1003 (recent Adrenalin branch, should support ROCm-for-WSL, but not independently verified against AMD's supported-driver list).

**Update (2026-08-10): GPU compute confirmed working end to end.** The project targets ROCm 7.2 (`gfx1100`/RX 7900 XTX is called out by name in AMD's official ROCm 7.2 WSL docs). User ran `sudo bash scripts/install_rocm_wsl.sh` (needed a fix — see script history / the `logname` note below — then a rerun completed cleanly, apt reported ROCm 7.2.0.70200 + `hsa-runtime-rocr4wsl-amdgpu` installed, `rocminfo` listed the `gfx1100` agent).

**Non-obvious problem hit after install**: even with ROCm installed and the user in the `render`/`video` groups, `torch.cuda.is_available()` still returned `False` (`RuntimeError: No CUDA GPUs are available` from `torch.cuda.init()`), while system `rocminfo` correctly detected the GPU (it even prints `WSL environment detected`). Root cause: **the PyTorch pip wheel bundles its own `libamdhip64.so`/`libhsa-runtime64.so`** inside `torch/lib/`, separate from the system ROCm install — and that bundled copy doesn't have WSL/DXCore support, only the native-Linux `/sys/class/kfd/kfd/topology/nodes` path (which doesn't exist in WSL, hence the `sysfs nodes path ... does not exist` warning that appears even when everything works). The system's `hsa-runtime-rocr4wsl-amdgpu` package (at `/opt/rocm-7.2.0/lib/libhsa-runtime64.so`) is the actual WSL-aware runtime.

**Fix**: force torch to load the system runtime instead of its bundled one via `LD_PRELOAD=/opt/rocm-7.2.0/lib/libhsa-runtime64.so`. Added to `.env` (alongside `HF_TOKEN`) so it's automatic via `uv run --env-file .env <script>` — **all script docstrings now say `--env-file .env`, this is required, not optional, for GPU scripts.** `HSA_OVERRIDE_GFX_VERSION=11.0.0` was tried too (commonly cited for RDNA3) but turned out to be unnecessary — `LD_PRELOAD` alone was sufficient; gfx1100 is natively recognized.

Verified: `uv run --env-file .env scripts/check_gpu.py` → `torch.cuda.is_available(): True`, device `AMD Radeon RX 7900 XTX — 23.9 GB VRAM`, ran a real 4096×4096 matmul on GPU successfully.

**Script gotcha (fixed)**: `scripts/install_rocm_wsl.sh` originally used `usermod -aG render,video "$(logname)"` to add the invoking user to the required groups — `logname` failed (`no login name`) when run via `sudo bash script.sh` in this WSL setup, which combined with `set -e` aborted the script before the group-add and `rocminfo` verification steps ran (apt/ROCm install itself had already completed by that point, so it wasn't wasted — just needed a rerun). Fixed to use `$SUDO_USER` instead, which `sudo` sets reliably.

## Python

- **Version manager**: pyenv 2.5.4 (was stale — last synced March 2025, so its list of installable versions stopped at 3.13.2/3.14.0a6; ran `git pull` in `~/.pyenv` to refresh)
- **System-wide default**: `pyenv global 3.13.15` — `python3` / `python3 -m pip` resolve to this outside the project.
- **The `gnn-experiment` uv project itself now runs on Python 3.14.7**, not 3.13 — see the uv section below for why. This only affects the project's own `.venv`; the pyenv global (3.13.15) is unchanged for anything outside this project.
- Python 3.15 is **not stable yet** as of 2026-08-10 (only alphas exist, `3.15.0a1`–`a5`+, on python.org) — not viable to build a real dependency stack on, even though some wheel builders (e.g. PyTorch) already publish forward-looking `cp315` wheels opportunistically.

### Known gotcha: bare `pip` does not follow pyenv
`~/.local/bin` is listed **before** `~/.pyenv/shims` in `$PATH`, and `~/.local/bin/pip` is a leftover pip tied to a **python3.10** install (dated June 2025). So:
- `python3 --version` / `python3 -m pip` → correctly use pyenv's 3.13.15 ✅
- bare `pip` / `pip3` → silently use the stale python3.10 pip ❌

**Always invoke `python3 -m pip ...` instead of bare `pip` in this environment**, or fix PATH ordering / remove `~/.local/bin/pip*` if you want bare `pip` to be trustworthy again.

## uv (2026-08-10)

- Was already installed at 0.11.7; upgraded via `uv self update` to **0.12.3**.
- Project is uv-native end to end: `uv init --package`, `uv python pin 3.13.15` (uv manages its own interpreter, separate from the pyenv one — it reused the pyenv 3.13.15 build since it's newer than anything in uv's own python-build-standalone registry at the time), `uv add ...` for every dependency, `.venv` created by `uv sync`. No manual `pip install` was used anywhere in this project.
- `requires-python` is pinned tight (`>=3.13,<3.14`), and `torch`/`triton-rocm` are sourced from an explicit `pytorch-rocm` index (`download.pytorch.org/whl/rocm7.2`) via `[tool.uv.sources]` in `pyproject.toml` — this is required because PyTorch's ROCm builds only exist for Linux/cp313, and a loose `requires-python` makes uv's universal resolver fail trying to satisfy other Python/OS combinations that don't have ROCm wheels.
- **uv quirk hit during setup**: `[tool.uv.sources]` overrides didn't apply to `triton-rocm` as long as it was only a *transitive* dependency of `torch` — uv kept resolving it against PyPI (where it tops out at `3.0.0rc1`) instead of the pinned ROCm index, even though the sources entry was present and correctly named. Fix: add `triton-rocm` as an explicit **direct** dependency (`uv add torch --index pytorch-rocm=...` followed by adding `triton-rocm` directly) — sources overrides reliably apply to direct/root dependencies. **Confirmed again** when adding vLLM: `torchvision` came in transitively (via vllm) from plain PyPI as a CUDA/generic build, silently mismatched against our ROCm `torch` build, and broke every `transformers` import (`RuntimeError: operator torchvision::nms does not exist`) — fixed the same way, by adding `torchvision` as an explicit direct dependency sourced from the `pytorch-rocm` index. **Rule of thumb for this project: any package sourced from a non-default index must be added as a direct dependency, even if it would otherwise only be pulled in transitively** — don't rely on `[tool.uv.sources]` alone for transitive deps.
- Also needed `[tool.uv] environments = ["sys_platform == 'linux'"]` — without it uv's universal resolver tries to satisfy win32/darwin too, which fails since ROCm wheels are Linux-only (and some custom package indexes return `403` instead of `404` for unknown projects on non-Linux forks, which uv treats as fatal rather than "not found here").

## LoRA/PEFT benchmark project (2026-08-10)

Project layout (all uv-managed):

- `src/gnn_experiment/peft_methods.py` — registry of PEFT configs: lora, dora, pissa, qlora, adalora, vera, ia3, prefix_tuning
- `src/gnn_experiment/bench.py` — loads a base model (default `Qwen/Qwen2.5-0.5B`), applies one PEFT method, runs a short training loop, reports wall time / peak VRAM / trainable-param % / final loss
- `scripts/run_benchmark.py` — CLI that loops over methods from `configs/benchmark.yaml`, writes `results/benchmark_results.csv`
- `scripts/check_gpu.py` — GPU smoke test (`uv run scripts/check_gpu.py`)
- `scripts/install_rocm_wsl.sh` — the sudo-gated system install, see GPU section above

Verified so far (no GPU yet, pending the ROCm install): `uv sync` resolves cleanly, all 8 PEFT method configs build without error (`adalora` needed a `total_step` fix, now wired to `max_steps`), `check_gpu.py` runs end to end and correctly reports no GPU. Not yet verified: an actual training run (needs ROCm installed + model/dataset download).

HF auth: `.env` (gitignored) holds `HF_TOKEN=` for gated models (e.g. Gemma) or higher HF Hub rate limits — not auto-loaded by plain `uv run`, since uv doesn't read `.env` from the working directory by default; use `uv run --env-file .env scripts/run_benchmark.py`, or `export HF_TOKEN=...` in the shell if you'd rather not pass the flag each time.

Known caveat: bitsandbytes (used for `qlora`) logged a warning that it couldn't find a precompiled `kernels-community` gemm binary for this exact torch 2.13/ROCm 7.2 combo and will fall back to a slower path — 4-bit quant works but may not be fully optimized on this stack yet. Worth rechecking after a bitsandbytes/kernels update.

## Python 3.13 → 3.14 migration + vLLM + chat (2026-08-10, same day)

User asked whether we could avoid being "stuck" on an old Python and use 3.14 (or 3.15) for everything, partly motivated by wanting vLLM for interactive chat with trained models — and AMD's dedicated consumer-GPU vLLM wheel (`vllm-rdna`) turns out to require Python **3.14**, not 3.13.

Verified empirically (not guessed) before switching: `torch==2.13.0+rocm7.2` (already installed), `triton-rocm==3.7.1`, and `pyarrow` all ship `cp314` wheels; `bitsandbytes` ships a version-agnostic (`py3-none-any`) wheel so Python version doesn't matter for it. On that basis, bumped the **whole project** to Python 3.14.7 (`requires-python = ">=3.14,<3.15"`, `uv python pin 3.14.7`) rather than keeping a second isolated env just for vLLM — `uv sync` resolved and installed all 94 packages cleanly on the first try, and the PEFT harness (`peft_methods.py` config builders, `check_gpu.py`) was re-verified working identically after the switch.

**vLLM install**: added a second explicit uv index, `vllm-rdna` (`https://rocm.frameworks.amd.com/whl-multi-arch/vllm-rdna`, `explicit = true`), sourcing `vllm` and `flash-attn` from it (both added as direct dependencies per the transitive-sources rule above). Required `[tool.uv] prerelease = "allow"` since the only available wheel is `vllm==0.23.1.dev1+rocm7.14.0.g9ddef7117.d20260715` (pre-release). Pulled in ~130 additional packages (full serving stack: FastAPI/uvicorn, opentelemetry, grpc, redis, cloud storage clients, etc.) — normal for vLLM, not a mistake.

**Two things flagged but not yet resolved (need a live GPU to check, blocked on the sudo ROCm install)**:

1. The only `vllm-rdna` wheel available (`0.23.1.dev1`) falls inside the version range (v0.21.0–v0.25.0) that AMD's own ROCm docs say has "significantly longer warmup times... on AMD Radeon GPUs" — their fix is "upgrade to v0.26.0+", which isn't available on this index yet.
2. Version-tag mismatch: our system ROCm install target is **7.2** (`scripts/install_rocm_wsl.sh`), but the vLLM wheel is tagged **rocm7.14.0** — unclear if these use compatible versioning schemes or if vLLM expects a different/newer system ROCm than what's about to be installed. Recheck after running `scripts/install_rocm_wsl.sh` and trying `vllm serve` for real.

**Chat**: per user's choice, built a plain `transformers`-based interactive terminal chat instead of standing up a vLLM server for now — `scripts/chat.py` (`uv run scripts/chat.py [--model ...] [--adapter path/to/trained/adapter]`), streams tokens via `TextStreamer`, keeps conversation history, optionally loads a PEFT adapter on top of the base model. vLLM is installed and available for a faster/serving-oriented chat path later (e.g. `vllm serve <model> --enable-lora`), once the two caveats above are checked against the real GPU.

## vLLM abandoned; llama.cpp + opencode eval harness built instead (2026-08-10, later same day)

**vLLM verdict: genuinely broken, not a config problem.** Chased it through four independent, compounding bugs — `vllm._C.abi3.so` ABI mismatch against pytorch.org's torch → switched to AMD's own exact matching `torch==2.12.0+rocm7.14.0` (long transitive-package whack-a-mole: `rocm`, `rocm-sdk-core/device-gfx1100/libraries`, `triton`, matched `torchvision`, all needed explicit `[tool.uv.sources]` entries, same pattern as the earlier `triton-rocm`/`torchvision` issue) → **still** ABI-mismatched (different symbol) → traced to AMD's bundled `_rocm_sdk_core` package only shipping a static `libhsakmt.a` (no runtime `.so`, built for native `/dev/kfd` which doesn't exist in WSL) → tried LD_PRELOADing the system's WSL-aware HSA runtime, but torch's own `_rocm_init` bypasses that and dlopens its own bundled libs directly, which have a **version-mismatched exported symbol** against each other in the same AMD package. That's AMD shipping internally-inconsistent binaries — not fixable from our side. User chose to try llama.cpp instead of digging further.

**llama.cpp: worked cleanly, first try, no fighting.** Built from source with HIP for gfx1100 (`serving/setup_llamacpp.sh`, ~15-20min on WSL's 4-core limit): `cmake -B build -G Ninja -DCMAKE_C_COMPILER=/opt/rocm-7.2.0/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm-7.2.0/llvm/bin/clang++ -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release`. `cmake`/`ninja` fetched via `uv tool install` (no sudo needed). Compiles against the *system* ROCm install directly — no AMD pip-wheel version-matching problems at all. `llama-server` (OpenAI-compatible endpoint) confirmed serving `Qwen/Qwen3.5-4B` (Q8_0 GGUF from `bartowski/Qwen_Qwen3.5-4B-GGUF`, ~4.6GB, base/unaltered model) at ~90 tok/s generation, ~1200 tok/s prompt processing on the RX 7900 XTX.

**opencode must be the WSL-native install, not the Windows one.** `opencode` on PATH originally resolved to `/mnt/c/Program Files/nodejs/opencode` — confirmed via testing that it runs **on the Windows side**, driving its `bash` tool through PowerShell against the WSL filesystem over `\\wsl.localhost\...` UNC paths. This cost a real eval run ~6x the time and tokens (136s/65.7K tokens vs 53s/10.9K tokens for the identical task) purely from the model fumbling `find`/`head`/`ls -la` failing under PowerShell before falling back to `Get-ChildItem`/`Get-Content`. Fixed with `npm install -g opencode-ai` using WSL's own node/npm (confirm `which opencode` resolves under `~/.nvm/...`) — despite the installed binary being literally named `opencode.exe`, it's a genuine ELF Linux binary (that's just the npm package's cross-platform naming convention). One other non-obvious thing: Windows→WSL `127.0.0.1` port forwarding worked by default in both cases (no WSL mirrored-networking config needed) — only the reverse direction (WSL→Windows, e.g. reaching Windows Ollama from WSL) failed earlier in this session.

**Eval harness**: `tests/opencode_evals/` — fully isolated from both `gnn-experiment` and `serving`'s own uv config (fixture `pyproject.toml`s are written via direct Python file I/O, never `uv init`, which is what caused uv to auto-register a nested project as a workspace member the one time it was tried — see project memory). Generates a fresh broken uv workspace (real `pydantic>=2.0` vs `pydantic<2.0` PubGrub conflict across workspace members) per run, drives opencode non-interactively (`opencode run <prompt> --dir <fixture> --model <provider>/<model> --auto --format json`), grades success by actually running `uv lock` in the fixture afterward (not by trusting the model's self-report), writes `runs/<timestamp>/{transcript.md,report.json}`.

**Token-accounting bug found and fixed**: opencode's per-step `tokens.total` field is the full context size at that point (including cached-read tokens), not an incremental cost — summing it across a multi-step run over-counts by ~10-16x (one run showed "1,024,650 tokens" that was actually 65,758). Correct total = `sum(step.tokens.input) + sum(step.tokens.output)` across steps, ignoring the pre-summed `total` field entirely.

**Verified end to end**: Qwen3.5-4B (unaltered base model) given the open-ended prompt "explain what is the problem with the dependencies, and fix it." — correctly diagnosed the `pydantic` version conflict, chose to upgrade the legacy constraint (not downgrade the modern one), applied a real file edit, and the fixture's `uv lock` genuinely exits 0 afterward. 52.7s, 10,945 tokens, on WSL-native opencode.
