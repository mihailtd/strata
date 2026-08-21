"""GPU smoke test. Run after installing ROCm (scripts/install_rocm_wsl.sh):

    uv run --env-file .env scripts/audit/check_gpu.py

Needs --env-file .env: torch's pip wheel bundles its own libhsa-runtime64.so
that doesn't know about WSL, so it can't enumerate the GPU on its own. .env
sets LD_PRELOAD to force the system ROCm install's WSL-aware runtime
(from the hsa-runtime-rocr4wsl-amdgpu package) instead. See SYSTEM.md.
"""

import torch


def main() -> None:
    print(f"torch {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print(
            "\nGPU not visible to torch. Likely causes:\n"
            "  - ROCm not installed yet: sudo bash scripts/install_rocm_wsl.sh\n"
            "  - Installed but this shell predates the render/video group change "
            "(open a new shell)\n"
            "  - Windows AMD driver is older than Adrenalin 26.1.1 for WSL2"
        )
        return

    device_count = torch.cuda.device_count()
    print(f"device count: {device_count}")
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024**3)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    used_gb = (total_bytes - free_bytes) / (1024**3)
    print(f"  [0] {name} — {total_gb:.1f} GB Total | Currently Used: {used_gb:.2f} GB | Free: {free_bytes/(1024**3):.2f} GB")

    from runtime.gpu_preflight import find_conflicting_processes
    conflicts = find_conflicting_processes()
    if conflicts:
        print(f"\n⚠️  ACTIVE GPU / PYTHON WORKLOADS DETECTED ({len(conflicts)}):")
        for c in conflicts:
            print(f"   • PID {c['pid']:<6} (CPU: {c['cpu_pct']}%, MEM: {c['mem_pct']}%): {c['cmd']}")
    else:
        print("\n✅ No conflicting GPU processes found. GPU is clean and ready.")

    print("\nRunning a matmul on the GPU...")
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print(f"result checksum: {c.sum().item():.2f}")
    print(f"peak VRAM used: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    print("\nGPU compute works.")


if __name__ == "__main__":
    main()
