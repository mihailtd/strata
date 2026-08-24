"""GPU smoke test & pre-flight guard audit for native CachyOS Linux.

Usage:
    uv run python scripts/audit/check_gpu.py

Validates ROCm / PyTorch hardware detection on AMD Radeon RX 7900 XTX (gfx1100),
verifies /dev/kfd compute device availability, scans for conflicting processes,
and executes a real 4096×4096 GPU matmul benchmark.
"""

from __future__ import annotations

import sys
import torch


def main() -> None:
    print(f"PyTorch Version: {torch.__version__}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print(
            "\n❌ GPU not visible to PyTorch. Likely causes on CachyOS:\n"
            "  - ROCm runtime not installed: check /opt/rocm and rocminfo\n"
            "  - User not in 'render' or 'video' groups: run 'groups'\n"
            "  - Missing /dev/kfd node or amdgpu driver not loaded"
        )
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f"Detected Compute Devices: {device_count}")
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        used_gb = (total_bytes - free_bytes) / (1024**3)
        print(f"  [{i}] {name} — {total_gb:.1f} GB Total | Currently Used: {used_gb:.2f} GB | Free: {free_bytes/(1024**3):.2f} GB")

    from runtime.gpu_preflight import find_conflicting_processes, get_gpu_vram_info
    vram_status = get_gpu_vram_info()
    conflicts = find_conflicting_processes()

    if conflicts:
        print(f"\n⚠️  ACTIVE GPU / COMPUTE WORKLOADS DETECTED ({len(conflicts)}):")
        for c in conflicts:
            kfd_tag = " [KFD Client]" if c.get("is_kfd_holder") else ""
            print(f"   • PID {c['pid']:<6} (CPU: {c['cpu_pct']}%, MEM: {c['mem_pct']}%){kfd_tag}: {c['cmd']}")
    else:
        print("\n✅ No conflicting GPU processes found. GPU is clean and ready for exclusive workloads.")

    print("\nRunning test 4096×4096 BF16 matmul on GPU...")
    torch.cuda.empty_cache()
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize()
    print(f"Result checksum: {c.float().sum().item():.2f}")
    print(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    print("\n✅ GPU compute verified successfully on native CachyOS.")


if __name__ == "__main__":
    main()
