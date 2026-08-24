"""GPU Pre-Flight Guard & Exclusivity Verification Module.

Prevents multi-process GPU collisions on single-GPU hardware (e.g. AMD Radeon RX 7900 XTX 24GB).
Scans VRAM memory and active processes before any GPU allocation occurs, emitting loud
warnings and safely halting to prevent OOM, ROCm driver crashes, or host BIOS restarts.
See DECISIONS.md §55.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from runtime.canon import CANON


def get_sysfs_vram_info() -> dict[str, Any] | None:
    """Reads raw hardware VRAM metrics directly from amdgpu sysfs (independent of PyTorch)."""
    try:
        from pathlib import Path
        # Search for primary discrete card (e.g. card0 / RX 7900 XTX)
        for card_dev in sorted(Path("/sys/class/drm").glob("card*/device")):
            used_path = card_dev / "mem_info_vram_used"
            total_path = card_dev / "mem_info_vram_total"
            if used_path.exists() and total_path.exists():
                total_bytes = int(total_path.read_text().strip())
                used_bytes = int(used_path.read_text().strip())
                total_gb = total_bytes / (1024**3)
                if total_gb >= 8.0:  # Identify discrete GPU with >= 8GB VRAM
                    used_gb = used_bytes / (1024**3)
                    free_gb = max(0.0, total_gb - used_gb)
                    return {
                        "sysfs_available": True,
                        "used_gb": round(used_gb, 2),
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(total_gb, 2),
                        "device_name": "AMD Discrete GPU (sysfs)",
                    }
    except Exception:
        pass
    return None


def get_gpu_vram_info() -> dict[str, Any]:
    """Queries hardware VRAM status via PyTorch/ROCm."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "cuda_available": False,
                "used_gb": 0.0,
                "free_gb": 0.0,
                "total_gb": 0.0,
                "device_name": "None",
            }

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        total_gb = total_bytes / (1024**3)
        free_gb = free_bytes / (1024**3)
        used_gb = (total_bytes - free_bytes) / (1024**3)
        device_name = torch.cuda.get_device_name(0)

        return {
            "cuda_available": True,
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "total_gb": round(total_gb, 2),
            "device_name": device_name,
        }
    except Exception as ex:
        return {
            "cuda_available": False,
            "error": str(ex),
            "used_gb": 0.0,
            "free_gb": 0.0,
            "total_gb": 0.0,
            "device_name": "Unknown",
        }


def _get_kfd_compute_pids() -> set[int]:
    """Finds PIDs holding the native Linux /dev/kfd device node (ROCm compute clients)."""
    pids = set()
    try:
        res = subprocess.run(
            ["lsof", "-t", "/dev/kfd"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            for p in res.stdout.strip().split():
                if p.isdigit():
                    pids.add(int(p))
    except Exception:
        pass
    return pids


def find_conflicting_processes() -> list[dict[str, Any]]:
    """Finds other Python/compute processes running concurrently that hold GPU state."""
    current_pid = os.getpid()
    parent_pid = os.getppid()
    conflicts = []
    seen_pids = set()
    kfd_pids = _get_kfd_compute_pids()

    try:
        res = subprocess.run(
            ["ps", "-eo", "pid,ppid,pcpu,pmem,args"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2.0,
        )
        if res.returncode == 0:
            for line in res.stdout.strip().split("\n")[1:]:
                parts = line.strip().split(None, 4)
                if len(parts) >= 5:
                    pid_str, ppid_str, cpu, mem, cmd = parts[0], parts[1], parts[2], parts[3], parts[4]
                    try:
                        pid = int(pid_str)
                        ppid = int(ppid_str)
                    except ValueError:
                        continue
                    if pid in (current_pid, parent_pid) or ppid == current_pid or pid in seen_pids:
                        continue
                    # Ignore internal IDE servers, pytest runner, language servers, and system daemons
                    if any(ignored in cmd for ignored in (
                        "antigravity-ide-server", "vscode-server", "subiquity",
                        "pylsp", "pyright", "pytest", "krunner", "plasmashell",
                        "plasma-systemmonitor", "Xwayland", "electron"
                    )):
                        continue

                    # Direct hardware match: process holds open /dev/kfd compute handle
                    is_kfd_holder = pid in kfd_pids

                    # Workload keyword match: python training/serving/eval scripts or standalone LLM servers
                    is_workload = (
                        ("python" in cmd and any(k in cmd for k in ("gnn", "train", "benchmark", "probe", "torch", "calibrate", "server")))
                        or any(server_bin in cmd for server_bin in ("llama-server", "vllm", "unsloth"))
                    )

                    if is_kfd_holder or is_workload:
                        seen_pids.add(pid)
                        conflicts.append({
                            "pid": pid,
                            "cpu_pct": cpu,
                            "mem_pct": mem,
                            "cmd": cmd[:90] + ("..." if len(cmd) > 90 else ""),
                            "is_kfd_holder": is_kfd_holder,
                        })
    except Exception:
        pass
    return conflicts


def check_gpu_availability(max_occupied_gb: float | None = None) -> dict[str, Any]:
    """Evaluates whether the GPU is clean and available for a new workload.

    Default threshold is CANON.GPU_SAFETY_THRESHOLD_GB (6.0 GB) to accommodate
    desktop compositing baseline and system headroom without false alarms.
    """
    threshold = CANON.GPU_SAFETY_THRESHOLD_GB if max_occupied_gb is None else max_occupied_gb
    vram = get_gpu_vram_info()
    conflicts = find_conflicting_processes()

    if not vram.get("cuda_available", False):
        return {
            "is_clean": True,
            "reason": "GPU not visible (CPU mode)",
            "vram": vram,
            "conflicts": conflicts,
        }

    used_gb = vram.get("used_gb", 0.0)
    is_occupied = used_gb > threshold
    has_conflicts = len(conflicts) > 0 and is_occupied

    return {
        "is_clean": not is_occupied,
        "is_occupied": is_occupied,
        "used_gb": used_gb,
        "total_gb": vram.get("total_gb", 24.0),
        "device_name": vram.get("device_name", "AMD GPU"),
        "conflicts": conflicts,
    }


def ensure_gpu_exclusive(
    max_occupied_gb: float | None = None,
    exit_on_conflict: bool = True,
    caller_name: str | None = None,
) -> bool:
    """Pre-flight check: ensures no other process is holding GPU memory.

    Plug-and-play zero-argument call: ensure_gpu_exclusive()
    Uses global CANON.GPU_SAFETY_THRESHOLD_GB (6.0 GB) by default.
    """
    threshold = CANON.GPU_SAFETY_THRESHOLD_GB if max_occupied_gb is None else max_occupied_gb
    status = check_gpu_availability(max_occupied_gb=threshold)
    if status.get("is_clean", True):
        return True

    used_gb = status.get("used_gb", 0.0)
    total_gb = status.get("total_gb", 24.0)
    dev_name = status.get("device_name", "AMD GPU")
    conflicts = status.get("conflicts", [])
    script_desc = caller_name or (os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "GPU Workload")

    conflict_lines = ""
    if conflicts:
        for c in conflicts:
            conflict_lines += f"║   • PID {c['pid']:<6} (CPU: {c['cpu_pct']}%, MEM: {c['mem_pct']}%): {c['cmd']:<40} ║\n"
    else:
        conflict_lines = "║   • Another process or background worker is currently holding VRAM.     ║\n"

    alert_box = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ ⚠️  GPU PRE-FLIGHT GUARD: VRAM CONFLICT DETECTED                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Hardware Target: {dev_name:<59} ║
║ Occupied VRAM  : {used_gb:>5.2f} GB / {total_gb:.2f} GB (Exceeds {threshold:.1f} GB safety threshold)   ║
║ Target Task    : {script_desc:<59} ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ ACTIVE CONFLICTING PROCESSES:                                                ║
{conflict_lines}╠══════════════════════════════════════════════════════════════════════════════╣
║ 🛑 DANGER: Concurrent GPU allocation causes OOM or Host Crash (DECISIONS §55)║
║                                                                              ║
║ {'ABORTING EXECUTION: Terminate active PID before starting this workload.' if exit_on_conflict else 'WARNING ONLY: Proceeding with concurrent allocation risks crash.'}   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(alert_box, file=sys.stderr, flush=True)

    if exit_on_conflict:
        sys.exit(1)

    return False
