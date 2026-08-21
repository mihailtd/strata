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


def find_conflicting_processes() -> list[dict[str, Any]]:
    """Finds other Python processes running concurrently that might hold GPU state."""
    current_pid = os.getpid()
    parent_pid = os.getppid()
    conflicts = []
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
                    if pid in (current_pid, parent_pid) or ppid == current_pid:
                        continue
                    # Ignore internal IDE servers and system daemons
                    if any(ignored in cmd for ignored in ("antigravity-ide-server", "vscode-server", "subiquity", "pylsp", "pyright", "pytest")):
                        continue
                    # Check if it's a python command in the current workspace or running torch
                    if "python" in cmd and any(k in cmd for k in ("gnn", "train", "benchmark", "probe", "torch", "calibrate", "server")):
                        conflicts.append({
                            "pid": pid,
                            "cpu_pct": cpu,
                            "mem_pct": mem,
                            "cmd": cmd[:90] + ("..." if len(cmd) > 90 else ""),
                        })
    except Exception:
        pass
    return conflicts


def check_gpu_availability(max_occupied_gb: float | None = None) -> dict[str, Any]:
    """Evaluates whether the GPU is clean and available for a new workload.

    Default threshold is CANON.GPU_SAFETY_THRESHOLD_GB (6.0 GB) to accommodate
    Windows/WSL2 DWM desktop compositing overhead without false alarms.
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
