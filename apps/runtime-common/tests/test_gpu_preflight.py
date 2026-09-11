"""Unit tests for GPU pre-flight guard and exclusivity checks (gpu_preflight.py)."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from runtime_common import gpu_preflight


def test_get_gpu_vram_info_cpu_fallback():
    # runtime-common deliberately has no torch dependency (gpu_preflight does a
    # lazy `import torch` inside the function body) -- mock the module in
    # sys.modules rather than `patch("torch.cuda.is_available", ...)`, which
    # would require torch to actually be importable here.
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": fake_torch}):
        info = gpu_preflight.get_gpu_vram_info()
        assert info["cuda_available"] is False
        assert info["used_gb"] == 0.0


def test_check_gpu_availability_clean_with_desktop_compositor():
    # Baseline desktop compositing typically occupies ~2-4 GB
    mock_vram = {
        "cuda_available": True,
        "used_gb": 4.2,
        "free_gb": 19.8,
        "total_gb": 24.0,
        "device_name": "AMD Radeon RX 7900 XTX",
    }
    with patch("runtime_common.gpu_preflight.get_gpu_vram_info", return_value=mock_vram):
        with patch("runtime_common.gpu_preflight.find_conflicting_processes", return_value=[]):
            status = gpu_preflight.check_gpu_availability(max_occupied_gb=6.0)
            assert status["is_clean"] is True
            assert status["is_occupied"] is False
            assert status["used_gb"] == 4.2


def test_check_gpu_availability_occupied():
    # Active model in VRAM occupies 14.5 GB (exceeding 6.0 GB)
    mock_vram = {
        "cuda_available": True,
        "used_gb": 14.5,
        "free_gb": 9.5,
        "total_gb": 24.0,
        "device_name": "AMD Radeon RX 7900 XTX",
    }
    mock_conflicts = [{"pid": 12345, "cpu_pct": "50.0", "mem_pct": "10.0", "cmd": "python scripts/train/train_expert.py"}]
    with patch("runtime_common.gpu_preflight.get_gpu_vram_info", return_value=mock_vram):
        with patch("runtime_common.gpu_preflight.find_conflicting_processes", return_value=mock_conflicts):
            status = gpu_preflight.check_gpu_availability(max_occupied_gb=6.0)
            assert status["is_clean"] is False
            assert status["is_occupied"] is True
            assert status["used_gb"] == 14.5
            assert len(status["conflicts"]) == 1


def test_ensure_gpu_exclusive_warning_mode():
    mock_vram = {
        "cuda_available": True,
        "used_gb": 14.5,
        "free_gb": 9.5,
        "total_gb": 24.0,
        "device_name": "AMD Radeon RX 7900 XTX",
    }
    with patch("runtime_common.gpu_preflight.get_gpu_vram_info", return_value=mock_vram):
        with patch("runtime_common.gpu_preflight.find_conflicting_processes", return_value=[]):
            result = gpu_preflight.ensure_gpu_exclusive(max_occupied_gb=6.0, exit_on_conflict=False)
            assert result is False


def test_ensure_gpu_exclusive_exit_mode():
    mock_vram = {
        "cuda_available": True,
        "used_gb": 14.5,
        "free_gb": 9.5,
        "total_gb": 24.0,
        "device_name": "AMD Radeon RX 7900 XTX",
    }
    with patch("runtime_common.gpu_preflight.get_gpu_vram_info", return_value=mock_vram):
        with patch("runtime_common.gpu_preflight.find_conflicting_processes", return_value=[]):
            with pytest.raises(SystemExit) as exc_info:
                gpu_preflight.ensure_gpu_exclusive(max_occupied_gb=6.0, exit_on_conflict=True)
            assert exc_info.value.code == 1
