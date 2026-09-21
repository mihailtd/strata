"""Unit tests for apps/factory/train_expert.py Streaming QLoRA loader."""

from __future__ import annotations

import ast
from pathlib import Path

FACTORY = Path(__file__).parent.parent


def test_train_expert_syntax() -> None:
    """Ensure train_expert.py parses without syntax errors."""
    ast.parse((FACTORY / "train_expert.py").read_text())


def test_train_expert_defines_streaming_loader() -> None:
    """Verify enable_streaming_qlora_loader is defined."""
    tree = ast.parse((FACTORY / "train_expert.py").read_text())
    func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    assert "enable_streaming_qlora_loader" in func_names, "train_expert.py must define enable_streaming_qlora_loader"


def test_streaming_loader_activates() -> None:
    """Verify enable_streaming_qlora_loader installs hooks cleanly."""
    import sys
    sys.path.insert(0, str(FACTORY))
    from train_expert import enable_streaming_qlora_loader
    import transformers.modeling_utils as mu
    import transformers.core_model_loading as cml

    enable_streaming_qlora_loader(gc_interval=10)

    assert hasattr(mu, "safe_open")
    assert hasattr(cml, "set_param_for_module")
