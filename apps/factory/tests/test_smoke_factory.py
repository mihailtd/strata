"""Smoke tests for apps/factory — quantization pipeline scripts.

Verifies Python syntax and public API surface of both factory scripts.
No GPU or model loading triggered.
"""

import ast
from pathlib import Path

FACTORY = Path(__file__).parent.parent


def test_leverage_quantization_exists() -> None:
    assert (FACTORY / "leverage_quantization.py").exists()


def test_ssi_quantization_exists() -> None:
    assert (FACTORY / "ssi_quantization.py").exists()


def test_leverage_quantization_syntax() -> None:
    ast.parse((FACTORY / "leverage_quantization.py").read_text())


def test_ssi_quantization_syntax() -> None:
    ast.parse((FACTORY / "ssi_quantization.py").read_text())


def _public_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and not node.name.startswith("_")
    ]


def test_leverage_quantization_has_public_api() -> None:
    names = _public_names(FACTORY / "leverage_quantization.py")
    assert len(names) > 0, "leverage_quantization.py must define at least one public class/function"


def test_ssi_quantization_has_public_api() -> None:
    names = _public_names(FACTORY / "ssi_quantization.py")
    assert len(names) > 0, "ssi_quantization.py must define at least one public class/function"


def test_no_hardcoded_fake_weights() -> None:
    """Zero-Mock invariant: factory scripts must not use torch.randn as fake weights."""
    import re
    for fname in ["leverage_quantization.py", "ssi_quantization.py"]:
        source = (FACTORY / fname).read_text()
        # torch.randn used as weight initialization is a red flag
        if "torch.randn" in source:
            # Allow if clearly inside a test/example comment block
            matches = re.findall(r"(?<!#\s)torch\.randn", source)
            assert not matches, (
                f"{fname}: torch.randn detected outside comments — "
                "factory scripts must use real pre-trained weights, not fake tensors "
                "(Zero-Mock invariant)"
            )
