import sys
from pathlib import Path

# apps/runtime-ipwf has no src/ layout and isn't pip-installed (package = false
# in pyproject.toml) -- its vendored modules (novel_peft.py, fused_norm.py,
# etc.) are plain top-level files imported as `from novel_peft import ...`
# by server.py itself. Put the project root on sys.path so tests can do the same.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
