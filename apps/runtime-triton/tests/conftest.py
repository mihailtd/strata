import sys
from pathlib import Path

# apps/runtime-triton has no src/ layout and isn't pip-installed (package = false
# in pyproject.toml) -- its vendored modules (native_27b_engine.py, triton_w4a16.py,
# etc.) are plain top-level files imported as `from native_27b_engine import ...`
# by server.py itself. Put the project root on sys.path so tests can do the same.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
