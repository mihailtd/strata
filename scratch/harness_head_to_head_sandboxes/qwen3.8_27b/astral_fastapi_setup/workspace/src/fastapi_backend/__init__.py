"""Console-script package.

The service itself lives in :mod:`app`; this package only provides the
``fastapi-backend`` command (``fastapi_backend.main``) required by the
``[project.scripts]`` entry point in ``pyproject.toml``.
"""
from fastapi_backend.__main__ import main  # noqa: F401  (console-script target)

__all__ = ["main"]
