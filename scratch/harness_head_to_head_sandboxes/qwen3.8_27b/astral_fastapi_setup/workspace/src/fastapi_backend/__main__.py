"""CLI entrypoint: ``python -m fastapi_backend`` (or the ``fastapi-backend`` script).

Thin wrapper that runs the ASGI app (``app.main:app``) under uvicorn so the
package name stays decoupled from the application package.
"""
from __future__ import annotations

import argparse
import sys

import uvicorn

from app.config import get_settings


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    debug = get_settings().debug
    parser = argparse.ArgumentParser(
        prog="fastapi-backend",
        description="Run the FastAPI backend with uvicorn.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0" if debug else "127.0.0.1",
        help="Bind host (default: 127.0.0.1, or 0.0.0.0 when APP_DEBUG=true)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--reload", action="store_true", help="Auto-reload on code changes (development only)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sys.exit(
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="debug" if get_settings().debug else "info",
        )
    )


if __name__ == "__main__":
    main()
