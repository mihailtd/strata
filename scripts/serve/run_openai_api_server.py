import argparse
import os
import sys
from pathlib import Path

import uvicorn

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT / "apps"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--vram-cap", dest="vram_cap", type=float, default=22.0, help="Hard VRAM cap in GB")
    args = parser.parse_args()

    print(f"Launching IMB Zero-Copy OpenAI API Server on http://{args.host}:{args.port}...")
    uvicorn.run("runtime.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

