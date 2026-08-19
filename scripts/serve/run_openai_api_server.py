import argparse
import os
import sys
from pathlib import Path

# Automatically ensure ROCm HSA runtime is preloaded for AMD Radeon RX 7900 XTX
rocm_hsa_lib = "/opt/rocm-7.2.0/lib/libhsa-runtime64.so"
if os.path.exists(rocm_hsa_lib) and rocm_hsa_lib not in os.environ.get("LD_PRELOAD", ""):
    current_preload = os.environ.get("LD_PRELOAD", "")
    os.environ["LD_PRELOAD"] = f"{rocm_hsa_lib}:{current_preload}".strip(":")
    # Re-exec process with updated environment so the dynamic linker loads libhsa
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

import uvicorn

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--vram-cap", dest="vram_cap", type=float, default=22.0, help="Hard VRAM cap in GB")
    args = parser.parse_args()

    print(f"Launching IMB Zero-Copy OpenAI API Server on http://{args.host}:{args.port}...")
    uvicorn.run("gnn_experiment.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

