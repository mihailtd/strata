#!/usr/bin/env bash
# One-time, system-wide ROCm-for-WSL install for the RX 7900 XTX (gfx1100).
# Must be run with sudo — Claude cannot run this itself (no passwordless sudo
# in this environment), so run it yourself:
#
#   sudo bash scripts/install_rocm_wsl.sh
#
# Prerequisite (do this on the Windows side first, not in WSL):
#   Install "AMD Software: Adrenalin Edition 26.1.1 for WSL2" (or newer) from
#   AMD's site, then restart Windows. Check your installed version in the
#   AMD Software app before running this script — if it's older than 26.1.1,
#   update it first or this install may not find a working GPU.
#
# Source: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/wsl/install-radeon.html

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "Run this with sudo: sudo bash $0" >&2
  exit 1
fi

ROCM_VERSION="7.2"
DEB_URL="https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/jammy/amdgpu-install_7.2.70200-1_all.deb"
DEB_FILE="/tmp/amdgpu-install_7.2.70200-1_all.deb"

echo "== Updating apt =="
apt update

echo "== Downloading amdgpu-install (ROCm ${ROCM_VERSION}) =="
wget -O "${DEB_FILE}" "${DEB_URL}"

echo "== Installing amdgpu-install package =="
apt install -y "${DEB_FILE}"

echo "== Installing ROCm + graphics userspace for WSL (--usecase=wsl,rocm --no-dkms) =="
amdgpu-install -y --usecase=wsl,rocm --no-dkms

TARGET_USER="${SUDO_USER:-}"
if [[ -z "${TARGET_USER}" ]]; then
  echo "WARNING: could not determine the invoking user (\$SUDO_USER is unset)." >&2
  echo "Add yourself to the render/video groups manually: sudo usermod -aG render,video <your-username>" >&2
else
  echo "== Adding ${TARGET_USER} to render/video groups (harmless if already a member) =="
  usermod -aG render,video "${TARGET_USER}"
fi

echo
echo "== Verifying: rocminfo should list a gfx1100 agent (RX 7900 XTX) =="
rocminfo | grep -A 5 "gfx1100" || echo "WARNING: gfx1100 not found in rocminfo output — check the log above for errors."

cat <<'EOF'

Done. Close and reopen your WSL shell (or run `newgrp render`) so the
render/video group membership takes effect, then verify from the project dir:

  uv run --env-file .env scripts/audit/check_gpu.py

Note the --env-file .env: torch's pip wheel bundles its own libhsa-runtime64.so
that doesn't know about WSL and will report no GPU without it. .env sets
LD_PRELOAD to force use of this system ROCm install's WSL-aware runtime
instead. See SYSTEM.md for details.

EOF
