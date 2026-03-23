#!/usr/bin/env bash
# Setup script: clones SWE-Lego into the correct location.
# Run this once after cloning auto-coder-trainer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_LEGO_DIR="${SCRIPT_DIR}/SWE-Lego"

if [ -d "${SWE_LEGO_DIR}" ]; then
    echo "[setup] SWE-Lego already present at ${SWE_LEGO_DIR}"
    exit 0
fi

echo "[setup] Cloning SWE-Lego..."
git clone --depth 1 https://github.com/SWE-Lego/SWE-Lego.git "${SWE_LEGO_DIR}"
echo "[setup] Done. SWE-Lego cloned to ${SWE_LEGO_DIR}"
