#!/usr/bin/env bash
# Setup script: clones SWE-Lego and installs LLaMA-Factory.
# Run this once after cloning auto-coder-trainer.
#
# Usage:
#   bash setup_swe_lego.sh                    # default (LLaMA-Factory 0.9.4.dev0)
#   bash setup_swe_lego.sh --qwen3.5          # use LLaMA-Factory 0.9.5+ for Qwen3.5
#   LLAMA_FACTORY_VERSION=0.9.5 bash setup_swe_lego.sh  # explicit version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWE_LEGO_DIR="${SCRIPT_DIR}/SWE-Lego"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Parse flags ---
QWEN35_MODE=false
for arg in "$@"; do
    case "$arg" in
        --qwen3.5|--qwen35) QWEN35_MODE=true ;;
    esac
done

# --- Resolve LLaMA-Factory version ---
if [ "$QWEN35_MODE" = true ]; then
    LLAMA_FACTORY_VERSION="${LLAMA_FACTORY_VERSION:-0.9.5}"
else
    LLAMA_FACTORY_VERSION="${LLAMA_FACTORY_VERSION:-0.9.4.dev0}"
fi

echo "[setup] Target LLaMA-Factory version: ${LLAMA_FACTORY_VERSION}"

# --- Step 1: Clone SWE-Lego ---
if [ -d "${SWE_LEGO_DIR}" ]; then
    echo "[setup] SWE-Lego already present at ${SWE_LEGO_DIR}"
else
    echo "[setup] Cloning SWE-Lego..."
    git clone --depth 1 https://github.com/SWE-Lego/SWE-Lego.git "${SWE_LEGO_DIR}"
    echo "[setup] SWE-Lego cloned to ${SWE_LEGO_DIR}"
fi

# --- Step 2: Install LLaMA-Factory ---
LLAMA_FACTORY_DIR="${SWE_LEGO_DIR}/LLaMA-Factory-${LLAMA_FACTORY_VERSION}"
if [ -d "${LLAMA_FACTORY_DIR}" ]; then
    echo "[setup] LLaMA-Factory ${LLAMA_FACTORY_VERSION} found at ${LLAMA_FACTORY_DIR}"
    echo "[setup] Installing in editable mode..."
    pip install -e "${LLAMA_FACTORY_DIR}" 2>&1 | tail -5
else
    echo "[setup] WARNING: LLaMA-Factory-${LLAMA_FACTORY_VERSION} not found in SWE-Lego."
    echo "[setup] Expected path: ${LLAMA_FACTORY_DIR}"
    if [ "$QWEN35_MODE" = true ]; then
        echo "[setup] For Qwen3.5, you may need to clone LLaMA-Factory main branch:"
        echo "[setup]   git clone https://github.com/hiyouga/LLaMA-Factory.git ${LLAMA_FACTORY_DIR}"
        echo "[setup]   pip install -e ${LLAMA_FACTORY_DIR}"
    fi
fi

# --- Step 3: Install Python dependencies ---
if [ "$QWEN35_MODE" = true ]; then
    echo "[setup] Installing Qwen3.5 dependencies..."
    pip install -e "${REPO_ROOT}[swe-lego-qwen35]" 2>&1 | tail -5
    echo ""
    echo "[setup] === Qwen3.5 dependency checklist ==="
    echo "[setup] Required versions for Qwen3.5-9B:"
    echo "[setup]   transformers >= 4.52.0  (for Qwen3_5ForCausalLM)"
    echo "[setup]   vllm         >= 0.17.0  (for Qwen3.5 architecture support)"
    echo "[setup]   LLaMA-Factory >= 0.9.5  (for qwen3 template + Qwen3.5 model class)"
    echo ""
    echo "[setup] Checking installed versions..."
    python -c "
import importlib, sys
checks = [
    ('transformers', '4.52.0'),
    ('vllm', '0.17.0'),
]
all_ok = True
for pkg, min_ver in checks:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        ok = ver >= min_ver
        status = 'OK' if ok else 'NEEDS UPGRADE'
        if not ok:
            all_ok = False
        print(f'  {pkg:20s} {ver:15s} (need >= {min_ver}) [{status}]')
    except ImportError:
        all_ok = False
        print(f'  {pkg:20s} NOT INSTALLED   (need >= {min_ver}) [MISSING]')
if not all_ok:
    print()
    print('  WARNING: Some dependencies need attention. See above.')
    sys.exit(1)
else:
    print()
    print('  All Qwen3.5 dependencies satisfied.')
" 2>&1 || true
else
    echo "[setup] Installing base SWE-Lego dependencies..."
    pip install -e "${REPO_ROOT}[swe-lego]" 2>&1 | tail -5
fi

# --- Step 4: Export env vars ---
echo ""
echo "[setup] === Environment variables ==="
echo "[setup] Add these to your shell or env.sh:"
echo ""
echo "  export LLAMA_FACTORY_DIR=${LLAMA_FACTORY_DIR}"
echo "  export ACT_SWE_LEGO_ROOT=${SWE_LEGO_DIR}"
echo ""
echo "[setup] Done."
