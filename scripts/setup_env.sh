#!/usr/bin/env bash
# Idempotent environment setup for a fresh Lambda H100 instance (Lambda Stack 24.04).
#
# Two-phase design:
#   Phase 1 writes the NCU profiling modprobe config. If it had to be (re)written,
#   a reboot is required before profiling works; the script still runs the install
#   steps below (they do not depend on the reboot) and prints REBOOT REQUIRED at
#   the end. It never reboots automatically.
#   Phase 2 (after reboot, or immediately if the config was already correct)
#   is validated by scripts/verify_env.sh.
#
# Versions and commands follow docs/g0/ (versions.txt, flashmla_build.log).
#
# Usage: scripts/setup_env.sh [--with-flashmla]

set -euo pipefail

WITH_FLASHMLA=0
for arg in "$@"; do
    case "$arg" in
        --with-flashmla) WITH_FLASHMLA=1 ;;
        *) echo "[setup] unknown argument: $arg" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NEED_REBOOT=0

# --- Step 1: NCU profiling permission (modprobe config) -----------------------
NCU_CONF=/etc/modprobe.d/nvidia-profiling.conf
NCU_CONF_CONTENT="options nvidia NVreg_RestrictProfilingToAdminUsers=0"

if [ -f "$NCU_CONF" ] && [ "$(cat "$NCU_CONF")" = "$NCU_CONF_CONTENT" ]; then
    echo "[setup] $NCU_CONF already correct, skipping"
else
    echo "[setup] writing $NCU_CONF"
    echo "$NCU_CONF_CONTENT" | sudo tee "$NCU_CONF" > /dev/null
    NEED_REBOOT=1
fi

# --- Step 2: apt packages (nsight-compute, pybind11-dev) ----------------------
# nsight-compute from the Lambda repo (2025.1.1.x) matches the ncu version
# recorded in docs/g0/versions.txt.
if dpkg -s nsight-compute pybind11-dev > /dev/null 2>&1; then
    echo "[setup] nsight-compute and pybind11-dev already installed, skipping"
else
    echo "[setup] apt update"
    sudo apt-get update -qq
    echo "[setup] installing nsight-compute and pybind11-dev"
    sudo apt-get install -y -qq nsight-compute pybind11-dev
fi

# --- Step 3: flashinfer -------------------------------------------------------
# docs/g0/flashinfer_smoke_raw.txt records import name "flashinfer" version
# 0.6.17. The PyPI distribution is "flashinfer-python" (verified on PyPI:
# 0.6.17 exists there; a bare "flashinfer" distribution does not exist).
FLASHINFER_VERSION=0.6.17
if python3 - <<EOF
import sys
try:
    import flashinfer
except Exception:
    sys.exit(1)
sys.exit(0 if flashinfer.__version__ == "$FLASHINFER_VERSION" else 1)
EOF
then
    echo "[setup] flashinfer $FLASHINFER_VERSION already installed, skipping"
else
    echo "[setup] pip installing flashinfer-python==$FLASHINFER_VERSION"
    python3 -m pip install "flashinfer-python==$FLASHINFER_VERSION"
fi

# --- Step 4 (optional): FlashMLA ----------------------------------------------
if [ "$WITH_FLASHMLA" = 1 ]; then
    FLASHMLA_DIR="$HOME/FlashMLA"
    FLASHMLA_SHA="$(cat "$REPO_ROOT/docs/g0/flashmla_sha.txt")"

    if [ -d "$FLASHMLA_DIR/.git" ]; then
        echo "[setup] $FLASHMLA_DIR already cloned, skipping clone"
    else
        echo "[setup] cloning FlashMLA to $FLASHMLA_DIR"
        git clone https://github.com/deepseek-ai/FlashMLA.git "$FLASHMLA_DIR"
    fi

    echo "[setup] checking out FlashMLA $FLASHMLA_SHA"
    git -C "$FLASHMLA_DIR" checkout --quiet "$FLASHMLA_SHA"
    git -C "$FLASHMLA_DIR" submodule update --init --recursive

    if python3 -c "import flash_mla" > /dev/null 2>&1; then
        echo "[setup] flash_mla already importable, skipping build"
    else
        # Editable pip install with MAX_JOBS=24, as recorded in
        # docs/g0/flashmla_build.log; SM100 disabled per project decision.
        echo "[setup] building FlashMLA (editable install, sm90 only)"
        (cd "$FLASHMLA_DIR" && \
            FLASH_MLA_DISABLE_SM100=1 MAX_JOBS=24 python3 -m pip install -v -e .)
    fi
fi

# --- Done ---------------------------------------------------------------------
if [ "$NEED_REBOOT" = 1 ]; then
    echo "[setup] REBOOT REQUIRED: sudo reboot, then run scripts/verify_env.sh"
    exit 0
fi
echo "[setup] done, run scripts/verify_env.sh"
