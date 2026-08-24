#!/usr/bin/env bash
# Profile a GQA decode kernel with Nsight Compute and extract key metrics.
# Usage: analysis/ncu/run_ncu.sh [KERNEL]   (KERNEL: v0 | v1, default v0)
set -euo pipefail

KERNEL="${1:-v0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REP="analysis/ncu/${KERNEL}_rep"

# MemoryWorkloadAnalysis_Tables is required for the per-request sector
# metrics (sectors/request); on NCU 2025.1 the base MemoryWorkloadAnalysis
# section no longer collects them.
ncu -k "regex:gqa_decode_${KERNEL}" \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables \
    --section Occupancy \
    --section WarpStateStats \
    --section LaunchStats \
    -f -o "$REP" \
    python bench/single_shape.py --ncu-mode --kernel "$KERNEL"

ncu --import "${REP}.ncu-rep" --page raw --csv > "${REP}_raw.csv"

python analysis/ncu/extract.py --kernel "$KERNEL"
echo "wrote ${REP}.csv"
