#!/usr/bin/env bash
# Profile the v0_naive kernel with Nsight Compute and extract key metrics.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REP="analysis/ncu/v0_rep"

ncu -k regex:gqa_decode_v0 \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --section WarpStateStats \
    --section LaunchStats \
    -f -o "$REP" \
    python bench/single_shape.py --ncu-mode

ncu --import "${REP}.ncu-rep" --csv > "${REP}_raw.csv"

python analysis/ncu/extract.py "${REP}_raw.csv" "${REP}.csv"
echo "wrote ${REP}.csv"
