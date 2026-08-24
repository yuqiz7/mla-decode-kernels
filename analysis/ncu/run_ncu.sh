#!/usr/bin/env bash
# Profile a GQA decode kernel with Nsight Compute and extract key metrics.
#
# Usage:
#   analysis/ncu/run_ncu.sh [KERNEL]                 (legacy form, KERNEL: v0|v1|v2)
#   analysis/ncu/run_ncu.sh --kernel v2 --B 32 --S 8192 --Hq 32 --Hkv 8 \
#       --bs 16 --num-splits 32 --out cellA_rep
#
# --out names the report/CSV files under analysis/ncu/ (default <kernel>_rep).
set -euo pipefail

KERNEL="v0"
OUT=""
BENCH_ARGS=()
if [[ $# -gt 0 && "$1" != --* ]]; then
    KERNEL="$1"; shift
fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel)     KERNEL="$2"; shift 2 ;;
        --out)        OUT="$2"; shift 2 ;;
        --B|--S|--Hq|--Hkv|--bs|--num-splits)
                      BENCH_ARGS+=("$1" "$2"); shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$OUT" ]] && OUT="${KERNEL}_rep"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REP="analysis/ncu/${OUT}"

# MemoryWorkloadAnalysis_Tables alone does not cover the per-request sector
# metric on NCU 2025.1, and the absolute byte counters are not in any of the
# sections, so they are requested explicitly via --metrics.
ncu -k "regex:gqa_decode_${KERNEL}" \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables \
    --section Occupancy \
    --section WarpStateStats \
    --section LaunchStats \
    --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld,dram__bytes,lts__t_bytes,lts__t_sector_hit_rate \
    -f -o "$REP" \
    python bench/single_shape.py --ncu-mode --kernel "$KERNEL" "${BENCH_ARGS[@]}"

ncu --import "${REP}.ncu-rep" --page raw --csv > "${REP}_raw.csv"

python analysis/ncu/extract.py "${REP}_raw.csv" --kernel "$KERNEL" -o "${REP}.csv"
echo "wrote ${REP}.csv"
