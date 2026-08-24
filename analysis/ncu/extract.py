"""Extract key metrics from an Nsight Compute report into a one-line CSV.

Usage: python analysis/ncu/extract.py [inputs ...] [--kernel v0] [-o OUTPUT]

Inputs may be .ncu-rep files (exported here via `ncu --import --page raw
--csv`) or pre-exported raw-page CSVs. With no inputs, defaults to
analysis/ncu/<kernel>_rep.ncu-rep, plus analysis/ncu/<kernel>_rep_tables.ncu-rep
if it exists (supplemental run holding MemoryWorkloadAnalysis_Tables metrics
when the main report lacks them). Later inputs only fill metrics missing from
earlier ones. Default output: analysis/ncu/<kernel>_rep.csv.

Raw-page metric names used (NCU 2025.1.1, H100):
  gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
  sm__warps_active.avg.pct_of_peak_sustained_active
  launch__occupancy_limit_{barriers,blocks,registers,shared_mem,warps}
  smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio
"""

import argparse
import csv
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_raw_rows(path):
    """Return one {column: value_str} dict per profiled kernel from a .ncu-rep
    or raw-page CSV.

    The raw page is wide-format: row 0 column names (leading identification
    columns such as "Kernel Name", then metric names), row 1 units, row 2+
    one row per profiled kernel.
    """
    if path.endswith(".ncu-rep"):
        r = subprocess.run(
            ["ncu", "--import", path, "--page", "raw", "--csv"],
            capture_output=True, text=True, check=True,
        )
        rows = list(csv.reader(io.StringIO(r.stdout)))
    else:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
    if len(rows) < 3:
        raise SystemExit(f"{path}: expected raw-page CSV with >= 3 rows")
    names = rows[0]
    return [dict(zip(names, r)) for r in rows[2:] if r]


def parse_float(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*",
                    help=".ncu-rep or raw-page CSV files (later ones fill gaps)")
    ap.add_argument("--kernel", default="v0",
                    help="kernel name used for default <kernel>_rep.* paths")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    if args.output is None:
        args.output = os.path.join(HERE, f"{args.kernel}_rep.csv")

    inputs = args.inputs
    if not inputs:
        inputs = [os.path.join(HERE, f"{args.kernel}_rep.ncu-rep")]
        tables_rep = os.path.join(HERE, f"{args.kernel}_rep_tables.ncu-rep")
        if os.path.exists(tables_rep):
            inputs.append(tables_rep)

    # A split-KV report holds two kernels: the seven columns come from the
    # partial kernel only; the merge kernel contributes its duration as a
    # trailing column.
    partial_pat = f"gqa_decode_{args.kernel}_partial"
    merge_pat = f"gqa_decode_{args.kernel}_merge"
    metrics = {}
    merge_row = None
    for path in inputs:
        rows = load_raw_rows(path)
        primary = next(
            (r for r in rows if partial_pat in r.get("Kernel Name", "")), rows[0]
        )
        if merge_row is None:
            merge_row = next(
                (r for r in rows if merge_pat in r.get("Kernel Name", "")), None
            )
        for name, value in primary.items():
            if name not in metrics or metrics[name] == "":
                metrics[name] = value

    def get(name):
        v = metrics.get(name, "")
        if v == "":
            print(f"warning: metric {name} not found in {inputs}", file=sys.stderr)
        return v

    dram_pct = get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
    sectors_per_req = get(
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
    )
    warps_active_pct = get("sm__warps_active.avg.pct_of_peak_sustained_active")

    # Occupancy limiter: the launch__occupancy_limit_* resource allowing the
    # fewest blocks per SM (ties are joined with '+').
    limits = []
    for name, value in metrics.items():
        if name.startswith("launch__occupancy_limit_"):
            v = parse_float(value)
            if v is not None:
                limits.append((v, name.removeprefix("launch__occupancy_limit_")))
    if limits:
        lo = min(v for v, _ in limits)
        occ_limiter = "+".join(sorted(n for v, n in limits if v == lo))
        occ_limiter += f" ({lo:g} blocks/SM)"
    else:
        occ_limiter = ""

    # Top-3 warp stall reasons by cycles per issued instruction. 'selected'
    # is the issuing warp itself, not a stall, so it is excluded.
    stalls = []
    prefix = "smsp__average_warps_issue_stalled_"
    suffix = "_per_issue_active.ratio"
    for name, value in metrics.items():
        if name.startswith(prefix) and name.endswith(suffix):
            reason = name[len(prefix):-len(suffix)]
            if reason == "selected":
                continue
            v = parse_float(value)
            if v is not None:
                stalls.append((v, reason))
    stalls.sort(reverse=True)
    top3 = [f"{reason}={v:.3f}" for v, reason in stalls[:3]]
    top3 += [""] * (3 - len(top3))

    header = [
        "dram_throughput_pct_of_peak",
        "l1tex_global_load_sectors_per_request",
        "warps_active_pct_of_peak",
        "occupancy_limiter",
        "stall_1",
        "stall_2",
        "stall_3",
    ]
    row = [dram_pct, sectors_per_req, warps_active_pct, occ_limiter, *top3]
    if merge_row is not None:
        header.append("merge_kernel_ms")
        row.append(merge_row.get("gpu__time_duration.avg", ""))
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
