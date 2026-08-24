"""Extract key metrics from an Nsight Compute report into a one-line CSV.

Usage: python analysis/ncu/extract.py [inputs ...] [-o OUTPUT]

Inputs may be .ncu-rep files (exported here via `ncu --import --page raw
--csv`) or pre-exported raw-page CSVs. With no inputs, defaults to
analysis/ncu/v0_rep.ncu-rep, plus analysis/ncu/v0_rep_tables.ncu-rep if it
exists (supplemental run holding MemoryWorkloadAnalysis_Tables metrics that
the main section set does not collect on NCU 2025.1). Later inputs only fill
metrics missing from earlier ones. Default output: analysis/ncu/v0_rep.csv.

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
DEFAULT_REP = os.path.join(HERE, "v0_rep.ncu-rep")
DEFAULT_TABLES_REP = os.path.join(HERE, "v0_rep_tables.ncu-rep")
DEFAULT_OUT = os.path.join(HERE, "v0_rep.csv")


def load_raw_metrics(path):
    """Return {metric_name: value_str} from a .ncu-rep or raw-page CSV.

    The raw page is wide-format: row 0 metric names, row 1 units, row 2+ one
    row per profiled kernel (we take the first).
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
    names, values = rows[0], rows[2]
    return dict(zip(names, values))


def parse_float(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*",
                    help=".ncu-rep or raw-page CSV files (later ones fill gaps)")
    ap.add_argument("-o", "--output", default=DEFAULT_OUT)
    args = ap.parse_args()

    inputs = args.inputs
    if not inputs:
        inputs = [DEFAULT_REP]
        if os.path.exists(DEFAULT_TABLES_REP):
            inputs.append(DEFAULT_TABLES_REP)

    metrics = {}
    for path in inputs:
        for name, value in load_raw_metrics(path).items():
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

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dram_throughput_pct_of_peak",
                "l1tex_global_load_sectors_per_request",
                "warps_active_pct_of_peak",
                "occupancy_limiter",
                "stall_1",
                "stall_2",
                "stall_3",
            ]
        )
        w.writerow(
            [dram_pct, sectors_per_req, warps_active_pct, occ_limiter, *top3]
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
