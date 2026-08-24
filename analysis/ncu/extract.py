"""Extract key metrics from an `ncu --import <rep> --csv` details-page dump.

Usage: python analysis/ncu/extract.py <raw_csv_in> <one_line_csv_out>

The details CSV is long-format: one row per (kernel, metric) with columns
including "Section Name", "Metric Name", "Metric Unit", "Metric Value".
Metric names differ across ncu versions (raw vs. human-readable), so matching
is by case-insensitive substring against both spellings.
"""

import csv
import sys


def parse_float(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def main():
    raw_path, out_path = sys.argv[1], sys.argv[2]
    rows = []
    with open(raw_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                (
                    r.get("Section Name", ""),
                    r.get("Metric Name", ""),
                    r.get("Metric Unit", ""),
                    r.get("Metric Value", ""),
                )
            )

    def find(*candidates):
        for section, name, unit, value in rows:
            low = name.lower()
            for c in candidates:
                if c.lower() in low:
                    return name, unit, value
        return None, None, None

    _, _, dram_pct = find(
        "dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM Throughput"
    )
    _, _, sectors_per_req = find(
        "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld",
        "Global Load Access Pattern",
        "Sectors Per Request",
    )
    _, _, warps_active_pct = find(
        "sm__warps_active.avg.pct_of_peak_sustained_active", "Achieved Occupancy"
    )

    # Occupancy limiter: the block-limit resource with the smallest block count.
    limits = []
    for section, name, unit, value in rows:
        if "block limit" in name.lower():
            v = parse_float(value)
            if v is not None:
                limits.append((v, name))
    occ_limiter = min(limits)[1] if limits else ""

    # Top-3 warp stall reasons by cycles/instruction.
    stalls = []
    for section, name, unit, value in rows:
        low = name.lower()
        if "stall" in low or "issue_stalled" in low:
            v = parse_float(value)
            if v is not None:
                stalls.append((v, name))
    stalls.sort(reverse=True)
    top3 = [f"{name}={v}" for v, name in stalls[:3]]
    top3 += [""] * (3 - len(top3))

    with open(out_path, "w", newline="") as f:
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
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
