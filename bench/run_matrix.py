"""W3 64-cell benchmark matrix: FlashInfer baseline vs v0/v1/v2.

Grid (design doc section 4): headtype {(32,8),(64,8)} x S {512,2048,8192,32768}
x B {1,8,32,128} x block {16,64} = 64 cells. Per cell: FlashInfer, v1, v2
over num_splits {1,4,8,16,32,64} (best recorded), and v0 only where it is
not prohibitively slow (S<=2048 and B<=8, kept for ablation-chain
completeness). Every cell starts with a 2-request correctness spot check of
both our kernel and FlashInfer against the fp32 reference.

Each cell writes its own JSON into the output directory and is skipped when
that file already exists, so an interrupted run resumes with
  python bench/run_matrix.py --out-dir bench/results/matrix_<ts>
A matrix_summary.csv is (re)built from all cell files at the end.
"""

import argparse
import csv
import itertools
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from binding import KERNELS, gqa_decode_v2
from common import D, kv_bytes_of, make_case, sol_ms_of, spot_check, time_kernel
from baselines.flashinfer_adapter import make_flashinfer_runner

HEADTYPES = [(32, 8), (64, 8)]
S_LIST = [512, 2048, 8192, 32768]
B_LIST = [1, 8, 32, 128]
BLOCK_LIST = [16, 64]
SPLITS_LIST = [1, 4, 8, 16, 32, 64]
POOL_FACTOR = 1.1
MEM_LIMIT_GIB = 70.0


def cell_name(Hq, Hkv, S, B, bs):
    return f"cell_{Hq}x{Hkv}_S{S}_B{B}_bs{bs}"


def memory_precheck():
    """Refuse to start if the largest cell cannot fit."""
    max_kv_gib = max(
        kv_bytes_of(B, S, Hkv)
        for (Hq, Hkv), S, B in itertools.product(HEADTYPES, S_LIST, B_LIST)
    ) / 2**30 * POOL_FACTOR
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"memory precheck: largest cell KV pool ~{max_kv_gib:.1f} GiB "
          f"(pool_factor {POOL_FACTOR}), device total {total_gib:.1f} GiB")
    if max_kv_gib > MEM_LIMIT_GIB:
        raise SystemExit(
            f"largest cell needs {max_kv_gib:.1f} GiB > {MEM_LIMIT_GIB} GiB limit"
        )


def run_cell(Hq, Hkv, S, B, bs):
    scale = 1.0 / math.sqrt(D)
    q, k_cache, v_cache, block_table, seq_lens = make_case(
        B, S, Hq, Hkv, bs, seed=0, pool_factor=POOL_FACTOR
    )
    case = (q, k_cache, v_cache, block_table, seq_lens)

    spot_check(KERNELS["v1"], *case, scale, label="v1")
    fi_run = make_flashinfer_runner(*case, scale)
    spot_check(lambda *a: fi_run(), *case, scale, label="flashinfer")

    warmup, runs = (10, 30) if (S == 32768 and B >= 32) else (10, 50)
    sol = sol_ms_of(B, S, Hkv)
    results = {}

    fi_ms = time_kernel(fi_run, warmup, runs)
    results["flashinfer"] = {"median_ms": fi_ms, "pct_of_sol": sol / fi_ms * 100}

    v1_ms = time_kernel(lambda: KERNELS["v1"](*case, scale), warmup, runs)
    results["v1"] = {"median_ms": v1_ms, "pct_of_sol": sol / v1_ms * 100}

    by_splits = {}
    for ns in SPLITS_LIST:
        by_splits[ns] = time_kernel(
            lambda: gqa_decode_v2(*case, scale, num_splits=ns), warmup, runs
        )
    best_splits = min(by_splits, key=by_splits.get)
    results["v2"] = {
        "by_splits": by_splits,
        "best_splits": best_splits,
        "median_ms": by_splits[best_splits],
        "pct_of_sol": sol / by_splits[best_splits] * 100,
    }

    if S <= 2048 and B <= 8:
        v0_ms = time_kernel(lambda: KERNELS["v0"](*case, scale), warmup, runs)
        results["v0"] = {"median_ms": v0_ms, "pct_of_sol": sol / v0_ms * 100}

    return {
        "Hq": Hq, "Hkv": Hkv, "S": S, "B": B, "block": bs,
        "warmup": warmup, "runs": runs,
        "kv_bytes": kv_bytes_of(B, S, Hkv), "sol_ms": sol,
        "spot_check": "passed",
        "results": results,
    }


def write_summary(out_dir):
    rows = []
    for fname in sorted(os.listdir(out_dir)):
        if not (fname.startswith("cell_") and fname.endswith(".json")):
            continue
        with open(os.path.join(out_dir, fname)) as f:
            cell = json.load(f)
        fi_ms = cell["results"]["flashinfer"]["median_ms"]
        for kernel, r in cell["results"].items():
            rows.append({
                "headtype": f"{cell['Hq']}x{cell['Hkv']}",
                "S": cell["S"], "B": cell["B"], "block": cell["block"],
                "kernel": kernel,
                "best_splits": r.get("best_splits", ""),
                "median_ms": r["median_ms"],
                "pct_of_sol": r["pct_of_sol"],
                # >1 means faster than the FlashInfer baseline
                "ratio_vs_flashinfer": fi_ms / r["median_ms"],
            })
    path = os.path.join(out_dir, "matrix_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None,
                    help="existing matrix dir to resume; default creates a new one")
    args = ap.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(os.path.dirname(__file__), "results", f"matrix_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"matrix output dir: {out_dir}")

    memory_precheck()

    cells = [
        (Hq, Hkv, S, B, bs)
        for (Hq, Hkv), S, B, bs
        in itertools.product(HEADTYPES, S_LIST, B_LIST, BLOCK_LIST)
    ]
    total = len(cells)
    durations = []
    for i, (Hq, Hkv, S, B, bs) in enumerate(cells, 1):
        name = cell_name(Hq, Hkv, S, B, bs)
        path = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(path):
            print(f"[{i}/{total}] {name}: exists, skipping")
            continue
        t0 = time.time()
        cell = run_cell(Hq, Hkv, S, B, bs)
        with open(path, "w") as f:
            json.dump(cell, f, indent=2)
        torch.cuda.empty_cache()
        dt = time.time() - t0
        durations.append(dt)
        remaining = (total - i) * (sum(durations) / len(durations))
        print(f"[{i}/{total}] {name}: done in {dt:.1f}s, "
              f"~{remaining/60:.1f} min remaining")

    write_summary(out_dir)


if __name__ == "__main__":
    main()
