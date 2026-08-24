"""Single-shape benchmark for the GQA decode kernels (--kernel v0|v1|v2)."""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from binding import KERNELS
from common import D, kv_bytes_of, make_case, sol_ms_of, spot_check, time_kernel


def query_clocks():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--S", type=int, default=8192)
    ap.add_argument("--Hq", type=int, default=32)
    ap.add_argument("--Hkv", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--kernel", choices=sorted(KERNELS.keys()), default="v0")
    ap.add_argument("--num-splits", type=int, default=8,
                    help="split-KV segment count (v2 only)")
    ap.add_argument("--ncu-mode", action="store_true",
                    help="run the kernel exactly once, no warmup, no timing")
    args = ap.parse_args()

    scale = 1.0 / math.sqrt(D)
    if args.kernel == "v2":
        kernel_fn = lambda *a: KERNELS["v2"](*a, num_splits=args.num_splits)
    else:
        kernel_fn = KERNELS[args.kernel]
    q, k_cache, v_cache, block_table, seq_lens = make_case(
        args.B, args.S, args.Hq, args.Hkv, args.bs
    )

    if args.ncu_mode:
        kernel_fn(q, k_cache, v_cache, block_table, seq_lens, scale)
        torch.cuda.synchronize()
        return

    nb, max_abs = spot_check(
        kernel_fn, q, k_cache, v_cache, block_table, seq_lens, scale
    )
    print(f"spot check on first {nb} requests passed (max-abs={max_abs:.4e})")

    median_ms = time_kernel(
        lambda: kernel_fn(q, k_cache, v_cache, block_table, seq_lens, scale),
        warmup=10, runs=50,
    )

    kv_bytes = kv_bytes_of(args.B, args.S, args.Hkv)
    sol_ms = sol_ms_of(args.B, args.S, args.Hkv)
    pct_of_sol = sol_ms / median_ms * 100.0
    clocks = query_clocks()

    print(f"median_ms   = {median_ms:.4f}")
    print(f"kv_bytes    = {kv_bytes}")
    print(f"sol_ms      = {sol_ms:.4f}")
    print(f"pct_of_sol  = {pct_of_sol:.2f}%")
    print(f"clocks (sm, mem) = {clocks}")

    result = {
        "kernel": args.kernel,
        "B": args.B, "S": args.S, "Hq": args.Hq, "Hkv": args.Hkv, "bs": args.bs,
        "num_splits": args.num_splits if args.kernel == "v2" else None,
        "median_ms": median_ms,
        "kv_bytes": kv_bytes,
        "sol_ms": sol_ms,
        "pct_of_sol": pct_of_sol,
        "clocks_sm_mem": clocks,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir,
        f"{args.kernel}_{args.B}x{args.S}x{args.Hq}x{args.Hkv}x{args.bs}_{ts}.json",
    )
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
