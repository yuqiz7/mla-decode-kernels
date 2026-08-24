"""Single-shape benchmark for the GQA decode kernels (--kernel v0|v1)."""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from binding import KERNELS
from reference import attention_ref

D = 128
HBM_PEAK_BYTES_PER_S = 3.35e12  # H100 SXM HBM3


def make_case(B, S, Hq, Hkv, bs, seed=0):
    torch.manual_seed(seed)
    device = "cuda"
    n_blocks = (S + bs - 1) // bs
    pool = B * n_blocks
    q = (torch.randn(B, Hq, D, device=device) * 0.5).to(torch.bfloat16)
    k_cache = torch.randn(pool, bs, Hkv, D, device=device, dtype=torch.bfloat16) * 0.5
    v_cache = torch.randn(pool, bs, Hkv, D, device=device, dtype=torch.bfloat16) * 0.5
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    perm = torch.randperm(pool).to(torch.int32)
    block_table = perm.reshape(B, n_blocks).to(device)
    seq_lens = torch.full((B,), S, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, block_table, seq_lens


def spot_check(kernel_fn, q, k_cache, v_cache, block_table, seq_lens, scale):
    nb = min(2, q.shape[0])
    out = kernel_fn(q, k_cache, v_cache, block_table, seq_lens, scale)
    torch.cuda.synchronize()
    ref = attention_ref(
        q[:nb], k_cache, v_cache, block_table[:nb], seq_lens[:nb], scale
    )
    out_f = out[:nb].float()
    max_abs = (out_f - ref).abs().max().item()
    assert torch.allclose(out_f, ref, atol=2e-2, rtol=2e-2), (
        f"spot check failed on first {nb} requests: max-abs={max_abs:.4e}"
    )
    print(f"spot check on first {nb} requests passed (max-abs={max_abs:.4e})")


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
    ap.add_argument("--ncu-mode", action="store_true",
                    help="run the kernel exactly once, no warmup, no timing")
    args = ap.parse_args()

    scale = 1.0 / math.sqrt(D)
    kernel_fn = KERNELS[args.kernel]
    q, k_cache, v_cache, block_table, seq_lens = make_case(
        args.B, args.S, args.Hq, args.Hkv, args.bs
    )

    if args.ncu_mode:
        kernel_fn(q, k_cache, v_cache, block_table, seq_lens, scale)
        torch.cuda.synchronize()
        return

    spot_check(kernel_fn, q, k_cache, v_cache, block_table, seq_lens, scale)

    for _ in range(10):
        kernel_fn(q, k_cache, v_cache, block_table, seq_lens, scale)
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(50):
        start.record()
        kernel_fn(q, k_cache, v_cache, block_table, seq_lens, scale)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    median_ms = times[len(times) // 2]

    kv_bytes = args.B * args.S * args.Hkv * D * 2 * 2  # K+V, bf16
    sol_ms = kv_bytes / HBM_PEAK_BYTES_PER_S * 1e3
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
