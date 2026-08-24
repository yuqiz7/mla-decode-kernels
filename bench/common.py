"""Shared benchmark pieces: case construction, CUDA-event timing, SOL math."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from reference import attention_ref

D = 128
HBM_PEAK_BYTES_PER_S = 3.35e12  # H100 SXM HBM3


def make_case(B, S, Hq, Hkv, bs, seed=0, pool_factor=1.0, fill_nan=False):
    """Uniform-seqlen paged decode case on CUDA.

    pool_factor scales the physical block pool beyond the exact need (the
    matrix runner uses 1.1 to keep large cells within memory while still
    exercising a shuffled, non-dense table). fill_nan poisons the unused
    storage (unallocated blocks and invalid tail slots) with NaN.
    """
    torch.manual_seed(seed)
    device = "cuda"
    n_blocks = (S + bs - 1) // bs
    pool = max(B * n_blocks, int(B * n_blocks * pool_factor))
    q = (torch.randn(B, Hq, D, device=device) * 0.5).to(torch.bfloat16)
    k_cache = torch.randn(pool, bs, Hkv, D, device=device, dtype=torch.bfloat16) * 0.5
    v_cache = torch.randn(pool, bs, Hkv, D, device=device, dtype=torch.bfloat16) * 0.5
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    perm = torch.randperm(pool).to(torch.int32)
    block_table = perm[: B * n_blocks].reshape(B, n_blocks).to(device)
    seq_lens = torch.full((B,), S, dtype=torch.int32, device=device)
    if fill_nan:
        used_mask = torch.zeros(pool, dtype=torch.bool)
        used_mask[perm[: B * n_blocks].long()] = True
        k_cache[~used_mask.to(device)] = float("nan")
        v_cache[~used_mask.to(device)] = float("nan")
        rem = S % bs
        if rem != 0:
            last = block_table[:, -1].long()
            k_cache[last, rem:] = float("nan")
            v_cache[last, rem:] = float("nan")
    return q, k_cache, v_cache, block_table, seq_lens


def time_kernel(fn, warmup=10, runs=50):
    """Median wall time of fn() in ms via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(runs):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]


def kv_bytes_of(B, S, Hkv):
    return B * S * Hkv * D * 2 * 2  # K+V, bf16


def sol_ms_of(B, S, Hkv):
    return kv_bytes_of(B, S, Hkv) / HBM_PEAK_BYTES_PER_S * 1e3


def spot_check(kernel_fn, q, k_cache, v_cache, block_table, seq_lens, scale,
               label="kernel"):
    """Compare the first min(2, B) requests against the fp32 reference."""
    nb = min(2, q.shape[0])
    out = kernel_fn(q, k_cache, v_cache, block_table, seq_lens, scale)
    torch.cuda.synchronize()
    ref = attention_ref(
        q[:nb], k_cache, v_cache, block_table[:nb], seq_lens[:nb], scale
    )
    out_f = out[:nb].float()
    max_abs = (out_f - ref).abs().max().item()
    assert torch.allclose(out_f, ref, atol=2e-2, rtol=2e-2), (
        f"{label} spot check failed on first {nb} requests: max-abs={max_abs:.4e}"
    )
    return nb, max_abs
