"""GPU correctness tests for the v0_naive GQA decode kernel.

Tolerance atol=2e-2, rtol=2e-2: bf16 output rounding alone is ~2^-8 ~= 0.4%
relative; on top of that the kernel's sequential online-softmax accumulation
order differs from the reference's batched softmax+matmul, adding accumulation-
order noise. 2e-2 is a conservative upper bound for both combined.
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from binding import gqa_decode_v0
from reference import attention_ref

ATOL = 2e-2
RTOL = 2e-2
D = 128


def make_case(B, S_list, Hq, Hkv, bs, seed=0):
    """Build a random paged-KV decode case on CUDA.

    The physical block pool is 2x the total needed blocks; block_table entries
    are drawn from the pool at random without repetition (shuffled order). All
    unused storage — the invalid tail slots of each request's last block and
    every unallocated physical block — is filled with NaN so any out-of-range
    read by the kernel poisons the output.
    """
    assert len(S_list) == B
    torch.manual_seed(seed)
    device = "cuda"
    n_blocks = [(s + bs - 1) // bs for s in S_list]
    pool = 2 * sum(n_blocks)
    max_blocks = max(n_blocks)

    q = (torch.randn(B, Hq, D, device=device) * 0.5).to(torch.bfloat16)
    k_cache = (torch.randn(pool, bs, Hkv, D, device=device) * 0.5).to(torch.bfloat16)
    v_cache = (torch.randn(pool, bs, Hkv, D, device=device) * 0.5).to(torch.bfloat16)

    perm = torch.randperm(pool)
    block_table = torch.full((B, max_blocks), -1, dtype=torch.int32, device=device)
    idx = 0
    used = []
    for b in range(B):
        blks = perm[idx : idx + n_blocks[b]]
        block_table[b, : n_blocks[b]] = blks.to(torch.int32).to(device)
        used.append(blks)
        idx += n_blocks[b]

    used_mask = torch.zeros(pool, dtype=torch.bool)
    used_mask[torch.cat(used)] = True
    k_cache[~used_mask.to(device)] = float("nan")
    v_cache[~used_mask.to(device)] = float("nan")
    for b in range(B):
        rem = S_list[b] % bs
        if rem != 0:
            last = int(block_table[b, n_blocks[b] - 1])
            k_cache[last, rem:] = float("nan")
            v_cache[last, rem:] = float("nan")

    seq_lens = torch.tensor(S_list, dtype=torch.int32, device=device)
    return q, k_cache, v_cache, block_table, seq_lens


def run_and_check(q, k_cache, v_cache, block_table, seq_lens, check_finite=False):
    scale = 1.0 / math.sqrt(D)
    out = gqa_decode_v0(q, k_cache, v_cache, block_table, seq_lens, scale)
    torch.cuda.synchronize()
    ref = attention_ref(q, k_cache, v_cache, block_table, seq_lens, scale)
    if check_finite:
        assert torch.isfinite(out.float()).all(), "kernel output has NaN/inf"
    out_f = out.float()
    ok = torch.allclose(out_f, ref, atol=ATOL, rtol=RTOL)
    if not ok:
        abs_err = (out_f - ref).abs()
        max_abs = abs_err.max().item()
        max_rel = (abs_err / ref.abs().clamp_min(1e-8)).max().item()
        pytest.fail(f"mismatch: max-abs={max_abs:.4e} max-rel={max_rel:.4e}")


CASES = {
    "a_base": dict(B=4, S_list=[100, 257, 512, 33], Hq=32, Hkv=8, bs=16),
    "b_long": dict(B=1, S_list=[1000], Hq=32, Hkv=8, bs=16),
    "c_len1": dict(B=1, S_list=[1], Hq=32, Hkv=8, bs=16),
    "d_exact_blocks": dict(B=2, S_list=[16, 64], Hq=32, Hkv=8, bs=16),
    "e_partial_blocks": dict(B=2, S_list=[17, 65], Hq=32, Hkv=8, bs=16),
    "f_bs64": dict(B=4, S_list=[100, 257, 512, 33], Hq=32, Hkv=8, bs=64),
    "g_hq64": dict(B=4, S_list=[100, 257, 512, 33], Hq=64, Hkv=8, bs=16),
    "i_8k": dict(B=2, S_list=[8192, 8192], Hq=32, Hkv=8, bs=16),
}


@pytest.mark.parametrize("name", list(CASES.keys()))
def test_gqa_decode_v0(name):
    cfg = CASES[name]
    case = make_case(cfg["B"], cfg["S_list"], cfg["Hq"], cfg["Hkv"], cfg["bs"], seed=0)
    run_and_check(*case)


def test_gqa_decode_v0_large_logits():
    cfg = CASES["a_base"]
    q, k_cache, v_cache, block_table, seq_lens = make_case(
        cfg["B"], cfg["S_list"], cfg["Hq"], cfg["Hkv"], cfg["bs"], seed=0
    )
    q = (q.float() * 50.0).to(torch.bfloat16)
    run_and_check(q, k_cache, v_cache, block_table, seq_lens, check_finite=True)
