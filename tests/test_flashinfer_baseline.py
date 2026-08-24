"""GPU tests: FlashInfer baseline vs our fp32 reference, plus the page-table
conversion helper. Tolerance matches our kernel harness (atol=rtol=2e-2)."""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))

from baselines.flashinfer_adapter import block_table_to_paged, flashinfer_decode
from reference import attention_ref
from test_gqa_decode import ATOL, CASES, D, RTOL, make_case


def test_block_table_to_paged():
    block_table = torch.tensor([[5, 1, 3], [0, 4, -1]], dtype=torch.int32,
                               device="cuda")
    seq_lens = torch.tensor([37, 22], dtype=torch.int32, device="cuda")
    indptr, indices, last_page_len = block_table_to_paged(
        block_table, seq_lens, page_size=16
    )
    assert indptr.tolist() == [0, 3, 5]
    assert indices.tolist() == [5, 1, 3, 0, 4]
    assert last_page_len.tolist() == [5, 6]
    for t in (indptr, indices, last_page_len):
        assert t.dtype == torch.int32 and t.device == block_table.device


@pytest.mark.parametrize(
    "name", ["a_base", "e_partial_blocks", "g_hq64", "f_bs64"]
)
def test_flashinfer_vs_reference(name):
    cfg = CASES[name]
    q, k_cache, v_cache, block_table, seq_lens = make_case(
        cfg["B"], cfg["S_list"], cfg["Hq"], cfg["Hkv"], cfg["bs"], seed=0
    )
    scale = 1.0 / math.sqrt(D)
    out = flashinfer_decode(q, k_cache, v_cache, block_table, seq_lens, scale)
    torch.cuda.synchronize()
    ref = attention_ref(q, k_cache, v_cache, block_table, seq_lens, scale)
    out_f = out.float()
    ok = torch.allclose(out_f, ref, atol=ATOL, rtol=RTOL)
    if not ok:
        abs_err = (out_f - ref).abs()
        max_abs = abs_err.max().item()
        max_rel = (abs_err / ref.abs().clamp_min(1e-8)).max().item()
        pytest.fail(f"mismatch: max-abs={max_abs:.4e} max-rel={max_rel:.4e}")
