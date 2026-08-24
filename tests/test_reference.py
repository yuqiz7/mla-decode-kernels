"""CPU test: attention_ref must match F.scaled_dot_product_attention in fp32."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from reference import check_against_sdpa


@pytest.mark.parametrize(
    "B,S,Hq,Hkv",
    [
        (2, [37, 100], 32, 8),
        (1, [1], 32, 8),
        (3, [16, 17, 64], 64, 8),
    ],
)
def test_reference_matches_sdpa(B, S, Hq, Hkv):
    max_abs = check_against_sdpa(B, S, Hq, Hkv)
    assert max_abs < 1e-5, f"max-abs vs SDPA = {max_abs:.3e}"
