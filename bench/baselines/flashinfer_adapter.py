"""FlashInfer paged-decode baseline adapter (flashinfer 0.6.17).

Layout: our separate k_cache/v_cache, NHD [num_blocks, block_size, Hkv, 128]
bf16, are fed zero-copy — BatchDecodeWithPagedKVCacheWrapper.run accepts a
(k_cache, v_cache) tuple in NHD layout directly, so no repacking is needed.

Timing contract (k1-007): plan() is CPU-side scheduling prep, done once per
batch shape and excluded from timing; only the returned run callable is timed.
"""

import torch
import flashinfer

_WORKSPACE_BYTES = 128 * 1024 * 1024
_wrapper = None


def _get_wrapper(device):
    """Lazily build one shared wrapper (128 MB float workspace).

    use_tensor_cores=True is FlashInfer's recommended configuration for GQA
    group sizes >= 4 (ours are 4 and 8), i.e. the production setting.
    """
    global _wrapper
    if _wrapper is None:
        ws = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        _wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws, kv_layout="NHD", use_tensor_cores=True
        )
    return _wrapper


def block_table_to_paged(block_table, seq_lens, page_size):
    """Convert block_table [B, max_blocks] + seq_lens [B] into FlashInfer's
    (kv_indptr [B+1], kv_indices [total_blocks], kv_last_page_len [B]),
    all int32 on block_table's device.

    kv_indices concatenates each request's first ceil(S/page_size) block ids
    in order; kv_last_page_len is the number of valid tokens in the last
    page, in [1, page_size].
    """
    B, max_blocks = block_table.shape
    device = block_table.device
    n_blocks = torch.div(seq_lens + page_size - 1, page_size,
                         rounding_mode="floor").to(torch.int32)
    indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(n_blocks, 0)
    valid = torch.arange(max_blocks, device=device)[None, :] < n_blocks[:, None]
    indices = block_table[valid].to(torch.int32)
    last_page_len = (seq_lens - (n_blocks - 1) * page_size).to(torch.int32)
    return indptr, indices, last_page_len


def make_flashinfer_runner(q, k_cache, v_cache, block_table, seq_lens, scale):
    """plan() now (untimed); return a zero-arg callable that only run()s."""
    B, Hq, D = q.shape
    page_size, Hkv = k_cache.shape[1], k_cache.shape[2]
    indptr, indices, last_page_len = block_table_to_paged(
        block_table, seq_lens, page_size
    )
    wrapper = _get_wrapper(q.device)
    wrapper.plan(
        indptr, indices, last_page_len, Hq, Hkv, D, page_size,
        sm_scale=scale, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    )
    return lambda: wrapper.run(q, (k_cache, v_cache))


def flashinfer_decode(q, k_cache, v_cache, block_table, seq_lens, scale):
    """One-shot convenience wrapper: plan + run, returns out [B, Hq, 128] bf16."""
    return make_flashinfer_runner(q, k_cache, v_cache, block_table, seq_lens,
                                  scale)()
