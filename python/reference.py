"""fp32 reference implementation of paged GQA decode attention."""

import torch
import torch.nn.functional as F


def gather_kv(k_cache, v_cache, block_table, seq_len, b):
    """Gather the first seq_len tokens of request b from the paged caches.

    Returns (K, V), each [seq_len, Hkv, 128].
    """
    bs = k_cache.shape[1]
    n = (seq_len + bs - 1) // bs
    blocks = block_table[b, :n].long()
    Hkv, D = k_cache.shape[2], k_cache.shape[3]
    K = k_cache[blocks].reshape(n * bs, Hkv, D)[:seq_len]
    V = v_cache[blocks].reshape(n * bs, Hkv, D)[:seq_len]
    return K, V


def attention_ref(q, k_cache, v_cache, block_table, seq_lens, scale):
    """fp32 reference: out[b, h] = softmax(K @ q * scale) @ V per (b, h).

    Returns out [B, Hq, 128] in fp32.
    """
    B, Hq, D = q.shape
    Hkv = k_cache.shape[2]
    q = q.float()
    out = torch.empty(B, Hq, D, dtype=torch.float32, device=q.device)
    for b in range(B):
        S = int(seq_lens[b])
        K, V = gather_kv(k_cache, v_cache, block_table, S, b)
        K = K.float()
        V = V.float()
        for h in range(Hq):
            kv = h // (Hq // Hkv)
            scores = K[:, kv, :] @ q[b, h] * scale  # [S]
            w = torch.softmax(scores, dim=0)
            out[b, h] = w @ V[:, kv, :]
    return out


def check_against_sdpa(B, S, Hq, Hkv):
    """Compare attention_ref with F.scaled_dot_product_attention on random fp32
    data with an identity block_table (unpaged layout). Returns max-abs error.
    """
    torch.manual_seed(0)
    D = 128
    bs = 16
    seq_lens = torch.tensor(S, dtype=torch.int32)
    max_S = int(seq_lens.max())
    max_blocks = (max_S + bs - 1) // bs
    num_blocks = B * max_blocks

    q = torch.randn(B, Hq, D, dtype=torch.float32)
    k_cache = torch.randn(num_blocks, bs, Hkv, D, dtype=torch.float32)
    v_cache = torch.randn(num_blocks, bs, Hkv, D, dtype=torch.float32)
    block_table = torch.arange(num_blocks, dtype=torch.int32).reshape(B, max_blocks)

    scale = 1.0 / (D ** 0.5)
    out_ref = attention_ref(q, k_cache, v_cache, block_table, seq_lens, scale)

    group = Hq // Hkv
    max_abs = 0.0
    for b in range(B):
        S_b = int(seq_lens[b])
        K, V = gather_kv(k_cache, v_cache, block_table, S_b, b)
        # [Hq, S_b, D] with KV heads expanded to match q heads (repeat_kv).
        K_exp = K.permute(1, 0, 2).repeat_interleave(group, dim=0)
        V_exp = V.permute(1, 0, 2).repeat_interleave(group, dim=0)
        q_b = q[b].unsqueeze(1)  # [Hq, 1, D]
        out_sdpa = F.scaled_dot_product_attention(q_b, K_exp, V_exp, scale=scale)
        err = (out_sdpa.squeeze(1) - out_ref[b]).abs().max().item()
        max_abs = max(max_abs, err)
    return max_abs
