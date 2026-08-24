# K1 week 1 notes: roofline groundwork for GQA decode

## H100 SXM5 spec values

| Resource | Value |
|---|---|
| SMs | 132 |
| Registers per SM | 256 KB |
| Shared memory per SM | 228 KB |
| L2 cache | 50 MB |
| HBM3 | 80 GB at 3.35 TB/s |
| bf16 tensor core | 989 TFLOPS |
| fp32 CUDA core | 67 TFLOPS |

Ridge points (FLOP/B): 989e12 / 3.35e12 ≈ **295** (tensor core),
67e12 / 3.35e12 = **20** (CUDA core).

## Decode arithmetic intensity (GQA, d = 128, bf16)

Per KV token per kv-head: K + V bytes = 2 × 128 × 2 B = 512 B. Each q-head
sharing that kv-head does 2·d (QK^T) + 2·d (PV) = 512 FLOP.

| Config | q-heads per kv-head | FLOP per KV token | Bytes | AI (FLOP/B) |
|---|---|---|---|---|
| 32Q / 8KV | 4 | 4 × 512 = 2048 | 512 | 4 |
| 64Q / 8KV | 8 | 8 × 512 = 4096 | 512 | 8 |

Both are far below even the CUDA-core ridge point of 20: decode is
memory-bound, and time is set by KV bytes over HBM bandwidth.

## Representative shape and speed-of-light

Shape: B = 32, S = 8192, H_kv = 8, d = 128, bf16.

| Quantity | Value |
|---|---|
| KV bytes = 32 × 8192 × 8 × 128 × 2 (K and V) × 2 B | 1,073,741,824 B |
| SOL time = 1,073,741,824 / 3.35e12 | **0.32 ms** |
| q + out + block_table bytes | ≈ 0.6 MB |
| Fraction of KV bytes | ≈ 0.05% |

The non-KV traffic is 0.05% of the total, which justifies modeling bytes as
KV-only in the roofline.

## KV cache per token (all layers)

| Model | Computation | Per token |
|---|---|---|
| Llama-3-8B | 32 layers × 8 kv-heads × 128 × 2 (K,V) × 2 B | 128 KB |
| Llama-2-7B | 32 layers × 32 kv-heads × 128 × 2 (K,V) × 2 B | 512 KB |
| DeepSeek-V3 (MLA) | 61 layers × 576 × 2 B | ≈ 68.6 KB |

## Paged KV layout (NHD) and index arithmetic

Layout: `[num_blocks, block_size, H_kv, 128]`, element index
`idx = ((p*bs + s)*H_kv + h)*d + j` where `p` is the physical block id and
`s = t % bs` the in-block slot.

Worked example: bs = 16, H_kv = 8, d = 128, h = 3, token t = 20,
block_table = [9, 2, 14]:

| Step | Value |
|---|---|
| Logical block = 20 // 16 | 1 |
| Physical block p = block_table[1] | 2 |
| Slot s = 20 % 16 | 4 |
| idx = ((2×16 + 4)×8 + 3)×128 + 0 | **37,248** |
| Byte offset (bf16) = 37,248 × 2 | **74,496** |

Overflow: at B = 128, S = 32768, H_kv = 8, d = 128 the cache holds
128 × 32768 × 8 × 128 = 2^32 elements — int32 index arithmetic overflows
exactly at this shape. Use `size_t` (or int64) for flattened indices.

## Online softmax trace

Scores s = [2, 5, 1, 6], 2-d values V = [1,0], [0,1], [1,1], [2,0].
State after each step (m = running max, l = running sum, acc unnormalized):

| Step | s_t | m | l | acc |
|---|---|---|---|---|
| t0 | 2 | 2 | 1.000000 | [1.000000, 0.000000] |
| t1 | 5 | 5 | 1.049787 | [0.049787, 1.000000] |
| t2 | 1 | 5 | 1.068103 | [0.068103, 1.018316] |
| t3 | 6 | 6 | 1.392933 | [2.025054, 0.374618] |

Final: l = **1.3929**, out = acc / l = **[1.4538, 0.2689]**.

### Split-KV merge check

Split 1 = {t0, t1}: m1 = 5, l1 = 1.049787, acc1 = [0.049787, 1.000000].
Split 2 = {t2, t3}: m2 = 6, l2 = 1.006738, acc2 = [2.006738, 0.006738].

Merge with M = max(m1, m2) = 6, α1 = e^(5−6) = 0.367879, α2 = 1:
l = α1·l1 + α2·l2 = 1.392933; acc = α1·acc1 + α2·acc2 = [2.025054, 0.374617];
out = [1.4538, 0.2689] — identical to the single-pass trace.

## Open item: FlashMLA bench AI (477) vs design estimate (242) — resolved

G0's headline FlashMLA row reports 759 TFLOPS at 1592 GB/s, i.e. 477 FLOP/B,
while the design estimate for 128 heads is AI ≈ h_q·(d+dv)/d =
128 × 1088 / 576 ≈ 242. The gap is entirely explained by `s_q`:

- The 759/1592 row in `docs/g0/flashmla_bench_raw.txt` is
  `TestParam(b=128, s_q=2, s_k=32768, h_q=128, h_kv=1, d=576, dv=512,
  is_causal=True)` — **s_q = 2**, i.e. two query tokens per request
  (MTP/speculative decode), both attending to the same KV.
- FlashMLA's accounting (`tests/test_flash_mla_dense_decoding.py:177-190` at
  the pinned SHA 15f13e5) is:
  FLOP = b·h_q·s_q·(2·d·mean_seqlen + 2·mean_seqlen·dv), and
  bytes = b·(s_q·h_q·d·2 + mean_seqlen·h_kv·d·2 + s_q·h_q·dv·2) —
  Q + KV (counted once) + output, with mean_seqlen the mean of the actual
  (varlen) cache_seqlens and no causal discount.
- So AI ≈ s_q·h_q·(d+dv)/d = 241.8 × s_q. For s_q = 2 that is 484, reduced to
  the observed 477 by the q/out bytes in the denominator.
- The s_q = 1 rows in the same log confirm the estimate directly: e.g.
  b=128, s_q=1, s_k=32768 gives 627 TFLOPS at 2612 GB/s = **240 FLOP/B ≈ 242**.

Conclusion: no discrepancy in the accounting. The design estimate assumed one
query token per request (s_q = 1); FlashMLA's headline number doubles FLOPs per
KV byte by batching 2 query tokens. When comparing our kernels against
FlashMLA, compare per-`s_q` rows.

## v0 -> v1 delta (B=32, S=8192, Hq=32, Hkv=8, bs=16, H100 SXM)

Extracted NCU rows (dram%, L1TEX global-load sectors/request, warps_active%,
occupancy limiter, top-3 stalls):

```
v0: 3.65, 1.67, 47.51, registers+warps (16 blocks/SM), long_scoreboard=6.716, short_scoreboard=2.241, wait=1.987
v1: 14.55, 5.67, 47.54, registers+warps (16 blocks/SM), long_scoreboard=11.023, wait=1.912, short_scoreboard=1.211
```

Bench: median 7.878 ms -> 2.005 ms, a **3.93x** speedup (4.07% -> 15.99% of
KV-bytes speed-of-light).

- **The 3.93x is almost pure warp overlap, not faster per-token work.** v0's
  single serial scan spends 7.878 ms / 8192 = 962 ns per token; v1's four
  concurrent per-warp scans each spend 2.005 ms / 2048 = 979 ns per token.
  Per-warp per-token latency is unchanged within 2% — v1 wins by running four
  dependency chains in parallel, exactly the k1-005 design intent (the two
  block-wide syncs per token that v0 paid are gone, but their cost was
  evidently hidden inside the load latency anyway).
- **DRAM traffic is 1x KV size in both**: 1.0027 GiB (v0, 122.4 GB/s x
  8.795 ms) vs 1.0026 GiB (v1, 487.9 GB/s x 2.206 ms) against a 1.000 GiB KV
  working set. L2 keeps absorbing the 4 q-heads' re-reads of the shared KV
  head; nothing extra was spent to overlap warps.
- **sectors/request 1.67 -> 5.67.** The v1 number is below the ideal 8 for
  256 B warp-coalesced loads, but the gap is bookkeeping, not miscoalescing:
  per token each warp issues two perfect 8-sector vector loads (K row, V row)
  plus one uniform 4 B `block_table` lookup (1 sector), and (8+8+1)/3 = 5.67.
- **long_scoreboard 6.72 -> 11.02 is a denominator effect.** The metric is
  stall cycles per *issued instruction*. v1 issues far fewer instructions per
  KV byte (one 8 B load replaces four 2 B loads; the shfl reduction replaces
  the smem reduction's ~14 instructions plus two barriers), so the same
  latency-bound waiting is divided by a much smaller instruction count.
  Total stall time fell with runtime (3.9x); latency hiding did not regress.

Remaining target: at ~488 GB/s sustained and ~650 ns DRAM latency the grid
keeps only ~0.3 MB in flight, versus the ~2 MB Little's-law requirement for
3.35 TB/s. Each warp still has just one token's K (256 B) outstanding at a
time because the online-softmax chain serializes consecutive tokens.
Directions: unroll multiple tokens per warp (issue several K rows before the
softmax update consumes them) or v2 split-KV (more CTAs per request so more
independent chains exist GPU-wide).
