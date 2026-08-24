# K1-005: v1 design — warp-per-token strided scan with vectorized loads

## Decision

Warp-per-token strided scan + 8 B vectorized loads + per-warp online state
merged once at the end (lesson-3 split-softmax formula). Each of the 4 warps
scans tokens `t = w, w+4, w+8, ...`, keeps its own fp32 `(m, l, acc)` state,
and the CTA merges the 4 states with a single `__syncthreads()` after the
loop.

## Why

NCU row 0 for v0 shows DRAM at 3.65% of peak with long_scoreboard-dominated
stalls (6.72 cyc/inst) and only ~512 B in flight per CTA (one 256 B K row +
one 256 B V row, 2 B per thread); the v0 loop also had two block-wide syncs
per token serializing the scan. Four warps in flight with 8 B loads per lane
raise both memory-level parallelism and bytes per instruction, and the loop
body needs zero `__syncthreads`.

## Rejected

- (b, kv-head) CTA regrouping: measured DRAM traffic is 0.96 GiB ≈ 1x KV
  size, proving L2 already absorbs the 4 q-heads' re-reads of the shared KV
  head, so regrouping saves no DRAM bytes.
- Per-token block reduction (v0): two block-wide syncs per token and a
  shared-memory round trip for every dot product.

## Trade-off

+2 KB smem (4x128 fp32 accumulators) and higher register use per lane (the
per-warp state and 4-wide vectors; watch the occupancy limiter — it was
registers at 47.5% achieved occupancy in v0). The merge cost is paid once per
CTA instead of per token.
