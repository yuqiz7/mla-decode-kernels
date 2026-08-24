# K1-006: v2 design — split-KV with a two-kernel partial/merge scheme

## Decision

The sequence is split into `num_splits` runtime-chosen segments. Each segment
runs the v1 warp-per-token scan in its own CTA and writes one per-segment fp32
partial `(m, l, acc)` state to a global workspace
(`ws_acc [B, Hq, num_splits, 128]`, `ws_m`/`ws_l [B, Hq, num_splits]`); a
second kernel merges the segments with the online-softmax merge identity and
writes the normalized bf16 output.

## Why

After v1 the grid is pinned at `B*Hq` CTAs — 12% of the 2112 resident CTA
slots at B=8 — and in-flight bytes sit at ~0.3 MB versus the ~2 MB
Little's-law requirement for HBM saturation. Splitting multiplies the CTA
count by `num_splits` at a measured-negligible cost: the workspace round trip
is ~8.6 MB ≈ 0.8% of the 1 GiB KV stream at the representative shape
(B=32, S=8192, Hq=32, num_splits=8).

## Rejected

- Atomic single-pass merge in global memory: ordering nondeterminism and
  contention on the shared `(m, l)` state.
- Persistent-CTA work stealing: complexity not justified at K1 scope.

## Trade-off

Second launch overhead per call. Output differs bitwise from v1 because the
summation order changes (within harness tolerance by design; `num_splits=1`
is bitwise-identical to v1). Optimal `num_splits` is shape-dependent — swept
in the W3 matrix.
