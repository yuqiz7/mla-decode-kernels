# K1 report: GQA paged-KV decode kernels on H100

**Scope.** Three hand-written CUDA decode-attention kernels for bf16 GQA with
a paged KV cache (v0 naive → v1 vectorized warp-scan → v2 split-KV),
benchmarked per-shape against FlashInfer 0.6.17 on an H100 SXM, with
NCU-backed attribution of every ladder step. d = 128 throughout; head
configurations 32Q/8KV and 64Q/8KV.

**Headline results** (representative shape B=32, S=8192, 32Q/8KV, block 16,
1 GiB of KV):

| Kernel | Median | % of KV-bytes speed-of-light | Step speedup |
|---|---|---|---|
| v0 naive | 7.878 ms | 4.07% | — |
| v1 vectorized | 2.005 ms | 15.99% | 3.93x |
| v2 split-KV (splits=8) | 1.191 ms | 26.91% | 1.68x |
| FlashInfer (tensor-core) | 0.368 ms | 87.0% | — |

At the shape v2 was designed for (B=8, S=32768 — same 1 GiB KV, starved
grid), v2 is **5.98x** faster than v1 (25.0% vs 4.2% of SoL), and at
(B=1, S=32768) the matrix shows **38.6x**. Saturated cells converge to a
steady ~0.32x of FlashInfer for 32Q (~0.167x for 64Q); the remaining gap is
attributed below to in-flight bytes and per-q-head L2 re-reads, not to
scheduling or DRAM waste. All three kernels pass the full correctness suite
and all four compute-sanitizer tools with zero findings.

## 1. Problem and roofline position

Single-token decode attention reads each request's entire KV cache once and
does 2·d FLOP (QK^T) + 2·d FLOP (PV) per q-head per KV token. Per KV token
per kv-head, K+V is 512 B in bf16, giving arithmetic intensity 4 FLOP/B at
32Q/8KV and 8 FLOP/B at 64Q/8KV — far below the H100's CUDA-core ridge of
~20 and tensor-core ridge of ~295 (docs/k1/notes_w1.md). Decode is therefore
memory-bound and the natural figure of merit is **percent of KV-bytes
speed-of-light**: `KV bytes / 3.35 TB/s` (non-KV traffic is ~0.05% of bytes
at the representative shape, so KV-only modeling is justified).

The KV cache is paged: layout NHD `[num_blocks, block_size, Hkv, 128]` with a
per-request `block_table` of physical block ids in arbitrary (shuffled)
order. Flattened indices are computed in `size_t` — int32 overflows exactly
at B=128, S=32768, Hkv=8, d=128 (2^32 elements).

## 2. Kernel ladder

Each step changes one thing so the delta is attributable (decision records in
`docs/decisions/`).

**v0 naive** (k1-001..004): one CTA per (batch, q-head), 128 threads, thread
j owns dimension j. Sequential online-softmax scan; per-token dot product via
a shared-memory tree reduction with two block-wide syncs per token. fp32
softmax state (m, l, acc) and accurate `expf` — v0's job is to match the
fp32 reference, not to be fast. GQA grouping is consecutive
(`kv_head = q_head // (Hq/Hkv)`, HF `repeat_kv` semantics) so correctness
checks against the HF-style reference are apples-to-apples.

**v1 vectorized** (k1-005): warp-per-token strided scan. Each of 4 warps
scans tokens `t = w, w+4, ...` with its own fp32 (m, l, acc) state; 8 B
vectorized loads (4 bf16 per lane); dot products via warp shuffle; the 4
per-warp states merge once after the loop with the split-softmax identity —
zero `__syncthreads` in the loop body. Rationale from v0's NCU row: DRAM at
3.65% of peak, long-scoreboard-dominated, ~512 B in flight per CTA; four
concurrent dependency chains and 4x wider loads attack both. (b, kv-head)
CTA regrouping was measured and rejected: DRAM traffic is already 1.0x KV
size, so L2 absorbs the 4 q-heads' re-reads and regrouping saves no DRAM
bytes.

**v2 split-KV** (k1-006): the sequence is cut into `num_splits` segments; a
partial kernel (grid (Hq, B, num_splits)) runs the v1 scan per segment and
writes fp32 (m, l, acc) partials to a global workspace
(`ws_acc [B, Hq, splits, 128]`, `ws_m`/`ws_l [B, Hq, splits]`); a merge
kernel combines segments with the online-softmax merge identity and writes
bf16 output. Rationale: after v1 the grid is pinned at B·Hq CTAs (0.48 waves
at the representative shape) with ~0.3 MB in flight versus the ~2 MB
Little's-law requirement for HBM saturation; splitting multiplies CTA count
at a workspace cost of ~0.8% of the KV stream. `num_splits=1` is bitwise
identical to v1 (asserted in tests), which closes the ablation: any sweep
gain is attributable to CTA count alone.

## 3. Methodology

**Environments.** Every benchmark and NCU number in this report was measured
on the Lambda H100 SXM instance recorded in `docs/g0/versions.txt` (driver
580.105.08, CUDA 12.8, torch 2.7.0, Ubuntu 24.04, Nsight Compute 2025.1.1.0),
with `RmProfilingAdminOnly=0` set for profiling. The compute-sanitizer runs
(§6) were done on a RunPod H100 SXM pod (environment captured in
`docs/k1/sanitizer/README.md`), while every benchmark and NCU number comes
from the Lambda environment in `docs/g0/versions.txt`; sanitizer runs are
correctness artifacts, not measurements, and the CUDA toolkit build is
identical on both machines.

**Timing.** CUDA events around the kernel call only; 10 warmup iterations,
median of 50 timed runs (`bench/common.py::time_kernel`). Inputs are built
once per cell: uniform seq_lens, shuffled non-dense block table (pool factor
1.1 in the matrix), scale 1/√d. Before any timing, both our kernel and the
baseline are spot-checked against the fp32 reference on the same inputs.

**Baseline fairness** (k1-007). FlashInfer is timed on `run()` only —
`plan()` is per-batch-shape CPU scheduling prep that production serving
amortizes across decode steps. Symmetrically, our v2 workspace allocation is
excluded (PyTorch caching allocator, warmed before timing). FlashInfer uses
`use_tensor_cores=True` (its recommended setting for GQA group size ≥ 4,
i.e. the production configuration), and our NHD separate-K/V layout feeds it
zero-copy, so no conversion cost exists to place on either side.
`ratio_vs_flashinfer` = FlashInfer median / kernel median; > 1 means faster
than the baseline.

**Profiling.** NCU on the isolated kernel (v2: partial kernel, with the
merge reported as a trailing column) extracting DRAM% of peak, L1TEX
sectors/request, achieved occupancy and limiter, top stalls, L2 hit rate,
and DRAM/L2 byte counters (`analysis/ncu/run_ncu.sh`, `extract.py`). NCU
replays a single cold launch while the bench reports a warmed median —
trends are compared within one tool, never absolutes across tools.

**Correctness harness.** 32 GPU tests (`tests/test_gqa_decode.py`) over all
three kernels: ragged batches, S=1, exact and partial last blocks, block
sizes 16/64, Hq 32/64, S=8192, num_splits ∈ {1, 3, 8, 32}, splits > S, and
large-logit finiteness. Tolerance atol=rtol=2e-2 (bf16 output rounding
~0.4% relative plus accumulation-order noise). All unused storage —
unallocated pool blocks and invalid tail slots of partial last blocks — is
filled with NaN, so any in-bounds-but-out-of-range read poisons the output
and fails the accuracy assert (the class of bug memcheck cannot see).

## 4. Results

### 4.1 The ladder, attributed (representative shape)

**v0 → v1 (3.93x) is almost pure warp overlap, not faster per-token work.**
v0's single serial scan spends 962 ns per token; each of v1's four
concurrent per-warp scans spends 979 ns per token — per-chain latency
unchanged within 2%. v1 wins by running four dependency chains in parallel.
DRAM traffic is 1.0x KV size in both. The sectors/request rise 1.67 → 5.67
matches the ideal mix exactly ((8+8+1)/3 for two perfect 256 B vector loads
plus a uniform block-table lookup); the long_scoreboard rise 6.72 → 11.02
cyc/inst is a denominator effect (far fewer instructions per KV byte), not a
latency-hiding regression.

**v1 → v2 (1.68x here, 5.98x at B=8/S=32768) is CTA supply.** Occupancy
47.5% → 89.6% with the limiter unchanged (registers+warps, 16 blocks/SM):
the per-CTA ceiling never moved; v1 simply undersupplied CTAs (0.48 waves →
~3.9 waves at splits=8). `not_selected` entering the top-3 stalls confirms a
surplus of runnable warps — the latency long_scoreboard used to expose is
now covered. v2(splits=1) benches at 15.91% vs v1's 15.99%: the split
machinery itself costs nothing measurable.

Splits sweep at the representative shape (% of SoL): 15.91 (1) → 24.09 (2)
→ 25.22 (4) → 26.91 (8) → 27.85 (16) → 28.24 (32) — monotonically rising
through 32; the merge kernel is 0.56% of total at splits=8, so split
overhead is nowhere near binding at this shape.

### 4.2 The matrix (64 cells) vs FlashInfer

Full grid: {32x8, 64x8} × S {512, 2048, 8192, 32768} × B {1, 8, 32, 128} ×
block {16, 64}, archived in `bench/results/matrix_20260824_213534/`. Key
readings (v2, block 16, ratio_vs_flashinfer):

- **The convergence wall:** every saturated 32Q cell converges to ~0.32
  (FlashInfer a steady ~3.1x faster once both sides have enough work); the
  64Q wall is ~0.167, half of 32Q.
- **v2/v1 gradient runs 38.6x → ~1.0x:** 38.6x at (32x8, S=32768, B=1) —
  the starvation cell v2 was built for — decaying to a ~1.0x floor at
  (64x8, S=512, large B) where the second launch + merge just breaks even.
- **argbest num_splits** is monotonically non-decreasing in S in every
  (headtype, B) column, but non-monotonic in B at fixed S (rollover
  competition), e.g. 16, 4, 8, 4 across B at 32x8/S=512.
- **Block size 16 vs 64 is performance-neutral:** median |diff| 0.49%; every
  B > 1 cell under 2%. The B=1 exceptions reflect starved-grid timing
  sensitivity, not paging cost.
- **B=1, S=512 near-tie (0.976) is a launch-overhead floor**, not a
  bandwidth result — KV SoL there is 2 µs against ~24 µs on both sides.

### 4.3 Why the wall: two mechanisms, both NCU-confirmed

**In-flight bytes (the 32Q wall).** At 901 GB/s sustained (splits=8) the
grid keeps ~0.6 MB in flight versus the ~2 MB Little's-law requirement for
3.35 TB/s. Each warp still has only one token's K row outstanding because
the online-softmax chain serializes consecutive tokens. FlashInfer's ~87%
of SoL shows the headroom; closing it needs multi-token unrolling per warp
or cp.async/TMA pipelines — out of K1's three-tier scope.

**Per-q-head L2 re-reads (the 64Q halving).** Going 32Q → 64Q at fixed KV,
FlashInfer's time barely moves (0.368 → 0.366 ms — tensor-core batching
makes extra q-heads nearly free) while our per-q-head scan doubles on-chip
work. The 3-cell NCU deep dive confirms the mechanism: DRAM bytes are flat
(+0.4%, both ~1x KV) while the partial kernel takes 1.96x longer, DRAM%
halves (24.75 → 12.71), and L2 hit rate rises 51.1% → 65.8% — the same DRAM
working set moved twice through the on-chip hierarchy. (L2 bytes grew 1.43x
rather than ~2x; L1 MSHR merging of concurrent same-sector misses is
recorded as a plausible, unverified mechanism.) k1-001's rejection of
(b, kv-head) regrouping covered only the DRAM dimension; regrouping or
tensor-core in-group batching would attack this L2 wall and is future work.

**Cost of splitting hard (B=1 rescue).** At (32x8, S=32768, B=1,
splits=64): 2048 CTAs reach 71.9% occupancy and 17.95% DRAM, but L2 hit
drops to 13.1% (64 segments share nothing) and the merge is ~11% of the
total — consistent with argbest stopping at 64.

## 5. Correctness and robustness

All 32 tests pass on both environments. The four compute-sanitizer tools
were each run over the full test suite (all three kernels plus the v2 merge
kernel, all edge cases) with `--error-exitcode 1`:

| Tool | Result |
|---|---|
| memcheck | 0 errors |
| racecheck | 0 hazards (0 errors, 0 warnings) |
| initcheck | 0 errors |
| synccheck | 0 errors |

Logs and the exact environment are in `docs/k1/sanitizer/`. Known
restriction (k1-004): `seq_len == 0` is unsupported (normalization divides
by the softmax sum); edge-case coverage starts at S=1.

## 6. Limitations and future work

- **No tensor-core path / q-head batching**: the per-q-head scalar scan is
  what halves 64Q throughput (L2 re-read wall) and caps saturated cells at
  ~0.32x FlashInfer. In-group batching à la FlashInfer is the single
  highest-leverage change.
- **In-flight bytes**: multi-token unroll per warp and cp.async/TMA
  double-buffering to close the Little's-law gap (~0.6 MB vs ~2 MB).
- **num_splits is swept, not planned**: a production kernel needs a
  heuristic (the matrix's argbest table is the data for one).
- **Uniform seq_lens in the bench matrix**: the correctness suite covers
  ragged batches, but matrix timing does not exercise varlen load balance.
- **K1 caps the ladder at three tiers by design**; the above are candidates
  for the MLA/DSA phases, where the MLA layout (d=576/dv=512, single latent
  head) changes the arithmetic-intensity picture entirely.

## Appendix: artifact map

| Artifact | Location |
|---|---|
| Kernels | `kernels/gqa_decode/v{0,1,2}*.cu` |
| Bindings / reference | `python/` |
| Tests (32) | `tests/test_gqa_decode.py` |
| Bench harness + results | `bench/`, `bench/results/matrix_20260824_213534/` |
| NCU scripts + extractions | `analysis/ncu/` |
| Working notes (W1–W3) | `docs/k1/notes_w1.md` |
| Decision records | `docs/decisions/k1-001..007` |
| Sanitizer runs | `docs/k1/sanitizer/` |
| Environment gate | `docs/g0/versions.txt` |
