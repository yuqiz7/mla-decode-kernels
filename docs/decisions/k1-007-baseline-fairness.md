# K1-007: baseline timing fairness — run() only

## Decision

Baseline timing includes FlashInfer's `run()` only. `plan()` — CPU-side
scheduling preparation — is excluded, mirroring how production serving
amortizes it across decode steps of the same batch. Our v2's workspace
`torch.empty` is excluded analogously (PyTorch caching allocator, warmed
before timing). Both sides get the same inputs and the same scale, and both
are verified against the same fp32 reference before any timing; layout and
page-table conversion happen outside the timed region (the NHD separate-K/V
layout feeds FlashInfer zero-copy, so there is no conversion cost to place
anywhere).

## Rejected

- Including `plan()` in the timed region: penalizes the baseline for a
  per-step cost production does not pay.
- cudaGraph wrapping of both sides: out of K1 scope.

## Trade-off

run-only timing slightly flatters both sides equally (launch overheads
remain included; per-step host prep does not). Documented here so the matrix
numbers are reproducible under the same contract.
