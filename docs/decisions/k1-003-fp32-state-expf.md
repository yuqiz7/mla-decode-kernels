# K1-003: fp32 softmax state and accurate expf

## Decision

The online-softmax state (running max m, running sum l, output accumulator
acc) is kept in fp32, and the accurate `expf` is used.

## Why

v0's job is to match the fp32 reference bit-for-close; fp32 state and accurate
`expf` remove numerics as a variable while the algorithm is being validated.

## Rejected

bf16 state and `__expf` (fast approximate exponential).

## Trade-off

v0 prioritizes matching the fp32 reference; speed is v1's job. If profiling
later shows `expf` or fp32 accumulation on the critical path, that becomes a
measured ablation rather than a day-one guess.
