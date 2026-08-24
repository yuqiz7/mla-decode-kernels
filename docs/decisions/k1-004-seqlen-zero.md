# K1-004: seq_len == 0 unsupported

## Decision

`seq_len == 0` is unsupported: the final normalization divides by the softmax
sum, which is zero when no key was processed.

## Why

Supporting it would require a special-case output definition (zeros? skip?)
that no consumer of the kernel currently needs.

## Rejected

Defining and handling a zero-length convention in the kernel.

## Trade-off

The harness does not test `seq_len == 0`; edge-case tests start at
`seq_len = 1`. If a serving integration later batches empty sequences, this
decision must be revisited before that integration.
