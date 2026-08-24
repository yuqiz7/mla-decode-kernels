# K1-001: v0 CTA mapping

## Decision

v0 maps one CTA to one (batch b, q-head h) pair, with 128 threads per CTA;
thread j owns head dimension j.

## Why

The simplest mapping to reason about and verify: every CTA runs the full
online-softmax loop over its own sequence independently, with no inter-head
coordination and a trivial thread-to-dimension assignment.

## Rejected

One CTA per (b, kv-head) computing its 4 q-heads together. Deferred to v1 as a
measurable step, not discarded — it amortizes K/V reads across the group.

## Trade-off

The 4 q-heads sharing a kv-head each re-read the same K/V (possibly absorbed
by L2). We accept the redundant traffic in v0 because it is the simplest
version to verify; the v0 → v1 delta then isolates the value of grouping.
