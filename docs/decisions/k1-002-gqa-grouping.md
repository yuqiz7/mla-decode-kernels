# K1-002: GQA head grouping rule

## Decision

`kv_head = q_head // (Hq / Hkv)` — consecutive grouping, matching Hugging Face
`repeat_kv` semantics.

## Why

This is the convention used by the HF reference implementations our tests
compare against, so correctness checks are apples-to-apples.

## Rejected

Strided/interleaved mapping (e.g. `kv_head = q_head % Hkv`).

## Trade-off

The reference implementation, the kernel, and the FlashInfer baseline must all
use this same rule; a mismatch anywhere produces wrong-but-plausible outputs
that only differ on grouped heads.
