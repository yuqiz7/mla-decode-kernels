# K1 compute-sanitizer runs

## Environment (RunPod H100 SXM pod — sanitizer runs only)

The four compute-sanitizer tool runs recorded in this directory were done on a
**RunPod H100 SXM pod**, not the Lambda instance recorded in
`docs/g0/versions.txt`. Every benchmark and NCU number in this repository
(bench/results, analysis/ncu, docs/k1/notes_w1.md, the report) comes from the
Lambda environment in `docs/g0/versions.txt`; nothing in this directory is a
performance measurement, so the environment difference does not affect any
reported number.

Captured on this pod (2026-08-26):

```
$ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
NVIDIA H100 80GB HBM3, 570.211.01

$ nvcc --version | grep Build
Build cuda_12.8.r12.8/compiler.35583870_0

$ python -c "import torch;print(torch.__version__, torch.version.cuda)"
2.8.0+cu128 12.8

$ compute-sanitizer --version
NVIDIA (R) Compute Sanitizer
Copyright (c) 2020-2025 NVIDIA Corporation
Version 2025.1.0.0 (build 35583870) (public-release)
```

Differences vs the Lambda environment (`docs/g0/versions.txt`): driver
570.211.01 vs 580.105.08 and torch 2.8.0 vs 2.7.0 (the `Version 2025.1.1.0`
line in versions.txt is Nsight Compute, which is unavailable on this pod and
not needed for sanitizer runs). The CUDA toolkit build
(cuda_12.8.r12.8/compiler.35583870_0) is identical, so the kernels compile to
the same SASS as on Lambda.

## What was run

Each of the four compute-sanitizer tools was run over the full GPU test suite
(`tests/test_gqa_decode.py`, 32 tests), which exercises all three kernels
(v0, v1, v2 including its merge kernel) across the edge-case grid: partial
last blocks, S=1, exact-block lengths, block sizes 16/64, Hq 32/64, S=8192,
num_splits ∈ {1, 3, 8, 32}, splits > S, and large-logit inputs. The command
for each tool was:

```
compute-sanitizer --tool <tool> --error-exitcode 1 \
    python -m pytest tests/test_gqa_decode.py -q
```

## Results — all four tools clean

| Tool | Result | Wall time | Log |
|---|---|---|---|
| memcheck | 0 errors | 8.6 s | `memcheck.log` |
| racecheck | 0 hazards (0 errors, 0 warnings) | 149.2 s | `racecheck.log` |
| initcheck | 0 errors | 5.1 s | `initcheck.log` |
| synccheck | 0 errors | 7.8 s | `synccheck.log` |

All 32 tests passed under every tool (so the numerical checks against the
reference also held while instrumented), and every run exited 0 with
`--error-exitcode 1` set.

Note the tests' NaN-poisoning discipline complements memcheck here: memcheck
catches reads outside allocations, while the tests fill every unallocated
pool block and every invalid tail slot with NaN — so an in-bounds but
out-of-range read (the kind memcheck cannot see, e.g. reading slot `rem` of a
partial last block) poisons the output and fails the accuracy assert instead.
