#!/usr/bin/env bash
# Backward-compatible forwarder; profiling lives in run_ncu.sh.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_ncu.sh" v0
