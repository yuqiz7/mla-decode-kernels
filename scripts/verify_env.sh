#!/usr/bin/env bash
# Environment gate check. Compares live versions against docs/g0/versions.txt
# (mismatches WARN but do not fail), then hard-checks the things that must
# work: NCU profiling as a normal user, and flashinfer import.
#
# Exit 0 with "VERIFY OK" on success; exit 1 on any hard failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSIONS_FILE="$REPO_ROOT/docs/g0/versions.txt"

# docs/g0/versions.txt line layout:
#   1: <driver>, <gpu name>
#   2: nvcc "Build ..." line
#   3: torch <ver> cuda <ver>
#   4: ncu "Version ..." line
#   5: OS description
EXP_DRIVER="$(sed -n '1p' "$VERSIONS_FILE" | cut -d',' -f1)"
EXP_NVCC="$(sed -n '2p' "$VERSIONS_FILE")"
EXP_TORCH="$(sed -n '3p' "$VERSIONS_FILE" | awk '{print $2}')"
EXP_TORCH_CUDA="$(sed -n '3p' "$VERSIONS_FILE" | awk '{print $4}')"
EXP_NCU="$(sed -n '4p' "$VERSIONS_FILE")"

check() { # check <label> <expected> <actual>
    if [ "$2" = "$3" ]; then
        echo "[verify] $1: $3 (matches docs/g0)"
    else
        echo "[verify] WARN: $1 mismatch: expected '$2', got '$3'"
    fi
}

# --- Step 1: version comparison (WARN only) -----------------------------------
ACT_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
check "driver" "$EXP_DRIVER" "$ACT_DRIVER"

ACT_NVCC="$(nvcc --version | grep '^Build' || true)"
check "nvcc" "$EXP_NVCC" "$ACT_NVCC"

ACT_TORCH="$(python3 -c 'import torch; print(torch.__version__)')"
ACT_TORCH_CUDA="$(python3 -c 'import torch; print(torch.version.cuda)')"
check "torch" "$EXP_TORCH" "$ACT_TORCH"
check "torch cuda" "$EXP_TORCH_CUDA" "$ACT_TORCH_CUDA"

ACT_NCU="$(ncu --version | grep '^Version' || true)"
check "ncu" "$EXP_NCU" "$ACT_NCU"

# --- Step 2: NCU profiling as a normal user -----------------------------------
NCU_TMP="$(mktemp -d /tmp/verify_env.XXXXXX)"
trap 'rm -rf "$NCU_TMP"' EXIT

cat > "$NCU_TMP/probe.cu" <<'EOF'
#include <cstdio>
__global__ void probe(float* x) { x[threadIdx.x] += 1.0f; }
int main() {
    float* x;
    cudaMalloc(&x, 128 * sizeof(float));
    probe<<<1, 128>>>(x);
    cudaError_t err = cudaDeviceSynchronize();
    printf("probe: %s\n", cudaGetErrorString(err));
    return err == cudaSuccess ? 0 : 1;
}
EOF

echo "[verify] compiling minimal CUDA kernel (-arch=sm_90a)"
nvcc -arch=sm_90a "$NCU_TMP/probe.cu" -o "$NCU_TMP/a.out"

echo "[verify] running ncu --section SpeedOfLight as $(whoami)"
NCU_OUT="$(cd "$NCU_TMP" && ncu --section SpeedOfLight ./a.out 2>&1 || true)"
if echo "$NCU_OUT" | grep -q ERR_NVGPUCTRPERM; then
    echo "$NCU_OUT" | grep -m1 ERR_NVGPUCTRPERM
    echo "[verify] FAIL: NCU profiling blocked for non-admin users" \
         "(check /etc/modprobe.d/nvidia-profiling.conf and reboot)"
    exit 1
fi
if ! echo "$NCU_OUT" | grep -q "probe"; then
    echo "$NCU_OUT" | tail -20
    echo "[verify] FAIL: ncu run did not produce expected kernel output"
    exit 1
fi
echo "[verify] ncu profiling works as normal user"

# --- Step 3: flashinfer import -------------------------------------------------
echo "[verify] importing flashinfer"
python3 -c "import flashinfer; print('[verify] flashinfer', flashinfer.__version__)"

echo "VERIFY OK"
