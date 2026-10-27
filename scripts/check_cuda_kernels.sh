#!/usr/bin/env bash
# Build Edifice's CUDA custom-call library and (optionally) run the GPU
# parity tests. This is the job GitHub's CPU runners cannot do; run it
# locally or from .github/workflows/cuda.yml on a self-hosted GPU runner.
#
#   scripts/check_cuda_kernels.sh            # compile-only (no GPU needed)
#   scripts/check_cuda_kernels.sh --test     # + GPU parity tests
#
# Env:
#   EXLA_GPU_ARCH   CUDA target (default sm_75; devenv sets sm_120 for RTX 5090)
#   MIX_ENV         defaults to test (the matrix is a test; priv/ is shared)
set -euo pipefail

cd "$(dirname "$0")/.."
ARCH="${EXLA_GPU_ARCH:-sm_75}"
export MIX_ENV="${MIX_ENV:-test}"
RUN_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --test) RUN_TESTS=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$*"; }

step "toolchain"
command -v nvcc >/dev/null || { echo "nvcc not found" >&2; exit 1; }
nvcc --version | tail -1
echo "GPU_ARCH=$ARCH MIX_ENV=$MIX_ENV"

step "deps + compile"
mix deps.get
mix compile --warnings-as-errors

step "syntax-check the kernel sources"
make -C native/cuda test-compile GPU_ARCH="$ARCH" -j"$(nproc)"

step "EXLA FFI handlers -> priv/cuda/libedifice_xla.so"
mix edifice.cuda.build --arch "$ARCH"

if [ "$RUN_TESTS" = 1 ]; then
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  step "GPU parity matrix (custom-call tier, op level and models)"
  EDIFICE_TEST_CLIENT=cuda mix test --only cuda_kernels
  step "custom-call f64-truth test"
  EDIFICE_TEST_CLIENT=cuda EDIFICE_FUSED_CUSTOM_CALL=1 \
    mix test test/edifice/cuda/fused_custom_call_gpu_test.exs --include exla_only
else
  echo
  echo "compile-only run finished; add --test on a GPU host to run the parity tests"
fi
