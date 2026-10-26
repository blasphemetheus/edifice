#!/usr/bin/env bash
# Build every CUDA artefact Edifice ships and (optionally) run the GPU parity
# matrix. This is the job GitHub's CPU runners cannot do; run it locally or
# from .github/workflows/cuda.yml on a self-hosted GPU runner.
#
#   scripts/check_cuda_kernels.sh            # compile-only (no GPU needed)
#   scripts/check_cuda_kernels.sh --test     # + GPU parity matrix
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

step "syntax-check every .cu (incl. test/bench harnesses)"
make -C native/cuda test-compile GPU_ARCH="$ARCH" -j"$(nproc)"

step "kernel libraries (f32 + bf16) -> priv/cuda"
make -C native/cuda all GPU_ARCH="$ARCH" -j"$(nproc)"

step "NIF bridge -> priv/libedifice_cuda_nif.so"
make -C c_src

step "EXLA FFI handlers -> priv/cuda/libedifice_xla.so"
mix edifice.cuda.build --arch "$ARCH"

step "standalone kernel harness (test_kernels)"
make -C native/cuda test-bin GPU_ARCH="$ARCH" -j"$(nproc)"

if [ "$RUN_TESTS" = 1 ]; then
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  step "GPU parity matrix (NIF, custom-call, bf16)"
  EDIFICE_TEST_CLIENT=cuda mix test --only cuda_kernels
  step "legacy custom-call GPU test"
  EDIFICE_TEST_CLIENT=cuda EDIFICE_FUSED_CUSTOM_CALL=1 \
    mix test test/edifice/cuda/fused_custom_call_gpu_test.exs --include exla_only
else
  echo
  echo "compile-only run finished; add --test on a GPU host to run the parity matrix"
fi
