{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

let
  cuda = pkgs.cudaPackages;
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    erlang_27
    elixir_1_18
    git
    tmux

    # CUDA
    cuda.cuda_nvcc
    cuda.cuda_nvrtc
    cuda.cuda_cudart
    cuda.cudnn
    cuda.libcublas
    cuda.libcusolver
    cuda.libcufft
    cuda.libcusparse
    cuda.libcurand
    cuda.libnvjitlink
    cuda.nccl
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    cuda.cuda_cudart
    cuda.cuda_nvrtc
    cuda.cudnn
    cuda.libcublas
    cuda.libcusolver
    cuda.libcufft
    cuda.libcusparse
    cuda.libcurand
    cuda.libnvjitlink
    cuda.nccl
    "/usr/lib/wsl"
  ];

  shellHook = ''
    export MIX_HOME="$PWD/.nix-mix"
    export HEX_HOME="$PWD/.nix-hex"
    export ERL_AFLAGS="-kernel shell_history enabled"
    mkdir -p "$MIX_HOME" "$HEX_HOME"
    export PATH="$MIX_HOME/bin:$MIX_HOME/escripts:$HEX_HOME/bin:$PATH"

    # EXLA CUDA target
    export EXLA_TARGET=cuda
    export XLA_TARGET=cuda12
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=${cuda.cuda_nvcc}"

    # XLA prebuilt binaries need NCCL 2.27+ and NVSHMEM 3.x (newer than nixpkgs)
    # Downloaded from PyPI: nvidia-nccl-cu12, nvidia-nvshmem-cu12
    export NCCL_LIB="$PWD/.nccl/nvidia/nccl/lib"
    export NVSHMEM_LIB="$PWD/.nvshmem/nvidia/nvshmem/lib"
    export CUDA_COMPAT="$PWD/.cuda-compat"
    export LD_LIBRARY_PATH="$NCCL_LIB:$NVSHMEM_LIB:$CUDA_COMPAT:$LD_LIBRARY_PATH"

    # EXLA 0.11+ CallbackServer spawns many processes during JIT
    export ERL_FLAGS="''${ERL_FLAGS:-} +P 4000000"
  '';
}
