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
  '';
}
