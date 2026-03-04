{ pkgs, lib, ... }:

let
  cuda = pkgs.cudaPackages;
in
{
  # Elixir + Erlang
  languages.elixir = {
    enable = true;
    package = pkgs.elixir_1_18;
  };
  languages.erlang = {
    enable = true;
    package = pkgs.erlang_27;
  };

  # Rust (for ortex/rustler NIFs)
  languages.rust.enable = true;

  # Python (for PyTorch fixture generation scripts)
  languages.python = {
    enable = true;
    package = pkgs.python312;
    venv.enable = true;
    venv.requirements = ''
      torch
      safetensors
      packaging
      numpy
    '';
  };

  # System packages
  packages = [
    pkgs.git
    pkgs.tmux

    # CUDA
    cuda.cuda_nvcc
    cuda.cuda_nvrtc
    cuda.cuda_cudart
    cuda.cuda_cccl
    cuda.cudnn
    cuda.libcublas
    cuda.libcusolver
    cuda.libcufft
    cuda.libcusparse
    cuda.libcurand
    cuda.libnvjitlink
    cuda.nccl
  ];

  # Environment variables
  env = {
    ERL_AFLAGS = "-kernel shell_history enabled";
    EXLA_TARGET = "cuda";
    XLA_TARGET = "cuda12";
    XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cuda.cuda_nvcc} --xla_gpu_per_fusion_autotune_cache_dir=$PWD/.cache/xla_autotune";
    ERL_FLAGS = "+P 4000000";
  };

  # Library path for CUDA
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
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
  ];

  # Shell setup
  enterShell = ''
    export MIX_HOME="$PWD/.nix-mix"
    export HEX_HOME="$PWD/.nix-hex"
    mkdir -p "$MIX_HOME" "$HEX_HOME"
    mkdir -p "$PWD/.cache/xla_autotune"
    export PATH="$MIX_HOME/bin:$MIX_HOME/escripts:$HEX_HOME/bin:$PATH"

    # XLA prebuilt binaries need NCCL 2.27+ and NVSHMEM 3.x
    export NCCL_LIB="$PWD/.nccl/nvidia/nccl/lib"
    export NVSHMEM_LIB="$PWD/.nvshmem/nvidia/nvshmem/lib"
    export CUDA_COMPAT="$PWD/.cuda-compat"
    export LD_LIBRARY_PATH="$NCCL_LIB:$NVSHMEM_LIB:$CUDA_COMPAT:$LD_LIBRARY_PATH"
  '';

}
