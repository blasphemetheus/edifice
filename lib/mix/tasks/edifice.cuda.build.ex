defmodule Mix.Tasks.Edifice.Cuda.Build do
  use Mix.Task

  @shortdoc "Build Edifice's EXLA FFI CUDA library"
  @moduledoc """
  Builds the f32 GRU and selective-scan forward/backward handlers.

      mix edifice.cuda.build --arch sm_120

  Requires nvcc and an installed EXLA with `EXLA.load_dylib/1` (EXLA 1.0+).
  `--arch` defaults to `EXLA_GPU_ARCH`, then `sm_75`. The build uses the
  headers and library from the resolved EXLA dependency's XLA installation.
  Enable loading with `EDIFICE_FUSED_CUSTOM_CALL=1` before starting the VM.
  """
  def run(args) do
    {opts, rest, invalid} = OptionParser.parse(args, strict: [arch: :string])
    if rest != [] or invalid != [], do: Mix.raise("Usage: mix edifice.cuda.build [--arch sm_120]")
    arch = opts[:arch] || System.get_env("EXLA_GPU_ARCH") || "sm_75"

    unless Regex.match?(~r/^sm_[0-9]+[a-z]?$/, arch),
      do: Mix.raise("Invalid CUDA architecture: #{arch}")

    Mix.Task.run("compile")

    exla =
      Mix.Project.deps_paths()[:exla] ||
        Mix.raise("EXLA must be installed to build CUDA handlers")

    xla = Path.join(exla, "cache/xla_extension")

    unless File.exists?(Path.join(xla, "include/xla/ffi/api/ffi.h")),
      do: Mix.raise("XLA headers are missing; compile the EXLA dependency first")

    root = Path.expand("../../..", __DIR__)
    output_dir = Application.app_dir(:edifice, "priv/cuda")
    # Separate object caches across XLA installations/header revisions.
    headers = Path.wildcard(Path.join(xla, "include/xla/ffi/api/*.h")) |> Enum.sort()

    fingerprint =
      :crypto.hash(:sha256, [xla | Enum.map(headers, &File.read!/1)])
      |> Base.encode16(case: :lower)

    {_, status} =
      System.cmd(
        "make",
        [
          "-C",
          Path.join(root, "native/cuda"),
          "ffi",
          "GPU_ARCH=#{arch}",
          "XLA_EXTENSION_DIR=#{xla}",
          "BUILD_DIR=#{output_dir}",
          "FFI_DIR=build/ffi/#{arch}/#{fingerprint}",
          "-j2"
        ],
        into: IO.stream()
      )

    if status != 0, do: Mix.raise("CUDA FFI build failed (exit #{status})")
    Mix.shell().info("Built #{Path.join(output_dir, "libedifice_xla.so")}")
  end
end
