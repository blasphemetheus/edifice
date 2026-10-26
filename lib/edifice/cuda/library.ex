defmodule Edifice.CUDA.Library do
  @moduledoc """
  Loads Edifice's CUDA FFI handlers into EXLA before custom-call compilation.

  Build with `mix edifice.cuda.build --arch sm_120` (choose your GPU target).
  Loading is opt-in through `EDIFICE_FUSED_CUSTOM_CALL=1`. By default the
  library is `priv/cuda/libedifice_xla.so`; set `EDIFICE_CUDA_LIBRARY` to an
  absolute path to use a separately built library. For a legacy EXLA build
  with Edifice handlers already linked, set it to `linked` instead.

  A loaded library remains resident for the VM's lifetime. Restart the VM
  after rebuilding kernels. EXLA remains optional for CPU consumers.
  """

  def ensure_loaded! do
    case :persistent_term.get(__MODULE__, nil) do
      :loaded ->
        :ok

      nil ->
        :global.trans(
          {__MODULE__, self()},
          fn ->
            case :persistent_term.get(__MODULE__, nil) do
              :loaded -> :ok
              nil -> load!()
            end
          end,
          [node()]
        )
    end
  end

  defp load! do
    # EXLA loads these handlers with RTLD_GLOBAL. Their XLA dependency exports
    # crypto symbols that can interfere with OTP crypto's lazy NIF load. Force
    # that NIF to bind first; merely starting the :crypto application does not
    # ensure that its module (and @on_load callback) has run.
    Code.ensure_loaded!(:crypto)

    case System.get_env("EDIFICE_CUDA_LIBRARY") do
      "linked" ->
        :ok

      override ->
        path =
          override || Path.join(to_string(:code.priv_dir(:edifice)), "cuda/libedifice_xla.so")

        unless Path.type(path) == :absolute and File.regular?(path),
          do:
            raise(
              "Edifice CUDA library not found at #{path}. Run mix edifice.cuda.build --arch <GPU target> before enabling EDIFICE_FUSED_CUSTOM_CALL=1."
            )

        case load_dylib(path) do
          :ok -> :ok
          other -> raise "Could not load Edifice CUDA library: #{inspect(other)}"
        end
    end

    :persistent_term.put(__MODULE__, :loaded)
    :ok
  end

  # EXLA 1.0 exposes `EXLA.load_dylib/1`; older builds only had the NIF.
  defp load_dylib(path) do
    cond do
      Code.ensure_loaded?(EXLA) and function_exported?(EXLA, :load_dylib, 1) ->
        apply(EXLA, :load_dylib, [path])

      Code.ensure_loaded?(EXLA.NIF) and function_exported?(EXLA.NIF, :load_dylib, 1) ->
        apply(EXLA.NIF, :load_dylib, [path])

      true ->
        raise "This EXLA build does not support loading CUDA libraries (needs EXLA.load_dylib/1)"
    end
  end
end
