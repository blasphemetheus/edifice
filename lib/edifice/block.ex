defmodule Edifice.Block do
  # EXLA is optional; retain undefined-function checks when it is installed.
  unless Code.ensure_loaded?(EXLA) do
    @compile {:no_warn_undefined, [EXLA.CustomCall]}
  end

  @moduledoc """
  Extensible-computation blocks for edifice's fused kernels, on the
  sanctioned `Nx.block/4` + `EXLA.CustomCall` protocol (nx PR #1709 /
  #1739) — the replacement for the removed `Nx.Shared.optional/4`
  (migration eval: exphil docs/planning/EVAL_SCAN_MIGRATION_2026-07-17).

  Every fused op shares one struct, `Edifice.Block.FusedOp` — `op` names
  the kernel (e.g. `:fused_mingru_scan`), `attrs` carries static kernel
  config. With no `defimpl EXLA.CustomCall, for: FusedOp`, the
  protocol's `Any` fallback returns `:skip` and EXLA compiles the pure
  default fun in-graph — identical semantics to the old `optional/4`
  fallback path, but on a supported API.

  The native tier uses a
  `defimpl EXLA.CustomCall, for: FusedOp` that pattern-matches `op` and
  returns `{:ok, %EXLA.CustomCall.Spec{call_target_name: ...}}` on CUDA
  after loading Edifice's FFI library, and `:skip` elsewhere.
  Build with `mix edifice.cuda.build` and opt in with
  `EDIFICE_FUSED_CUSTOM_CALL=1`. `native_impl?/0` reports protocol
  availability, not whether the shared library has been built or loaded.
  """

  defmodule FusedOp do
    @moduledoc "Identifies one fused kernel invocation for `Nx.block/4`."
    defstruct [:op, attrs: %{}]
  end

  @doc """
  Run `op` through `Nx.block/4` with `fallback` as the default
  implementation. `fallback` keeps `optional/4`'s convention (arity =
  length(args); no struct argument) so callsites migrate mechanically.
  """
  def run(op, args, output, fallback, attrs \\ %{})
      when is_atom(op) and is_list(args) and is_function(fallback) do
    Nx.block(%FusedOp{op: op, attrs: attrs}, args, output, wrap(fallback, length(args)))
  end

  @doc """
  Whether a real (non-Any) `EXLA.CustomCall` implementation is
  registered for `FusedOp` — i.e. an in-graph native tier exists.
  """
  def native_impl? do
    Code.ensure_loaded?(EXLA.CustomCall) and
      EXLA.CustomCall.impl_for(%FusedOp{op: :probe}) != EXLA.CustomCall.Any
  rescue
    _ -> false
  end

  # Nx.block's default fun receives the struct first and demands exact
  # arity; the old fallbacks don't take a struct. Bridge per arity.
  defp wrap(f, 1), do: fn _s, a -> f.(a) end
  defp wrap(f, 2), do: fn _s, a, b -> f.(a, b) end
  defp wrap(f, 3), do: fn _s, a, b, c -> f.(a, b, c) end
  defp wrap(f, 4), do: fn _s, a, b, c, d -> f.(a, b, c, d) end
  defp wrap(f, 5), do: fn _s, a, b, c, d, e -> f.(a, b, c, d, e) end
  defp wrap(f, 6), do: fn _s, a, b, c, d, e, g -> f.(a, b, c, d, e, g) end
  defp wrap(f, 7), do: fn _s, a, b, c, d, e, g, h -> f.(a, b, c, d, e, g, h) end
  defp wrap(f, 8), do: fn _s, a, b, c, d, e, g, h, i -> f.(a, b, c, d, e, g, h, i) end
end
