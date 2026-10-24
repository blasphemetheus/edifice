defmodule Edifice.CUDA.LibraryTest do
  use ExUnit.Case, async: false

  alias Edifice.Block.FusedOp

  setup do
    variables = ["EDIFICE_FUSED_CUSTOM_CALL", "EDIFICE_CUDA_LIBRARY"]
    saved = Map.new(variables, &{&1, System.get_env(&1)})
    loaded = :persistent_term.get(Edifice.CUDA.Library, nil)
    :persistent_term.erase(Edifice.CUDA.Library)
    System.put_env("EDIFICE_FUSED_CUSTOM_CALL", "1")

    System.put_env(
      "EDIFICE_CUDA_LIBRARY",
      "/tmp/edifice-absent-#{System.unique_integer([:positive])}.so"
    )

    on_exit(fn ->
      for {key, value} <- saved do
        if value, do: System.put_env(key, value), else: System.delete_env(key)
      end

      if loaded,
        do: :persistent_term.put(Edifice.CUDA.Library, loaded),
        else: :persistent_term.erase(Edifice.CUDA.Library)
    end)
  end

  test "supported CUDA calls give an actionable error when the library is missing" do
    assert_raise RuntimeError, ~r/mix edifice.cuda.build/, fn ->
      EXLA.CustomCall.call(%FusedOp{op: :fused_gru_scan}, nil, [Nx.template({1, 4, 24}, :f32)], %{
        platform: :cuda
      })
    end
  end

  test "disabled, host, unsupported dtype and oversized calls do not require a library" do
    block = %FusedOp{op: :fused_gru_scan}
    args = [Nx.template({1, 4, 24}, :f32)]
    assert :skip == EXLA.CustomCall.call(block, nil, args, %{platform: :host})

    assert :skip ==
             EXLA.CustomCall.call(block, nil, [Nx.template({1, 4, 24}, :bf16)], %{platform: :cuda})

    assert :skip ==
             EXLA.CustomCall.call(block, nil, [Nx.template({1, 4, 3075}, :f32)], %{
               platform: :cuda
             })

    assert :skip ==
             EXLA.CustomCall.call(
               %FusedOp{op: :fused_selective_scan},
               nil,
               [Nx.template({1, 4, 8}, :f32), nil, Nx.template({8, 33}, :f32)],
               %{platform: :cuda}
             )

    assert :skip ==
             EXLA.CustomCall.call(
               %FusedOp{op: :fused_gru_scan_backward},
               nil,
               [Nx.template({1, 257, 24}, :f32)],
               %{platform: :cuda}
             )

    System.delete_env("EDIFICE_FUSED_CUSTOM_CALL")
    assert :skip == EXLA.CustomCall.call(block, nil, args, %{platform: :cuda})
  end
end
