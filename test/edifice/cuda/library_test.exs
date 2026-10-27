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

  @tag :exla_only
  test "a cold VM loads crypto before globally loading CUDA handlers" do
    library = Path.expand("../../../priv/cuda/libedifice_xla.so", __DIR__)
    assert File.regular?(library), "build the CUDA handler library before running this test"

    # Use Erlang, not the Elixir CLI/compiler: compiling an Elixir script can
    # itself load crypto and accidentally hide this load-order regression.
    script = """
    application:ensure_all_started(elixir),
    false = code:is_loaded(crypto),
    ok = 'Elixir.Edifice.CUDA.Library':'ensure_loaded!'(),
    {file, _} = code:is_loaded(crypto),
    16 = byte_size(crypto:strong_rand_bytes(16)),
    ok = 'Elixir.Edifice.CUDA.Library':'ensure_loaded!'(),
    io:format("crypto remains usable after CUDA handlers~n"),
    halt().
    """

    args =
      ["+S", "2:2", "-noshell", "-pa"] ++
        Enum.map(:code.get_path(), &to_string/1) ++
        ["-pa", Path.dirname(to_string(:code.which(Edifice.CUDA.Library))), "-eval", script]

    {output, status} =
      System.cmd(Path.join(to_string(:code.root_dir()), "bin/erl"), args,
        stderr_to_stdout: true,
        env: [{"EDIFICE_CUDA_LIBRARY", library}, {"ERL_CRASH_DUMP", "/dev/null"}]
      )

    assert status == 0, output
    assert output =~ "crypto remains usable after CUDA handlers"
  end

  # The handlers read and write row-major and cannot see layouts; without
  # these attributes XLA's layout assignment may hand them e.g. a T-major
  # GRU result (corrupt at batch > 1). `linked` skips loading so this runs
  # on CPU-only CI.
  test "every native op pins row-major layouts on its operands and results" do
    System.put_env("EDIFICE_CUDA_LIBRARY", "linked")
    client = %{platform: :cuda}
    {b, t, h, s} = {2, 8, 16, 8}
    wx = Nx.template({b, t, 3 * h}, :f32)
    r = Nx.template({h, 3 * h}, :f32)
    h0 = Nx.template({b, h}, :f32)
    bhn = Nx.template({h}, :f32)
    x = Nx.template({b, t, h}, :f32)
    a = Nx.template({h, s}, :f32)
    bc = Nx.template({b, t, s}, :f32)

    cases = [
      {:fused_gru_scan, [wx, r, h0, bhn], x,
       "[dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, " <>
         "dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>]",
       "[dense<[2, 1, 0]> : tensor<3xindex>]"},
      {:fused_gru_scan_backward, [wx, r, h0, bhn, x, x], {wx, wx, h0},
       "[dense<[2, 1, 0]> : tensor<3xindex>, dense<[1, 0]> : tensor<2xindex>, " <>
         "dense<[1, 0]> : tensor<2xindex>, dense<[0]> : tensor<1xindex>, " <>
         "dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]",
       "[dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, " <>
         "dense<[1, 0]> : tensor<2xindex>]"},
      {:fused_selective_scan, [x, x, a, bc, bc], x,
       "[dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, " <>
         "dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, " <>
         "dense<[2, 1, 0]> : tensor<3xindex>]", "[dense<[2, 1, 0]> : tensor<3xindex>]"},
      {:fused_selective_scan_backward, [x, x, a, bc, bc, x], {x, x, bc, bc},
       "[dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, " <>
         "dense<[1, 0]> : tensor<2xindex>, dense<[2, 1, 0]> : tensor<3xindex>, " <>
         "dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]",
       "[dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>, " <>
         "dense<[2, 1, 0]> : tensor<3xindex>, dense<[2, 1, 0]> : tensor<3xindex>]"}
    ]

    assert Enum.map(cases, &elem(&1, 0)) |> Enum.sort() ==
             Enum.sort(Edifice.Block.native_ops())

    for {op, args, out, operand_layouts, result_layouts} <- cases do
      assert {:ok, %EXLA.CustomCall.Spec{attributes: attrs}} =
               EXLA.CustomCall.call(%FusedOp{op: op}, out, args, client),
             "#{op}"

      assert attrs == [
               {"operand_layouts", operand_layouts},
               {"result_layouts", result_layouts}
             ],
             "#{op}"
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
