defmodule Edifice.CUDA.DispatchGatingTest do
  @moduledoc """
  Pins the tier-selection logic in `Edifice.CUDA.FusedScan`, `AutoTune`,
  `Recurrent` and `Edifice.CUDA.NIF` without needing a GPU, the NIF
  library or the FFI library.

  Three regressions this guards against (all shipped at one point):

  1. The custom-call tier was reported available for EVERY op whenever it
     was enabled, although only `Edifice.Block.native_ops/0` have handlers.
     The other ops were routed into `Nx.block` where the protocol returned
     `:skip`, i.e. they silently ran the pure fallback and shadowed the NIF.
  2. `function_exported?(Edifice.CUDA.NIF, :fused_mingru_scan, 7)` was used
     as "NIF loaded": the stubs always exist, so it was always true.
  3. `Recurrent` tested `fused_lstm_scan/7` against an arity-8 stub, so the
     fused parameter layout could never be chosen through the NIF.
  """
  use ExUnit.Case, async: false

  alias Edifice.CUDA.{AutoTune, FusedScan, NIF}

  @kernels ~w(mingru minlstm elu_gru real_gru diag_linear liquid linear_scan
    delta_net_scan gated_delta_net_scan delta_product_scan slstm_scan lstm_scan
    gru_scan ttt_scan selective_scan kda_scan rla_scan flash_attention
    laser_attention fox_attention reservoir_scan titans_scan miras_scan gsa_scan
    mingru_block minlstm_block linear_block lstm_block gru_block)a

  @block_ops ~w(fused_mingru_scan fused_minlstm_scan fused_elu_gru_scan
    fused_real_gru_scan fused_diag_linear_scan fused_liquid_scan fused_linear_scan
    fused_delta_net_scan fused_gated_delta_net_scan fused_delta_product_scan
    fused_slstm_scan fused_lstm_scan fused_gru_scan fused_ttt_scan
    fused_selective_scan fused_kda_scan fused_rla_scan fused_flash_attention
    fused_laser_attention fused_fox_attention fused_reservoir_scan
    fused_titans_scan fused_miras_scan fused_gsa_scan fused_mingru_block_scan
    fused_minlstm_block_scan fused_linear_block_scan fused_lstm_block_scan
    fused_gru_block_scan)a

  defp with_env(pairs, fun) do
    saved = Enum.map(pairs, fn {k, _} -> {k, System.get_env(k)} end)

    Enum.each(pairs, fn
      {k, nil} -> System.delete_env(k)
      {k, v} -> System.put_env(k, v)
    end)

    try do
      fun.()
    after
      Enum.each(saved, fn
        {k, nil} -> System.delete_env(k)
        {k, v} -> System.put_env(k, v)
      end)
    end
  end

  describe "FusedScan.block_op/1" do
    test "maps every dispatch kernel name onto the FusedOp atom used by Block.run" do
      mapped = Enum.map(@kernels, &FusedScan.block_op/1)
      assert mapped == @block_ops
    end

    test "is the identity on FusedOp atoms" do
      for op <- Edifice.Block.native_ops(), do: assert(FusedScan.block_op(op) == op)
    end
  end

  describe "Edifice.Block.native_ops/0" do
    test "lists exactly the handlers linked into libedifice_xla.so" do
      makefile = File.read!(Path.expand("../../../native/cuda/Makefile", __DIR__))

      # FFI_SOURCES := a.cu b.cu \\
      #                c.cu
      from_makefile =
        makefile
        |> String.split("\n")
        |> Enum.drop_while(&(not String.starts_with?(&1, "FFI_SOURCES :=")))
        |> Enum.reduce_while([], fn line, acc ->
          if String.ends_with?(line, "\\"), do: {:cont, [line | acc]}, else: {:halt, [line | acc]}
        end)
        |> Enum.reverse()
        |> Enum.join(" ")
        |> String.replace(["FFI_SOURCES :=", "\\"], " ")
        |> String.split()
        |> Enum.map(&(&1 |> String.trim_trailing(".cu") |> String.to_atom()))
        |> Enum.sort()

      assert Enum.sort(Edifice.Block.native_ops()) == from_makefile
    end
  end

  describe "custom-call tier gating" do
    test "is off for every op without the activation flag" do
      with_env([{"EDIFICE_FUSED_CUSTOM_CALL", nil}], fn ->
        refute FusedScan.custom_call_available?()
        for k <- @kernels, do: refute(FusedScan.custom_call_available?(k), "#{k}")
      end)
    end

    test "with the flag, only ops with a native handler report the tier" do
      with_env([{"EDIFICE_FUSED_CUSTOM_CALL", "1"}], fn ->
        impl? = Edifice.Block.native_impl?()
        assert FusedScan.custom_call_available?() == impl?
        assert FusedScan.custom_call_available?(:selective_scan) == impl?
        assert FusedScan.custom_call_available?(:fused_selective_scan_backward) == impl?

        # The GRU handler was retired on 2026-10-26 (slower than the XLA
        # fallback at every size); it must not be routed to the tier.
        for k <- @kernels -- [:selective_scan] do
          refute FusedScan.custom_call_available?(k), "#{k} has no FFI handler"
        end
      end)
    end

    test "EDIFICE_DISABLE_FUSED=1 and the force-fallback flag win" do
      with_env([{"EDIFICE_FUSED_CUSTOM_CALL", "1"}, {"EDIFICE_DISABLE_FUSED", "1"}], fn ->
        refute FusedScan.custom_call_available?(:selective_scan)
      end)

      with_env([{"EDIFICE_FUSED_CUSTOM_CALL", "1"}], fn ->
        Process.put(:__edifice_force_fallback__, true)

        try do
          refute FusedScan.custom_call_available?(:selective_scan)
        after
          Process.delete(:__edifice_force_fallback__)
        end
      end)
    end

    test "AutoTune never picks the custom-call tier for an op without a handler" do
      t = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {1, 4, 8})

      with_env([{"EDIFICE_FUSED_CUSTOM_CALL", "1"}, {"EDIFICE_AUTOTUNE", "0"}], fn ->
        refute AutoTune.use_fused?(:mingru, t)
        refute AutoTune.use_fused?(:lstm_scan, t)
        refute AutoTune.use_fused?(:gru_scan, t)
        assert AutoTune.use_fused?(:selective_scan, t) == Edifice.Block.native_impl?()
      end)
    end
  end

  describe "NIF load status" do
    test "loaded?/0 reflects :erlang.load_nif, not the presence of the stubs" do
      assert is_boolean(NIF.loaded?())
      # The stubs are always exported; that was the broken probe.
      assert function_exported?(NIF, :fused_mingru_scan, 7)
      assert function_exported?(NIF, :fused_lstm_scan, 8)

      so = Path.join(to_string(:code.priv_dir(:edifice)), "libedifice_cuda_nif.so")
      unless File.exists?(so), do: refute(NIF.loaded?())
    end
  end

  describe "Recurrent.fused_rnn_available?/1" do
    test "LSTM has no custom-call handler, so only a loaded NIF can enable it" do
      with_env([{"EDIFICE_FUSED_CUSTOM_CALL", "1"}], fn ->
        assert Edifice.Recurrent.fused_rnn_available?(:lstm) == NIF.loaded?()
      end)
    end

    test "GRU no longer has a custom-call handler, so it follows the NIF like LSTM" do
      # Retired on 2026-10-26; enabling the tier must not flip the layout.
      with_env([{"EDIFICE_FUSED_CUSTOM_CALL", "1"}], fn ->
        assert Edifice.Recurrent.fused_rnn_available?(:gru) == NIF.loaded?()
      end)

      with_env([{"EDIFICE_FUSED_CUSTOM_CALL", nil}], fn ->
        assert Edifice.Recurrent.fused_rnn_available?(:gru) == NIF.loaded?()
      end)
    end

    test "other cell types never take the fused layout" do
      refute Edifice.Recurrent.fused_rnn_available?(:min_gru)
    end
  end
end
