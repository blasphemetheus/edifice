defmodule Edifice.StatefulStepEquivalenceTest do
  @moduledoc """
  THE correctness pin for `Edifice.Stateful` (HANDOFF_BOT_RUNTIME.md task 2):
  iterated `step/3` must reproduce the full-sequence forward at every prefix
  length. If this equivalence holds, everything downstream (latency numbers,
  rollback behavior, the bot's frame loop) is trustworthy.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  import Edifice.StatefulCase

  @moduletag :stateful
  @moduletag timeout: 120_000

  setup do
    # Pin the pure-Nx fallback paths (no fused CUDA kernels) so step and
    # forward compare the same math
    Process.put(:__edifice_force_fallback__, true)
    :ok
  end

  describe "MinGRU step == forward" do
    test "single layer, embed == hidden" do
      assert_step_matches_forward(:min_gru,
        embed_dim: 8,
        hidden_size: 8,
        num_layers: 1,
        dropout: 0.0
      )
    end

    test "three layers, embed != hidden (input projection)" do
      assert_step_matches_forward(:min_gru,
        embed_dim: 12,
        hidden_size: 8,
        num_layers: 3,
        dropout: 0.0
      )
    end

    test "batch of 2" do
      assert_step_matches_forward(:min_gru,
        [embed_dim: 8, hidden_size: 8, num_layers: 2, dropout: 0.0],
        batch: 2
      )
    end

    property "holds across random seeds and sequence lengths" do
      check all(
              seed <- StreamData.integer(0..1_000),
              seq_len <- StreamData.integer(1..10),
              batch <- StreamData.integer(1..2),
              max_runs: 8
            ) do
        assert_step_matches_forward(
          :min_gru,
          [embed_dim: 6, hidden_size: 6, num_layers: 2, dropout: 0.0],
          seed: seed,
          seq_len: seq_len,
          batch: batch
        )
      end
    end
  end

  describe "Mamba step == forward" do
    @mamba_base [hidden_size: 8, state_size: 4, expand_factor: 2, dropout: 0.0]

    test "single layer, embed == hidden" do
      assert_step_matches_forward(
        :mamba,
        @mamba_base ++ [embed_dim: 8, num_layers: 1, conv_size: 4]
      )
    end

    test "two layers, embed != hidden (input projection), conv_size 3" do
      assert_step_matches_forward(
        :mamba,
        @mamba_base ++ [embed_dim: 12, num_layers: 2, conv_size: 3]
      )
    end

    test "batch of 2" do
      assert_step_matches_forward(
        :mamba,
        @mamba_base ++ [embed_dim: 8, num_layers: 2, conv_size: 4],
        batch: 2
      )
    end

    test "conv edge: sequences shorter than conv_size" do
      # t < conv_size exercises the zero ring buffer == causal left-pad
      assert_step_matches_forward(
        :mamba,
        @mamba_base ++ [embed_dim: 8, num_layers: 1, conv_size: 4],
        seq_len: 2
      )
    end

    test "seq_len 40 crosses the Blelloch-scan branch" do
      # The forward switches from sequential_scan to blelloch_scan above
      # seq_len 32; different summation order → slightly looser tolerance.
      # This transitively pins step == blelloch == sequential.
      assert_step_matches_forward(
        :mamba,
        @mamba_base ++ [embed_dim: 8, num_layers: 1, conv_size: 4],
        seq_len: 40,
        atol: 1.0e-4
      )
    end

    property "holds across random seeds and sequence lengths" do
      check all(
              seed <- StreamData.integer(0..1_000),
              seq_len <- StreamData.integer(1..8),
              max_runs: 5
            ) do
        assert_step_matches_forward(
          :mamba,
          @mamba_base ++ [embed_dim: 6, num_layers: 1, conv_size: 3],
          seed: seed,
          seq_len: seq_len
        )
      end
    end
  end

  describe "GatedSSM step == forward (scan_mode: :causal)" do
    @gated_base [
      hidden_size: 8,
      state_size: 4,
      expand_factor: 2,
      dropout: 0.0,
      scan_mode: :causal
    ]

    test "single layer, embed == hidden" do
      assert_step_matches_forward(
        :gated_ssm,
        @gated_base ++ [embed_dim: 8, num_layers: 1, conv_size: 4]
      )
    end

    test "two layers, embed != hidden, conv_size 3" do
      assert_step_matches_forward(
        :gated_ssm,
        @gated_base ++ [embed_dim: 12, num_layers: 2, conv_size: 3]
      )
    end

    test "init_state raises without scan_mode: :causal" do
      err =
        assert_raise ArgumentError, fn ->
          Edifice.init_state(:gated_ssm, %{}, embed_dim: 8, hidden_size: 8)
        end

      assert err.message =~ "scan_mode: :causal"
    end

    test "gated_ssm default build is unchanged (:legacy regression guard)" do
      # The default MUST keep legacy semantics: anything trained on main
      # before this change would silently drift otherwise
      opts = [
        embed_dim: 8,
        hidden_size: 8,
        state_size: 4,
        num_layers: 1,
        dropout: 0.0,
        window_size: 6
      ]

      {params, predict_default, x} = Edifice.StatefulCase.build_forward(:gated_ssm, opts, 6, 1, 3)

      legacy_model = Edifice.build(:gated_ssm, Keyword.put(opts, :scan_mode, :legacy) |> Keyword.put(:seq_len, nil))
      {_, predict_legacy} = Axon.build(legacy_model, mode: :inference)

      causal_model = Edifice.build(:gated_ssm, Keyword.put(opts, :scan_mode, :causal) |> Keyword.put(:seq_len, nil))
      {_, predict_causal} = Axon.build(causal_model, mode: :inference)

      out_default = predict_default.(params, x)
      out_legacy = predict_legacy.(params, x)
      out_causal = predict_causal.(params, x)

      # default == explicit :legacy, bitwise
      assert Nx.to_binary(out_default) == Nx.to_binary(out_legacy)

      # :causal computes a genuinely different function (sanity: flag threads)
      refute Nx.to_binary(out_default) == Nx.to_binary(out_causal)
    end
  end

  describe "GRU/LSTM step == forward (Axon layout)" do
    # The riskiest equivalence: Axon's initial hidden state is glorot-sampled
    # from a stored RNG key (NOT zeros), bhn sits inside the reset product,
    # and the edifice builders use layer-norm epsilon 1e-6. A wrong h0 fails
    # at k=1 immediately.

    test "GRU single layer" do
      assert_step_matches_forward(:gru,
        embed_dim: 8,
        hidden_size: 8,
        num_layers: 1,
        dropout: 0.0
      )
    end

    test "GRU two layers, embed != hidden" do
      assert_step_matches_forward(:gru,
        embed_dim: 12,
        hidden_size: 8,
        num_layers: 2,
        dropout: 0.0
      )
    end

    test "LSTM single layer" do
      assert_step_matches_forward(:lstm,
        embed_dim: 8,
        hidden_size: 8,
        num_layers: 1,
        dropout: 0.0
      )
    end

    test "LSTM two layers, batch of 2" do
      assert_step_matches_forward(
        :lstm,
        [embed_dim: 8, hidden_size: 8, num_layers: 2, dropout: 0.0],
        batch: 2
      )
    end
  end

  describe "GRU/LSTM fused-layout step math" do
    # The fused graph layout only builds on the CUDA-fork machine, but its
    # CPU fallback scan is public — pin the step math against it with
    # synthetic weights, no fork needed.

    test "fused GRU step matches gru_scan fallback" do
      key = Nx.Random.key(11)
      {wx_seq, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {2, 6, 3 * 4})
      {r_kernel, _} = Nx.Random.uniform(key, -1.0, 1.0, shape: {4, 3 * 4})

      full = Edifice.CUDA.FusedScan.gru_scan(wx_seq, r_kernel)

      {_, outs} =
        Enum.reduce(0..5, {Nx.broadcast(0.0, {2, 4}), []}, fn t, {h, acc} ->
          wx_t = wx_seq |> Nx.slice_along_axis(t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          h_new = fused_gru_step(wx_t, r_kernel, h)
          {h_new, [h_new | acc]}
        end)

      stepped = outs |> Enum.reverse() |> Nx.stack(axis: 1)

      assert Nx.all_close(full, stepped, atol: 1.0e-6) |> Nx.to_number() == 1
    end

    test "fused LSTM step matches lstm_scan fallback" do
      key = Nx.Random.key(12)
      {wx_seq, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {2, 6, 4 * 4})
      {r_kernel, _} = Nx.Random.uniform(key, -1.0, 1.0, shape: {4, 4 * 4})

      full = Edifice.CUDA.FusedScan.lstm_scan(wx_seq, r_kernel)

      zeros = Nx.broadcast(0.0, {2, 4})

      {_, outs} =
        Enum.reduce(0..5, {{zeros, zeros}, []}, fn t, {{h, c}, acc} ->
          wx_t = wx_seq |> Nx.slice_along_axis(t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          {h_new, c_new} = fused_lstm_step(wx_t, r_kernel, h, c)
          {{h_new, c_new}, [h_new | acc]}
        end)

      stepped = outs |> Enum.reverse() |> Nx.stack(axis: 1)

      assert Nx.all_close(full, stepped, atol: 1.0e-6) |> Nx.to_number() == 1
    end
  end

  # Single-frame twins of the fused-layout cell steps in Edifice.Recurrent —
  # exercised through synthetic params shaped like the fused graph layout
  defp fused_gru_step(wx_t, r_kernel, h) do
    params = fused_params(:gru, wx_t, r_kernel)
    state = %{h: Nx.new_axis(h, 1)}
    {out, _state} = Edifice.Recurrent.step(params, state, wx_t)
    out
  end

  defp fused_lstm_step(wx_t, r_kernel, h, c) do
    params = fused_params(:lstm, wx_t, r_kernel)
    state = %{h: Nx.new_axis(h, 1), c: Nx.new_axis(c, 1)}
    {out, new_state} = Edifice.Recurrent.step(params, state, wx_t)
    {out, new_state.c |> Nx.squeeze(axes: [1])}
  end

  # Identity input_proj (wx is precomputed in these synthetic tests) and no
  # layer norms, so the step reduces to exactly the cell recurrence
  defp fused_params(cell, wx_t, r_kernel) do
    gate_size = Nx.axis_size(wx_t, 1)

    %{
      "#{cell}_1_input_proj" => %{
        "kernel" => Nx.eye(gate_size),
        "bias" => Nx.broadcast(0.0, {gate_size})
      },
      "#{cell}_1_fused_scan" => %{"#{cell}_1_recurrent_kernel" => r_kernel}
    }
  end
end
