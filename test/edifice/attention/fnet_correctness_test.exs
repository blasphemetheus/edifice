defmodule Edifice.Attention.FNetCorrectnessTest do
  @moduledoc """
  Correctness tests for FNet 2D FFT fix.
  Verifies output is real-valued, FFT mixing is parameter-free,
  and output shape is preserved.
  """
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.FNet

  @batch 2
  @embed_dim 32
  @hidden_size 16
  @seq_len 4

  @base_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_layers: 1,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  # ============================================================================
  # FFT Mixing: Real-Valued Output
  # ============================================================================

  describe "FFT mixing produces real-valued output" do
    test "fourier_mixing_real output has no imaginary component" do
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})

      dft_seq = FNet.dft_real_matrix(@seq_len)
      dft_hidden = FNet.dft_real_matrix(@hidden_size)
      output = FNet.fourier_mixing_real(input, dft_seq, dft_hidden)

      # Output type should be real (f32), not complex
      assert Nx.type(output) == {:f, 32}

      # Output should be finite (no NaN/Inf from DFT)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "fourier_mixing_real preserves shape [batch, seq, hidden]" do
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})

      dft_seq = FNet.dft_real_matrix(@seq_len)
      dft_hidden = FNet.dft_real_matrix(@hidden_size)
      output = FNet.fourier_mixing_real(input, dft_seq, dft_hidden)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end

    test "full model output is real-valued and finite" do
      model = FNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.type(output) == {:f, 32}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # FFT Mixing: Parameter-Free
  # ============================================================================

  describe "FFT mixing is parameter-free" do
    test "same params produce same output (FFT is deterministic)" do
      model = FNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "no attention weight parameters in model" do
      model = FNet.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # FNet should not have Q, K, V projection parameters
      qkv_keys =
        Enum.filter(param_keys, fn k ->
          String.contains?(k, "q_proj") or
            String.contains?(k, "k_proj") or
            String.contains?(k, "v_proj")
        end)

      assert qkv_keys == [],
             "FNet should not have Q/K/V projection params (FFT is parameter-free), found: #{inspect(qkv_keys)}"
    end
  end

  # ============================================================================
  # Output Shape
  # ============================================================================

  describe "output shape" do
    test "output shape is [batch, hidden_size]" do
      model = FNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "different inputs produce different outputs" do
      model = FNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      {input2, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different inputs should produce different outputs"
    end
  end
end
