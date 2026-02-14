defmodule Edifice.Recurrent.DeltaNetCorrectnessTest do
  @moduledoc """
  Correctness tests for DeltaNet multi-head fix.
  Verifies model builds and runs with num_heads option,
  output shape is [batch, hidden_size], and different num_heads values work.
  """
  use ExUnit.Case, async: true

  alias Edifice.Recurrent.DeltaNet

  @batch 2
  @embed_dim 32
  @hidden_size 16
  @seq_len 4

  @base_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_layers: 1,
    num_heads: 2,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  # ============================================================================
  # Model Builds and Runs with num_heads
  # ============================================================================

  describe "model builds and runs with num_heads" do
    test "builds an Axon model without error" do
      model = DeltaNet.build(@base_opts)
      assert %Axon{} = model
    end

    test "forward pass completes without error" do
      model = DeltaNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Output Shape
  # ============================================================================

  describe "output shape" do
    test "output shape is [batch, hidden_size]" do
      model = DeltaNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is deterministic" do
      model = DeltaNet.build(@base_opts)
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

    test "different inputs produce different outputs" do
      model = DeltaNet.build(@base_opts)
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

  # ============================================================================
  # Different num_heads Values Work
  # ============================================================================

  describe "different num_heads values" do
    test "num_heads=1 builds and runs" do
      opts = Keyword.put(@base_opts, :num_heads, 1)
      model = DeltaNet.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "num_heads=2 builds and runs" do
      opts = Keyword.put(@base_opts, :num_heads, 2)
      model = DeltaNet.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "num_heads=4 builds and runs" do
      opts = Keyword.put(@base_opts, :num_heads, 4)
      model = DeltaNet.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "all num_heads values produce same output shape" do
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      for num_heads <- [1, 2, 4] do
        opts = Keyword.put(@base_opts, :num_heads, num_heads)
        model = DeltaNet.build(opts)
        {init_fn, predict_fn} = Axon.build(model, mode: :inference)

        params =
          init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

        output = predict_fn.(params, input)

        assert Nx.shape(output) == {@batch, @hidden_size},
               "num_heads=#{num_heads} should produce [batch, hidden_size] output"
      end
    end
  end
end
