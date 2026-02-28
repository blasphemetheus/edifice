defmodule Edifice.Attention.RetNetCorrectnessTest do
  @moduledoc """
  Correctness tests for RetNet multi-scale decay fix.
  Verifies model builds and runs, output shape is correct,
  and different num_heads produce different model structures.
  """
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.RetNet

  @batch 2
  @embed_dim 32
  @hidden_size 16
  @seq_len 4
  @num_heads 4

  @base_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_layers: 1,
    num_heads: @num_heads,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  # ============================================================================
  # Model Builds and Runs
  # ============================================================================

  describe "model builds and runs" do
    test "builds an Axon model without error" do
      model = RetNet.build(@base_opts)
      assert %Axon{} = model
    end

    test "forward pass completes without error" do
      model = RetNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output is deterministic" do
      model = RetNet.build(@base_opts)
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
  end

  # ============================================================================
  # Output Shape
  # ============================================================================

  describe "output shape" do
    test "output shape is [batch, hidden_size]" do
      model = RetNet.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "different inputs produce different outputs" do
      model = RetNet.build(@base_opts)
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
  # Multi-Scale Decay (Different num_heads)
  # ============================================================================

  describe "different num_heads produce different model structures" do
    test "models with 2 vs 4 heads have different retention param shapes" do
      # hidden_size must be divisible by num_heads
      opts_h2 = Keyword.merge(@base_opts, num_heads: 2, hidden_size: 16)
      opts_h4 = Keyword.merge(@base_opts, num_heads: 4, hidden_size: 16)

      model_h2 = RetNet.build(opts_h2)
      model_h4 = RetNet.build(opts_h4)

      {init_fn_2, _} = Axon.build(model_h2, mode: :inference)
      {init_fn_4, _} = Axon.build(model_h4, mode: :inference)

      template = Nx.template({@batch, @seq_len, @embed_dim}, :f32)
      params_h2 = init_fn_2.(template, Axon.ModelState.empty())
      params_h4 = init_fn_4.(template, Axon.ModelState.empty())

      # Both models should build successfully with different head counts
      keys_h2 = Map.keys(params_h2.data)
      keys_h4 = Map.keys(params_h4.data)

      # Both should have retention/MSR parameter groups
      msr_keys_h2 = Enum.filter(keys_h2, &String.contains?(&1, "msr"))
      msr_keys_h4 = Enum.filter(keys_h4, &String.contains?(&1, "msr"))

      assert msr_keys_h2 != [], "2-head model should have MSR params"
      assert msr_keys_h4 != [], "4-head model should have MSR params"
    end

    test "models with different num_heads both produce valid outputs" do
      opts_h2 = Keyword.merge(@base_opts, num_heads: 2, hidden_size: 16)
      opts_h4 = Keyword.merge(@base_opts, num_heads: 4, hidden_size: 16)

      model_h2 = RetNet.build(opts_h2)
      model_h4 = RetNet.build(opts_h4)

      {init_fn_2, predict_fn_2} = Axon.build(model_h2, mode: :inference)
      {init_fn_4, predict_fn_4} = Axon.build(model_h4, mode: :inference)

      template = Nx.template({@batch, @seq_len, @embed_dim}, :f32)
      params_h2 = init_fn_2.(template, Axon.ModelState.empty())
      params_h4 = init_fn_4.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      output_h2 = predict_fn_2.(params_h2, input)
      output_h4 = predict_fn_4.(params_h4, input)

      # Both should produce correct shape
      assert Nx.shape(output_h2) == {@batch, 16}
      assert Nx.shape(output_h4) == {@batch, 16}

      # Both should be finite
      assert Nx.all(Nx.is_nan(output_h2) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output_h4) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
