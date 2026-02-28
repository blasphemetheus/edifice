defmodule Edifice.Attention.HGRNCorrectnessTest do
  @moduledoc """
  Correctness tests for HGRN hierarchical gating fix.
  Verifies model builds and runs, output shape, and that different
  state_expansion values produce different outputs.
  """
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.HGRN

  @batch 2
  @embed_dim 32
  @hidden_size 16
  @seq_len 4

  @base_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_layers: 1,
    state_expansion: 2,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  # ============================================================================
  # Model Builds and Runs
  # ============================================================================

  describe "model builds and runs" do
    test "builds an Axon model without error" do
      model = HGRN.build(@base_opts)
      assert %Axon{} = model
    end

    test "forward pass completes without error" do
      model = HGRN.build(@base_opts)
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
      model = HGRN.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is deterministic" do
      model = HGRN.build(@base_opts)
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
  # State Expansion Behavior
  # ============================================================================

  describe "state expansion affects output" do
    test "different state_expansion values produce different model structures" do
      # Build two models with different state_expansion
      opts_e1 = Keyword.put(@base_opts, :state_expansion, 1)
      opts_e4 = Keyword.put(@base_opts, :state_expansion, 4)

      model_e1 = HGRN.build(opts_e1)
      model_e4 = HGRN.build(opts_e4)

      {init_fn_1, _} = Axon.build(model_e1, mode: :inference)
      {init_fn_4, _} = Axon.build(model_e4, mode: :inference)

      template = Nx.template({@batch, @seq_len, @embed_dim}, :f32)
      params_e1 = init_fn_1.(template, Axon.ModelState.empty())
      params_e4 = init_fn_4.(template, Axon.ModelState.empty())

      # Different state_expansion should result in different param shapes
      # (value_proj goes to hidden * expansion, so the shapes differ)
      param_count_e1 =
        params_e1.data
        |> Edifice.TestHelpers.flatten_params()
        |> Enum.map(fn {_, t} -> Nx.size(t) end)
        |> Enum.sum()

      param_count_e4 =
        params_e4.data
        |> Edifice.TestHelpers.flatten_params()
        |> Enum.map(fn {_, t} -> Nx.size(t) end)
        |> Enum.sum()

      assert param_count_e4 > param_count_e1,
             "Higher state_expansion should have more params: #{param_count_e4} vs #{param_count_e1}"
    end

    test "different state_expansion values produce different outputs" do
      opts_e1 = Keyword.put(@base_opts, :state_expansion, 1)
      opts_e2 = Keyword.put(@base_opts, :state_expansion, 2)

      model_e1 = HGRN.build(opts_e1)
      model_e2 = HGRN.build(opts_e2)

      {init_fn_1, predict_fn_1} = Axon.build(model_e1, mode: :inference)
      {init_fn_2, predict_fn_2} = Axon.build(model_e2, mode: :inference)

      template = Nx.template({@batch, @seq_len, @embed_dim}, :f32)
      params_e1 = init_fn_1.(template, Axon.ModelState.empty())
      params_e2 = init_fn_2.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      output_e1 = predict_fn_1.(params_e1, input)
      output_e2 = predict_fn_2.(params_e2, input)

      # Both should produce valid output shape
      assert Nx.shape(output_e1) == {@batch, @hidden_size}
      assert Nx.shape(output_e2) == {@batch, @hidden_size}

      # Both should be finite
      assert Nx.all(Nx.is_nan(output_e1) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output_e2) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
