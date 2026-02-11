defmodule Edifice.Attention.GLACorrectnessTest do
  @moduledoc """
  Correctness tests for GLA data-dependent decay fix.
  Verifies model builds and runs, output shape is [batch, hidden_size],
  and data-dependent gating produces valid outputs.
  """
  use ExUnit.Case, async: true

  alias Edifice.Attention.GLA

  @batch 2
  @embed_size 32
  @hidden_size 16
  @seq_len 4
  @num_heads 2
  @head_dim 8

  @base_opts [
    embed_size: @embed_size,
    hidden_size: @hidden_size,
    num_layers: 1,
    num_heads: @num_heads,
    head_dim: @head_dim,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  # ============================================================================
  # Model Builds and Runs
  # ============================================================================

  describe "model builds and runs" do
    test "builds an Axon model without error" do
      model = GLA.build(@base_opts)
      assert %Axon{} = model
    end

    test "forward pass completes without error" do
      model = GLA.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
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
      model = GLA.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is deterministic" do
      model = GLA.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "different inputs produce different outputs" do
      model = GLA.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})
      {input2, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_size})

      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different inputs should produce different outputs"
    end
  end

  # ============================================================================
  # Data-Dependent Gating
  # ============================================================================

  describe "data-dependent gating" do
    test "model has gate projection parameters" do
      model = GLA.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # GLA should have gate projection (data-dependent gating is the key innovation)
      gate_keys = Enum.filter(param_keys, &String.contains?(&1, "g_proj"))

      assert length(gate_keys) > 0,
             "GLA should have gate projection params for data-dependent gating, got: #{inspect(param_keys)}"
    end

    test "model has Q, K, V, and G projections" do
      model = GLA.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      for proj <- ["q_proj", "k_proj", "v_proj", "g_proj"] do
        matching = Enum.filter(param_keys, &String.contains?(&1, proj))

        assert length(matching) > 0,
               "GLA should have #{proj} params, got: #{inspect(param_keys)}"
      end
    end
  end
end
