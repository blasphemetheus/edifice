defmodule Edifice.SSM.HyenaCoverageTest do
  @moduledoc """
  Coverage tests for Edifice.SSM.Hyena.
  Covers option variations and code branches not exercised by existing tests.
  """
  use ExUnit.Case, async: true

  alias Edifice.SSM.Hyena

  @batch 2
  @seq_len 8
  @embed_dim 16
  @hidden_size 16
  @filter_size 8

  # ============================================================================
  # Input projection branch: embed_dim == hidden_size (skip projection)
  # ============================================================================

  describe "input projection branch" do
    test "skips input projection when embed_dim == hidden_size" do
      opts = [
        embed_dim: @hidden_size,
        hidden_size: @hidden_size,
        order: 2,
        filter_size: @filter_size,
        num_layers: 1,
        window_size: @seq_len,
        seq_len: @seq_len,
        dropout: 0.0
      ]

      model = Hyena.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      # Should NOT have input_projection parameter
      param_keys = Map.keys(params.data)
      refute Enum.any?(param_keys, &String.contains?(&1, "input_projection"))

      key = Nx.Random.key(100)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "uses input projection when embed_dim != hidden_size" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: 32,
        order: 2,
        filter_size: @filter_size,
        num_layers: 1,
        window_size: @seq_len,
        seq_len: @seq_len,
        dropout: 0.0
      ]

      model = Hyena.build(opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)
      assert Enum.any?(param_keys, &String.contains?(&1, "input_projection"))
    end
  end

  # ============================================================================
  # Dropout branch: dropout > 0 vs dropout == 0
  # ============================================================================

  describe "dropout branch" do
    test "dropout=0 skips dropout layers" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        order: 2,
        filter_size: @filter_size,
        num_layers: 1,
        window_size: @seq_len,
        seq_len: @seq_len,
        dropout: 0.0
      ]

      model = Hyena.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(101)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "dropout > 0 includes dropout layers" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        order: 2,
        filter_size: @filter_size,
        num_layers: 1,
        window_size: @seq_len,
        seq_len: @seq_len,
        dropout: 0.2
      ]

      model = Hyena.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(102)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  # ============================================================================
  # Order variations
  # ============================================================================

  describe "order variations" do
    test "order=1 produces correct output" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        order: 1,
        filter_size: @filter_size,
        num_layers: 1,
        window_size: @seq_len,
        seq_len: @seq_len,
        dropout: 0.0
      ]

      model = Hyena.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(103)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "order=3 produces correct output with 4 projections" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        order: 3,
        filter_size: @filter_size,
        num_layers: 1,
        window_size: @seq_len,
        seq_len: @seq_len,
        dropout: 0.0
      ]

      model = Hyena.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      # order=3 should have filter0, filter1, filter2
      param_keys = Map.keys(params.data)
      assert Enum.any?(param_keys, &String.contains?(&1, "filter0"))
      assert Enum.any?(param_keys, &String.contains?(&1, "filter1"))
      assert Enum.any?(param_keys, &String.contains?(&1, "filter2"))

      key = Nx.Random.key(104)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Multiple layers
  # ============================================================================

  describe "multiple layers" do
    test "num_layers=3 stacks correctly" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        order: 2,
        filter_size: @filter_size,
        num_layers: 3,
        window_size: @seq_len,
        seq_len: @seq_len,
        dropout: 0.0
      ]

      model = Hyena.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      # Should have blocks 1, 2, 3
      param_keys = Map.keys(params.data)
      assert Enum.any?(param_keys, &String.contains?(&1, "hyena_block_1"))
      assert Enum.any?(param_keys, &String.contains?(&1, "hyena_block_2"))
      assert Enum.any?(param_keys, &String.contains?(&1, "hyena_block_3"))

      key = Nx.Random.key(105)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Filter size variations
  # ============================================================================

  describe "filter size variations" do
    test "small filter_size=4" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        order: 2,
        filter_size: 4,
        num_layers: 1,
        window_size: @seq_len,
        seq_len: @seq_len,
        dropout: 0.0
      ]

      model = Hyena.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      # Check filter MLP first dense layer has filter_size=4 output
      dense1_params = params.data["hyena_block_1_filter0_dense1"]
      kernel = dense1_params["kernel"]
      assert elem(Nx.shape(kernel), 1) == 4

      key = Nx.Random.key(106)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "larger filter_size=32" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        order: 2,
        filter_size: 32,
        num_layers: 1,
        window_size: @seq_len,
        seq_len: @seq_len,
        dropout: 0.0
      ]

      model = Hyena.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      dense1_params = params.data["hyena_block_1_filter0_dense1"]
      kernel = dense1_params["kernel"]
      assert elem(Nx.shape(kernel), 1) == 32

      key = Nx.Random.key(107)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  # ============================================================================
  # Sequence length variations
  # ============================================================================

  describe "sequence length variations" do
    test "short seq_len=4" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        order: 2,
        filter_size: @filter_size,
        num_layers: 1,
        window_size: 4,
        seq_len: 4,
        dropout: 0.0
      ]

      model = Hyena.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, 4, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(108)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 4, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "longer seq_len=16" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        order: 2,
        filter_size: @filter_size,
        num_layers: 1,
        window_size: 16,
        seq_len: 16,
        dropout: 0.0
      ]

      model = Hyena.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, 16, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(109)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 16, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Utility functions
  # ============================================================================

  describe "output_size/1" do
    test "returns custom hidden_size" do
      assert Hyena.output_size(hidden_size: 128) == 128
    end

    test "returns default hidden_size" do
      assert Hyena.output_size() == 256
    end

    test "returns hidden_size from opts list" do
      assert Hyena.output_size(hidden_size: 64, other: :stuff) == 64
    end
  end

  describe "param_count/1" do
    test "returns integer for default opts" do
      count =
        Hyena.param_count(
          embed_dim: 287,
          hidden_size: 256,
          order: 2,
          filter_size: 64,
          num_layers: 4
        )

      assert is_integer(count)
      assert count > 0
    end

    test "param_count varies with order" do
      base =
        Hyena.param_count(
          embed_dim: 32,
          hidden_size: 32,
          order: 1,
          filter_size: 16,
          num_layers: 1
        )

      more =
        Hyena.param_count(
          embed_dim: 32,
          hidden_size: 32,
          order: 3,
          filter_size: 16,
          num_layers: 1
        )

      assert more > base
    end

    test "param_count when embed_dim == hidden_size excludes input proj" do
      same =
        Hyena.param_count(
          embed_dim: 32,
          hidden_size: 32,
          order: 2,
          filter_size: 16,
          num_layers: 1
        )

      diff =
        Hyena.param_count(
          embed_dim: 64,
          hidden_size: 32,
          order: 2,
          filter_size: 16,
          num_layers: 1
        )

      assert diff > same
    end
  end

  describe "recommended_defaults/0" do
    test "returns expected keyword list" do
      defaults = Hyena.recommended_defaults()

      assert Keyword.keyword?(defaults)
      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :order) == 2
      assert Keyword.get(defaults, :filter_size) == 64
      assert Keyword.get(defaults, :num_layers) == 4
      assert Keyword.get(defaults, :window_size) == 60
      assert Keyword.get(defaults, :dropout) == 0.1
    end
  end

  # ============================================================================
  # build_hyena_block/2 directly
  # ============================================================================

  describe "build_hyena_block/2" do
    test "builds a standalone block" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden_size})

      block =
        Hyena.build_hyena_block(input,
          hidden_size: @hidden_size,
          order: 2,
          filter_size: @filter_size,
          dropout: 0.0,
          seq_len: @seq_len,
          name: "test_block"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(110)
      {input_data, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      # Block preserves sequence shape for residual
      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "block with dropout=0.3" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden_size})

      block =
        Hyena.build_hyena_block(input,
          hidden_size: @hidden_size,
          order: 2,
          filter_size: @filter_size,
          dropout: 0.3,
          seq_len: @seq_len,
          name: "dropout_block"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(111)
      {input_data, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end
  end
end
