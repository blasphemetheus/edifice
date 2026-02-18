defmodule Edifice.Attention.RetNetCoverageTest do
  @moduledoc """
  Additional coverage tests for Edifice.Attention.RetNet.
  Targets uncovered code paths: recurrent_retention_step, init_retention_state,
  param_count, recommended_defaults, default accessors, dropout > 0 path,
  embed_dim == hidden_size branch, build_retnet_block directly, different
  head counts and layer counts.
  """
  use ExUnit.Case, async: true

  alias Edifice.Attention.RetNet

  @batch 2
  @seq_len 8
  @embed_dim 16
  @hidden_size 16
  @num_heads 4

  # ============================================================================
  # Default accessor functions
  # ============================================================================

  describe "default accessor functions" do
    test "default_hidden_size/0" do
      assert RetNet.default_hidden_size() == 256
    end

    test "default_num_layers/0" do
      assert RetNet.default_num_layers() == 6
    end

    test "default_num_heads/0" do
      assert RetNet.default_num_heads() == 4
    end

    test "default_expand_factor/0" do
      assert RetNet.default_expand_factor() == 2
    end

    test "default_dropout/0" do
      assert RetNet.default_dropout() == 0.0
    end

    test "eps/0" do
      assert RetNet.eps() == 1.0e-6
    end
  end

  # ============================================================================
  # embed_dim == hidden_size (skips input projection)
  # ============================================================================

  describe "embed_dim == hidden_size" do
    test "skips input projection" do
      model =
        RetNet.build(
          embed_dim: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 1,
          num_heads: @num_heads,
          seq_len: @seq_len,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)
      refute Enum.any?(param_keys, &(&1 == "input_projection"))

      input = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  # ============================================================================
  # dropout > 0 path
  # ============================================================================

  describe "dropout > 0" do
    test "adds dropout between blocks" do
      model =
        RetNet.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 3,
          num_heads: @num_heads,
          seq_len: @seq_len,
          dropout: 0.1
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # build_retnet_block/2 directly
  # ============================================================================

  describe "build_retnet_block/2" do
    test "builds a single RetNet block" do
      input = Axon.input("retnet_input", shape: {nil, nil, @hidden_size})

      block =
        RetNet.build_retnet_block(input,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          expand_factor: 2,
          layer_idx: 1
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{"retnet_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"retnet_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end

    @tag :slow
    test "with default opts" do
      input = Axon.input("retnet_input", shape: {nil, nil, 256})

      block = RetNet.build_retnet_block(input)

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{"retnet_input" => Nx.template({@batch, @seq_len, 256}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, 256})
      output = predict_fn.(params, %{"retnet_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, 256}
    end
  end

  # ============================================================================
  # build_multi_scale_retention/2 directly
  # ============================================================================

  describe "build_multi_scale_retention/2" do
    test "builds MSR layer" do
      input = Axon.input("msr_input", shape: {nil, nil, @hidden_size})

      msr =
        RetNet.build_multi_scale_retention(input,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          name: "test_msr"
        )

      {init_fn, predict_fn} = Axon.build(msr, mode: :inference)

      params =
        init_fn.(
          %{"msr_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"msr_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end

    @tag :slow
    test "with default opts" do
      input = Axon.input("msr_input", shape: {nil, nil, 256})

      msr = RetNet.build_multi_scale_retention(input)

      {init_fn, predict_fn} = Axon.build(msr, mode: :inference)

      params =
        init_fn.(
          %{"msr_input" => Nx.template({@batch, @seq_len, 256}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, 256})
      output = predict_fn.(params, %{"msr_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, 256}
    end
  end

  # ============================================================================
  # recurrent_retention_step/5
  # ============================================================================

  describe "recurrent_retention_step/5" do
    test "produces correct shapes with batched input" do
      batch = 2
      head_dim = 4

      q = Nx.broadcast(0.5, {batch, head_dim})
      k = Nx.broadcast(0.3, {batch, head_dim})
      v = Nx.broadcast(0.7, {batch, head_dim})
      state = Nx.broadcast(0.0, {batch, head_dim, head_dim})
      gamma = Nx.tensor(0.9)

      {output, new_state} = RetNet.recurrent_retention_step(q, k, v, state, gamma)

      assert Nx.shape(output) == {batch, head_dim}
      assert Nx.shape(new_state) == {batch, head_dim, head_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(new_state) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "state accumulates across multiple steps" do
      batch = 2
      head_dim = 4
      state = Nx.broadcast(0.0, {batch, head_dim, head_dim})
      gamma = Nx.tensor(0.95)

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, shape: {batch, head_dim})
      {k, key} = Nx.Random.uniform(key, shape: {batch, head_dim})
      {v, _key} = Nx.Random.uniform(key, shape: {batch, head_dim})

      {out1, state1} = RetNet.recurrent_retention_step(q, k, v, state, gamma)
      {out2, state2} = RetNet.recurrent_retention_step(q, k, v, state1, gamma)

      # State should grow (non-zero after updates)
      assert Nx.to_number(Nx.reduce_max(Nx.abs(state2))) > Nx.to_number(Nx.reduce_max(Nx.abs(state1)))
      # Outputs should differ due to different state
      assert Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(out1, out2)))) > 0
    end
  end

  describe "different expand_factors" do
    test "expand_factor = 4" do
      model =
        RetNet.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          num_heads: @num_heads,
          expand_factor: 4,
          seq_len: @seq_len,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "multi-layer with dropout between blocks" do
    test "4 layers with dropout" do
      model =
        RetNet.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 4,
          num_heads: @num_heads,
          seq_len: @seq_len,
          dropout: 0.15
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "different inputs produce different outputs" do
    test "verifying model sensitivity to input" do
      model =
        RetNet.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          num_heads: @num_heads,
          seq_len: @seq_len,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      {input2, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6
    end
  end

  # ============================================================================
  # init_retention_state/3
  # ============================================================================

  describe "init_retention_state/3" do
    test "returns zero-initialized state with correct shape" do
      state = RetNet.init_retention_state(2, 4, 8)
      assert Nx.shape(state) == {2, 4, 8, 8}
      assert Nx.to_number(Nx.reduce_max(Nx.abs(state))) == 0.0
    end

    test "with batch_size=1" do
      state = RetNet.init_retention_state(1, 2, 4)
      assert Nx.shape(state) == {1, 2, 4, 4}
    end
  end

  # ============================================================================
  # param_count/1
  # ============================================================================

  describe "param_count/1" do
    test "returns a positive integer" do
      count =
        RetNet.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 2,
          expand_factor: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "more layers increases param count" do
      count_1 =
        RetNet.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          expand_factor: 2
        )

      count_4 =
        RetNet.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 4,
          expand_factor: 2
        )

      assert count_4 > count_1
    end

    test "embed_dim == hidden_size has no input projection cost" do
      count_same =
        RetNet.param_count(
          embed_dim: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 1,
          expand_factor: 2
        )

      count_diff =
        RetNet.param_count(
          embed_dim: @embed_dim + 8,
          hidden_size: @hidden_size,
          num_layers: 1,
          expand_factor: 2
        )

      assert count_diff > count_same
    end
  end

  # ============================================================================
  # output_size/1
  # ============================================================================

  describe "output_size/1" do
    test "returns default when no opts given" do
      assert RetNet.output_size() == 256
    end

    test "returns custom hidden_size" do
      assert RetNet.output_size(hidden_size: 128) == 128
    end
  end

  # ============================================================================
  # recommended_defaults/0
  # ============================================================================

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = RetNet.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :expand_factor)
      assert Keyword.has_key?(defaults, :window_size)
      assert Keyword.has_key?(defaults, :dropout)
    end
  end

  # ============================================================================
  # Different head count configurations
  # ============================================================================

  describe "different head counts" do
    test "with 2 heads" do
      model =
        RetNet.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          num_heads: 2,
          seq_len: @seq_len,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "with 8 heads and larger hidden_size" do
      hidden = 32

      model =
        RetNet.build(
          embed_dim: @embed_dim,
          hidden_size: hidden,
          num_layers: 1,
          num_heads: 8,
          seq_len: @seq_len,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, hidden}
    end
  end
end
