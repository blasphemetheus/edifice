defmodule Edifice.SSM.S4CoverageTest do
  @moduledoc """
  Coverage tests for Edifice.SSM.S4.
  Covers option variations, code branches, utility functions, and
  the build_s4_block/2 standalone not exercised by existing tests.
  """
  use ExUnit.Case, async: true

  alias Edifice.SSM.S4

  @batch 2
  @seq_len 8
  @embed_dim 16
  @hidden_size 16
  @state_size 8

  # ============================================================================
  # Input projection branch: embed_dim == hidden_size (skip projection)
  # ============================================================================

  describe "input projection branch" do
    test "skips input projection when embed_dim == hidden_size" do
      opts = [
        embed_dim: @hidden_size,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 1,
        dropout: 0.0,
        window_size: @seq_len,
        seq_len: @seq_len
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      # Should NOT have input_projection parameter
      param_keys = Map.keys(params.data)
      refute Enum.any?(param_keys, &String.contains?(&1, "input_projection"))

      key = Nx.Random.key(400)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "uses input projection when embed_dim != hidden_size" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: 32,
        state_size: @state_size,
        num_layers: 1,
        dropout: 0.0,
        window_size: @seq_len,
        seq_len: @seq_len
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)
      assert Enum.any?(param_keys, &String.contains?(&1, "input_projection"))

      key = Nx.Random.key(401)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, 32}
    end
  end

  # ============================================================================
  # Dropout branch
  # ============================================================================

  describe "dropout branch" do
    test "dropout=0 skips dropout layers" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 1,
        dropout: 0.0,
        window_size: @seq_len,
        seq_len: @seq_len
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(402)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "dropout > 0 includes dropout in block" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 1,
        dropout: 0.2,
        window_size: @seq_len,
        seq_len: @seq_len
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(403)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # State size variations
  # ============================================================================

  describe "state size variations" do
    test "small state_size=4" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        state_size: 4,
        num_layers: 1,
        dropout: 0.0,
        window_size: @seq_len,
        seq_len: @seq_len
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      # B and C projections should project to state_size=4
      b_proj = params.data["s4_block_1_b_proj"]
      kernel = b_proj["kernel"]
      assert elem(Nx.shape(kernel), 1) == 4

      key = Nx.Random.key(410)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "larger state_size=16" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        state_size: 16,
        num_layers: 1,
        dropout: 0.0,
        window_size: @seq_len,
        seq_len: @seq_len
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      b_proj = params.data["s4_block_1_b_proj"]
      kernel = b_proj["kernel"]
      assert elem(Nx.shape(kernel), 1) == 16

      key = Nx.Random.key(411)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
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
        state_size: @state_size,
        num_layers: 3,
        dropout: 0.0,
        window_size: @seq_len,
        seq_len: @seq_len
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)
      assert Enum.any?(param_keys, &String.contains?(&1, "s4_block_1"))
      assert Enum.any?(param_keys, &String.contains?(&1, "s4_block_2"))
      assert Enum.any?(param_keys, &String.contains?(&1, "s4_block_3"))

      key = Nx.Random.key(420)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "num_layers=1 with dropout > 0" do
      # Single layer with dropout -- dropout is inside the block not between layers
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 1,
        dropout: 0.3,
        window_size: @seq_len,
        seq_len: @seq_len
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(421)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  # ============================================================================
  # build_s4_block/2 directly
  # ============================================================================

  describe "build_s4_block/2" do
    test "builds a standalone S4 block" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden_size})

      block =
        S4.build_s4_block(input,
          hidden_size: @hidden_size,
          state_size: @state_size,
          dropout: 0.0,
          name: "test_s4_block"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(430)
      {input_data, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      # Block output preserves sequence dimension (before last_timestep extraction)
      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "block with dropout" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, @hidden_size})

      block =
        S4.build_s4_block(input,
          hidden_size: @hidden_size,
          state_size: @state_size,
          dropout: 0.2,
          name: "dropout_s4_block"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(431)
      {input_data, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, input_data)

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end

    @tag :slow
    @tag timeout: 120_000
    test "block uses defaults for missing options" do
      input = Axon.input("state_sequence", shape: {nil, @seq_len, 256})

      block = S4.build_s4_block(input, [])

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, 256}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(432)
      {input_data, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, 256})
      output = predict_fn.(params, input_data)

      # Default hidden_size=256
      assert Nx.shape(output) == {@batch, @seq_len, 256}
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
        state_size: @state_size,
        num_layers: 1,
        dropout: 0.0,
        window_size: 4,
        seq_len: 4
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, 4, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(440)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 4, @embed_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "longer seq_len=16" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 1,
        dropout: 0.0,
        window_size: 16,
        seq_len: 16
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, 16, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(441)
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
      assert S4.output_size(hidden_size: 128) == 128
    end

    test "returns default hidden_size" do
      assert S4.output_size() == 256
    end

    test "returns hidden_size from opts list with extra keys" do
      assert S4.output_size(hidden_size: 64, state_size: 32) == 64
    end
  end

  describe "param_count/1" do
    test "returns integer for typical opts" do
      count =
        S4.param_count(
          embed_dim: 287,
          hidden_size: 256,
          state_size: 64,
          num_layers: 4
        )

      assert is_integer(count)
      assert count > 0
    end

    test "param_count varies with state_size" do
      small =
        S4.param_count(
          embed_dim: 32,
          hidden_size: 32,
          state_size: 4,
          num_layers: 1
        )

      large =
        S4.param_count(
          embed_dim: 32,
          hidden_size: 32,
          state_size: 32,
          num_layers: 1
        )

      assert large > small
    end

    test "param_count varies with num_layers" do
      one =
        S4.param_count(
          embed_dim: 32,
          hidden_size: 32,
          state_size: 8,
          num_layers: 1
        )

      four =
        S4.param_count(
          embed_dim: 32,
          hidden_size: 32,
          state_size: 8,
          num_layers: 4
        )

      assert four > one
      # Should be exactly 4x per-layer params (same input proj since embed==hidden)
      assert four == 4 * one
    end

    test "param_count when embed_dim == hidden_size excludes input proj" do
      same =
        S4.param_count(
          embed_dim: 32,
          hidden_size: 32,
          state_size: 8,
          num_layers: 1
        )

      diff =
        S4.param_count(
          embed_dim: 64,
          hidden_size: 32,
          state_size: 8,
          num_layers: 1
        )

      assert diff > same
    end
  end

  describe "recommended_defaults/0" do
    test "returns expected keyword list" do
      defaults = S4.recommended_defaults()

      assert Keyword.keyword?(defaults)
      assert Keyword.get(defaults, :hidden_size) == 256
      assert Keyword.get(defaults, :state_size) == 64
      assert Keyword.get(defaults, :num_layers) == 4
      assert Keyword.get(defaults, :window_size) == 60
      assert Keyword.get(defaults, :dropout) == 0.1
    end
  end

  # ============================================================================
  # SSM correctness: HiPPO init properties
  # ============================================================================

  describe "HiPPO initialization properties" do
    test "SSM with varying input amplitudes produces different outputs" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 1,
        dropout: 0.0,
        window_size: @seq_len,
        seq_len: @seq_len
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(500)
      {base_input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})

      # Small scale
      input_small = Nx.multiply(base_input, 0.1)
      output_small = predict_fn.(params, input_small)

      # Large scale
      input_large = Nx.multiply(base_input, 2.0)
      output_large = predict_fn.(params, input_large)

      # Both should be finite
      assert Nx.all(Nx.is_nan(output_small) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(output_large) |> Nx.logical_not()) |> Nx.to_number() == 1

      # Different scale inputs should produce different outputs
      diff =
        Nx.subtract(output_small, output_large)
        |> Nx.abs()
        |> Nx.reduce_max()
        |> Nx.to_number()

      assert diff > 1.0e-6,
             "Different input scales should produce different outputs, got diff #{diff}"
    end

    test "zero input produces near-zero output" do
      opts = [
        embed_dim: @embed_dim,
        hidden_size: @hidden_size,
        state_size: @state_size,
        num_layers: 1,
        dropout: 0.0,
        window_size: @seq_len,
        seq_len: @seq_len
      ]

      model = S4.build(opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.0, {@batch, @seq_len, @embed_dim})
      output = predict_fn.(params, input)

      # Zero input through SSM should produce small output (biases may cause non-zero)
      max_abs = Nx.reduce_max(Nx.abs(output)) |> Nx.to_number()
      assert max_abs < 10.0, "Zero input should produce small output, got max abs #{max_abs}"
    end
  end
end
