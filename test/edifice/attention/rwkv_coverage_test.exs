defmodule Edifice.Attention.RWKVCoverageTest do
  @moduledoc """
  Additional coverage tests for Edifice.Attention.RWKV.
  Targets uncovered code paths: init_cache, param_count, recommended_defaults,
  embed_dim == hidden_size branch, dropout=0 branch, build_rwkv_block directly,
  build_time_mixing directly, build_channel_mixing directly, different head sizes.
  """
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.RWKV

  @batch 2
  @seq_len 8
  @embed_dim 16
  @hidden_size 16
  @head_size 8

  # ============================================================================
  # embed_dim == hidden_size (skips input projection)
  # ============================================================================

  describe "embed_dim == hidden_size" do
    test "skips input projection" do
      model =
        RWKV.build(
          embed_dim: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 1,
          head_size: @head_size,
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
  # dropout = 0 branch
  # ============================================================================

  describe "dropout = 0" do
    test "builds and runs without dropout layers" do
      model =
        RWKV.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          head_size: @head_size,
          seq_len: @seq_len,
          dropout: 0.0
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
  # build_rwkv_block/2 directly
  # ============================================================================

  describe "build_rwkv_block/2" do
    test "builds a single RWKV block" do
      input = Axon.input("rwkv_input", shape: {nil, nil, @hidden_size})

      block =
        RWKV.build_rwkv_block(input,
          hidden_size: @hidden_size,
          head_size: @head_size,
          num_heads: div(@hidden_size, @head_size),
          dropout: 0.0,
          name: "test_rwkv_block"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{"rwkv_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"rwkv_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end

    test "with dropout > 0" do
      input = Axon.input("rwkv_input", shape: {nil, nil, @hidden_size})

      block =
        RWKV.build_rwkv_block(input,
          hidden_size: @hidden_size,
          head_size: @head_size,
          num_heads: div(@hidden_size, @head_size),
          dropout: 0.2,
          name: "test_rwkv_block_drop"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{"rwkv_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"rwkv_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end
  end

  # ============================================================================
  # build_time_mixing/2 directly
  # ============================================================================

  describe "build_time_mixing/2" do
    test "builds time mixing sub-block" do
      input = Axon.input("tm_input", shape: {nil, nil, @hidden_size})

      time_mix =
        RWKV.build_time_mixing(input,
          hidden_size: @hidden_size,
          head_size: @head_size,
          num_heads: div(@hidden_size, @head_size),
          dropout: 0.0,
          name: "test_time_mix"
        )

      {init_fn, predict_fn} = Axon.build(time_mix, mode: :inference)

      params =
        init_fn.(
          %{"tm_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"tm_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end

    test "with dropout > 0" do
      input = Axon.input("tm_input", shape: {nil, nil, @hidden_size})

      time_mix =
        RWKV.build_time_mixing(input,
          hidden_size: @hidden_size,
          head_size: @head_size,
          num_heads: div(@hidden_size, @head_size),
          dropout: 0.2,
          name: "test_time_mix_drop"
        )

      {init_fn, predict_fn} = Axon.build(time_mix, mode: :inference)

      params =
        init_fn.(
          %{"tm_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"tm_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end
  end

  # ============================================================================
  # build_channel_mixing/2 directly
  # ============================================================================

  describe "build_channel_mixing/2" do
    test "builds channel mixing sub-block" do
      input = Axon.input("cm_input", shape: {nil, nil, @hidden_size})

      channel_mix =
        RWKV.build_channel_mixing(input,
          hidden_size: @hidden_size,
          dropout: 0.0,
          name: "test_channel_mix"
        )

      {init_fn, predict_fn} = Axon.build(channel_mix, mode: :inference)

      params =
        init_fn.(
          %{"cm_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"cm_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end

    test "with dropout > 0" do
      input = Axon.input("cm_input", shape: {nil, nil, @hidden_size})

      channel_mix =
        RWKV.build_channel_mixing(input,
          hidden_size: @hidden_size,
          dropout: 0.2,
          name: "test_channel_mix_drop"
        )

      {init_fn, predict_fn} = Axon.build(channel_mix, mode: :inference)

      params =
        init_fn.(
          %{"cm_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"cm_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end
  end

  # ============================================================================
  # Different head sizes
  # ============================================================================

  describe "different head sizes" do
    test "with head_size = 4" do
      model =
        RWKV.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          head_size: 4,
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

    test "with head_size = hidden_size (single head)" do
      model =
        RWKV.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          head_size: @hidden_size,
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

  # ============================================================================
  # Multi-layer configurations
  # ============================================================================

  describe "multi-layer" do
    test "with 3 layers" do
      model =
        RWKV.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 3,
          head_size: @head_size,
          seq_len: @seq_len,
          dropout: 0.0
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
  # init_cache/1
  # ============================================================================

  describe "init_cache/1" do
    test "returns cache with correct structure and shapes" do
      cache =
        RWKV.init_cache(
          batch_size: @batch,
          hidden_size: @hidden_size,
          head_size: @head_size,
          num_layers: 2
        )

      assert is_map(cache)
      assert cache.step == 0
      assert cache.config.hidden_size == @hidden_size
      assert cache.config.head_size == @head_size
      assert cache.config.num_heads == div(@hidden_size, @head_size)
      assert cache.config.num_layers == 2

      # Check layer caches
      assert Map.has_key?(cache.layers, "layer_1")
      assert Map.has_key?(cache.layers, "layer_2")

      layer_1 = cache.layers["layer_1"]
      num_heads = div(@hidden_size, @head_size)

      assert Nx.shape(layer_1.wkv_numerator) == {@batch, num_heads, @head_size}
      assert Nx.shape(layer_1.wkv_denominator) == {@batch, num_heads, @head_size}
      assert Nx.shape(layer_1.last_token_time) == {@batch, @hidden_size}
      assert Nx.shape(layer_1.last_token_channel) == {@batch, @hidden_size}
    end

    test "with default options" do
      cache = RWKV.init_cache()

      assert cache.config.hidden_size == 256
      assert cache.config.head_size == 64
      assert cache.config.num_heads == 4
      assert cache.config.num_layers == 6
      assert cache.step == 0
    end

    test "with custom batch_size" do
      cache =
        RWKV.init_cache(
          batch_size: 4,
          hidden_size: 32,
          head_size: 8,
          num_layers: 1
        )

      layer_1 = cache.layers["layer_1"]
      assert Nx.shape(layer_1.wkv_numerator) == {4, 4, 8}
      assert Nx.shape(layer_1.last_token_time) == {4, 32}
    end
  end

  # ============================================================================
  # param_count/1
  # ============================================================================

  describe "param_count/1" do
    test "returns a positive integer" do
      count =
        RWKV.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "more layers increases param count" do
      count_1 =
        RWKV.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1
        )

      count_4 =
        RWKV.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 4
        )

      assert count_4 > count_1
    end

    test "embed_dim == hidden_size has no input projection cost" do
      count_same =
        RWKV.param_count(
          embed_dim: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 1
        )

      count_diff =
        RWKV.param_count(
          embed_dim: @embed_dim + 8,
          hidden_size: @hidden_size,
          num_layers: 1
        )

      assert count_diff > count_same
    end
  end

  # ============================================================================
  # output_size/1
  # ============================================================================

  describe "output_size/1" do
    test "returns default when no opts given" do
      assert RWKV.output_size() == 256
    end

    test "returns custom hidden_size" do
      assert RWKV.output_size(hidden_size: 128) == 128
    end
  end

  # ============================================================================
  # recommended_defaults/0
  # ============================================================================

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = RWKV.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :head_size)
      assert Keyword.has_key?(defaults, :window_size)
      assert Keyword.has_key?(defaults, :dropout)
    end
  end
end
