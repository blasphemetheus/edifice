defmodule Edifice.Attention.GLACoverageTest do
  @moduledoc """
  Additional coverage tests for Edifice.Attention.GLA.
  Targets uncovered code paths: build_gla_block directly, build_gated_linear_attention
  directly, param_count, recommended_defaults, embed_dim == hidden_size branch,
  dropout=0 branch, different head/layer configurations.
  """
  use ExUnit.Case, async: true

  alias Edifice.Attention.GLA

  @batch 2
  @seq_len 8
  @embed_dim 16
  @hidden_size 16
  @num_heads 2
  @head_dim 8

  # ============================================================================
  # embed_dim == hidden_size (skips input projection)
  # ============================================================================

  describe "embed_dim == hidden_size" do
    test "skips input projection when embed_dim == hidden_size" do
      model =
        GLA.build(
          embed_dim: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 1,
          num_heads: @num_heads,
          head_dim: @head_dim,
          seq_len: @seq_len,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @hidden_size}, :f32), Axon.ModelState.empty())

      # Should NOT have input_projection in params
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
        GLA.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          num_heads: @num_heads,
          head_dim: @head_dim,
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
  # build_gla_block/2 directly
  # ============================================================================

  describe "build_gla_block/2" do
    test "builds a single GLA block with correct output shape" do
      input = Axon.input("gla_input", shape: {nil, nil, @hidden_size})

      block =
        GLA.build_gla_block(input,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          head_dim: @head_dim,
          expand_factor: 2,
          dropout: 0.0,
          name: "test_gla_block"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{"gla_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"gla_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end

    test "builds a GLA block with different expand_factor" do
      input = Axon.input("gla_input", shape: {nil, nil, @hidden_size})

      block =
        GLA.build_gla_block(input,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          head_dim: @head_dim,
          expand_factor: 4,
          dropout: 0.0,
          name: "test_gla_expand4"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{"gla_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"gla_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end
  end

  # ============================================================================
  # build_gated_linear_attention/2 directly
  # ============================================================================

  describe "build_gated_linear_attention/2" do
    test "builds gated linear attention layer" do
      input = Axon.input("gla_input", shape: {nil, nil, @hidden_size})

      attn_out =
        GLA.build_gated_linear_attention(input,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          head_dim: @head_dim,
          dropout: 0.0,
          name: "test_gla"
        )

      {init_fn, predict_fn} = Axon.build(attn_out, mode: :inference)

      params =
        init_fn.(
          %{"gla_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"gla_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end

    test "with dropout > 0" do
      input = Axon.input("gla_input", shape: {nil, nil, @hidden_size})

      attn_out =
        GLA.build_gated_linear_attention(input,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          head_dim: @head_dim,
          dropout: 0.2,
          name: "test_gla_drop"
        )

      {init_fn, predict_fn} = Axon.build(attn_out, mode: :inference)

      params =
        init_fn.(
          %{"gla_input" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @seq_len, @hidden_size})
      output = predict_fn.(params, %{"gla_input" => input_data})

      assert Nx.shape(output) == {@batch, @seq_len, @hidden_size}
    end
  end

  # ============================================================================
  # Multi-layer configurations
  # ============================================================================

  describe "multi-layer" do
    test "with 3 layers" do
      model =
        GLA.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 3,
          num_heads: @num_heads,
          head_dim: @head_dim,
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
  # param_count/1
  # ============================================================================

  describe "param_count/1" do
    test "returns a positive integer" do
      count =
        GLA.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 2,
          num_heads: @num_heads,
          head_dim: @head_dim,
          expand_factor: 2
        )

      assert is_integer(count)
      assert count > 0
    end

    test "more layers increases param count" do
      count_1 =
        GLA.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 1,
          num_heads: @num_heads,
          head_dim: @head_dim,
          expand_factor: 2
        )

      count_4 =
        GLA.param_count(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_layers: 4,
          num_heads: @num_heads,
          head_dim: @head_dim,
          expand_factor: 2
        )

      assert count_4 > count_1
    end

    test "embed_dim == hidden_size has no input projection cost" do
      count_same =
        GLA.param_count(
          embed_dim: @hidden_size,
          hidden_size: @hidden_size,
          num_layers: 1,
          num_heads: @num_heads,
          head_dim: @head_dim,
          expand_factor: 2
        )

      count_diff =
        GLA.param_count(
          embed_dim: @embed_dim + 8,
          hidden_size: @hidden_size,
          num_layers: 1,
          num_heads: @num_heads,
          head_dim: @head_dim,
          expand_factor: 2
        )

      assert count_diff > count_same
    end
  end

  # ============================================================================
  # output_size/1 edge cases
  # ============================================================================

  describe "output_size/1" do
    test "returns default when no opts given" do
      assert GLA.output_size() == 256
    end

    test "returns custom hidden_size" do
      assert GLA.output_size(hidden_size: 128) == 128
    end
  end

  # ============================================================================
  # recommended_defaults/0
  # ============================================================================

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = GLA.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :head_dim)
      assert Keyword.has_key?(defaults, :expand_factor)
      assert Keyword.has_key?(defaults, :window_size)
      assert Keyword.has_key?(defaults, :dropout)
    end
  end
end
