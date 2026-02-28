defmodule Edifice.Attention.PerceiverCoverageTest do
  @moduledoc """
  Additional coverage tests for Edifice.Attention.Perceiver.
  Targets uncovered code paths: param_count, recommended_defaults,
  multiple cross-attention layers, dropout=0 branch, different
  num_latents, build_cross_attention_block directly,
  build_self_attention_block directly.
  """
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.Perceiver

  @batch 2
  @seq_len 8
  @input_dim 16
  @latent_dim 16
  @num_latents 4
  @num_heads 4

  # ============================================================================
  # Multiple cross-attention layers
  # ============================================================================

  describe "multiple cross-attention layers" do
    test "with 2 cross-attention layers" do
      model =
        Perceiver.build(
          input_dim: @input_dim,
          latent_dim: @latent_dim,
          num_latents: @num_latents,
          num_layers: 4,
          num_cross_layers: 2,
          num_heads: @num_heads,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @input_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @input_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @latent_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "with 3 cross-attention layers and 6 total self-attention layers" do
      model =
        Perceiver.build(
          input_dim: @input_dim,
          latent_dim: @latent_dim,
          num_latents: @num_latents,
          num_layers: 6,
          num_cross_layers: 3,
          num_heads: @num_heads,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @input_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @input_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @latent_dim}
    end
  end

  # ============================================================================
  # Dropout = 0 branch
  # ============================================================================

  describe "dropout = 0" do
    test "skips dropout layers when dropout is 0" do
      model =
        Perceiver.build(
          input_dim: @input_dim,
          latent_dim: @latent_dim,
          num_latents: @num_latents,
          num_layers: 1,
          num_cross_layers: 1,
          num_heads: @num_heads,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @input_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @input_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @latent_dim}
    end
  end

  # ============================================================================
  # Dropout > 0 branch
  # ============================================================================

  describe "dropout > 0" do
    test "includes dropout layers when dropout > 0" do
      model =
        Perceiver.build(
          input_dim: @input_dim,
          latent_dim: @latent_dim,
          num_latents: @num_latents,
          num_layers: 1,
          num_cross_layers: 1,
          num_heads: @num_heads,
          dropout: 0.2
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @input_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @input_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @latent_dim}
    end
  end

  # ============================================================================
  # Different num_latents
  # ============================================================================

  describe "different num_latents" do
    test "with 2 latents" do
      model =
        Perceiver.build(
          input_dim: @input_dim,
          latent_dim: @latent_dim,
          num_latents: 2,
          num_layers: 1,
          num_cross_layers: 1,
          num_heads: @num_heads,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @input_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @input_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @latent_dim}
    end

    test "with 16 latents" do
      model =
        Perceiver.build(
          input_dim: @input_dim,
          latent_dim: @latent_dim,
          num_latents: 16,
          num_layers: 1,
          num_cross_layers: 1,
          num_heads: @num_heads,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @input_dim}, :f32), Axon.ModelState.empty())

      input = Nx.broadcast(0.5, {@batch, @seq_len, @input_dim})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @latent_dim}
    end
  end

  # ============================================================================
  # build_cross_attention_block/3 directly
  # ============================================================================

  describe "build_cross_attention_block/3" do
    test "builds cross-attention block with correct shapes" do
      latents_input = Axon.input("latents", shape: {nil, nil, @latent_dim})
      kv_input = Axon.input("kv", shape: {nil, nil, @latent_dim})

      block =
        Perceiver.build_cross_attention_block(latents_input, kv_input,
          latent_dim: @latent_dim,
          num_heads: @num_heads,
          dropout: 0.0,
          name: "test_cross"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{
            "latents" => Nx.template({@batch, @num_latents, @latent_dim}, :f32),
            "kv" => Nx.template({@batch, @seq_len, @latent_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      latents_data = Nx.broadcast(0.5, {@batch, @num_latents, @latent_dim})
      kv_data = Nx.broadcast(0.3, {@batch, @seq_len, @latent_dim})

      output =
        predict_fn.(params, %{"latents" => latents_data, "kv" => kv_data})

      assert Nx.shape(output) == {@batch, @num_latents, @latent_dim}
    end

    test "with dropout > 0" do
      latents_input = Axon.input("latents", shape: {nil, nil, @latent_dim})
      kv_input = Axon.input("kv", shape: {nil, nil, @latent_dim})

      block =
        Perceiver.build_cross_attention_block(latents_input, kv_input,
          latent_dim: @latent_dim,
          num_heads: @num_heads,
          dropout: 0.2,
          name: "test_cross_drop"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{
            "latents" => Nx.template({@batch, @num_latents, @latent_dim}, :f32),
            "kv" => Nx.template({@batch, @seq_len, @latent_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      latents_data = Nx.broadcast(0.5, {@batch, @num_latents, @latent_dim})
      kv_data = Nx.broadcast(0.3, {@batch, @seq_len, @latent_dim})

      output =
        predict_fn.(params, %{"latents" => latents_data, "kv" => kv_data})

      assert Nx.shape(output) == {@batch, @num_latents, @latent_dim}
    end
  end

  # ============================================================================
  # build_self_attention_block/2 directly
  # ============================================================================

  describe "build_self_attention_block/2" do
    test "builds self-attention block" do
      input = Axon.input("self_input", shape: {nil, nil, @latent_dim})

      block =
        Perceiver.build_self_attention_block(input,
          latent_dim: @latent_dim,
          num_heads: @num_heads,
          dropout: 0.0,
          name: "test_self"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{"self_input" => Nx.template({@batch, @num_latents, @latent_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @num_latents, @latent_dim})
      output = predict_fn.(params, %{"self_input" => input_data})

      assert Nx.shape(output) == {@batch, @num_latents, @latent_dim}
    end

    test "with dropout > 0" do
      input = Axon.input("self_input", shape: {nil, nil, @latent_dim})

      block =
        Perceiver.build_self_attention_block(input,
          latent_dim: @latent_dim,
          num_heads: @num_heads,
          dropout: 0.2,
          name: "test_self_drop"
        )

      {init_fn, predict_fn} = Axon.build(block, mode: :inference)

      params =
        init_fn.(
          %{"self_input" => Nx.template({@batch, @num_latents, @latent_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input_data = Nx.broadcast(0.5, {@batch, @num_latents, @latent_dim})
      output = predict_fn.(params, %{"self_input" => input_data})

      assert Nx.shape(output) == {@batch, @num_latents, @latent_dim}
    end
  end

  # ============================================================================
  # param_count/1
  # ============================================================================

  describe "param_count/1" do
    test "returns a positive integer" do
      count =
        Perceiver.param_count(
          input_dim: @input_dim,
          latent_dim: @latent_dim,
          num_latents: @num_latents,
          num_layers: 2,
          num_cross_layers: 1
        )

      assert is_integer(count)
      assert count > 0
    end

    test "more layers increases param count" do
      count_1 =
        Perceiver.param_count(
          input_dim: @input_dim,
          latent_dim: @latent_dim,
          num_latents: @num_latents,
          num_layers: 1,
          num_cross_layers: 1
        )

      count_4 =
        Perceiver.param_count(
          input_dim: @input_dim,
          latent_dim: @latent_dim,
          num_latents: @num_latents,
          num_layers: 4,
          num_cross_layers: 1
        )

      assert count_4 > count_1
    end

    test "more cross-layers increases param count" do
      count_1 =
        Perceiver.param_count(
          input_dim: @input_dim,
          latent_dim: @latent_dim,
          num_latents: @num_latents,
          num_layers: 2,
          num_cross_layers: 1
        )

      count_3 =
        Perceiver.param_count(
          input_dim: @input_dim,
          latent_dim: @latent_dim,
          num_latents: @num_latents,
          num_layers: 2,
          num_cross_layers: 3
        )

      assert count_3 > count_1
    end
  end

  # ============================================================================
  # output_size/1
  # ============================================================================

  describe "output_size/1" do
    test "returns default when no opts given" do
      assert Perceiver.output_size() == 256
    end

    test "returns custom latent_dim" do
      assert Perceiver.output_size(latent_dim: 128) == 128
    end
  end

  # ============================================================================
  # recommended_defaults/0
  # ============================================================================

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = Perceiver.recommended_defaults()
      assert Keyword.has_key?(defaults, :latent_dim)
      assert Keyword.has_key?(defaults, :num_latents)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :num_cross_layers)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :dropout)
    end
  end
end
