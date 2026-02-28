defmodule Edifice.Blocks.ModelBuilderTest do
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @batch 2
  @seq_len 8

  defp simple_block_builder(input, opts) do
    name = "block_#{opts[:layer_idx]}"
    Axon.dense(input, opts[:hidden_size], name: "#{name}_dense")
  end

  defp transformer_block_builder(input, opts) do
    layer_idx = opts[:layer_idx]
    hidden_size = opts[:hidden_size]
    name = "block_#{layer_idx}"

    attn_fn = fn x, attn_name ->
      Axon.dense(x, hidden_size, name: "#{attn_name}_proj")
    end

    TransformerBlock.layer(input,
      attention_fn: attn_fn,
      hidden_size: hidden_size,
      name: name
    )
  end

  describe "build_sequence_model/1 with :last_timestep" do
    test "produces [batch, hidden_size] output" do
      model =
        ModelBuilder.build_sequence_model(
          embed_dim: 16,
          hidden_size: 32,
          num_layers: 2,
          seq_len: @seq_len,
          block_builder: &transformer_block_builder/2,
          output_mode: :last_timestep
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, 16}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, 16})
      output = predict_fn.(params, test_input)

      assert Nx.shape(output) == {@batch, 32}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end

  describe "build_sequence_model/1 with :all" do
    test "produces [batch, seq_len, hidden_size] output" do
      model =
        ModelBuilder.build_sequence_model(
          embed_dim: 32,
          hidden_size: 32,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: &simple_block_builder/2,
          output_mode: :all
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(1.0, {@batch, @seq_len, 32}))

      assert Nx.shape(output) == {@batch, @seq_len, 32}
    end
  end

  describe "build_sequence_model/1 with :mean_pool" do
    test "produces [batch, hidden_size] output" do
      model =
        ModelBuilder.build_sequence_model(
          embed_dim: 32,
          hidden_size: 32,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: &simple_block_builder/2,
          output_mode: :mean_pool
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(1.0, {@batch, @seq_len, 32}))

      assert Nx.shape(output) == {@batch, 32}
    end
  end

  describe "build_sequence_model/1 options" do
    test "projects input when embed_dim != hidden_size" do
      model =
        ModelBuilder.build_sequence_model(
          embed_dim: 16,
          hidden_size: 64,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: &simple_block_builder/2,
          output_mode: :last_timestep
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, 16}, :f32), Axon.ModelState.empty())

      # Should have input_projection in params
      assert Map.has_key?(params.data, "input_projection")
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, 16}))
      assert Nx.shape(output) == {@batch, 64}
    end

    test "no projection when embed_dim == hidden_size" do
      model =
        ModelBuilder.build_sequence_model(
          embed_dim: 32,
          hidden_size: 32,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: &simple_block_builder/2,
          output_mode: :last_timestep
        )

      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, 32}, :f32), Axon.ModelState.empty())

      refute Map.has_key?(params.data, "input_projection")
    end

    test "final_norm: false skips normalization" do
      model =
        ModelBuilder.build_sequence_model(
          embed_dim: 32,
          hidden_size: 32,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: &simple_block_builder/2,
          output_mode: :all,
          final_norm: false
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, 32}, :f32), Axon.ModelState.empty())

      refute Map.has_key?(params.data, "final_norm")
      output = predict_fn.(params, Nx.broadcast(1.0, {@batch, @seq_len, 32}))
      assert Nx.shape(output) == {@batch, @seq_len, 32}
    end

    test "handles batch_size=1" do
      model =
        ModelBuilder.build_sequence_model(
          embed_dim: 16,
          hidden_size: 32,
          num_layers: 1,
          seq_len: @seq_len,
          block_builder: &simple_block_builder/2,
          output_mode: :last_timestep
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, @seq_len, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {1, @seq_len, 16}))

      assert Nx.shape(output) == {1, 32}
    end
  end

  describe "build_vision_model/1" do
    test "produces correct output shape" do
      block_builder = fn input, opts ->
        name = "vit_block_#{opts[:layer_idx]}"
        Axon.dense(input, opts[:hidden_size], name: "#{name}_dense")
      end

      model =
        ModelBuilder.build_vision_model(
          image_size: 32,
          patch_size: 8,
          in_channels: 3,
          hidden_size: 64,
          num_layers: 1,
          block_builder: block_builder
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 3, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 3, 32, 32}))

      # Mean pool over 16 patches -> [batch, 64]
      assert Nx.shape(output) == {@batch, 64}
    end

    test "with classifier head" do
      block_builder = fn input, opts ->
        name = "vit_block_#{opts[:layer_idx]}"
        Axon.dense(input, opts[:hidden_size], name: "#{name}_dense")
      end

      model =
        ModelBuilder.build_vision_model(
          image_size: 32,
          patch_size: 8,
          in_channels: 3,
          hidden_size: 64,
          num_layers: 1,
          block_builder: block_builder,
          num_classes: 10
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 3, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 3, 32, 32}))

      assert Nx.shape(output) == {@batch, 10}
    end
  end
end
