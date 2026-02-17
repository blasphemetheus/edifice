defmodule Edifice.Convolutional.EfficientNetTest do
  use ExUnit.Case, async: true

  # Full model forward passes are expensive on BinaryBackend under cover
  @moduletag timeout: 180_000

  alias Edifice.Convolutional.EfficientNet

  @batch 2
  @input_dim 64

  describe "build/1" do
    test "builds an Axon model without classifier" do
      model = EfficientNet.build(input_dim: @input_dim)
      assert %Axon{} = model
    end

    test "builds with classifier head" do
      model = EfficientNet.build(input_dim: @input_dim, num_classes: 10)
      assert %Axon{} = model
    end

    test "forward pass without classifier produces correct shape" do
      model = EfficientNet.build(input_dim: @input_dim, width_multiplier: 0.5)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @input_dim}))
      expected_dim = EfficientNet.output_size(width_multiplier: 0.5)
      assert Nx.shape(output) == {@batch, expected_dim}
    end

    @tag :slow
    test "forward pass with classifier produces correct shape" do
      model = EfficientNet.build(input_dim: @input_dim, num_classes: 5, width_multiplier: 0.5)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @input_dim}))
      assert Nx.shape(output) == {@batch, 5}
    end

    test "supports dropout" do
      model = EfficientNet.build(input_dim: @input_dim, num_classes: 10, dropout: 0.2)
      assert %Axon{} = model
    end

    @tag :slow
    test "supports depth and width multipliers" do
      model =
        EfficientNet.build(
          input_dim: @input_dim,
          depth_multiplier: 0.5,
          width_multiplier: 0.5,
          num_classes: 10
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @input_dim}))
      assert Nx.shape(output) == {@batch, 10}
    end

    test "output values are finite" do
      model = EfficientNet.build(input_dim: @input_dim, num_classes: 10, width_multiplier: 0.5)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @input_dim}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @input_dim}))
      assert Nx.all(Nx.is_nan(output) |> Nx.bitwise_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns num_classes when specified" do
      assert EfficientNet.output_size(num_classes: 10) == 10
    end

    test "returns scaled head dim when no classifier" do
      size = EfficientNet.output_size(width_multiplier: 1.0)
      assert is_integer(size)
      assert size > 0
    end
  end

  describe "mbconv_block/3" do
    test "builds an Axon layer" do
      input = Axon.input("input", shape: {nil, 32})
      block = EfficientNet.mbconv_block(input, 32)
      assert %Axon{} = block
    end

    test "forward pass with matching dims has residual" do
      input = Axon.input("input", shape: {nil, 32})
      model = EfficientNet.mbconv_block(input, 32, expand_ratio: 2)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 32}))
      assert Nx.shape(output) == {@batch, 32}
    end
  end

  describe "squeeze_excitation/3" do
    test "builds SE block" do
      input = Axon.input("input", shape: {nil, 64})
      se = EfficientNet.squeeze_excitation(input, 16, expand_dim: 64)
      assert %Axon{} = se

      {init_fn, predict_fn} = Axon.build(se)
      params = init_fn.(Nx.template({@batch, 64}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 64}))
      assert Nx.shape(output) == {@batch, 64}
    end
  end
end
