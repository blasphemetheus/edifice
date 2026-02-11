defmodule Edifice.Convolutional.ResNetTest do
  use ExUnit.Case, async: true

  alias Edifice.Convolutional.ResNet

  @batch_size 2

  describe "build/1 with residual blocks" do
    test "builds ResNet-18 style model with correct output shape" do
      num_classes = 10

      model =
        ResNet.build(
          input_shape: {nil, 32, 32, 3},
          num_classes: num_classes,
          block_sizes: [2, 2, 2, 2],
          initial_channels: 16
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 32, 32, 3}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, 32, 32, 3}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, num_classes}
    end

    test "supports custom num_classes" do
      num_classes = 100

      model =
        ResNet.build(
          input_shape: {nil, 32, 32, 3},
          num_classes: num_classes,
          block_sizes: [1, 1, 1, 1],
          initial_channels: 8
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 32, 32, 3}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, 32, 32, 3}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, num_classes}
    end
  end

  describe "build/1 with bottleneck blocks" do
    test "builds bottleneck ResNet with correct output shape" do
      num_classes = 10

      model =
        ResNet.build(
          input_shape: {nil, 32, 32, 3},
          num_classes: num_classes,
          block_sizes: [1, 1, 1, 1],
          block_type: :bottleneck,
          initial_channels: 8
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 32, 32, 3}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, 32, 32, 3}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, num_classes}
    end
  end

  describe "build/1 with dropout" do
    test "supports dropout option" do
      model =
        ResNet.build(
          input_shape: {nil, 32, 32, 3},
          num_classes: 10,
          block_sizes: [1, 1, 1, 1],
          initial_channels: 8,
          dropout: 0.5
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 32, 32, 3}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, 32, 32, 3}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 10}
    end
  end

  describe "output_size/1" do
    test "returns num_classes" do
      assert ResNet.output_size(num_classes: 1000) == 1000
    end

    test "returns default when not specified" do
      assert ResNet.output_size() == 10
    end
  end
end
