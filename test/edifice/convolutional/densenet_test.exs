defmodule Edifice.Convolutional.DenseNetTest do
  use ExUnit.Case, async: true

  alias Edifice.Convolutional.DenseNet

  @batch_size 2

  describe "build/1" do
    test "builds compact DenseNet with correct output shape" do
      num_classes = 10

      model =
        DenseNet.build(
          input_shape: {nil, 32, 32, 3},
          num_classes: num_classes,
          growth_rate: 8,
          block_config: [2, 2, 2, 2],
          compression: 0.5
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 32, 32, 3}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, 32, 32, 3}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, num_classes}
    end

    test "supports custom num_classes" do
      num_classes = 50

      model =
        DenseNet.build(
          input_shape: {nil, 32, 32, 3},
          num_classes: num_classes,
          growth_rate: 8,
          block_config: [2, 2],
          compression: 0.5
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 32, 32, 3}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, 32, 32, 3}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, num_classes}
    end

    test "supports dropout option" do
      model =
        DenseNet.build(
          input_shape: {nil, 32, 32, 3},
          num_classes: 10,
          growth_rate: 8,
          block_config: [2, 2],
          dropout: 0.2
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
      assert DenseNet.output_size(num_classes: 1000) == 1000
    end

    test "returns default when not specified" do
      assert DenseNet.output_size() == 10
    end
  end
end
