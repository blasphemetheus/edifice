defmodule Edifice.Convolutional.ConvTest do
  use ExUnit.Case, async: true

  alias Edifice.Convolutional.Conv

  @batch 2

  describe "build_conv1d/1" do
    test "builds an Axon model" do
      model = Conv.build_conv1d(input_size: 8, channels: [16], seq_len: 10)
      assert %Axon{} = model
    end

    test "forward pass produces correct shape" do
      model = Conv.build_conv1d(input_size: 8, channels: [16], seq_len: 10)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 10, 8}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 10, 8}))
      {b, _, ch} = Nx.shape(output)
      assert b == @batch
      assert ch == 16
    end

    test "supports multiple layers" do
      model = Conv.build_conv1d(input_size: 8, channels: [16, 32], seq_len: 10)
      assert %Axon{} = model
    end

    test "supports max pooling" do
      model =
        Conv.build_conv1d(input_size: 8, channels: [16], seq_len: 10, pooling: :max)

      assert %Axon{} = model
    end

    test "supports avg pooling" do
      model =
        Conv.build_conv1d(input_size: 8, channels: [16], seq_len: 10, pooling: :avg)

      assert %Axon{} = model
    end

    test "supports custom activation" do
      model =
        Conv.build_conv1d(input_size: 8, channels: [16], seq_len: 10, activation: :gelu)

      assert %Axon{} = model
    end

    test "supports no dropout" do
      model = Conv.build_conv1d(input_size: 8, channels: [16], seq_len: 10, dropout: 0.0)
      assert %Axon{} = model
    end
  end

  describe "build/1" do
    test "aliases build_conv1d" do
      model = Conv.build(input_size: 8, channels: [16], seq_len: 10)
      assert %Axon{} = model
    end
  end

  describe "conv1d_block/3" do
    test "builds a single conv1d block" do
      input = Axon.input("input", shape: {nil, 10, 8})
      block = Conv.conv1d_block(input, 16)
      assert %Axon{} = block
    end

    test "supports custom options" do
      input = Axon.input("input", shape: {nil, 10, 8})

      block =
        Conv.conv1d_block(input, 16,
          kernel_size: 5,
          activation: :gelu,
          padding: :valid,
          strides: 2
        )

      assert %Axon{} = block
    end
  end

  describe "build_conv2d/1" do
    test "builds a 2D conv model" do
      model =
        Conv.build_conv2d(
          input_shape: {nil, 8, 8, 3},
          channels: [8]
        )

      assert %Axon{} = model
    end

    test "forward pass produces correct shape" do
      model =
        Conv.build_conv2d(
          input_shape: {nil, 8, 8, 3},
          channels: [8]
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 8, 8, 3}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 8, 8, 3}))
      {b, _, _, ch} = Nx.shape(output)
      assert b == @batch
      assert ch == 8
    end

    test "supports pooling" do
      model =
        Conv.build_conv2d(
          input_shape: {nil, 8, 8, 3},
          channels: [8],
          pooling: :max
        )

      assert %Axon{} = model
    end
  end

  describe "conv2d_block/3" do
    test "builds a single conv2d block" do
      input = Axon.input("input", shape: {nil, 8, 8, 3})
      block = Conv.conv2d_block(input, 16)
      assert %Axon{} = block
    end
  end

  describe "output_size/1" do
    test "returns last channel count" do
      assert Conv.output_size(channels: [32, 64, 128]) == 128
      assert Conv.output_size(channels: [16]) == 16
    end

    test "default channels" do
      assert Conv.output_size() == 256
    end
  end

  describe "kernel_sizes validation" do
    test "raises on mismatched kernel_sizes list" do
      assert_raise ArgumentError, ~r/kernel_sizes list length/, fn ->
        Conv.build_conv1d(input_size: 8, channels: [16, 32], kernel_sizes: [3])
      end
    end
  end
end
