defmodule Edifice.Blocks.DepthwiseConvTest do
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.Blocks.DepthwiseConv

  @batch 2
  @seq_len 16
  @channels 32

  describe "layer/3 with defaults" do
    test "produces correct output shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @channels})
      model = DepthwiseConv.layer(input, @channels)

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @channels}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @channels}))

      assert Nx.shape(output) == {@batch, @seq_len, @channels}
    end

    test "output values are finite" do
      input = Axon.input("input", shape: {nil, @seq_len, @channels})
      model = DepthwiseConv.layer(input, @channels)

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @channels}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @channels})
      output = predict_fn.(params, test_input)

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "handles batch_size=1" do
      input = Axon.input("input", shape: {nil, @seq_len, @channels})
      model = DepthwiseConv.layer(input, @channels)

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({1, @seq_len, @channels}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {1, @seq_len, @channels}))

      assert Nx.shape(output) == {1, @seq_len, @channels}
    end
  end

  describe "layer/3 with custom kernel_size" do
    test "kernel_size=3 produces correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @channels})
      model = DepthwiseConv.layer(input, @channels, 3)

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @channels}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @channels}))

      assert Nx.shape(output) == {@batch, @seq_len, @channels}
    end
  end

  describe "options" do
    test "out_channels differs from input channels" do
      input = Axon.input("input", shape: {nil, @seq_len, @channels})
      model = DepthwiseConv.layer(input, @channels, 5, out_channels: 64)

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @channels}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @channels}))

      assert Nx.shape(output) == {@batch, @seq_len, 64}
    end

    test "use_norm=false skips normalization" do
      input = Axon.input("input", shape: {nil, @seq_len, @channels})
      model = DepthwiseConv.layer(input, @channels, 5, use_norm: false)

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @channels}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @channels}))

      assert Nx.shape(output) == {@batch, @seq_len, @channels}
    end

    test "padding :same produces correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @channels})
      model = DepthwiseConv.layer(input, @channels, 5, padding: :same)

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @seq_len, @channels}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @channels}))

      assert Nx.shape(output) == {@batch, @seq_len, @channels}
    end
  end
end
