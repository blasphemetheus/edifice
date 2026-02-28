defmodule Edifice.Utils.CommonTest do
  use ExUnit.Case, async: true
  @moduletag :utils

  alias Edifice.Utils.Common

  describe "align_dim/2" do
    test "returns dim unchanged when already aligned" do
      assert Common.align_dim(256, 8) == 256
      assert Common.align_dim(64, 8) == 64
      assert Common.align_dim(16, 16) == 16
    end

    test "rounds up to next multiple" do
      assert Common.align_dim(287, 8) == 288
      assert Common.align_dim(1, 8) == 8
      assert Common.align_dim(15, 16) == 16
      assert Common.align_dim(17, 16) == 32
    end

    test "default alignment is 8" do
      assert Common.align_dim(10) == 16
      assert Common.align_dim(8) == 8
    end

    test "handles zero" do
      assert Common.align_dim(0, 8) == 0
    end
  end

  describe "last_timestep/1" do
    test "extracts last timestep from 3D tensor" do
      tensor = Nx.iota({2, 5, 3}, type: :f32)
      result = Common.last_timestep(tensor)
      assert Nx.shape(result) == {2, 3}
      # Last timestep is index 4 for each batch
      expected = Nx.slice_along_axis(tensor, 4, 1, axis: 1) |> Nx.squeeze(axes: [1])
      assert Nx.equal(result, expected) |> Nx.all() |> Nx.to_number() == 1
    end

    test "works with single timestep" do
      tensor = Nx.broadcast(1.0, {3, 1, 8})
      result = Common.last_timestep(tensor)
      assert Nx.shape(result) == {3, 8}
    end
  end

  describe "last_timestep_layer/2" do
    test "builds an Axon layer" do
      input = Axon.input("input", shape: {nil, 10, 32})
      layer = Common.last_timestep_layer(input)
      assert %Axon{} = layer
    end

    test "produces correct output shape" do
      input = Axon.input("input", shape: {nil, 10, 32})
      model = Common.last_timestep_layer(input)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 10, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {2, 10, 32}))
      assert Nx.shape(output) == {2, 32}
    end

    test "accepts custom name" do
      input = Axon.input("input", shape: {nil, 10, 32})
      layer = Common.last_timestep_layer(input, name: "my_layer")
      assert %Axon{} = layer
    end
  end

  describe "ffn_block/3" do
    test "builds dense -> activation block" do
      input = Axon.input("input", shape: {nil, 16})
      block = Common.ffn_block(input, 32)
      assert %Axon{} = block
    end

    test "forward pass produces correct shape" do
      input = Axon.input("input", shape: {nil, 16})
      model = Common.ffn_block(input, 32)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({4, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {4, 16}))
      assert Nx.shape(output) == {4, 32}
    end

    test "supports dropout" do
      input = Axon.input("input", shape: {nil, 16})
      block = Common.ffn_block(input, 32, dropout: 0.5)
      assert %Axon{} = block
    end

    test "supports layer norm" do
      input = Axon.input("input", shape: {nil, 16})
      block = Common.ffn_block(input, 32, layer_norm: true)
      assert %Axon{} = block

      {init_fn, predict_fn} = Axon.build(block)
      params = init_fn.(Nx.template({4, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {4, 16}))
      assert Nx.shape(output) == {4, 32}
    end
  end

  describe "rms_norm/3" do
    test "normalizes tensor" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0])
      result = Common.rms_norm(x, gamma)
      assert Nx.shape(result) == {2, 3}
      assert Nx.all(Nx.is_nan(result) |> Nx.bitwise_not()) |> Nx.to_number() == 1
    end

    test "respects gamma scaling" do
      x = Nx.tensor([[1.0, 2.0, 3.0]])
      gamma_ones = Nx.tensor([1.0, 1.0, 1.0])
      gamma_twos = Nx.tensor([2.0, 2.0, 2.0])
      result_ones = Common.rms_norm(x, gamma_ones)
      result_twos = Common.rms_norm(x, gamma_twos)
      # result_twos should be 2x result_ones
      ratio = Nx.divide(result_twos, result_ones)
      assert Nx.all(Nx.less(Nx.abs(Nx.subtract(ratio, 2.0)), 1.0e-4)) |> Nx.to_number() == 1
    end
  end

  describe "silu/1" do
    test "computes x * sigmoid(x)" do
      x = Nx.tensor([0.0, 1.0, -1.0])
      result = Common.silu(x)
      # silu(0) = 0 * 0.5 = 0
      assert Nx.to_number(result[0]) |> abs() < 1.0e-5
      # silu(1) = 1 * sigmoid(1) ≈ 0.731
      assert abs(Nx.to_number(result[1]) - 0.731) < 0.01
    end
  end

  describe "gelu/1" do
    test "computes GELU activation" do
      x = Nx.tensor([0.0, 1.0, -1.0])
      result = Common.gelu(x)
      # gelu(0) = 0
      assert Nx.to_number(result[0]) |> abs() < 1.0e-5
      # gelu(1) ≈ 0.841
      assert abs(Nx.to_number(result[1]) - 0.841) < 0.01
    end
  end
end
