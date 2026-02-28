defmodule Edifice.Blocks.Upsample2xTest do
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.Blocks.Upsample2x

  describe "layer/2" do
    test "doubles spatial dimensions" do
      input = Axon.input("input", shape: {nil, 4, 4, 8})
      model = Upsample2x.layer(input, "upsample")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 4, 4, 8}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(1.0, {2, 4, 4, 8}))

      assert Nx.shape(output) == {2, 8, 8, 8}
    end

    test "preserves channel dimension" do
      input = Axon.input("input", shape: {nil, 2, 3, 16})
      model = Upsample2x.layer(input, "up")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 2, 3, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {1, 2, 3, 16}))

      assert Nx.shape(output) == {1, 4, 6, 16}
    end

    test "nearest-neighbor: each pixel replicated 2x2" do
      # Create a 1x1 input per channel, should become 2x2
      input = Axon.input("input", shape: {nil, 1, 1, 2})
      model = Upsample2x.layer(input, "up")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 1, 1, 2}, :f32), Axon.ModelState.empty())

      test_input = Nx.tensor([[[[3.0, 7.0]]]])
      output = predict_fn.(params, test_input)

      assert Nx.shape(output) == {1, 2, 2, 2}

      # All 4 pixels should have the same value
      assert_in_delta Nx.to_number(output[0][0][0][0]), 3.0, 1.0e-5
      assert_in_delta Nx.to_number(output[0][0][1][0]), 3.0, 1.0e-5
      assert_in_delta Nx.to_number(output[0][1][0][0]), 3.0, 1.0e-5
      assert_in_delta Nx.to_number(output[0][1][1][0]), 3.0, 1.0e-5

      assert_in_delta Nx.to_number(output[0][0][0][1]), 7.0, 1.0e-5
    end

    test "handles batch_size=1" do
      input = Axon.input("input", shape: {nil, 4, 4, 3})
      model = Upsample2x.layer(input, "up")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 4, 4, 3}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {1, 4, 4, 3}))

      assert Nx.shape(output) == {1, 8, 8, 3}
    end
  end
end
