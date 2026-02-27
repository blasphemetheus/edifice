defmodule Edifice.Blocks.SoftpickTest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.Softpick

  describe "compute/2" do
    test "output bounded in (-1, 1)" do
      x = Nx.tensor([[-3.0, -1.0, 0.0, 2.0, 5.0]])
      result = Softpick.compute(x)

      max_val = Nx.to_number(Nx.reduce_max(result))
      min_val = Nx.to_number(Nx.reduce_min(result))

      assert max_val < 1.0
      assert min_val > -1.0
    end

    test "preserves sign" do
      x = Nx.tensor([[-2.0, -1.0, 0.0, 1.0, 3.0]])
      result = Softpick.compute(x)

      # Negative inputs should produce negative outputs
      assert Nx.to_number(result[0][0]) < 0
      assert Nx.to_number(result[0][1]) < 0
      # Zero input should produce zero output
      assert_in_delta Nx.to_number(result[0][2]), 0.0, 1.0e-6
      # Positive inputs should produce positive outputs
      assert Nx.to_number(result[0][3]) > 0
      assert Nx.to_number(result[0][4]) > 0
    end

    test "zero input produces zero output" do
      x = Nx.tensor([[0.0, 0.0, 0.0]])
      result = Softpick.compute(x)

      max_val = Nx.to_number(Nx.reduce_max(Nx.abs(result)))
      assert max_val < 1.0e-6
    end

    test "handles single element" do
      x = Nx.tensor([[5.0]])
      result = Softpick.compute(x)

      # 5.0 / (1 + 5.0) = 5/6 ≈ 0.833
      assert_in_delta Nx.to_number(result[0][0]), 5.0 / 6.0, 1.0e-5
    end

    test "maintains relative ordering" do
      x = Nx.tensor([[1.0, 3.0, 2.0]])
      result = Softpick.compute(x)

      v0 = Nx.to_number(result[0][0])
      v1 = Nx.to_number(result[0][1])
      v2 = Nx.to_number(result[0][2])

      assert v1 > v2
      assert v2 > v0
    end

    test "custom axis" do
      # [2, 3] tensor, normalize along axis 0
      x = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      result = Softpick.compute(x, axis: 0)

      assert Nx.shape(result) == {2, 3}
    end
  end

  describe "layer/2" do
    test "builds and runs in Axon model" do
      input = Axon.input("input", shape: {nil, 8, 32})
      model = Softpick.layer(input)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(1.0, {2, 8, 32}))

      assert Nx.shape(output) == {2, 8, 32}
    end
  end

  describe "build/1" do
    test "builds a full model" do
      model = Softpick.build(embed_dim: 16, hidden_size: 32, num_layers: 1, window_size: 8)

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({2, 8, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {2, 8, 16}))

      assert Nx.shape(output) == {2, 32}
    end
  end
end
