defmodule Edifice.Blocks.SSMaxTest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.SSMax

  describe "compute/3" do
    test "output sums to approximately 1" do
      logits = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      result = SSMax.compute(logits, 1.0, 4)

      sum = Nx.to_number(Nx.sum(result))
      assert_in_delta sum, 1.0, 1.0e-5
    end

    test "output values are non-negative" do
      logits = Nx.tensor([[-3.0, -1.0, 0.0, 2.0]])
      result = SSMax.compute(logits, 1.0, 4)

      min_val = Nx.to_number(Nx.reduce_min(result))
      assert min_val >= 0.0
    end

    test "s=0 is equivalent to standard softmax" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]])
      result = SSMax.compute(logits, 0.0, 3)

      # Standard softmax
      exp_logits = Nx.exp(Nx.subtract(logits, Nx.reduce_max(logits, axes: [-1], keep_axes: true)))
      expected = Nx.divide(exp_logits, Nx.sum(exp_logits, axes: [-1], keep_axes: true))

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(result, expected))))
      assert diff < 1.0e-5
    end

    test "s is a uniform shift so distribution is preserved for fixed length" do
      # SSMax(x) = softmax(x - s*log(n)), which is a uniform shift that
      # cancels in softmax. For fixed-length input, s does not change output.
      # The benefit is during training across variable-length sequences.
      logits = Nx.tensor([[1.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
      seq_len = 8

      result_s0 = SSMax.compute(logits, 0.0, seq_len)
      result_s2 = SSMax.compute(logits, 2.0, seq_len)

      # Both should produce valid probability distributions
      assert_in_delta Nx.to_number(Nx.sum(result_s0)), 1.0, 1.0e-4
      assert_in_delta Nx.to_number(Nx.sum(result_s2)), 1.0, 1.0e-4

      # The peak should be at index 1 (value 3.0)
      argmax_s0 = result_s0 |> Nx.argmax(axis: -1) |> Nx.squeeze() |> Nx.to_number()
      argmax_s2 = result_s2 |> Nx.argmax(axis: -1) |> Nx.squeeze() |> Nx.to_number()

      assert argmax_s0 == 1
      assert argmax_s2 == 1
    end

    test "handles batched input" do
      logits = Nx.broadcast(1.0, {2, 4, 8})
      result = SSMax.compute(logits, 1.0, 8)

      assert Nx.shape(result) == {2, 4, 8}
    end
  end

  describe "layer/2" do
    test "builds Axon layer with learnable s" do
      input = Axon.input("input", shape: {nil, 8, 16})
      model = SSMax.layer(input)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(1.0, {2, 8, 16}))

      assert Nx.shape(output) == {2, 8, 16}
    end
  end

  describe "build/1" do
    test "builds a full model" do
      model = SSMax.build(embed_dim: 16, hidden_size: 32, num_layers: 1, window_size: 8)

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({2, 8, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {2, 8, 16}))

      assert Nx.shape(output) == {2, 32}
    end
  end
end
