defmodule Edifice.Blocks.RMSNormTest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.RMSNorm

  @batch 2
  @seq_len 8
  @hidden 32

  describe "layer/2" do
    test "produces correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = RMSNorm.layer(input, hidden_size: @hidden, name: "test_rms")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @seq_len, @hidden}))

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end

    test "handles batch_size=1" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = RMSNorm.layer(input, hidden_size: @hidden, name: "test_rms")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {1, @seq_len, @hidden}))

      assert Nx.shape(output) == {1, @seq_len, @hidden}
    end
  end

  describe "apply/3" do
    test "output has unit RMS when gamma=1" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      result = RMSNorm.apply(x, gamma)

      rms = Nx.sqrt(Nx.mean(Nx.pow(result, 2), axes: [-1]))
      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(rms, 1.0))))
      assert diff < 1.0e-4
    end

    test "scale parameter works (gamma=2 doubles output)" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      gamma_1 = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      gamma_2 = Nx.tensor([2.0, 2.0, 2.0, 2.0])

      result_1 = RMSNorm.apply(x, gamma_1)
      result_2 = RMSNorm.apply(x, gamma_2)

      diff =
        Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(Nx.multiply(result_1, 2.0), result_2))))

      assert diff < 1.0e-5
    end

    test "invariant to input scaling" do
      x = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0])

      result_1x = RMSNorm.apply(x, gamma)
      result_5x = RMSNorm.apply(Nx.multiply(x, 5.0), gamma)

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(result_1x, result_5x))))
      assert diff < 1.0e-4
    end

    test "handles zero input without NaN" do
      x = Nx.tensor([[0.0, 0.0, 0.0, 0.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      result = RMSNorm.apply(x, gamma)

      refute Nx.any(Nx.is_nan(result)) |> Nx.to_number() == 1
    end

    test "preserves batch dimension independence" do
      x = Nx.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
      gamma = Nx.tensor([1.0, 1.0, 1.0])
      result = RMSNorm.apply(x, gamma)

      result_row0 = RMSNorm.apply(Nx.tensor([[1.0, 2.0, 3.0]]), gamma)
      result_row1 = RMSNorm.apply(Nx.tensor([[10.0, 20.0, 30.0]]), gamma)

      diff0 = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(result[0], result_row0[0]))))
      diff1 = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(result[1], result_row1[0]))))

      assert diff0 < 1.0e-5
      assert diff1 < 1.0e-5
    end
  end
end
