defmodule Edifice.Utils.QuantizationTest do
  use ExUnit.Case, async: true
  import Nx.Testing
  alias Edifice.Utils.Quantization

  defp weights,
    do: %{"dense" => %{"kernel" => Nx.tensor([[1.0], [0.4], [0.4]]), "bias" => Nx.tensor([0.2])}}

  defp correlated do
    Nx.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.9], [0.0, 0.0, :math.sqrt(0.19)]])
  end

  test "off-diagonal Hessian compensates error and improves calibration output error over RTN" do
    params = weights()
    x = correlated()
    result = Quantization.gptq(params, [x], bits: 2, group_size: -1)
    # H has correlation 0.9 between channels 1 and 2. Rounding 0.4 to zero
    # pushes the remaining 0.4 above 0.5; diagonal-only rounding cannot do this.
    assert_equal(
      result.quantized_params["dense"]["kernel"],
      Nx.tensor([[1], [0], [1]], type: :s8)
    )

    assert_equal(result.quantized_params["dense"]["bias"], params["dense"]["bias"])
    restored = Quantization.dequantize(result)["dense"]["kernel"]
    rtn = params |> Quantization.quantize_weights(bits: 2) |> Quantization.dequantize()
    target = Nx.dot(x, params["dense"]["kernel"])

    error = fn w ->
      Nx.subtract(Nx.dot(x, w), target) |> Nx.pow(2) |> Nx.sum() |> Nx.to_number()
    end

    assert error.(restored) < error.(rtn["dense"]["kernel"])
    independent = Quantization.gptq(params, [Nx.eye(3)], bits: 2, group_size: -1)

    assert_equal(
      independent.quantized_params["dense"]["kernel"],
      Nx.tensor([[1], [0], [0]], type: :s8)
    )
  end

  test "block size and calibration batching do not change quantization" do
    expected = Quantization.gptq(weights(), [correlated()], bits: 2, block_size: 3)
    batches = for i <- 0..2, do: Nx.slice_along_axis(correlated(), i, 1, axis: 0)

    for block_size <- [1, 2, 8] do
      actual = Quantization.gptq(weights(), batches, bits: 2, block_size: block_size)

      assert_equal(
        actual.quantized_params["dense"]["kernel"],
        expected.quantized_params["dense"]["kernel"]
      )
    end
  end

  test "full paths keep calibration for same-named kernels distinct" do
    kernel = weights()["dense"]
    params = %{"a" => kernel, "b" => kernel}

    result =
      Quantization.gptq(
        params,
        %{
          ["a", "kernel"] => [correlated()],
          ["b", "kernel"] => [Nx.eye(3)]
        },
        bits: 2
      )

    assert_equal(result.quantized_params["a"]["kernel"], Nx.tensor([[1], [0], [1]], type: :s8))
    assert_equal(result.quantized_params["b"]["kernel"], Nx.tensor([[1], [0], [0]], type: :s8))

    assert_raise ArgumentError, ~r/full parameter paths/, fn ->
      Quantization.gptq(params, [correlated()])
    end

    assert_raise ArgumentError, ~r/missing or empty/, fn ->
      Quantization.gptq(params, %{["a", "kernel"] => [correlated()]})
    end
  end

  test "static groups, rectangular kernels and sequence activations round-trip" do
    kernel = Nx.tensor([[1.0, 2.0], [0.4, 0.8], [0.03, 0.06]])
    params = %{"kernel" => kernel}
    x = Nx.reshape(Nx.eye(3), {1, 3, 3})
    result = Quantization.gptq(params, [x], bits: 2, group_size: 2, block_size: 1)
    assert_all_close(result.scales["kernel"], Nx.tensor([[1.0, 2.0], [1.0, 2.0], [0.03, 0.06]]))

    assert_all_close(
      Quantization.dequantize(result)["kernel"],
      Nx.tensor([[1.0, 2.0], [0.0, 0.0], [0.03, 0.06]])
    )
  end

  test "dead and rank-deficient calibration channels remain finite" do
    for x <- [Nx.broadcast(0.0, {2, 3}), Nx.tensor([[1.0, 1.0, 0.0]])] do
      result = Quantization.gptq(weights(), [x]) |> Quantization.dequantize()
      kernel = result["dense"]["kernel"]
      assert Nx.to_number(Nx.any(Nx.is_nan(kernel))) == 0
      assert Nx.to_number(Nx.any(Nx.is_infinity(kernel))) == 0
      assert Nx.to_number(kernel[2][0]) == 0
    end
  end

  test "bad calibration and unsupported options fail explicitly" do
    for batches <- [[], [Nx.eye(2)], [Nx.tensor([[0.0, :nan, 1.0]])]] do
      assert_raise ArgumentError, fn -> Quantization.gptq(weights(), batches) end
    end

    for opts <- [[bits: 1], [group_size: 0], [block_size: 0], [dampening: 0], [typo: true]] do
      assert_raise ArgumentError, fn -> Quantization.gptq(weights(), [correlated()], opts) end
    end

    assert_raise ArgumentError, ~r/rank-two/, fn ->
      Quantization.gptq(%{"conv" => Nx.broadcast(1.0, {2, 2, 2})}, %{})
    end
  end
end
