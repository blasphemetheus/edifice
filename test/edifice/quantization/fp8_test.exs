defmodule Edifice.Quantization.FP8Test do
  use ExUnit.Case, async: true
  import ExUnit.CaptureIO

  alias Edifice.Quantization.FP8

  @params %{
    "layer1" => %{
      "kernel" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]]),
      "bias" => Nx.tensor([0.5, -0.5])
    },
    "layer2" => %{
      "kernel" => Nx.tensor([[0.1, -0.2], [0.3, 0.4]]),
      "bias" => Nx.tensor([0.0, 0.0])
    }
  }

  describe "quantize/2" do
    test "quantizes float tensors to e4m3fn by default" do
      q = FP8.quantize(@params)

      # Each tensor should be a quantized map
      kernel = q["layer1"]["kernel"]
      assert %{tensor: tensor, scale: scale, original_type: {:f, 32}} = kernel
      assert Nx.type(tensor) == {:f8_e4m3fn, 8}
      assert Nx.type(scale) == {:f, 32}
    end

    test "quantizes to e5m2 format" do
      q = FP8.quantize(@params, format: :e5m2)

      kernel = q["layer1"]["kernel"]
      assert %{tensor: tensor, scale: _scale, original_type: {:f, 32}} = kernel
      assert Nx.type(tensor) == {:f, 8}
    end

    test "preserves shape" do
      q = FP8.quantize(@params)

      kernel = q["layer1"]["kernel"]
      assert Nx.shape(kernel.tensor) == {2, 2}

      bias = q["layer1"]["bias"]
      assert Nx.shape(bias.tensor) == {2}
    end

    test "skip patterns keep tensors in original precision" do
      q = FP8.quantize(@params, skip: ["layer2"])

      # layer1 should be quantized
      assert %{tensor: _, scale: _, original_type: _} = q["layer1"]["kernel"]

      # layer2 should be unchanged
      assert %Nx.Tensor{} = q["layer2"]["kernel"]
      assert Nx.type(q["layer2"]["kernel"]) == {:f, 32}
    end

    test "handles zero tensors" do
      params = %{"w" => Nx.tensor([0.0, 0.0, 0.0])}
      q = FP8.quantize(params)
      assert %{tensor: _, scale: scale, original_type: _} = q["w"]
      assert Nx.to_number(scale) == 1.0
    end
  end

  describe "dequantize/1" do
    test "round-trip preserves values approximately" do
      q = FP8.quantize(@params)
      deq = FP8.dequantize(q)

      original_kernel = @params["layer1"]["kernel"]
      restored_kernel = deq["layer1"]["kernel"]

      assert Nx.type(restored_kernel) == {:f, 32}
      assert Nx.shape(restored_kernel) == Nx.shape(original_kernel)

      # FP8 E4M3FN has only 3 mantissa bits — quantization error can be
      # significant for values that don't map cleanly to the FP8 grid
      assert Nx.all_close(original_kernel, restored_kernel, atol: 0.5)
             |> Nx.to_number() == 1
    end

    test "preserves type from skip patterns" do
      q = FP8.quantize(@params, skip: ["layer2"])
      deq = FP8.dequantize(q)

      # Skipped tensors pass through unchanged
      assert Nx.equal(deq["layer2"]["kernel"], @params["layer2"]["kernel"])
             |> Nx.all()
             |> Nx.to_number() == 1
    end
  end

  describe "wrap_inference/1" do
    test "wraps predict function to auto-dequantize" do
      predict_fn = fn params, input ->
        # Simple linear: input @ kernel + bias
        kernel = params["layer1"]["kernel"]
        bias = params["layer1"]["bias"]
        Nx.add(Nx.dot(input, kernel), bias)
      end

      q_params = FP8.quantize(@params)
      q_predict = FP8.wrap_inference(predict_fn)

      input = Nx.tensor([[1.0, 0.0]])
      result = q_predict.(q_params, input)

      # Should get approximately [1.0, 2.0] + [0.5, -0.5] = [1.5, 1.5]
      expected = predict_fn.(@params, input)
      assert Nx.all_close(result, expected, atol: 0.2) |> Nx.to_number() == 1
    end
  end

  describe "report/2" do
    test "prints memory savings summary" do
      output =
        capture_io(fn ->
          result = FP8.report(@params)
          assert result.original_bytes > 0
          assert result.quantized_bytes > 0
          assert result.ratio > 1.0
          assert result.tensor_count == 4
        end)

      assert output =~ "[FP8]"
      assert output =~ "Original:"
      assert output =~ "Quantized:"
      assert output =~ "reduction"
    end
  end

  describe "estimate_savings/2" do
    test "estimates without quantizing" do
      result = FP8.estimate_savings(@params)

      # 4 tensors: 2x{2,2} + 2x{2} = 8+2+4+2 = 12 elements * 4 bytes = 48 bytes original
      # FP8: 12 elements * 1 byte + 4 scales * 4 bytes = 28 bytes
      assert result.original_bytes == 48
      assert result.quantized_bytes == 28
      assert result.tensor_count == 4
      assert result.ratio > 1.0
    end

    test "estimates for bf16 params" do
      bf16_params = %{
        "w" => Nx.as_type(Nx.tensor([1.0, 2.0, 3.0, 4.0]), {:bf, 16})
      }

      result = FP8.estimate_savings(bf16_params)
      # 4 elements * 2 bytes = 8 bytes original
      # 4 elements * 1 byte + 4 scale = 8 bytes quantized
      assert result.original_bytes == 8
      assert result.quantized_bytes == 8
    end
  end

  describe "e4m3fn precision" do
    test "handles values within representable range" do
      # E4M3FN range is [-448, 448], only 3 mantissa bits
      # Large values have significant quantization steps (e.g., 200 → 208)
      t = Nx.tensor([100.0, -200.0, 0.001, 448.0])
      q = FP8.quantize(%{"w" => t})
      deq = FP8.dequantize(q)

      # Relative tolerance is more appropriate for FP8 than absolute
      assert Nx.all_close(deq["w"], t, atol: 10.0) |> Nx.to_number() == 1
    end
  end
end
