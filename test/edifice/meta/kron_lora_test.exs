defmodule Edifice.Meta.KronLoRATest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  alias Edifice.Meta.KronLoRA

  @moduletag timeout: 120_000

  describe "build/1" do
    test "builds standalone Kron-LoRA adapter" do
      model = KronLoRA.build(input_size: 32, output_size: 32, rank: 4)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = KronLoRA.build(input_size: 32, output_size: 64, rank: 4, d_a1: 2, d_a2: 4)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({2, 32})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, 64}
    end

    test "output values are finite" do
      model = KronLoRA.build(input_size: 16, output_size: 16, rank: 4, d_a1: 2, d_a2: 2)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({1, 16})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert_finite!(output)
    end

    test "B1 initialized to zeros produces near-zero output" do
      model = KronLoRA.build(input_size: 32, output_size: 32, rank: 4)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({2, 32})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      # B1 is zero-initialized, so delta starts at zero
      max_abs = Nx.reduce_max(Nx.abs(output)) |> Nx.to_number()
      assert max_abs < 1.0e-5
    end

    test "asymmetric input/output dimensions" do
      model = KronLoRA.build(input_size: 16, output_size: 32, rank: 4, d_a1: 2, d_a2: 2)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({3, 16})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {3, 32}
      assert_finite!(output)
    end

    test "custom Kronecker dimensions" do
      # d_A1=4, d_A2=4: larger Kronecker factor, smaller LoRA blocks
      model = KronLoRA.build(input_size: 64, output_size: 64, rank: 4, d_a1: 4, d_a2: 4)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({2, 64})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, 64}
      assert_finite!(output)
    end
  end

  describe "wrap/3" do
    test "wraps an existing dense layer" do
      input = Axon.input("input", shape: {nil, 32})
      original = Axon.dense(input, 32, name: "base_dense")

      adapted =
        KronLoRA.wrap(input, original,
          input_size: 32,
          output_size: 32,
          rank: 4,
          name: "kl_adapter"
        )

      assert %Axon{} = adapted
    end

    test "produces correct output shape when wrapping" do
      input_node = Axon.input("input", shape: {nil, 16})
      original = Axon.dense(input_node, 16, name: "base")

      adapted =
        KronLoRA.wrap(input_node, original,
          input_size: 16,
          output_size: 16,
          rank: 4
        )

      {init_fn, predict_fn} = Axon.build(adapted)

      input = random_tensor({2, 16})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, 16}
      assert_finite!(output)
    end
  end

  describe "kron_lora_delta/3" do
    test "builds delta component" do
      input = Axon.input("input", shape: {nil, 32})

      delta =
        KronLoRA.kron_lora_delta(input, 64,
          input_size: 32,
          rank: 4,
          d_a1: 2,
          d_a2: 4,
          name: "delta"
        )

      assert %Axon{} = delta
    end
  end

  describe "parameter efficiency" do
    test "fewer parameters than equivalent LoRA" do
      # Kron-LoRA: A_kron(2*2) + B2(4*16) + B1(16*4) = 4 + 64 + 64 = 132
      # LoRA r=4:  A(4*32) + B(4*32) = 128 + 128 = 256
      kron_model = KronLoRA.build(input_size: 32, output_size: 32, rank: 4)
      {init_fn, _predict_fn} = Axon.build(kron_model)
      input = random_tensor({1, 32})
      params = init_fn.(input, Axon.ModelState.empty())

      kron_param_count = count_params(params.data)

      # LoRA equivalent: rank * (input_size + output_size) = 4 * (32 + 32) = 256
      lora_param_count = 4 * (32 + 32)

      assert kron_param_count < lora_param_count
    end
  end

  describe "output_size/1" do
    test "returns output_size" do
      assert KronLoRA.output_size(output_size: 128) == 128
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = KronLoRA.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.has_key?(defaults, :rank)
      assert Keyword.has_key?(defaults, :alpha)
      assert Keyword.has_key?(defaults, :d_a1)
      assert Keyword.has_key?(defaults, :d_a2)
    end
  end

  describe "validation" do
    test "raises on indivisible input_size" do
      assert_raise ArgumentError, ~r/input_size.*divisible.*d_a1/, fn ->
        KronLoRA.build(input_size: 33, output_size: 32, rank: 4, d_a1: 2)
      end
    end

    test "raises on indivisible output_size" do
      assert_raise ArgumentError, ~r/output_size.*divisible.*d_a2/, fn ->
        KronLoRA.build(input_size: 32, output_size: 33, rank: 4, d_a2: 2)
      end
    end
  end

  # Helper to count params in nested model state
  defp count_params(map) when is_map(map) do
    Enum.reduce(map, 0, fn {_k, v}, acc ->
      case v do
        %Nx.Tensor{} -> acc + Nx.size(v)
        m when is_map(m) -> acc + count_params(m)
        _ -> acc
      end
    end)
  end
end
