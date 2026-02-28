defmodule Edifice.Meta.PiSSATest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  alias Edifice.Meta.PiSSA

  @moduletag timeout: 120_000

  describe "build/1" do
    test "builds standalone PiSSA adapter" do
      model = PiSSA.build(input_size: 32, output_size: 32, rank: 4)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = PiSSA.build(input_size: 32, output_size: 64, rank: 4)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({2, 32})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, 64}
    end

    test "output values are finite" do
      model = PiSSA.build(input_size: 16, output_size: 16, rank: 4)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({1, 16})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert_finite!(output)
    end

    test "B initialized to zeros produces near-zero output" do
      model = PiSSA.build(input_size: 32, output_size: 32, rank: 4)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({2, 32})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      max_abs = Nx.reduce_max(Nx.abs(output)) |> Nx.to_number()
      assert max_abs < 1.0e-5
    end

    test "asymmetric input/output dimensions" do
      model = PiSSA.build(input_size: 16, output_size: 48, rank: 8)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({3, 16})
      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {3, 48}
      assert_finite!(output)
    end
  end

  describe "wrap/3" do
    test "wraps an existing dense layer" do
      input = Axon.input("input", shape: {nil, 32})
      original = Axon.dense(input, 32, name: "base_dense")

      adapted =
        PiSSA.wrap(input, original,
          output_size: 32,
          rank: 4,
          name: "pissa_adapter"
        )

      assert %Axon{} = adapted
    end

    test "produces correct output shape when wrapping" do
      input_node = Axon.input("input", shape: {nil, 16})
      original = Axon.dense(input_node, 16, name: "base")

      adapted =
        PiSSA.wrap(input_node, original,
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

  describe "pissa_delta/3" do
    test "builds delta component" do
      input = Axon.input("input", shape: {nil, 32})
      delta = PiSSA.pissa_delta(input, 64, rank: 4, name: "delta")
      assert %Axon{} = delta
    end
  end

  describe "decompose/2" do
    test "decomposes weight matrix via SVD" do
      key = Nx.Random.key(42)
      {weight, _key} = Nx.Random.normal(key, shape: {16, 32}, type: :f32)
      {a_init, b_init, w_residual} = PiSSA.decompose(weight, rank: 4)

      # A: [input_size, rank] = [32, 4]
      assert Nx.shape(a_init) == {32, 4}
      # B: [rank, output_size] = [4, 16]
      assert Nx.shape(b_init) == {4, 16}
      # W_res: [output_size, input_size] = [16, 32]
      assert Nx.shape(w_residual) == {16, 32}
    end

    test "decomposition reconstructs original weight" do
      key = Nx.Random.key(43)
      {weight, _key} = Nx.Random.normal(key, shape: {8, 8}, type: :f32)
      {a_init, b_init, w_residual} = PiSSA.decompose(weight, rank: 8)

      # Full rank decomposition: W_res + B * A should equal W
      # a_init is [8, 8] (transposed A), b_init is [8, 8] (transposed B)
      # In original convention: B_matrix = transpose(b_init), A_matrix = transpose(a_init)
      b_matrix = Nx.transpose(b_init)
      a_matrix = Nx.transpose(a_init)
      reconstructed = Nx.add(w_residual, Nx.dot(b_matrix, a_matrix))

      diff = Nx.subtract(weight, reconstructed)
      max_diff = Nx.reduce_max(Nx.abs(diff)) |> Nx.to_number()
      assert max_diff < 1.0e-3
    end

    test "low-rank decomposition residual is smaller" do
      key = Nx.Random.key(44)
      {weight, _key} = Nx.Random.normal(key, shape: {16, 16}, type: :f32)
      {_a, _b, w_res_r2} = PiSSA.decompose(weight, rank: 2)
      {_a, _b, w_res_r8} = PiSSA.decompose(weight, rank: 8)

      # Higher rank captures more of W, so residual should have smaller norm
      norm_r2 = Nx.sum(Nx.multiply(w_res_r2, w_res_r2)) |> Nx.to_number()
      norm_r8 = Nx.sum(Nx.multiply(w_res_r8, w_res_r8)) |> Nx.to_number()

      assert norm_r8 < norm_r2
    end
  end

  describe "output_size/1" do
    test "returns output_size" do
      assert PiSSA.output_size(output_size: 128) == 128
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = PiSSA.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.has_key?(defaults, :rank)
      assert Keyword.has_key?(defaults, :alpha)
    end
  end
end
