defmodule Edifice.Meta.VeRATest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  alias Edifice.Meta.VeRA

  @moduletag timeout: 120_000

  describe "build/1" do
    test "builds standalone VeRA adapter" do
      model = VeRA.build(input_size: 32, output_size: 32, rank: 16)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = VeRA.build(input_size: 32, output_size: 64, rank: 16)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({2, 32})
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, 64}
    end

    test "output values are finite" do
      model = VeRA.build(input_size: 16, output_size: 16, rank: 8)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({1, 16})
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert_finite!(output)
    end

    test "works with batch size 1" do
      model = VeRA.build(input_size: 16, output_size: 16, rank: 8)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({1, 16})
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, 16}
    end

    test "b initialized to zeros produces near-zero output" do
      model = VeRA.build(input_size: 16, output_size: 16, rank: 8)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({2, 16})
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      # b is initialized to zeros, so output should be near zero
      max_abs = Nx.reduce_max(Nx.abs(output)) |> Nx.to_number()
      assert max_abs < 1.0e-5
    end
  end

  describe "wrap/3" do
    test "wraps an existing dense layer" do
      input = Axon.input("input", shape: {nil, 32})
      original = Axon.dense(input, 32, name: "base_dense")

      adapted = VeRA.wrap(input, original, output_size: 32, rank: 16, name: "vera_adapter")
      assert %Axon{} = adapted
    end

    test "produces correct output shape when wrapping" do
      input_node = Axon.input("input", shape: {nil, 16})
      original = Axon.dense(input_node, 16, name: "base")

      adapted = VeRA.wrap(input_node, original, output_size: 16, rank: 8)
      {init_fn, predict_fn} = Axon.build(adapted)

      input = random_tensor({2, 16})
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, 16}
      assert_finite!(output)
    end
  end

  describe "vera_delta/3" do
    test "builds delta component" do
      input = Axon.input("input", shape: {nil, 32})
      delta = VeRA.vera_delta(input, 64, rank: 16, name: "delta")
      assert %Axon{} = delta
    end
  end

  describe "output_size/1" do
    test "returns output_size" do
      assert VeRA.output_size(output_size: 128) == 128
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = VeRA.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.has_key?(defaults, :rank)
      assert Keyword.has_key?(defaults, :d_initial)
    end
  end
end
