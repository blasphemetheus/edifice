defmodule Edifice.Attention.LASERTest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  alias Edifice.Attention.LASER

  @moduletag timeout: 120_000

  describe "build/1" do
    test "builds model with default options" do
      model = LASER.build(embed_dim: 32, hidden_size: 16, num_heads: 2, num_layers: 1)
      assert %Axon{} = model
    end

    test "builds model with custom options" do
      model =
        LASER.build(
          embed_dim: 64,
          hidden_size: 32,
          num_heads: 4,
          num_layers: 2,
          dropout: 0.1,
          causal: true
        )

      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = LASER.build(embed_dim: 32, hidden_size: 16, num_heads: 2, num_layers: 1)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({2, 8, 32})
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, 16}
    end

    test "output values are finite" do
      model = LASER.build(embed_dim: 16, hidden_size: 8, num_heads: 2, num_layers: 1)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({1, 4, 16})
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert_finite!(output)
    end

    test "works with batch size 1" do
      model = LASER.build(embed_dim: 16, hidden_size: 8, num_heads: 2, num_layers: 1)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({1, 4, 16})
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, 8}
    end

    test "works with non-causal mode" do
      model =
        LASER.build(embed_dim: 16, hidden_size: 8, num_heads: 2, num_layers: 1, causal: false)

      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({2, 4, 16})
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, 8}
      assert_finite!(output)
    end

    test "works with multiple layers" do
      model = LASER.build(embed_dim: 16, hidden_size: 8, num_heads: 2, num_layers: 3)
      {init_fn, predict_fn} = Axon.build(model)

      input = random_tensor({2, 4, 16})
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {2, 8}
      assert_finite!(output)
    end
  end

  describe "self_attention/2" do
    test "builds a self-attention layer" do
      input = Axon.input("input", shape: {nil, 4, 16})
      attn = LASER.self_attention(input, hidden_size: 16, num_heads: 2, name: "test_laser")
      assert %Axon{} = attn
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert LASER.output_size(hidden_size: 128) == 128
    end

    test "returns default when no option" do
      assert LASER.output_size([]) == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = LASER.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
    end
  end
end
