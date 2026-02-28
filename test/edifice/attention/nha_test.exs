defmodule Edifice.Attention.NHATest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.NHA

  import Edifice.TestHelpers

  @moduletag :attention

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @num_heads 2

  describe "build/1" do
    test "builds model with default options" do
      model = NHA.build(embed_dim: @embed_dim)
      assert %Axon{} = model
    end

    test "builds model with custom options" do
      model =
        NHA.build(
          embed_dim: 64,
          hidden_size: 64,
          num_heads: 4,
          num_layers: 2,
          dropout: 0.0,
          gate_init: 0.7
        )

      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model =
        NHA.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          num_layers: 2,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({@batch, @seq_len, @embed_dim}, 42)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      # ModelBuilder outputs [batch, hidden_size] from last position
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "batch=1 forward pass" do
      model =
        NHA.build(
          embed_dim: 16,
          hidden_size: 16,
          num_heads: 2,
          num_layers: 1,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({1, 4, 16}, 99)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, 16}
    end

    test "output values are finite" do
      model =
        NHA.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          num_layers: 1,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({@batch, 6, @embed_dim}, 77)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert_finite!(output)
    end
  end

  describe "build_nha_attention/2" do
    test "builds attention layer" do
      input = Axon.input("input", shape: {nil, nil, @hidden_size})

      attn =
        NHA.build_nha_attention(input,
          hidden_size: @hidden_size,
          num_heads: @num_heads,
          gate_init: 0.5,
          name: "test_attn"
        )

      assert %Axon{} = attn
    end

    test "gate_init near 0 biases toward linear attention" do
      model =
        NHA.build(
          embed_dim: 16,
          hidden_size: 16,
          num_heads: 2,
          num_layers: 1,
          dropout: 0.0,
          gate_init: 0.01
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({@batch, 4, 16}, 55)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, 16}
      assert_finite!(output)
    end

    test "gate_init near 1 biases toward softmax attention" do
      model =
        NHA.build(
          embed_dim: 16,
          hidden_size: 16,
          num_heads: 2,
          num_layers: 1,
          dropout: 0.0,
          gate_init: 0.99
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({@batch, 4, 16}, 66)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, 16}
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert NHA.output_size(hidden_size: 128) == 128
    end

    test "returns default when no options" do
      assert NHA.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = NHA.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert defaults[:hidden_size] == 256
      assert defaults[:num_heads] == 4
      assert defaults[:gate_init] == 0.5
    end
  end

  describe "registry integration" do
    test "builds via Edifice.build/2" do
      model =
        Edifice.build(:nha,
          embed_dim: 16,
          hidden_size: 16,
          num_heads: 2,
          num_layers: 1,
          dropout: 0.0
        )

      assert %Axon{} = model
    end
  end
end
