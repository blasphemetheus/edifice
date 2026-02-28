defmodule Edifice.Memory.MemoryLayerTest do
  use ExUnit.Case, async: true

  alias Edifice.Memory.MemoryLayer

  import Edifice.TestHelpers

  @moduletag :memory

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32

  describe "build/1" do
    test "builds model with default options" do
      model = MemoryLayer.build(embed_dim: @embed_dim)
      assert %Axon{} = model
    end

    test "forward pass produces correct shape" do
      model =
        MemoryLayer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 2,
          num_layers: 2,
          memory_size: 64,
          key_dim: 16,
          top_k: 4,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({@batch, @seq_len, @embed_dim}, 42)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "batch=1 forward pass" do
      model =
        MemoryLayer.build(
          embed_dim: 16,
          hidden_size: 16,
          num_heads: 2,
          num_layers: 1,
          memory_size: 16,
          key_dim: 8,
          top_k: 2,
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
        MemoryLayer.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 2,
          num_layers: 1,
          memory_size: 64,
          key_dim: 16,
          top_k: 4,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({@batch, @seq_len, @embed_dim}, 77)
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert MemoryLayer.output_size(hidden_size: 512) == 512
    end

    test "returns default" do
      assert MemoryLayer.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = MemoryLayer.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert defaults[:memory_size] == 1024
      assert defaults[:top_k] == 32
    end
  end

  describe "registry integration" do
    test "builds via Edifice.build/2" do
      model =
        Edifice.build(:memory_layer,
          embed_dim: 16,
          hidden_size: 16,
          num_heads: 2,
          num_layers: 1,
          memory_size: 16,
          key_dim: 8,
          top_k: 2,
          dropout: 0.0
        )

      assert %Axon{} = model
    end
  end
end
