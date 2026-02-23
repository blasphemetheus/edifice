defmodule Edifice.Attention.FlashLinearAttentionTest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.FlashLinearAttention

  @batch 2
  @seq_len 16
  @embed_dim 32
  @hidden_size 32
  @num_heads 4
  @num_layers 2
  @chunk_size 8

  @base_opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_layers: @num_layers,
    chunk_size: @chunk_size,
    window_size: @seq_len
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1 with ELU feature map" do
    test "builds and runs with correct output shape" do
      opts = Keyword.put(@base_opts, :feature_map, :elu)
      model = FlashLinearAttention.build(opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      opts = Keyword.put(@base_opts, :feature_map, :elu)
      model = FlashLinearAttention.build(opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1 with ReLU feature map" do
    test "builds and runs with correct output shape" do
      opts = Keyword.put(@base_opts, :feature_map, :relu)
      model = FlashLinearAttention.build(opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "build/1 with identity feature map" do
    test "builds and runs with correct output shape" do
      opts = Keyword.put(@base_opts, :feature_map, :identity)
      model = FlashLinearAttention.build(opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert FlashLinearAttention.output_size(@base_opts) == @hidden_size
    end
  end
end
