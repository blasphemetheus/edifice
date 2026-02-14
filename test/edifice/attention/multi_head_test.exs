defmodule Edifice.Attention.MultiHeadTest do
  use ExUnit.Case, async: true

  alias Edifice.Attention.MultiHead

  @batch 4
  @seq_len 12
  @embed_dim 64
  @num_heads 4
  @head_dim 8
  @hidden_dim @num_heads * @head_dim
  @num_layers 2

  @opts [
    embed_dim: @embed_dim,
    num_heads: @num_heads,
    head_dim: @head_dim,
    num_layers: @num_layers,
    window_size: @seq_len
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build_sliding_window/1" do
    test "builds an Axon model" do
      model = MultiHead.build_sliding_window(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = MultiHead.build_sliding_window(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_dim}
    end

    test "output contains finite values" do
      model = MultiHead.build_sliding_window(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns num_heads * head_dim" do
      assert MultiHead.output_size(@opts) == @hidden_dim
    end
  end
end
