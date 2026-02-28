defmodule Edifice.Recurrent.GatedDeltaNetTest do
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Recurrent.GatedDeltaNet

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @num_heads 2
  @num_layers 2

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_layers: @num_layers,
    window_size: @seq_len,
    dropout: 0.0,
    use_short_conv: true
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = GatedDeltaNet.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = GatedDeltaNet.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = GatedDeltaNet.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works without short convolution" do
      opts = Keyword.put(@opts, :use_short_conv, false)
      model = GatedDeltaNet.build(opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @embed_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, random_input())

      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert GatedDeltaNet.output_size(@opts) == @hidden_size
    end
  end
end
