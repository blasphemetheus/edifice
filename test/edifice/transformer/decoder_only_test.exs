defmodule Edifice.Transformer.DecoderOnlyTest do
  use ExUnit.Case, async: true

  alias Edifice.Transformer.DecoderOnly

  @batch 2
  @seq_len 16
  @embed_dim 32
  @hidden_size 64
  @num_heads 8
  @num_kv_heads 2

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_kv_heads: @num_kv_heads,
    num_layers: 2,
    window_size: @seq_len
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  defp template, do: %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}

  describe "build/1" do
    test "builds an Axon model" do
      model = DecoderOnly.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = DecoderOnly.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = DecoderOnly.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with different num_layers" do
      model = DecoderOnly.build(Keyword.put(@opts, :num_layers, 1))
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "works without RoPE" do
      model = DecoderOnly.build(Keyword.put(@opts, :use_rope, false))
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert DecoderOnly.output_size(@opts) == @hidden_size
    end
  end
end
