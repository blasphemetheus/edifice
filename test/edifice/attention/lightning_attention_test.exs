defmodule Edifice.Attention.LightningAttentionTest do
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.LightningAttention
  alias Edifice.Transformer.DecoderOnly

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @block_size 4

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: 4,
    num_layers: 2,
    block_size: @block_size,
    dropout: 0.0,
    window_size: @seq_len
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  defp template,
    do: %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}

  describe "build/1" do
    test "builds an Axon model" do
      model = LightningAttention.build(@opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      model = LightningAttention.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = LightningAttention.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert LightningAttention.output_size(@opts) == @hidden_size
    end

    test "returns default when not specified" do
      assert LightningAttention.output_size([]) == 256
    end
  end

  describe "integration with decoder_only" do
    test "decoder_only with lightning attention builds" do
      model =
        DecoderOnly.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 4,
          num_layers: 2,
          attention_type: :lightning,
          block_size: @block_size,
          window_size: @seq_len,
          dropout: 0.0
        )

      assert %Axon{} = model
    end

    test "decoder_only with lightning attention produces correct output" do
      model =
        DecoderOnly.build(
          embed_dim: @embed_dim,
          hidden_size: @hidden_size,
          num_heads: 4,
          num_layers: 2,
          attention_type: :lightning,
          block_size: @block_size,
          window_size: @seq_len,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
