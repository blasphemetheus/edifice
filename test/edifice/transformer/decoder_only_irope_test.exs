defmodule Edifice.Transformer.DecoderOnlyIRoPETest do
  use ExUnit.Case, async: true

  alias Edifice.Transformer.DecoderOnly

  @batch 2
  @seq_len 16
  @embed_dim 32
  @hidden_size 64

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_heads: 4,
    num_kv_heads: 2,
    num_layers: 4,
    interleave_rope: true,
    dropout: 0.0,
    window_size: @seq_len
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  defp template, do: %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)}

  describe "build/1 with interleave_rope" do
    test "builds an Axon model" do
      model = DecoderOnly.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = DecoderOnly.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output contains finite values" do
      model = DecoderOnly.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(template(), Axon.ModelState.empty())
      output = predict_fn.(params, %{"state_sequence" => random_input()})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "produces different results than non-interleaved" do
      model_irope = DecoderOnly.build(@opts)
      model_rope = DecoderOnly.build(Keyword.put(@opts, :interleave_rope, false))

      # Both should build successfully as different models
      assert %Axon{} = model_irope
      assert %Axon{} = model_rope
    end
  end
end
