defmodule Edifice.SSM.SSTransformerTest do
  use ExUnit.Case, async: true
  @moduletag :ssm

  alias Edifice.SSM.SSTransformer

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @num_layers 2

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    state_size: 8,
    num_layers: @num_layers,
    num_heads: 4,
    head_dim: 8,
    expand_factor: 2,
    conv_size: 4,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    test "produces correct output shape [batch, hidden_size]" do
      model = SSTransformer.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is finite" do
      model = SSTransformer.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert SSTransformer.output_size(@opts) == @hidden_size
    end
  end
end
