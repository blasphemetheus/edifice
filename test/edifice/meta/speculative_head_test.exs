defmodule Edifice.Meta.SpeculativeHeadTest do
  use ExUnit.Case, async: true
  @moduletag :meta

  alias Edifice.Meta.SpeculativeHead

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @vocab_size 64
  @num_predictions 3
  @num_layers 2

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    vocab_size: @vocab_size,
    num_predictions: @num_predictions,
    num_layers: @num_layers,
    num_heads: 4,
    num_kv_heads: 2,
    head_hidden: 16,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    test "returns container with N prediction keys" do
      model = SpeculativeHead.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})

      assert is_map(output)
      assert map_size(output) == @num_predictions

      for i <- 1..@num_predictions do
        assert Map.has_key?(output, String.to_atom("pred_#{i}"))
      end
    end

    test "each head has correct shape [batch, seq_len, vocab_size]" do
      model = SpeculativeHead.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})

      for i <- 1..@num_predictions do
        key = String.to_atom("pred_#{i}")
        assert Nx.shape(output[key]) == {@batch, @seq_len, @vocab_size}
      end
    end

    test "output contains finite values" do
      model = SpeculativeHead.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})

      for i <- 1..@num_predictions do
        key = String.to_atom("pred_#{i}")
        assert Nx.all(Nx.is_nan(output[key]) |> Nx.logical_not()) |> Nx.to_number() == 1
      end
    end
  end

  describe "accept_reject/2" do
    test "delegates to SpeculativeDecoding" do
      draft = Nx.tensor([1, 2, 3, 4])
      verifier = Nx.tensor([1, 2, 5, 4])

      result = SpeculativeHead.accept_reject(draft, verifier)
      assert Nx.to_number(result) == 2
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert SpeculativeHead.output_size(@opts) == @hidden_size
    end
  end
end
