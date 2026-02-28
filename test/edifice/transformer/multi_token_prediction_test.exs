defmodule Edifice.Transformer.MultiTokenPredictionTest do
  use ExUnit.Case, async: true
  @moduletag :transformer

  alias Edifice.Transformer.MultiTokenPrediction

  @batch 2
  @seq_len 12
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
    seq_len: @seq_len
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    test "returns an Axon container" do
      model = MultiTokenPrediction.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces container with N prediction keys" do
      model = MultiTokenPrediction.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})

      # Should have N prediction keys
      assert is_map(output)
      assert Map.has_key?(output, :pred_1)
      assert Map.has_key?(output, :pred_2)
      assert Map.has_key?(output, :pred_3)
      assert map_size(output) == @num_predictions
    end

    test "each prediction head has correct shape [batch, seq_len, vocab_size]" do
      model = MultiTokenPrediction.build(@opts)

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
      model = MultiTokenPrediction.build(@opts)

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

  describe "output_size/1" do
    test "returns hidden_size" do
      assert MultiTokenPrediction.output_size(@opts) == @hidden_size
    end
  end
end
