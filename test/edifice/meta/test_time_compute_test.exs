defmodule Edifice.Meta.TestTimeComputeTest do
  use ExUnit.Case, async: true

  alias Edifice.Meta.TestTimeCompute

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32
  @num_layers 2

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    num_layers: @num_layers,
    num_heads: 4,
    num_kv_heads: 2,
    scorer_hidden: 16,
    seq_len: @seq_len,
    dropout: 0.0
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    test "returns container with backbone and scores keys" do
      model = TestTimeCompute.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})

      assert is_map(output)
      assert Map.has_key?(output, :backbone)
      assert Map.has_key?(output, :scores)
    end

    test "backbone has correct shape [batch, seq_len, hidden_size]" do
      model = TestTimeCompute.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output.backbone) == {@batch, @seq_len, @hidden_size}
    end

    test "scores has correct shape [batch, seq_len, 1]" do
      model = TestTimeCompute.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output.scores) == {@batch, @seq_len, 1}
    end

    test "scores are finite" do
      model = TestTimeCompute.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.all(Nx.is_nan(output.scores) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "select_best_of_n/2" do
    test "returns argmax per batch element" do
      # N=3 candidates, batch=2
      scores = Nx.tensor([[0.1, 0.5], [0.9, 0.2], [0.3, 0.8]])
      result = TestTimeCompute.select_best_of_n(scores)

      assert Nx.to_flat_list(result) == [1, 2]
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert TestTimeCompute.output_size(@opts) == @hidden_size
    end
  end
end
