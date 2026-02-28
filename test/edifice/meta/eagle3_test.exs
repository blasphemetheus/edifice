defmodule Edifice.Meta.Eagle3Test do
  use ExUnit.Case, async: true
  @moduletag :meta

  alias Edifice.Meta.Eagle3

  @batch 2
  @seq_len 8
  @hidden_size 64
  @vocab_size 128
  @num_heads 4
  @num_kv_heads 2

  @base_opts [
    hidden_size: @hidden_size,
    vocab_size: @vocab_size,
    num_heads: @num_heads,
    num_kv_heads: @num_kv_heads,
    intermediate_size: 128
  ]

  defp random_inputs do
    key = Nx.Random.key(42)
    {tok, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden_size})
    {low, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden_size})
    {mid, key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden_size})
    {high, _key} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden_size})

    %{
      "token_embeddings" => tok,
      "features_low" => low,
      "features_mid" => mid,
      "features_high" => high
    }
  end

  defp build_and_predict(opts) do
    model = Eagle3.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "token_embeddings" => Nx.template({@batch, @seq_len, @hidden_size}, :f32),
      "features_low" => Nx.template({@batch, @seq_len, @hidden_size}, :f32),
      "features_mid" => Nx.template({@batch, @seq_len, @hidden_size}, :f32),
      "features_high" => Nx.template({@batch, @seq_len, @hidden_size}, :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())
    predict_fn.(params, random_inputs())
  end

  describe "build/1" do
    test "builds model and produces correct output shape" do
      output = build_and_predict(@base_opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "output contains finite values" do
      output = build_and_predict(@base_opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      refute Nx.to_number(Nx.any(Nx.is_infinity(output))) == 1
    end

    test "works with equal Q and KV heads (MHA)" do
      opts = Keyword.merge(@base_opts, num_kv_heads: @num_heads)
      output = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "works with single feature level" do
      opts = Keyword.put(@base_opts, :num_feature_levels, 1)
      output = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end

    test "works with two feature levels" do
      opts = Keyword.put(@base_opts, :num_feature_levels, 2)
      output = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, @seq_len, @vocab_size}
    end
  end

  describe "map_to_target_vocab/2" do
    test "maps draft tokens to target vocabulary" do
      # Draft vocab 4 -> target vocab 10
      d2t = Nx.tensor([5, 8, 2, 9])
      draft_tokens = Nx.tensor([0, 2, 1, 3])

      target_tokens = Eagle3.map_to_target_vocab(draft_tokens, d2t)
      assert Nx.to_list(target_tokens) == [5, 2, 8, 9]
    end

    test "works with batched input" do
      d2t = Nx.tensor([10, 20, 30])
      draft_tokens = Nx.tensor([[0, 1], [2, 0]])

      target_tokens = Eagle3.map_to_target_vocab(draft_tokens, d2t)
      assert Nx.to_list(target_tokens) == [[10, 20], [30, 10]]
    end
  end

  describe "accept_reject/2" do
    test "counts consecutive matches" do
      draft = Nx.tensor([1, 2, 3, 4])
      verify = Nx.tensor([1, 2, 5, 4])

      accepted = Eagle3.accept_reject(draft, verify)
      assert Nx.to_number(accepted) == 2
    end

    test "returns 0 when first token mismatches" do
      draft = Nx.tensor([1, 2, 3])
      verify = Nx.tensor([9, 2, 3])

      accepted = Eagle3.accept_reject(draft, verify)
      assert Nx.to_number(accepted) == 0
    end

    test "returns full length when all match" do
      draft = Nx.tensor([1, 2, 3])
      verify = Nx.tensor([1, 2, 3])

      accepted = Eagle3.accept_reject(draft, verify)
      assert Nx.to_number(accepted) == 3
    end
  end

  describe "recommended_extract_layers/1" do
    test "returns layer indices for llama_8b" do
      {low, mid, high} = Eagle3.recommended_extract_layers(:llama_8b)
      assert low < mid
      assert mid < high
    end

    test "returns layer indices for llama_70b" do
      {low, mid, high} = Eagle3.recommended_extract_layers(:llama_70b)
      assert low == 20
      assert mid == 50
      assert high == 79
    end
  end

  describe "output_size/1" do
    test "returns vocab_size" do
      assert Eagle3.output_size(vocab_size: 32_000) == 32_000
    end
  end
end
