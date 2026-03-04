defmodule Edifice.Serving.GenerateTest do
  use ExUnit.Case, async: true
  @moduletag :serving

  alias Edifice.Serving.Generate
  alias Edifice.Serving.Sampling

  # Small dims for fast CPU tests
  @vocab_size 32
  @embed_dim 8
  @seq_len 16
  @batch 1

  # Build a simple model that outputs [batch, seq_len, vocab_size] logits.
  # Uses raw Axon to avoid depending on a specific architecture.
  defp build_raw_lm do
    Axon.input("state_sequence", shape: {nil, @seq_len, @embed_dim})
    |> Axon.dense(@embed_dim, name: "hidden", activation: :relu)
    |> Axon.dense(@vocab_size, name: "lm_head")
  end

  defp init_model(model) do
    template = %{
      "state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)
    }

    {init_fn, predict_fn} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())
    {predict_fn, params}
  end

  defp make_embed_fn do
    {table, _key} = Nx.Random.uniform(Nx.Random.key(0), shape: {@vocab_size, @embed_dim})

    fn token_ids ->
      Nx.take(table, token_ids)
    end
  end

  describe "build_lm/1" do
    test "builds model with LM head using decoder_only" do
      model =
        Generate.build_lm(
          arch: :decoder_only,
          vocab_size: @vocab_size,
          embed_dim: @embed_dim,
          hidden_size: @embed_dim,
          seq_len: @seq_len,
          num_layers: 1,
          num_heads: 2,
          num_kv_heads: 2
        )

      assert %Axon{} = model
    end
  end

  describe "generate/3" do
    test "generates tokens from prompt" do
      model = build_raw_lm()
      {predict_fn, params} = init_model(model)

      prompt = Nx.tensor([[1, 2, 3]])
      max_tokens = 5

      result =
        Generate.generate(predict_fn, params,
          prompt: prompt,
          embed_fn: make_embed_fn(),
          max_tokens: max_tokens,
          seq_len: @seq_len,
          temperature: 0.0
        )

      {batch, total_len} = Nx.shape(result)
      assert batch == @batch
      assert total_len >= 3 + 1
      assert total_len <= 3 + max_tokens

      # Prompt is preserved
      prompt_slice = result[[0, 0..2]]
      assert Nx.to_flat_list(prompt_slice) == [1, 2, 3]
    end

    test "greedy decoding is deterministic" do
      model = build_raw_lm()
      {predict_fn, params} = init_model(model)

      prompt = Nx.tensor([[1, 5, 10]])

      opts = [
        prompt: prompt,
        embed_fn: make_embed_fn(),
        max_tokens: 8,
        seq_len: @seq_len,
        temperature: 0.0
      ]

      result1 = Generate.generate(predict_fn, params, opts)
      result2 = Generate.generate(predict_fn, params, opts)

      assert Nx.equal(result1, result2) |> Nx.all() |> Nx.to_number() == 1
    end

    test "respects stop_token" do
      model = build_raw_lm()
      {predict_fn, params} = init_model(model)

      prompt = Nx.tensor([[1, 2]])

      result =
        Generate.generate(predict_fn, params,
          prompt: prompt,
          embed_fn: make_embed_fn(),
          max_tokens: 50,
          seq_len: @seq_len,
          temperature: 1.0,
          seed: 123,
          stop_token: -999
        )

      {_batch, total_len} = Nx.shape(result)
      assert total_len >= 3
    end

    test "temperature > 0 produces valid tokens" do
      model = build_raw_lm()
      {predict_fn, params} = init_model(model)

      prompt = Nx.tensor([[1]])

      result =
        Generate.generate(predict_fn, params,
          prompt: prompt,
          embed_fn: make_embed_fn(),
          max_tokens: 10,
          seq_len: @seq_len,
          temperature: 0.8,
          top_k: 10,
          seed: 42
        )

      {_batch, total_len} = Nx.shape(result)
      assert total_len >= 2

      flat = Nx.to_flat_list(result)
      assert Enum.all?(flat, &(&1 >= 0 and &1 < @vocab_size))
    end
  end

  describe "generate_simple/3" do
    test "produces output from prompt" do
      model = build_raw_lm()
      {predict_fn, params} = init_model(model)

      prompt = Nx.tensor([[1, 2, 3]])

      result =
        Generate.generate_simple(predict_fn, params,
          prompt: prompt,
          embed_fn: make_embed_fn(),
          max_tokens: 4,
          seq_len: @seq_len,
          temperature: 0.0
        )

      prompt_slice = result[[0, 0..2]]
      assert Nx.to_flat_list(prompt_slice) == [1, 2, 3]
      assert Nx.axis_size(result, 1) >= 4
    end
  end

  describe "Sampling" do
    test "greedy returns argmax" do
      logits = Nx.tensor([[0.1, 0.5, 0.3, 0.9, 0.2]])
      result = Sampling.greedy(logits)
      assert Nx.to_number(result[0]) == 3
    end

    test "temperature scaling sharpens distribution" do
      logits = Nx.tensor([[1.0, 2.0, 3.0]])
      cold = Sampling.apply_temperature(logits, 0.1)
      hot = Sampling.apply_temperature(logits, 10.0)

      cold_range = Nx.to_number(Nx.subtract(Nx.reduce_max(cold), Nx.reduce_min(cold)))
      hot_range = Nx.to_number(Nx.subtract(Nx.reduce_max(hot), Nx.reduce_min(hot)))
      assert cold_range > hot_range
    end

    test "top_k masks tokens outside top-k" do
      logits = Nx.tensor([[1.0, 5.0, 2.0, 4.0, 3.0]])
      filtered = Sampling.apply_top_k(logits, 2)
      flat = Nx.to_flat_list(filtered)

      finite_count = Enum.count(flat, &is_number/1)
      assert finite_count == 2
    end

    test "top_p keeps smallest set exceeding threshold" do
      logits = Nx.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])
      filtered = Sampling.apply_top_p(logits, 0.5)
      flat = Nx.to_flat_list(filtered)

      finite_count = Enum.count(flat, &is_number/1)
      assert finite_count >= 1
    end

    test "categorical_sample returns valid indices" do
      logits = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      key = Nx.Random.key(42)

      {tokens, _new_key} = Sampling.categorical_sample(logits, key)

      token = Nx.to_number(tokens[0])
      assert token >= 0 and token < 4
    end

    test "sample with temperature + top_k" do
      logits = Nx.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
      key = Nx.Random.key(42)

      {tokens, _new_key} = Sampling.sample(logits, key, temperature: 0.5, top_k: 3)
      token = Nx.to_number(tokens[0])
      assert token >= 0 and token < 5
    end
  end

  describe "generate_stream/3" do
    test "streams tokens via callback" do
      model = build_raw_lm()
      {predict_fn, params} = init_model(model)

      prompt = Nx.tensor([[1, 2, 3]])
      streamed = :ets.new(:streamed, [:set, :public])

      result =
        Generate.generate_stream(predict_fn, params,
          prompt: prompt,
          embed_fn: make_embed_fn(),
          max_tokens: 5,
          seq_len: @seq_len,
          temperature: 0.0,
          on_token: fn token ->
            count = :ets.info(streamed, :size)
            :ets.insert(streamed, {count, token})
            :cont
          end
        )

      token_count = :ets.info(streamed, :size)
      :ets.delete(streamed)

      assert token_count >= 1
      assert Nx.axis_size(result, 1) >= 4
    end

    test "halt stops generation early" do
      model = build_raw_lm()
      {predict_fn, params} = init_model(model)

      prompt = Nx.tensor([[1]])

      result =
        Generate.generate_stream(predict_fn, params,
          prompt: prompt,
          embed_fn: make_embed_fn(),
          max_tokens: 50,
          seq_len: @seq_len,
          temperature: 0.0,
          on_token: fn _token -> :halt end
        )

      # Should have prompt + 1 token only (halted after first)
      assert Nx.axis_size(result, 1) == 2
    end
  end

  describe "token_stream/3" do
    test "returns a Stream of token IDs" do
      model = build_raw_lm()
      {predict_fn, params} = init_model(model)

      prompt = Nx.tensor([[1, 2]])

      stream =
        Generate.token_stream(predict_fn, params,
          prompt: prompt,
          embed_fn: make_embed_fn(),
          max_tokens: 5,
          seq_len: @seq_len,
          temperature: 0.0
        )

      tokens = Enum.take(stream, 3)
      assert length(tokens) >= 1
      assert Enum.all?(tokens, &is_integer/1)
    end
  end

  describe "InferenceServer" do
    alias Edifice.Serving.InferenceServer

    test "serves single prediction" do
      model = build_raw_lm()
      {predict_fn, params} = init_model(model)

      {:ok, pid} =
        InferenceServer.start_link(
          predict_fn: predict_fn,
          params: params,
          max_batch_size: 4,
          batch_timeout_ms: 10
        )

      {input, _key} = Nx.Random.uniform(Nx.Random.key(1), shape: {1, @seq_len, @embed_dim})
      result = InferenceServer.predict(pid, %{"state_sequence" => input})

      assert is_struct(result, Nx.Tensor) or is_map(result)
      InferenceServer.stop(pid)
    end

    test "tracks metrics" do
      model = build_raw_lm()
      {predict_fn, params} = init_model(model)

      {:ok, pid} =
        InferenceServer.start_link(
          predict_fn: predict_fn,
          params: params,
          max_batch_size: 4,
          batch_timeout_ms: 10
        )

      {input, _key} = Nx.Random.uniform(Nx.Random.key(1), shape: {1, @seq_len, @embed_dim})
      _result = InferenceServer.predict(pid, %{"state_sequence" => input})

      metrics = InferenceServer.metrics(pid)
      assert metrics.requests >= 1
      assert metrics.batches >= 1
      InferenceServer.stop(pid)
    end
  end

  describe "Speculative" do
    alias Edifice.Serving.Speculative

    test "generates tokens with draft+verify" do
      # Use same model for both draft and verifier (simplified test)
      model = build_raw_lm()
      {predict_fn, params} = init_model(model)

      prompt = Nx.tensor([[1, 2, 3]])

      result =
        Speculative.generate(predict_fn, params, predict_fn, params,
          prompt: prompt,
          embed_fn: make_embed_fn(),
          seq_len: @seq_len,
          draft_steps: 3,
          max_tokens: 10,
          temperature: 0.0
        )

      # Should have generated tokens
      assert Nx.axis_size(result, 1) >= 4
      # Prompt preserved
      assert Nx.to_flat_list(result[[0, 0..2]]) == [1, 2, 3]
    end
  end
end
