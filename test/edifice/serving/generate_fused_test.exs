defmodule Edifice.Serving.GenerateFusedTest do
  use ExUnit.Case, async: true

  alias Edifice.Serving.GenerateFused

  @embed_dim 8
  @vocab_size 16
  @seq_len 16
  @max_tokens 10

  # Simple embedding: one-hot scaled by 0.1
  defp make_embed_fn do
    table = Nx.eye(@vocab_size) |> Nx.slice_along_axis(0, @embed_dim, axis: 1) |> Nx.as_type(:f32)
    fn token_ids -> Nx.take(table, token_ids) end
  end

  # Simple model: dense layer (input -> logits)
  # Returns constant logits regardless of input (for deterministic testing)
  defp build_constant_logit_model(hot_token) do
    # predict_fn that ignores input and returns logits with `hot_token` as highest
    logits_row = Nx.broadcast(0.0, {@vocab_size})
    logits_row = Nx.indexed_put(logits_row, Nx.tensor([[hot_token]]), Nx.tensor([10.0]))

    predict_fn = fn _params, input ->
      batch_size = Nx.axis_size(input["state_sequence"], 0)
      Nx.broadcast(Nx.reshape(logits_row, {1, 1, @vocab_size}), {batch_size, @seq_len, @vocab_size})
    end

    {predict_fn, %{}}
  end

  describe "greedy generation (low temperature)" do
    test "generates constant token with near-greedy temperature" do
      {predict_fn, params} = build_constant_logit_model(5)
      embed_fn = make_embed_fn()

      tokens = GenerateFused.generate(predict_fn, params,
        prompt: Nx.tensor([[1, 2, 3]]),
        embed_fn: embed_fn,
        seq_len: @seq_len,
        max_tokens: @max_tokens,
        temperature: 0.01,
        seed: 42
      )

      assert Nx.shape(tokens) == {1, @max_tokens}

      # All tokens should be 5 (the hot token)
      token_list = Nx.to_flat_list(tokens)
      assert Enum.all?(token_list, &(&1 == 5))
    end

    test "different hot tokens produce different outputs" do
      embed_fn = make_embed_fn()

      {predict_fn_7, params_7} = build_constant_logit_model(7)
      tokens_7 = GenerateFused.generate(predict_fn_7, params_7,
        prompt: Nx.tensor([[0]]),
        embed_fn: embed_fn,
        seq_len: @seq_len,
        max_tokens: 5,
        temperature: 0.01,
        seed: 1
      )

      {predict_fn_3, params_3} = build_constant_logit_model(3)
      tokens_3 = GenerateFused.generate(predict_fn_3, params_3,
        prompt: Nx.tensor([[0]]),
        embed_fn: embed_fn,
        seq_len: @seq_len,
        max_tokens: 5,
        temperature: 0.01,
        seed: 1
      )

      # tokens_7 should be all 7s, tokens_3 should be all 3s
      assert Enum.all?(Nx.to_flat_list(tokens_7), &(&1 == 7))
      assert Enum.all?(Nx.to_flat_list(tokens_3), &(&1 == 3))
    end
  end

  describe "temperature sampling" do
    test "higher temperature produces more varied tokens" do
      {predict_fn, params} = build_constant_logit_model(5)
      embed_fn = make_embed_fn()

      # Low temp: should be mostly 5
      tokens_low = GenerateFused.generate(predict_fn, params,
        prompt: Nx.tensor([[0]]),
        embed_fn: embed_fn,
        seq_len: @seq_len,
        max_tokens: 20,
        temperature: 0.01,
        seed: 42
      )

      # High temp: should have variety
      tokens_high = GenerateFused.generate(predict_fn, params,
        prompt: Nx.tensor([[0]]),
        embed_fn: embed_fn,
        seq_len: @seq_len,
        max_tokens: 20,
        temperature: 5.0,
        seed: 42
      )

      unique_low = tokens_low |> Nx.to_flat_list() |> Enum.uniq() |> length()
      unique_high = tokens_high |> Nx.to_flat_list() |> Enum.uniq() |> length()

      # High temp should produce more unique tokens
      assert unique_high >= unique_low
    end
  end

  describe "stop token" do
    test "pads with zeros after stop token" do
      # Model always outputs token 5, but stop_token=5 means first token stops
      {predict_fn, params} = build_constant_logit_model(5)
      embed_fn = make_embed_fn()

      tokens = GenerateFused.generate(predict_fn, params,
        prompt: Nx.tensor([[0]]),
        embed_fn: embed_fn,
        seq_len: @seq_len,
        max_tokens: @max_tokens,
        temperature: 0.01,
        stop_token: 5,
        seed: 42
      )

      token_list = Nx.to_flat_list(tokens)

      # First token is 5 (generated), rest should be 0 (stopped)
      assert hd(token_list) == 5
      assert Enum.all?(tl(token_list), &(&1 == 0))
    end
  end

  describe "batch generation" do
    test "handles batch_size > 1" do
      {predict_fn, params} = build_constant_logit_model(4)
      embed_fn = make_embed_fn()

      tokens = GenerateFused.generate(predict_fn, params,
        prompt: Nx.tensor([[1, 2], [3, 4]]),
        embed_fn: embed_fn,
        seq_len: @seq_len,
        max_tokens: 5,
        temperature: 0.01,
        seed: 42
      )

      assert Nx.shape(tokens) == {2, 5}

      # Both batch items should get token 4
      assert Enum.all?(Nx.to_flat_list(tokens), &(&1 == 4))
    end
  end

  describe "sample_token/3 defn" do
    test "greedy with very low temperature" do
      logits = Nx.tensor([[0.1, 0.5, 0.3, 0.9]])
      key = Nx.Random.key(42)

      {token, _key} = GenerateFused.sample_token(logits, key, 0.01)
      assert token |> Nx.squeeze() |> Nx.to_number() == 3
    end
  end
end
