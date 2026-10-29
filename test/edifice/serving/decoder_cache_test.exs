defmodule Edifice.Serving.DecoderCacheTest do
  use ExUnit.Case, async: true
  import Nx.Testing
  alias Edifice.Serving.Generate
  alias Edifice.Transformer.DecoderOnly.Cache

  @opts [
    arch: :decoder_only,
    embed_dim: 6,
    hidden_size: 8,
    num_heads: 4,
    num_kv_heads: 2,
    num_layers: 2,
    seq_len: 8,
    vocab_size: 12,
    dropout: 0.0
  ]

  for rope_opts <- [
        [use_rope: true],
        [use_rope: false],
        [interleave_rope: true, final_norm: false]
      ] do
    test "cached chunks match full backbone and support rollback: #{inspect(rope_opts)}" do
      opts = Keyword.merge(@opts, unquote(rope_opts))
      model = Edifice.Transformer.DecoderOnly.build(Keyword.put(opts, :output_mode, :all))
      {init, predict} = Axon.build(model, seed: 42)
      x = Nx.sin(Nx.iota({2, 8, 6}, type: :f32))
      input = %{"state_sequence" => x}
      params = init.(input, Axon.ModelState.empty())
      expected = predict.(params, input)
      cache = Cache.build(opts)

      {prefix, state} =
        cache.predict.(params, Nx.slice_along_axis(x, 0, 3, axis: 1), cache.init.(2))

      saved = Edifice.Stateful.snapshot(state)

      {outputs, final} =
        Enum.map_reduce(3..7, state, fn i, state ->
          cache.predict.(params, Nx.slice_along_axis(x, i, 1, axis: 1), state)
        end)

      assert_all_close(Nx.concatenate([prefix | outputs], axis: 1), expected, atol: 1.0e-5)
      {replayed, _} = cache.predict.(params, Nx.slice_along_axis(x, 3, 1, axis: 1), saved)
      assert_all_close(replayed, hd(outputs))
      assert Nx.to_number(final.position) == 8
      assert Nx.shape(elem(final.layers[1], 0)) == {2, 2, 8, 2}

      assert_raise ArgumentError, ~r/capacity/, fn ->
        cache.predict.(params, Nx.slice_along_axis(x, 0, 1, axis: 1), final)
      end
    end
  end

  test "generation matches uncached sampling, including window overflow and zero tokens" do
    model = Generate.build_lm(@opts)
    {init, predict} = Axon.build(model, seed: 42)
    params = init.(%{"token_ids" => Nx.template({2, 8}, :s64)}, Axon.ModelState.empty())
    cache = Generate.build_cache(@opts)

    for temperature <- [0.0, 0.7] do
      opts = [
        prompt: Nx.tensor([[1, 2, 3], [2, 4, 5]]),
        seq_len: 8,
        max_tokens: 9,
        temperature: temperature,
        seed: 42
      ]

      expected = Generate.generate(predict, params, opts)

      actual =
        Generate.generate(
          fn _, _ -> flunk("full predictor called on cache path") end,
          params,
          Keyword.put(opts, :cache, cache)
        )

      assert_equal(actual, expected)

      assert_equal(
        Generate.generate(predict, params, Keyword.put(opts, :max_tokens, 0)),
        opts[:prompt]
      )
    end
  end

  @tag :exla_only
  test "compiled host cache preserves full-model logits" do
    model = Generate.build_lm(@opts)
    {init, predict} = Axon.build(model, seed: 42, compiler: EXLA, client: :host)
    tokens = Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    params = init.(%{"token_ids" => tokens}, Axon.ModelState.empty())
    expected = predict.(params, %{"token_ids" => tokens})
    cache = Generate.build_cache(@opts ++ [compiler: EXLA, client: :host])

    {outputs, _} =
      Enum.map_reduce(0..7, cache.init.(1), fn i, state ->
        cache.predict.(params, Nx.slice_along_axis(tokens, i, 1, axis: 1), state)
      end)

    assert_all_close(Nx.concatenate(outputs, axis: 1), expected, atol: 1.0e-5)

    opts = [prompt: Nx.tensor([[1, 2, 3]]), seq_len: 8, max_tokens: 7, temperature: 0.0]
    generated = Generate.generate(predict, params, opts) |> Nx.to_flat_list() |> Enum.drop(3)

    assert Enum.to_list(Generate.token_stream(predict, params, Keyword.put(opts, :cache, cache))) ==
             generated
  end

  test "streaming rejects incompatible batch sizes and cache capacities before prediction" do
    no_predict = fn _, _ -> flunk("invalid options reached prediction") end
    opts = [prompt: Nx.tensor([[1], [2]]), seq_len: 8, max_tokens: 1, on_token: fn _ -> :cont end]

    assert_raise ArgumentError, ~r/single prompt/, fn ->
      Generate.generate_stream(no_predict, %{}, opts)
    end

    assert_raise ArgumentError, ~r/single prompt/, fn ->
      Generate.token_stream(no_predict, %{}, opts) |> Enum.to_list()
    end

    opts =
      Keyword.merge(opts,
        prompt: Nx.tensor([[1]]),
        cache: Generate.build_cache(@opts),
        seq_len: 4
      )

    assert_raise ArgumentError, ~r/capacity/, fn ->
      Generate.generate_stream(no_predict, %{}, opts)
    end

    assert_raise ArgumentError, ~r/capacity/, fn ->
      Generate.token_stream(no_predict, %{}, opts) |> Enum.to_list()
    end
  end

  test "both streaming APIs match seeded generation through cache overflow" do
    model = Generate.build_lm(@opts)
    {init, predict} = Axon.build(model, seed: 42)
    params = init.(%{"token_ids" => Nx.template({1, 8}, :s64)}, Axon.ModelState.empty())
    cache = Generate.build_cache(@opts)

    for temperature <- [0.0, 0.7],
        prompt <- [Nx.tensor([[1, 2, 3]]), Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])] do
      opts = [prompt: prompt, seq_len: 8, max_tokens: 9, temperature: temperature, seed: 42]
      expected = Generate.generate(predict, params, opts)

      expected_tokens =
        expected |> Nx.slice_along_axis(Nx.axis_size(prompt, 1), 9, axis: 1) |> Nx.to_flat_list()

      cached_opts = Keyword.put(opts, :cache, cache)
      no_full = fn _, _ -> flunk("stream called the full predictor with a cache") end
      assert Enum.to_list(Generate.token_stream(no_full, params, cached_opts)) == expected_tokens
      assert Enum.to_list(Generate.token_stream(predict, params, opts)) == expected_tokens

      actual =
        Generate.generate_stream(
          no_full,
          params,
          Keyword.put(cached_opts, :on_token, fn token -> send(self(), {:token, token}) end)
        )

      assert_equal(actual, expected)
      for token <- expected_tokens, do: assert_receive({:token, ^token})
      refute_receive {:token, _}
    end
  end

  test "lazy cache does one prefill then only the requested decode steps" do
    model = Generate.build_lm(@opts)
    {init, _predict} = Axon.build(model, seed: 42)
    params = init.(%{"token_ids" => Nx.template({1, 8}, :s64)}, Axon.ModelState.empty())
    cache = Generate.build_cache(@opts)

    traced = %{
      cache
      | predict: fn p, chunk, state ->
          send(self(), {:chunk, Nx.axis_size(chunk, 1)})
          cache.predict.(p, chunk, state)
        end
    }

    opts = [
      prompt: Nx.tensor([[1, 2, 3]]),
      seq_len: 8,
      max_tokens: 10,
      temperature: 0.0,
      cache: traced
    ]

    no_full = fn _, _ -> flunk("full prediction") end
    stream = Generate.token_stream(no_full, params, opts)
    refute_receive {:chunk, _}
    assert length(Enum.take(stream, 2)) == 2
    assert_receive {:chunk, 3}
    assert_receive {:chunk, 1}
    refute_receive {:chunk, _}

    result =
      Generate.generate_stream(no_full, params, Keyword.put(opts, :on_token, fn _ -> :halt end))

    assert Nx.axis_size(result, 1) == 4
    assert_receive {:chunk, 3}
    refute_receive {:chunk, _}
  end

  test "stream limits and stop tokens never cause unwanted prediction" do
    predict = fn _, %{"token_ids" => tokens} ->
      send(self(), :prediction)
      Nx.broadcast(Nx.tensor([[[0.0, 9.0, 0.0]]]), {1, Nx.axis_size(tokens, 1), 3})
    end

    opts = [
      prompt: Nx.tensor([[2]]),
      seq_len: 4,
      max_tokens: 0,
      temperature: 0.0,
      on_token: fn token -> send(self(), {:token, token}) end
    ]

    assert_equal(Generate.generate_stream(predict, %{}, opts), opts[:prompt])
    assert Enum.to_list(Generate.token_stream(predict, %{}, opts)) == []
    refute_receive :prediction
    refute_receive {:token, _}

    for extra <- [[max_tokens: 1], [max_tokens: 5, stop_token: 1]] do
      result = Generate.generate_stream(predict, %{}, Keyword.merge(opts, extra))
      assert_equal(result, Nx.tensor([[2, 1]]))
      assert_receive :prediction
      assert_receive {:token, 1}
      refute_receive :prediction
    end

    assert Enum.to_list(
             Generate.token_stream(
               predict,
               %{},
               Keyword.merge(opts, max_tokens: 5, stop_token: 1)
             )
           ) == []

    assert_receive :prediction
    refute_receive :prediction
    stopped = Keyword.merge(opts, max_tokens: 5, stop_token: 2)
    assert_equal(Generate.generate_stream(predict, %{}, stopped), opts[:prompt])
    assert Enum.to_list(Generate.token_stream(predict, %{}, stopped)) == []
    refute_receive :prediction
  end

  test "finished rows remain stopped while other rows decode" do
    predict = fn _params, %{"token_ids" => tokens} ->
      next = Nx.tensor([[[9.0, 0.0, 0.0]], [[0.0, 9.0, 0.0]]])
      Nx.broadcast(next, {2, Nx.axis_size(tokens, 1), 3})
    end

    out =
      Generate.generate(predict, %{},
        prompt: Nx.tensor([[2], [1]]),
        seq_len: 4,
        max_tokens: 3,
        stop_token: 2,
        temperature: 0.0
      )

    assert_equal(out, Nx.tensor([[2, 2, 2, 2], [1, 1, 1, 1]]))
  end

  test "external embeddings and oversized prompts retain the same context" do
    embed = fn ids ->
      Nx.new_axis(Nx.as_type(ids, :f32), -1)
      |> Nx.broadcast({Nx.axis_size(ids, 0), Nx.axis_size(ids, 1), 6})
    end

    opts = Keyword.merge(@opts, embedding: :external, embed_fn: embed)
    model = Generate.build_lm(opts)
    {init, predict} = Axon.build(model, seed: 42)
    params = init.(%{"state_sequence" => Nx.template({1, 8, 6}, :f32)}, Axon.ModelState.empty())

    gen = [
      prompt: Nx.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]]),
      seq_len: 8,
      max_tokens: 2,
      embed_fn: embed,
      temperature: 0.0
    ]

    assert_equal(
      Generate.generate(predict, params, gen),
      Generate.generate(predict, params, Keyword.put(gen, :cache, Generate.build_cache(opts)))
    )

    expected = Generate.generate(predict, params, gen) |> Nx.to_flat_list() |> Enum.drop(9)

    assert Enum.to_list(
             Generate.token_stream(
               predict,
               params,
               Keyword.put(gen, :cache, Generate.build_cache(opts))
             )
           ) == expected

    assert_raise ArgumentError, fn -> Cache.build(Keyword.put(@opts, :yarn, true)) end
  end
end
