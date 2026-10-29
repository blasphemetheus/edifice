defmodule Edifice.Serving.Generate do
  @moduledoc """
  Autoregressive generation loop for Edifice language models.

  Takes a compiled Axon model (with `compiler: EXLA` for best performance),
  runs the sequence through the model each step, and samples new tokens using
  configurable strategies (greedy, temperature, top-k, top-p).

  ## Architecture

  ```
  prompt tokens → model forward (full seq) → logits at last position → sample token₁
                                                                            ↓
  prompt ++ token₁ → model forward (full seq) → logits → sample token₂
                                                                            ↓
                                                                      ... repeat
  ```

  By default each step re-runs the full padded sequence. For GQA decoder-only
  models, pass a cache built by `build_cache/1` to project only new tokens.
  Cached decoding preserves causal context and rotary positions. At capacity
  it prefills the retained window again, matching the full re-run behavior.

  ## Usage

      # 1. Build a causal LM (token embedding + decoder + LM head)
      model = Edifice.Serving.Generate.build_lm(
        arch: :decoder_only,
        embed_dim: 256,
        hidden_size: 256,
        vocab_size: 32_000,
        num_layers: 4,
        num_heads: 8,
        seq_len: 512
      )

      # 2. Compile for fast inference
      {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
      template = %{"token_ids" => Nx.template({1, 512}, :s64)}
      params = init_fn.(template, Axon.ModelState.empty())

      # 3. Generate — token IDs go straight in; the model embeds them
      tokens = Edifice.Serving.Generate.generate(predict_fn, params,
        prompt: Nx.tensor([[1, 45, 892]]),
        seq_len: 512,
        max_tokens: 100,
        temperature: 0.7,
        top_k: 50
      )

  Models built with `embedding: :external` (or any model taking a
  `"state_sequence"` float input) still work: pass an `:embed_fn` and the
  loop embeds token IDs outside the model as before.

  ## Performance Notes

  - Use `compiler: EXLA` to compile prediction and cached decoding
  - The generation loop runs outside the JIT boundary (each step is a separate
    JIT call), but the per-step forward pass is fully compiled
  - Full prediction repeats quadratic attention over the context each step.
    The cached path projects only new tokens until the window fills.
  - All generation APIs support KV caching, including callback and lazy streams.
  """

  alias Edifice.Serving.Sampling

  @default_max_tokens 128
  @default_temperature 1.0
  @default_top_k 0
  @default_top_p 1.0
  @default_seed 42

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a language model: token embedding + backbone + LM head.

  This wraps any Edifice sequence architecture with a trainable token
  embedding and a vocabulary projection head for autoregressive generation.

  ## Options

    - `:arch` - Architecture name (atom) passed to `Edifice.build/2`
    - `:vocab_size` - Vocabulary size for embedding + LM head (required)
    - `:embed_dim` - Token embedding dimension (required)
    - `:hidden_size` - Model hidden dimension (default: embed_dim)
    - `:embedding` - `:trainable` (default) prepends an `Axon.embedding`
      so the model takes integer token IDs and the embedding table lives
      in `Axon.ModelState` (i.e. it trains). `:external` preserves the
      legacy behavior: the model takes pre-embedded float vectors via the
      `"state_sequence"` input and you supply an `:embed_fn` at generation
      time (that table does NOT receive gradients).
    - All other options are forwarded to `Edifice.build/2`

  ## Returns

    With `embedding: :trainable` (default), an Axon model that takes
    `[batch, seq_len]` integer token IDs via the `"token_ids"` input and
    outputs `[batch, seq_len, vocab_size]` logits. With
    `embedding: :external`, the model takes `[batch, seq_len, embed_dim]`
    floats via `"state_sequence"`.
  """
  def build_lm(opts) do
    arch = Keyword.fetch!(opts, :arch)
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    _hidden_size = Keyword.get(opts, :hidden_size, embed_dim)
    seq_len = Keyword.get(opts, :seq_len, 128)
    embedding = Keyword.get(opts, :embedding, :trainable)

    # Build opts for the backbone — force output_mode: :all for seq output
    backbone_opts =
      opts
      |> Keyword.drop([:arch, :vocab_size, :embedding])
      |> Keyword.put(:output_mode, :all)
      |> Keyword.put(:seq_len, seq_len)

    backbone_opts =
      case embedding do
        :external ->
          backbone_opts

        :trainable ->
          embedded =
            Axon.input("token_ids", shape: {nil, seq_len})
            |> Axon.embedding(vocab_size, embed_dim, name: "token_embedding")

          Keyword.put(backbone_opts, :input, embedded)
      end

    model =
      Edifice.build(arch, backbone_opts)
      |> Axon.dense(vocab_size, name: "lm_head", use_bias: false)

    if embedding == :trainable and
         not Map.has_key?(Axon.get_inputs(model), "token_ids") do
      raise ArgumentError,
            "architecture #{inspect(arch)} ignored the :input override " <>
              "(it does not route through ModelBuilder.build_sequence_model), " <>
              "so the token embedding would be silently dropped. " <>
              "Use embedding: :external with this architecture."
    end

    model
  end

  # ============================================================================
  # Generation Loop
  # ============================================================================

  @doc """
  Run autoregressive generation.

  Uses full-context prediction by default. Pass `:cache` from `build_cache/1`
  for incremental GQA decoding with the same trained parameters.

  ## Options

    - `:cache` - Optional reusable decoder from `build_cache/1`
    - `:prompt` - `[batch, prompt_len]` tensor of token IDs (required)
    - `:seq_len` - Model's expected sequence length dimension (required for padding)
    - `:embed_fn` - Function `token_ids -> [batch, seq, embed_dim]`.
      Only for models taking a `"state_sequence"` float input (e.g. built
      with `embedding: :external`). Omit for models with a trainable
      embedding — token IDs are fed directly via `"token_ids"`.
    - `:pad_token` - Token ID used to right-pad up to `seq_len` when feeding
      token IDs directly (default: 0)
    - `:max_tokens` - Maximum tokens to generate (default: 128)
    - `:temperature` - Sampling temperature (default: 1.0)
    - `:top_k` - Top-k filtering (default: 0 = disabled)
    - `:top_p` - Nucleus sampling threshold (default: 1.0 = disabled)
    - `:seed` - PRNG seed (default: 42)
    - `:stop_token` - Stop generation at this token ID (default: nil)

  ## Parameters

    - `predict_fn` - Compiled prediction function from `Axon.build/2`
    - `params` - Model parameters from init_fn
    - `opts` - Generation options (see above)

  ## Returns

    `[batch, prompt_len + generated_len]` tensor of token IDs.
  """
  def generate(predict_fn, params, opts) do
    generate_simple(predict_fn, params, opts)
  end

  @doc """
  Build a reusable GQA decoder cache from the same options as `build_lm/1`.

      cache = Generate.build_cache(model_opts ++ [compiler: EXLA])
      Generate.generate(predict_fn, params, prompt: prompt, seq_len: 512, cache: cache)

  Supports `arch: :decoder_only` with GQA and no YaRN. Use `:embed_fn` here
  for external embeddings. New-token embedding, attention, FFN and LM head
  run in one compiled call. At capacity the generation loop resets and
  prefills the retained window to preserve full-context semantics.
  """
  def build_cache(opts) do
    unless Keyword.get(opts, :arch, :decoder_only) == :decoder_only,
      do: raise(ArgumentError, "KV generation currently supports :decoder_only")

    before =
      if Keyword.get(opts, :embedding, :trainable) == :external do
        embed = Keyword.fetch!(opts, :embed_fn)
        fn _params, tokens -> embed.(tokens) end
      else
        fn params, tokens -> Nx.take(params["token_embedding"]["kernel"], tokens) end
      end

    after_fn = fn params, output -> Edifice.Stateful.Ops.dense(output, params["lm_head"]) end

    Edifice.Transformer.DecoderOnly.Cache.build(
      Keyword.merge(opts, input_transform: before, output_transform: after_fn)
    )
  end

  @doc """
  Run generation with an optional KV cache, otherwise re-running the full
  padded sequence each step.

  Same options as `generate/3`, which currently delegates here.
  """
  def generate_simple(predict_fn, params, opts) do
    {ctx, acc, max_tokens} = init_generation(predict_fn, params, opts)

    if max_tokens == 0 do
      elem(acc, 0)
    else
      {tokens, _, _, _} =
        Enum.reduce_while(1..max_tokens, acc, fn _step, acc ->
          simple_step(ctx, acc)
        end)

      tokens
    end
  end

  defp init_generation(predict_fn, params, opts) do
    prompt = Keyword.fetch!(opts, :prompt)
    max_tokens = Keyword.get(opts, :max_tokens, @default_max_tokens)
    seq_len = Keyword.fetch!(opts, :seq_len)
    cache = Keyword.get(opts, :cache)
    stop_token = Keyword.get(opts, :stop_token, nil)
    validate_simple_opts!(prompt, max_tokens, seq_len, cache)

    batch_size = Nx.axis_size(prompt, 0)
    temperature = Keyword.get(opts, :temperature, @default_temperature)

    ctx = %{
      predict_fn: predict_fn,
      params: params,
      cache: cache,
      embed_fn: Keyword.get(opts, :embed_fn),
      pad_token: Keyword.get(opts, :pad_token, 0),
      seq_len: seq_len,
      batch_size: batch_size,
      stop_token: stop_token,
      temperature: temperature,
      sampling_opts: [
        temperature: temperature,
        top_k: Keyword.get(opts, :top_k, @default_top_k),
        top_p: Keyword.get(opts, :top_p, @default_top_p)
      ]
    }

    key = Nx.Random.key(Keyword.get(opts, :seed, @default_seed))

    stopped =
      if stop_token,
        do: Nx.equal(Nx.take(prompt, Nx.axis_size(prompt, 1) - 1, axis: 1), stop_token),
        else: Nx.broadcast(0, {batch_size})

    {ctx, {prompt, key, stopped, nil}, max_tokens}
  end

  defp validate_simple_opts!(prompt, max_tokens, seq_len, cache) do
    unless is_integer(max_tokens) and max_tokens >= 0 and is_integer(seq_len) and seq_len > 0 and
             Nx.axis_size(prompt, 1) > 0,
           do:
             raise(
               ArgumentError,
               "generation needs a nonempty prompt, positive seq_len and nonnegative max_tokens"
             )

    if cache && cache.max_seq_len != seq_len,
      do: raise(ArgumentError, "seq_len must match the decoder cache capacity")

    :ok
  end

  defp simple_step(ctx, {tokens, key, stopped, state}) do
    if Nx.to_number(Nx.all(stopped)) == 1 do
      {:halt, {tokens, key, stopped, state}}
    else
      {step_logits, state} = next_logits(ctx, tokens, state)
      {next_token, key} = sample_token(step_logits, key, ctx.temperature, ctx.sampling_opts)
      next_token = Nx.reshape(next_token, {ctx.batch_size})
      {next_token, stopped} = apply_stop_token(next_token, stopped, ctx.stop_token)
      new_tokens = Nx.concatenate([tokens, Nx.new_axis(next_token, 1)], axis: 1)
      {:cont, {new_tokens, key, stopped, state}}
    end
  end

  # Logits for the next position: incremental via the KV cache when present
  # (re-prefilling the retained window at capacity), else a full re-run.
  defp next_logits(%{cache: nil} = ctx, tokens, _state) do
    current_len = Nx.axis_size(tokens, 1)
    inputs = model_inputs(tokens, ctx.embed_fn, ctx.seq_len, ctx.pad_token)
    logits = ctx.predict_fn.(ctx.params, inputs)
    {Nx.take(logits, min(current_len, ctx.seq_len) - 1, axis: 1), nil}
  end

  defp next_logits(%{cache: cache, seq_len: seq_len} = ctx, tokens, state) do
    current_len = Nx.axis_size(tokens, 1)

    {chunk, state} =
      if state == nil or Nx.to_number(state.position) >= seq_len do
        count = min(current_len, seq_len)

        {Nx.slice_along_axis(tokens, current_len - count, count, axis: 1),
         cache.init.(ctx.batch_size)}
      else
        {Nx.slice_along_axis(tokens, current_len - 1, 1, axis: 1), state}
      end

    {logits, state} = cache.predict.(ctx.params, chunk, state)
    {Nx.take(logits, Nx.axis_size(logits, 1) - 1, axis: 1), state}
  end

  defp apply_stop_token(next_token, stopped, nil), do: {next_token, stopped}

  defp apply_stop_token(next_token, stopped, stop_token) do
    token = Nx.select(stopped, stop_token, next_token)
    {token, Nx.logical_or(stopped, Nx.equal(token, stop_token))}
  end

  # ============================================================================
  # Streaming Generation
  # ============================================================================

  @doc """
  Generate tokens with a per-token callback for streaming output.

  Same options as `generate/3`, including `:cache`, for a single prompt, plus:

    - `:on_token` - `fn token_id :: integer() -> :cont | :halt` callback.
      Called with each generated token ID (scalar integer). Return `:halt`
      to stop generation early, `:cont` (or anything else) to continue.

  ## Returns

    `[1, prompt_len + generated_len]` tensor of token IDs. A generated stop token
    is included in the result and passed to the callback. A prompt already ending
    in the stop token produces no new tokens.
  """
  def generate_stream(predict_fn, params, opts) do
    on_token = Keyword.fetch!(opts, :on_token)
    {ctx, acc, max_tokens} = init_generation(predict_fn, params, opts)
    validate_stream_batch!(ctx)

    if max_tokens == 0 do
      elem(acc, 0)
    else
      {tokens, _, _, _} =
        Enum.reduce_while(1..max_tokens, acc, fn _step, acc ->
          stream_step(ctx, acc, on_token)
        end)

      tokens
    end
  end

  defp stream_step(ctx, acc, on_token) do
    case simple_step(ctx, acc) do
      {:halt, acc} ->
        {:halt, acc}

      {:cont, acc} ->
        token = last_token(elem(acc, 0))
        if on_token.(token) == :halt, do: {:halt, acc}, else: {:cont, acc}
    end
  end

  @doc """
  Return a `Stream` that lazily generates integer token IDs for a single prompt.

  Accepts the same options as `generate/3`, including `:cache`. Stop tokens
  are not emitted. Prediction happens only when the next token is requested;
  halting enumeration does not predict an extra token.
  """
  def token_stream(predict_fn, params, opts) do
    Stream.resource(
      fn ->
        {ctx, acc, remaining} = init_generation(predict_fn, params, opts)
        validate_stream_batch!(ctx)
        {ctx, acc, remaining}
      end,
      &next_stream_token/1,
      fn _state -> :ok end
    )
  end

  defp next_stream_token({_, _, 0} = state), do: {:halt, state}

  defp next_stream_token({ctx, acc, remaining} = state) do
    case simple_step(ctx, acc) do
      {:halt, _} ->
        {:halt, state}

      {:cont, acc} ->
        token = last_token(elem(acc, 0))
        next = {ctx, acc, remaining - 1}

        if token == ctx.stop_token, do: {:halt, next}, else: {[token], next}
    end
  end

  defp validate_stream_batch!(%{batch_size: 1}), do: :ok

  defp validate_stream_batch!(_) do
    raise ArgumentError, "streaming generation requires a single prompt (batch size 1)"
  end

  defp last_token(tokens) do
    tokens |> Nx.take(Nx.axis_size(tokens, 1) - 1, axis: 1) |> Nx.squeeze() |> Nx.to_number()
  end

  defp sample_token(logits, key, temperature, sampling_opts) do
    if temperature == 0.0 do
      {Sampling.greedy(logits), key}
    else
      Sampling.sample(logits, key, sampling_opts)
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  # Build the model input map for a token sequence. With an embed_fn the
  # model takes pre-embedded floats ("state_sequence"); without one the
  # model embeds internally and takes raw token IDs ("token_ids").
  defp model_inputs(tokens, nil, seq_len, pad_token) do
    %{"token_ids" => pad_tokens(tokens, seq_len, pad_token)}
  end

  defp model_inputs(tokens, embed_fn, seq_len, _pad_token) do
    %{"state_sequence" => pad_and_embed(tokens, embed_fn, seq_len)}
  end

  defp pad_tokens(tokens, seq_len, pad_token) do
    current_len = Nx.axis_size(tokens, 1)

    if current_len >= seq_len do
      # Truncate to seq_len (take last seq_len positions)
      Nx.slice_along_axis(tokens, current_len - seq_len, seq_len, axis: 1)
    else
      batch = Nx.axis_size(tokens, 0)

      pad =
        Nx.broadcast(Nx.tensor(pad_token, type: Nx.type(tokens)), {batch, seq_len - current_len})

      Nx.concatenate([tokens, pad], axis: 1)
    end
  end

  defp pad_and_embed(token_ids, embed_fn, seq_len) do
    embedded = embed_fn.(token_ids)
    current_len = Nx.axis_size(embedded, 1)

    if current_len >= seq_len do
      # Truncate to seq_len (take last seq_len positions)
      Nx.slice_along_axis(embedded, current_len - seq_len, seq_len, axis: 1)
    else
      # Pad with zeros
      {batch, _len, dim} = Nx.shape(embedded)

      pad =
        Nx.broadcast(0.0, {batch, seq_len - current_len, dim}) |> Nx.as_type(Nx.type(embedded))

      Nx.concatenate([embedded, pad], axis: 1)
    end
  end
end
