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

  Every step re-runs the full (padded) sequence. A KV-cached decode path is
  not yet implemented — `blocks/kv_cache.ex` exists but is not wired into
  any architecture. Until it is, full re-run is the only decode strategy
  that is correct for context-dependent models.

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

  - Use `compiler: EXLA` for cached graph compilation (97x speedup)
  - The generation loop runs outside the JIT boundary (each step is a separate
    JIT call), but the per-step forward pass is fully compiled
  - Full re-run each step is O(n²) over the generation; acceptable at small
    seq_len, and the price of correctness until KV caching is wired up
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

  Currently delegates to `generate_simple/3` (full re-run each step). The
  previous "cached" decode path embedded only the newest token into an
  otherwise-zeros buffer — the model saw no prior context, which is
  semantically wrong for any context-dependent architecture. A real
  KV-cached path can reclaim this entry point once `blocks/kv_cache.ex`
  is wired into the architectures.

  ## Options

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
  Run generation by re-running the full (padded) sequence each step.

  Same options as `generate/3`, which currently delegates here.
  """
  def generate_simple(predict_fn, params, opts) do
    prompt = Keyword.fetch!(opts, :prompt)
    embed_fn = Keyword.get(opts, :embed_fn)
    pad_token = Keyword.get(opts, :pad_token, 0)
    max_tokens = Keyword.get(opts, :max_tokens, @default_max_tokens)
    temperature = Keyword.get(opts, :temperature, @default_temperature)
    top_k = Keyword.get(opts, :top_k, @default_top_k)
    top_p = Keyword.get(opts, :top_p, @default_top_p)
    seed = Keyword.get(opts, :seed, @default_seed)
    stop_token = Keyword.get(opts, :stop_token, nil)
    seq_len = Keyword.fetch!(opts, :seq_len)

    batch_size = Nx.axis_size(prompt, 0)
    key = Nx.Random.key(seed)
    sampling_opts = [temperature: temperature, top_k: top_k, top_p: top_p]

    # Start with prompt, grow each step
    {tokens, _key} =
      Enum.reduce_while(1..max_tokens, {prompt, key}, fn _step, {tokens, key} ->
        current_len = Nx.axis_size(tokens, 1)

        if stop_token && token_matches_stop?(Nx.slice_along_axis(tokens, current_len - 1, 1, axis: 1), stop_token) do
          {:halt, {tokens, key}}
        else
          logits = predict_fn.(params, model_inputs(tokens, embed_fn, seq_len, pad_token))

          # Take logits at the last real position
          pos = min(current_len - 1, seq_len - 1)
          step_logits = logits[[.., pos, ..]]

          {next_token, key} =
            if temperature == 0.0 do
              {Sampling.greedy(step_logits), key}
            else
              Sampling.sample(step_logits, key, sampling_opts)
            end

          next_token = Nx.reshape(next_token, {batch_size, 1})
          new_tokens = Nx.concatenate([tokens, next_token], axis: 1)
          {:cont, {new_tokens, key}}
        end
      end)

    tokens
  end

  # ============================================================================
  # Streaming Generation
  # ============================================================================

  @doc """
  Generate tokens with a per-token callback for streaming output.

  Same options as `generate/3`, plus:

    - `:on_token` - `fn token_id :: integer() -> :cont | :halt` callback.
      Called with each generated token ID (scalar integer). Return `:halt`
      to stop generation early, `:cont` (or anything else) to continue.

  ## Returns

    `[batch, prompt_len + generated_len]` tensor of token IDs (same as `generate/3`).
  """
  def generate_stream(predict_fn, params, opts) do
    prompt = Keyword.fetch!(opts, :prompt)
    embed_fn = Keyword.get(opts, :embed_fn)
    pad_token = Keyword.get(opts, :pad_token, 0)
    max_tokens = Keyword.get(opts, :max_tokens, @default_max_tokens)
    temperature = Keyword.get(opts, :temperature, @default_temperature)
    top_k = Keyword.get(opts, :top_k, @default_top_k)
    top_p = Keyword.get(opts, :top_p, @default_top_p)
    seed = Keyword.get(opts, :seed, @default_seed)
    stop_token = Keyword.get(opts, :stop_token, nil)
    seq_len = Keyword.fetch!(opts, :seq_len)
    on_token = Keyword.fetch!(opts, :on_token)

    batch_size = Nx.axis_size(prompt, 0)
    prompt_len = Nx.axis_size(prompt, 1)

    key = Nx.Random.key(seed)
    sampling_opts = [temperature: temperature, top_k: top_k, top_p: top_p]

    # Prefill
    logits = predict_fn.(params, model_inputs(prompt, embed_fn, seq_len, pad_token))
    last_logits = logits[[.., prompt_len - 1, ..]]

    {next_token, key} = sample_token(last_logits, key, temperature, sampling_opts)
    next_token = Nx.reshape(next_token, {batch_size, 1})
    tokens = Nx.concatenate([prompt, next_token], axis: 1)

    # Stream first token
    token_val = next_token |> Nx.squeeze() |> Nx.to_number()
    halt? = on_token.(token_val) == :halt

    # Decode loop — full-sequence re-run each step
    {tokens, _key} =
      if halt? do
        {tokens, key}
      else
        Enum.reduce_while(2..max_tokens, {tokens, key}, fn _step, {tokens, key} ->
          current_len = Nx.axis_size(tokens, 1)
          last = Nx.slice_along_axis(tokens, current_len - 1, 1, axis: 1)

          if stop_token && token_matches_stop?(last, stop_token) do
            {:halt, {tokens, key}}
          else
            logits = predict_fn.(params, model_inputs(tokens, embed_fn, seq_len, pad_token))

            pos = min(current_len - 1, seq_len - 1)
            step_logits = logits[[.., pos, ..]]

            {next_token, key} = sample_token(step_logits, key, temperature, sampling_opts)
            next_token = Nx.reshape(next_token, {batch_size, 1})

            token_val = next_token |> Nx.squeeze() |> Nx.to_number()
            new_tokens = Nx.concatenate([tokens, next_token], axis: 1)

            case on_token.(token_val) do
              :halt -> {:halt, {new_tokens, key}}
              _ -> {:cont, {new_tokens, key}}
            end
          end
        end)
      end

    tokens
  end

  @doc """
  Return a `Stream` that lazily generates tokens one at a time.

  Each element in the stream is an integer token ID.

  ## Options

    Same as `generate/3`.
  """
  def token_stream(predict_fn, params, opts) do
    Stream.resource(
      fn -> init_stream_state(predict_fn, params, opts) end,
      fn state -> next_stream_token(state) end,
      fn _state -> :ok end
    )
  end

  defp init_stream_state(predict_fn, params, opts) do
    prompt = Keyword.fetch!(opts, :prompt)
    embed_fn = Keyword.get(opts, :embed_fn)
    pad_token = Keyword.get(opts, :pad_token, 0)
    max_tokens = Keyword.get(opts, :max_tokens, @default_max_tokens)
    temperature = Keyword.get(opts, :temperature, @default_temperature)
    top_k = Keyword.get(opts, :top_k, @default_top_k)
    top_p = Keyword.get(opts, :top_p, @default_top_p)
    seed = Keyword.get(opts, :seed, @default_seed)
    stop_token = Keyword.get(opts, :stop_token, nil)
    seq_len = Keyword.fetch!(opts, :seq_len)

    batch_size = Nx.axis_size(prompt, 0)
    prompt_len = Nx.axis_size(prompt, 1)
    key = Nx.Random.key(seed)
    sampling_opts = [temperature: temperature, top_k: top_k, top_p: top_p]

    # Prefill
    logits = predict_fn.(params, model_inputs(prompt, embed_fn, seq_len, pad_token))
    last_logits = logits[[.., prompt_len - 1, ..]]
    {next_token, key} = sample_token(last_logits, key, temperature, sampling_opts)
    next_token = Nx.reshape(next_token, {batch_size, 1})

    %{
      predict_fn: predict_fn,
      params: params,
      embed_fn: embed_fn,
      pad_token: pad_token,
      seq_len: seq_len,
      batch_size: batch_size,
      temperature: temperature,
      sampling_opts: sampling_opts,
      stop_token: stop_token,
      max_tokens: max_tokens,
      key: key,
      tokens_generated: 0,
      pending_token: next_token,
      tokens: Nx.concatenate([prompt, next_token], axis: 1)
    }
  end

  defp next_stream_token(%{pending_token: nil} = state) do
    {:halt, state}
  end

  defp next_stream_token(state) do
    %{
      pending_token: current_token,
      tokens_generated: count,
      max_tokens: max_tokens,
      stop_token: stop_token
    } = state

    token_val = current_token |> Nx.squeeze() |> Nx.to_number()

    cond do
      count >= max_tokens ->
        {:halt, state}

      stop_token && token_val == stop_token ->
        {:halt, state}

      true ->
        # Generate next token from the full accumulated sequence
        %{
          predict_fn: predict_fn,
          params: params,
          embed_fn: embed_fn,
          pad_token: pad_token,
          seq_len: seq_len,
          batch_size: batch_size,
          temperature: temperature,
          sampling_opts: sampling_opts,
          key: key,
          tokens: tokens
        } = state

        current_len = Nx.axis_size(tokens, 1)

        logits = predict_fn.(params, model_inputs(tokens, embed_fn, seq_len, pad_token))
        pos = min(current_len - 1, seq_len - 1)
        step_logits = logits[[.., pos, ..]]

        {next_token, key} = sample_token(step_logits, key, temperature, sampling_opts)
        next_token = Nx.reshape(next_token, {batch_size, 1})

        new_state = %{
          state
          | key: key,
            tokens_generated: count + 1,
            pending_token: next_token,
            tokens: Nx.concatenate([tokens, next_token], axis: 1)
        }

        {[token_val], new_state}
    end
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
      pad = Nx.broadcast(0.0, {batch, seq_len - current_len, dim}) |> Nx.as_type(Nx.type(embedded))
      Nx.concatenate([embedded, pad], axis: 1)
    end
  end

  defp token_matches_stop?(token_tensor, stop_token) do
    token_tensor
    |> Nx.reshape({:auto})
    |> Nx.to_flat_list()
    |> Enum.all?(&(&1 == stop_token))
  end
end
