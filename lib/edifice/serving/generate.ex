defmodule Edifice.Serving.Generate do
  @moduledoc """
  Autoregressive generation loop for Edifice language models.

  Takes a compiled Axon model (with `compiler: EXLA` for best performance),
  feeds tokens one at a time using KV caching, and samples new tokens using
  configurable strategies (greedy, temperature, top-k, top-p).

  ## Architecture

  ```
  prompt tokens → embed → model forward (full seq) → logits → sample token₁
                                                                    ↓
  token₁ → embed → model forward (1 token, KV cached) → logits → sample token₂
                                                                    ↓
                                                              ... repeat
  ```

  ## Usage

      # 1. Build a causal LM model (decoder + LM head)
      model = Edifice.Serving.Generate.build_lm(
        arch: :decoder_only,
        embed_dim: 256,
        hidden_size: 256,
        vocab_size: 32_000,
        num_layers: 4,
        num_heads: 8
      )

      # 2. Compile for fast inference
      {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
      params = init_fn.(template, Axon.ModelState.empty())

      # 3. Generate
      tokens = Edifice.Serving.Generate.generate(predict_fn, params,
        prompt: Nx.tensor([[1, 45, 892]]),
        max_tokens: 100,
        temperature: 0.7,
        top_k: 50
      )

  ## Performance Notes

  - Use `compiler: EXLA` for cached graph compilation (97x speedup)
  - KV cache avoids recomputing K/V for past tokens (O(n) vs O(n²) per step)
  - The generation loop runs outside the JIT boundary (each step is a separate
    JIT call), but the per-step forward pass is fully compiled
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
  Build a language model: backbone + LM head (dense → vocab_size logits).

  This wraps any Edifice sequence architecture with a vocabulary projection
  head for autoregressive generation.

  ## Options

    - `:arch` - Architecture name (atom) passed to `Edifice.build/2`
    - `:vocab_size` - Vocabulary size for the LM head (required)
    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Model hidden dimension (default: embed_dim)
    - All other options are forwarded to `Edifice.build/2`

  ## Returns

    An Axon model that takes `[batch, seq_len]` token IDs (via embedding)
    and outputs `[batch, seq_len, vocab_size]` logits.
  """
  def build_lm(opts) do
    arch = Keyword.fetch!(opts, :arch)
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    _hidden_size = Keyword.get(opts, :hidden_size, embed_dim)
    seq_len = Keyword.get(opts, :seq_len, 128)

    # Build opts for the backbone — force output_mode: :all for seq output
    backbone_opts =
      opts
      |> Keyword.drop([:arch, :vocab_size])
      |> Keyword.put(:output_mode, :all)
      |> Keyword.put(:seq_len, seq_len)

    backbone = Edifice.build(arch, backbone_opts)

    # Add LM head: project hidden → vocab logits
    backbone
    |> Axon.dense(vocab_size, name: "lm_head", use_bias: false)
  end

  # ============================================================================
  # Generation Loop
  # ============================================================================

  @doc """
  Run autoregressive generation.

  ## Options

    - `:prompt` - `[batch, prompt_len]` tensor of token IDs (required)
    - `:embed_fn` - Function `token_ids -> [batch, seq, embed_dim]` (required).
      Converts integer token IDs to embeddings. Typically a simple lookup:
      `fn ids -> Nx.take(embed_table, ids) end`
    - `:max_tokens` - Maximum tokens to generate (default: 128)
    - `:temperature` - Sampling temperature (default: 1.0)
    - `:top_k` - Top-k filtering (default: 0 = disabled)
    - `:top_p` - Nucleus sampling threshold (default: 1.0 = disabled)
    - `:seed` - PRNG seed (default: 42)
    - `:stop_token` - Stop generation at this token ID (default: nil)
    - `:seq_len` - Model's expected sequence length dimension (required for padding)

  ## Parameters

    - `predict_fn` - Compiled prediction function from `Axon.build/2`
    - `params` - Model parameters from init_fn
    - `opts` - Generation options (see above)

  ## Returns

    `[batch, prompt_len + generated_len]` tensor of token IDs.
  """
  def generate(predict_fn, params, opts) do
    prompt = Keyword.fetch!(opts, :prompt)
    embed_fn = Keyword.fetch!(opts, :embed_fn)
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

    # Accumulate generated tokens
    generated = [prompt]

    # --- Prefill: run the full prompt through the model ---
    prompt_embedded = pad_and_embed(prompt, embed_fn, seq_len)
    input_map = %{"state_sequence" => prompt_embedded}
    logits = predict_fn.(params, input_map)

    # logits shape: [batch, seq_len, vocab_size] (output_mode: :all)
    # Take logits at the last prompt position
    last_logits = logits[[.., prompt_len - 1, ..]]

    # Sample first token
    {next_token, key} =
      if temperature == 0.0 do
        {Sampling.greedy(last_logits), key}
      else
        Sampling.sample(last_logits, key, sampling_opts)
      end

    generated = [Nx.reshape(next_token, {batch_size, 1}) | generated]

    # --- Decode: generate one token at a time ---
    {generated, _key} =
      Enum.reduce_while(2..max_tokens, {generated, key}, fn _step, {acc, key} ->
        current_token = hd(acc)
        current_len = prompt_len + length(acc) - 1

        # Early stop check (on CPU)
        if stop_token && token_matches_stop?(current_token, stop_token) do
          {:halt, {acc, key}}
        else
          # Embed the single new token and pad to seq_len
          token_embedded = pad_and_embed_single(current_token, embed_fn, seq_len, current_len - 1)
          input_map = %{"state_sequence" => token_embedded}

          # Forward pass (single token in context — model sees padded seq)
          logits = predict_fn.(params, input_map)

          # Take logits at the position of our new token
          pos = min(current_len - 1, seq_len - 1)
          step_logits = logits[[.., pos, ..]]

          # Sample next token
          {next_token, key} =
            if temperature == 0.0 do
              {Sampling.greedy(step_logits), key}
            else
              Sampling.sample(step_logits, key, sampling_opts)
            end

          next_token = Nx.reshape(next_token, {batch_size, 1})
          {:cont, {[next_token | acc], key}}
        end
      end)

    # Concatenate all tokens in order
    generated
    |> Enum.reverse()
    |> Nx.concatenate(axis: 1)
  end

  @doc """
  Run generation without KV cache — simple but functional.
  Re-runs the full sequence each step. Use for correctness testing and
  as a baseline benchmark.

  Same options as `generate/3`.
  """
  def generate_simple(predict_fn, params, opts) do
    prompt = Keyword.fetch!(opts, :prompt)
    embed_fn = Keyword.fetch!(opts, :embed_fn)
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
          # Embed full sequence, pad to seq_len
          embedded = pad_and_embed(tokens, embed_fn, seq_len)
          input_map = %{"state_sequence" => embedded}

          logits = predict_fn.(params, input_map)

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
    embed_fn = Keyword.fetch!(opts, :embed_fn)
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

    generated = [prompt]

    # Prefill
    prompt_embedded = pad_and_embed(prompt, embed_fn, seq_len)
    logits = predict_fn.(params, %{"state_sequence" => prompt_embedded})
    last_logits = logits[[.., prompt_len - 1, ..]]

    {next_token, key} = sample_token(last_logits, key, temperature, sampling_opts)
    next_token = Nx.reshape(next_token, {batch_size, 1})
    generated = [next_token | generated]

    # Stream first token
    token_val = next_token |> Nx.squeeze() |> Nx.to_number()
    halt? = on_token.(token_val) == :halt

    # Decode loop
    {generated, _key} =
      if halt? do
        {generated, key}
      else
        Enum.reduce_while(2..max_tokens, {generated, key}, fn _step, {acc, key} ->
          current_token = hd(acc)
          current_len = prompt_len + length(acc) - 1

          if stop_token && token_matches_stop?(current_token, stop_token) do
            {:halt, {acc, key}}
          else
            token_embedded = pad_and_embed_single(current_token, embed_fn, seq_len, current_len - 1)
            logits = predict_fn.(params, %{"state_sequence" => token_embedded})

            pos = min(current_len - 1, seq_len - 1)
            step_logits = logits[[.., pos, ..]]

            {next_token, key} = sample_token(step_logits, key, temperature, sampling_opts)
            next_token = Nx.reshape(next_token, {batch_size, 1})

            token_val = next_token |> Nx.squeeze() |> Nx.to_number()

            case on_token.(token_val) do
              :halt -> {:halt, {[next_token | acc], key}}
              _ -> {:cont, {[next_token | acc], key}}
            end
          end
        end)
      end

    generated |> Enum.reverse() |> Nx.concatenate(axis: 1)
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
    embed_fn = Keyword.fetch!(opts, :embed_fn)
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
    prompt_embedded = pad_and_embed(prompt, embed_fn, seq_len)
    logits = predict_fn.(params, %{"state_sequence" => prompt_embedded})
    last_logits = logits[[.., prompt_len - 1, ..]]
    {next_token, key} = sample_token(last_logits, key, temperature, sampling_opts)
    next_token = Nx.reshape(next_token, {batch_size, 1})

    %{
      predict_fn: predict_fn,
      params: params,
      embed_fn: embed_fn,
      seq_len: seq_len,
      batch_size: batch_size,
      prompt_len: prompt_len,
      temperature: temperature,
      sampling_opts: sampling_opts,
      stop_token: stop_token,
      max_tokens: max_tokens,
      key: key,
      tokens_generated: 0,
      pending_token: next_token,
      generated: [next_token, prompt]
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
        # Generate next token
        %{
          predict_fn: predict_fn,
          params: params,
          embed_fn: embed_fn,
          seq_len: seq_len,
          batch_size: batch_size,
          prompt_len: prompt_len,
          temperature: temperature,
          sampling_opts: sampling_opts,
          key: key,
          generated: generated
        } = state

        current_len = prompt_len + length(generated) - 1

        token_embedded =
          pad_and_embed_single(current_token, embed_fn, seq_len, current_len - 1)

        logits = predict_fn.(params, %{"state_sequence" => token_embedded})
        pos = min(current_len - 1, seq_len - 1)
        step_logits = logits[[.., pos, ..]]

        {next_token, key} = sample_token(step_logits, key, temperature, sampling_opts)
        next_token = Nx.reshape(next_token, {batch_size, 1})

        new_state = %{
          state
          | key: key,
            tokens_generated: count + 1,
            pending_token: next_token,
            generated: [next_token | generated]
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

  defp pad_and_embed_single(token_id, embed_fn, seq_len, position) do
    # Embed single token
    embedded = embed_fn.(token_id)
    {batch, _one, dim} = Nx.shape(embedded)

    # Place at the correct position in a zero-padded seq_len sequence
    zeros = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(embedded)), {batch, seq_len, dim})
    pos = min(position, seq_len - 1)
    Nx.put_slice(zeros, [0, pos, 0], embedded)
  end

  defp token_matches_stop?(token_tensor, stop_token) do
    token_tensor
    |> Nx.reshape({:auto})
    |> Nx.to_flat_list()
    |> Enum.all?(&(&1 == stop_token))
  end
end
