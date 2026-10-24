defmodule Edifice.Serving.GenerateFused do
  @moduledoc """
  Optimized autoregressive generation with fused defn operations.

  Compared to `Edifice.Serving.Generate`, this module fuses the sampling
  pipeline (temperature scaling, softmax, Gumbel-max sampling, stop-token
  check) into a single `defn` function. Each decode step is one JIT call
  covering all post-forward-pass work.

  Every forward pass receives the full retained context (prompt and generated
  tokens), using the most recent `seq_len` tokens when the context overflows.
  There is no KV cache; only sampling and buffer updates are fused. The loop
  runs in Elixir, with one full model prediction per generated token.

  ## Performance Improvements Over Generate

  - **Fused sampling**: temperature + softmax + Gumbel + argmax in one JIT call
  - **Fused token management**: token write + stop check in the same call
  - **No intermediate tensor materializations** between sampling steps

  ## Usage

      tokens = Edifice.Serving.GenerateFused.generate(predict_fn, params,
        prompt: Nx.tensor([[1, 45, 892]]),
        embed_fn: embed_fn,
        seq_len: 128,
        max_tokens: 50,
        temperature: 0.7
      )
  """

  import Nx.Defn

  deftransformp(get_opt(opts, key, default), do: Keyword.get(opts, key, default))
  deftransformp(get_opt!(opts, key), do: Keyword.fetch!(opts, key))

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Run autoregressive generation with fused sampling.

  ## Options

    * `:prompt` - `[batch, prompt_len]` tensor of token IDs (required)
    * `:embed_fn` - Function `token_ids -> embeddings` (required)
    * `:max_tokens` - Maximum tokens to generate (required)
    * `:seq_len` - Model sequence length (required)
    * `:temperature` - Sampling temperature (default: 1.0). Use 0.01 for greedy.
    * `:stop_token` - Stop token ID (default: nil)
    * `:seed` - PRNG seed (default: 42)

  Returns only generated tokens, shape `{batch, max_tokens}`. Rows that emit
  `stop_token` retain that token and are zero-padded thereafter. `max_tokens`
  must be positive. Context beyond `seq_len` uses a sliding window.
  """
  def generate(predict_fn, params, opts) do
    prompt = Keyword.fetch!(opts, :prompt)
    embed_fn = Keyword.fetch!(opts, :embed_fn)
    max_tokens = Keyword.fetch!(opts, :max_tokens)
    seq_len = Keyword.fetch!(opts, :seq_len)
    temperature = Keyword.get(opts, :temperature, 1.0)
    stop_token = Keyword.get(opts, :stop_token, nil)
    seed = Keyword.get(opts, :seed, 42)

    unless Nx.rank(prompt) == 2 and Nx.axis_size(prompt, 1) > 0,
      do: raise(ArgumentError, "prompt must have shape {batch, nonempty_sequence}")

    unless is_integer(max_tokens) and max_tokens > 0,
      do: raise(ArgumentError, "max_tokens must be a positive integer")

    unless is_integer(seq_len) and seq_len > 0,
      do: raise(ArgumentError, "seq_len must be a positive integer")

    unless is_number(temperature) and temperature > 0,
      do: raise(ArgumentError, "temperature must be positive")

    batch_size = Nx.axis_size(prompt, 0)
    key = Nx.Random.key(seed)
    tokens = Nx.broadcast(Nx.s32(0), {batch_size, max_tokens})
    stopped = Nx.broadcast(Nx.u8(0), {batch_size})

    {tokens, _, _, _} =
      Enum.reduce_while(0..(max_tokens - 1), {tokens, key, stopped, prompt}, fn step,
                                                                                {tokens, key,
                                                                                 stopped,
                                                                                 context} ->
        if Nx.all(stopped) |> Nx.to_number() == 1 do
          {:halt, {tokens, key, stopped, context}}
        else
          length = Nx.axis_size(context, 1)
          retained = min(length, seq_len)
          context = Nx.slice_along_axis(context, length - retained, retained, axis: 1)
          embedded = embed_and_pad(context, embed_fn, seq_len)
          logits = predict_fn.(params, %{"state_sequence" => embedded})
          step_logits = logits[[.., retained - 1, ..]]

          {tokens, key, stopped} =
            fused_sample_and_write(step_logits, tokens, key, stopped, Nx.s32(step),
              temperature: temperature,
              stop_token: stop_token || -1,
              batch_size: batch_size,
              max_tokens: max_tokens
            )

          next_token = Nx.slice_along_axis(tokens, step, 1, axis: 1)
          context = Nx.concatenate([context, next_token], axis: 1)
          {:cont, {tokens, key, stopped, context}}
        end
      end)

    tokens
  end

  # ============================================================================
  # Fused sampling + token write + stop check (single defn call per step)
  # ============================================================================

  @doc """
  Fused operation: sample token, write to buffer, check stop condition.

  Combines temperature scaling, softmax, Gumbel-max sampling, buffer
  write, and stop-token detection into a single defn function.
  """
  defn fused_sample_and_write(logits, tokens, key, stopped, step, opts \\ []) do
    temperature = get_opt(opts, :temperature, 1.0)
    stop_token = get_opt(opts, :stop_token, -1)
    batch_size = get_opt!(opts, :batch_size)
    max_tokens = get_opt!(opts, :max_tokens)

    # Temperature + softmax + Gumbel-max sampling
    scaled = Nx.divide(Nx.as_type(logits, :f32), temperature)
    max_val = Nx.reduce_max(scaled, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(scaled, max_val)
    exp_vals = Nx.exp(shifted)
    probs = Nx.divide(exp_vals, Nx.sum(exp_vals, axes: [-1], keep_axes: true))

    {uniform, key} = Nx.Random.uniform(key, shape: Nx.shape(probs), type: :f32)
    gumbel = Nx.negate(Nx.log(Nx.negate(Nx.log(Nx.add(uniform, 1.0e-10)))))

    next_token =
      Nx.argmax(Nx.add(Nx.log(Nx.add(probs, 1.0e-10)), gumbel), axis: -1)
      |> Nx.as_type(:s32)

    # Write token at position `step` (masked by PREVIOUS stopped flag)
    # Write first, then update stop — so the stop token itself IS written
    pos_indices = Nx.iota({max_tokens}, type: :s32)
    mask = Nx.equal(pos_indices, step)
    mask_2d = Nx.broadcast(Nx.reshape(mask, {1, max_tokens}), {batch_size, max_tokens})

    token_broadcast =
      Nx.broadcast(Nx.reshape(next_token, {batch_size, 1}), {batch_size, max_tokens})

    # Write 0 if already stopped, token if not
    write_val =
      Nx.select(
        Nx.broadcast(Nx.reshape(stopped, {batch_size, 1}), {batch_size, max_tokens}),
        Nx.broadcast(Nx.s32(0), {batch_size, max_tokens}),
        token_broadcast
      )

    tokens = Nx.select(mask_2d, write_val, tokens)

    # NOW check stop: update stopped flag for subsequent steps
    stop_match = Nx.equal(next_token, stop_token)
    stopped = Nx.max(stopped, Nx.as_type(stop_match, :u8))

    {tokens, key, stopped}
  end

  @doc """
  Temperature + Gumbel-max categorical sampling.

  Standalone defn function for use in custom generation loops.
  """
  defn sample_token(logits, key, temperature) do
    scaled = Nx.divide(Nx.as_type(logits, :f32), temperature)
    max_val = Nx.reduce_max(scaled, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(scaled, max_val)
    exp_vals = Nx.exp(shifted)
    probs = Nx.divide(exp_vals, Nx.sum(exp_vals, axes: [-1], keep_axes: true))

    {uniform, key} = Nx.Random.uniform(key, shape: Nx.shape(probs), type: :f32)
    gumbel = Nx.negate(Nx.log(Nx.negate(Nx.log(Nx.add(uniform, 1.0e-10)))))

    token =
      Nx.argmax(Nx.add(Nx.log(Nx.add(probs, 1.0e-10)), gumbel), axis: -1)
      |> Nx.as_type(:s32)

    {token, key}
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp embed_and_pad(token_ids, embed_fn, seq_len) do
    embedded = embed_fn.(token_ids)
    current_len = Nx.axis_size(embedded, 1)

    if current_len >= seq_len do
      Nx.slice_along_axis(embedded, current_len - seq_len, seq_len, axis: 1)
    else
      {batch, _len, dim} = Nx.shape(embedded)

      pad =
        Nx.broadcast(0.0, {batch, seq_len - current_len, dim}) |> Nx.as_type(Nx.type(embedded))

      Nx.concatenate([embedded, pad], axis: 1)
    end
  end
end
