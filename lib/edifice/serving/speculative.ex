defmodule Edifice.Serving.Speculative do
  @moduledoc """
  Speculative decoding generation pipeline.

  Orchestrates a small draft model and a large verifier model for
  faster autoregressive generation. The draft model proposes K tokens
  cheaply, the verifier checks them in one forward pass, and the longest
  matching prefix is accepted.

  ## How It Works

  ```
  1. Draft model generates K tokens autoregressively (fast, small model)
  2. Verifier scores the K draft tokens in one forward pass (parallel)
  3. Accept longest prefix where draft == verifier (accept_reject)
  4. Append accepted tokens, resume from divergence point
  5. Repeat until max_tokens
  ```

  ## Usage

      # Build draft and verifier as LM models
      draft_model = Generate.build_lm(arch: :min_gru, vocab_size: 32000, embed_dim: 128, ...)
      verifier_model = Generate.build_lm(arch: :decoder_only, vocab_size: 32000, embed_dim: 512, ...)

      # Compile both
      {_, draft_fn} = Axon.build(draft_model, compiler: EXLA)
      {_, verify_fn} = Axon.build(verifier_model, compiler: EXLA)

      tokens = Speculative.generate(
        draft_fn, draft_params, verify_fn, verify_params,
        prompt: prompt, embed_fn: embed_fn, seq_len: 128,
        draft_steps: 4, max_tokens: 100
      )
  """

  alias Edifice.Serving.Sampling
  alias Edifice.Meta.SpeculativeDecoding

  @default_draft_steps 4
  @default_max_tokens 128
  @default_temperature 1.0
  @default_seed 42

  @doc """
  Run speculative decoding generation.

  ## Parameters

    - `draft_fn` - Compiled draft model prediction function
    - `draft_params` - Draft model parameters
    - `verify_fn` - Compiled verifier model prediction function
    - `verify_params` - Verifier model parameters
    - `opts` - Generation options

  ## Options

    - `:prompt` - `[batch, prompt_len]` tensor of token IDs (required)
    - `:embed_fn` - Token ID → embedding function (required)
    - `:seq_len` - Model sequence length (required)
    - `:draft_steps` - Number of draft tokens per speculation round (default: 4)
    - `:max_tokens` - Maximum total tokens to generate (default: 128)
    - `:temperature` - Sampling temperature (default: 1.0)
    - `:top_k` - Top-k filtering (default: 0)
    - `:top_p` - Nucleus sampling (default: 1.0)
    - `:seed` - PRNG seed (default: 42)
    - `:stop_token` - Stop token ID (default: nil)
    - `:on_accept` - Optional callback `fn accepted_count, round -> :ok` for monitoring

  ## Returns

    `[batch, prompt_len + generated_len]` tensor of token IDs.
  """
  def generate(draft_fn, draft_params, verify_fn, verify_params, opts) do
    prompt = Keyword.fetch!(opts, :prompt)
    embed_fn = Keyword.fetch!(opts, :embed_fn)
    seq_len = Keyword.fetch!(opts, :seq_len)
    draft_steps = Keyword.get(opts, :draft_steps, @default_draft_steps)
    max_tokens = Keyword.get(opts, :max_tokens, @default_max_tokens)
    temperature = Keyword.get(opts, :temperature, @default_temperature)
    top_k = Keyword.get(opts, :top_k, 0)
    top_p = Keyword.get(opts, :top_p, 1.0)
    seed = Keyword.get(opts, :seed, @default_seed)
    stop_token = Keyword.get(opts, :stop_token, nil)
    on_accept = Keyword.get(opts, :on_accept, nil)

    batch_size = Nx.axis_size(prompt, 0)
    key = Nx.Random.key(seed)
    sampling_opts = [temperature: temperature, top_k: top_k, top_p: top_p]

    # Initial state: just the prompt tokens
    do_speculative_loop(
      draft_fn, draft_params, verify_fn, verify_params,
      prompt, embed_fn, seq_len, draft_steps, max_tokens,
      temperature, sampling_opts, key, stop_token, on_accept,
      batch_size, 0
    )
  end

  defp do_speculative_loop(
         draft_fn, draft_params, verify_fn, verify_params,
         tokens, embed_fn, seq_len, draft_steps, max_tokens,
         temperature, sampling_opts, key, stop_token, on_accept,
         batch_size, round
       ) do
    current_len = Nx.axis_size(tokens, 1)
    _remaining = max_tokens - (current_len - Nx.axis_size(tokens, 1)) + max_tokens

    if current_len >= max_tokens + Nx.axis_size(tokens, 1) do
      tokens
    else
      # Step 1: Draft K tokens autoregressively from the draft model
      {draft_tokens, key} =
        draft_k_tokens(draft_fn, draft_params, tokens, embed_fn, seq_len,
          draft_steps, temperature, sampling_opts, key, batch_size)

      # Step 2: Verify all draft tokens with the verifier in one forward pass
      verifier_tokens =
        verify_draft(verify_fn, verify_params, tokens, draft_tokens, embed_fn,
          seq_len, temperature, sampling_opts, batch_size)

      # Step 3: Accept-reject
      accepted_count =
        SpeculativeDecoding.accept_reject(draft_tokens, verifier_tokens)
        |> Nx.squeeze()
        |> Nx.to_number()

      if on_accept, do: on_accept.(accepted_count, round)

      # Step 4: Append accepted tokens + first verifier correction
      accepted = Nx.slice_along_axis(verifier_tokens, 0, min(accepted_count + 1, draft_steps), axis: 1)
      accepted = Nx.reshape(accepted, {batch_size, Nx.axis_size(accepted, 1)})
      tokens = Nx.concatenate([tokens, accepted], axis: 1)

      # Check stop token
      if stop_token && token_at_end_matches?(tokens, stop_token) do
        tokens
      else
        new_len = Nx.axis_size(tokens, 1)

        if new_len >= max_tokens do
          Nx.slice_along_axis(tokens, 0, max_tokens, axis: 1)
        else
          do_speculative_loop(
            draft_fn, draft_params, verify_fn, verify_params,
            tokens, embed_fn, seq_len, draft_steps, max_tokens,
            temperature, sampling_opts, key, stop_token, on_accept,
            batch_size, round + 1
          )
        end
      end
    end
  end

  # Generate K tokens from the draft model autoregressively
  defp draft_k_tokens(draft_fn, draft_params, context, embed_fn, seq_len,
         k, temperature, sampling_opts, key, batch_size) do
    {draft_tokens, key} =
      Enum.reduce(1..k, {[], key}, fn _step, {acc, key} ->
        # Build full sequence: context + previously drafted tokens
        prev =
          case acc do
            [] -> context
            _ ->
              drafted = acc |> Enum.reverse() |> Nx.concatenate(axis: 1)
              Nx.concatenate([context, drafted], axis: 1)
          end

        current_len = Nx.axis_size(prev, 1)
        embedded = pad_and_embed(prev, embed_fn, seq_len)
        logits = draft_fn.(draft_params, %{"state_sequence" => embedded})

        pos = min(current_len - 1, seq_len - 1)
        step_logits = logits[[.., pos, ..]]

        {token, key} = sample_token(step_logits, key, temperature, sampling_opts)
        token = Nx.reshape(token, {batch_size, 1})
        {[token | acc], key}
      end)

    tokens = draft_tokens |> Enum.reverse() |> Nx.concatenate(axis: 1)
    {tokens, key}
  end

  # Verify draft tokens: run verifier on context + draft_tokens
  defp verify_draft(verify_fn, verify_params, context, draft_tokens, embed_fn,
         seq_len, temperature, _sampling_opts, batch_size) do
    k = Nx.axis_size(draft_tokens, 1)
    full_seq = Nx.concatenate([context, draft_tokens], axis: 1)
    context_len = Nx.axis_size(context, 1)

    embedded = pad_and_embed(full_seq, embed_fn, seq_len)
    logits = verify_fn.(verify_params, %{"state_sequence" => embedded})

    # Extract verifier's token predictions at each draft position
    # Position i in draft corresponds to logits at context_len + i - 1
    verifier_tokens =
      for i <- 0..(k - 1) do
        pos = min(context_len + i - 1, seq_len - 1)
        step_logits = logits[[.., pos, ..]]

        if temperature == 0.0 do
          Sampling.greedy(step_logits)
        else
          # Use deterministic sampling for verification (greedy from verifier)
          Sampling.greedy(step_logits)
        end
      end
      |> Enum.map(&Nx.reshape(&1, {batch_size, 1}))
      |> Nx.concatenate(axis: 1)

    verifier_tokens
  end

  defp pad_and_embed(token_ids, embed_fn, seq_len) do
    embedded = embed_fn.(token_ids)
    current_len = Nx.axis_size(embedded, 1)

    if current_len >= seq_len do
      Nx.slice_along_axis(embedded, current_len - seq_len, seq_len, axis: 1)
    else
      {batch, _len, dim} = Nx.shape(embedded)
      pad = Nx.broadcast(0.0, {batch, seq_len - current_len, dim}) |> Nx.as_type(Nx.type(embedded))
      Nx.concatenate([embedded, pad], axis: 1)
    end
  end

  defp sample_token(logits, key, temperature, sampling_opts) do
    if temperature == 0.0 do
      {Sampling.greedy(logits), key}
    else
      Sampling.sample(logits, key, sampling_opts)
    end
  end

  defp token_at_end_matches?(tokens, stop_token) do
    last_pos = Nx.axis_size(tokens, 1) - 1
    last = tokens[[.., last_pos]] |> Nx.squeeze() |> Nx.to_number()
    last == stop_token
  end
end
