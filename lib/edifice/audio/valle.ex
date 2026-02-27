defmodule Edifice.Audio.VALLE do
  @moduledoc """
  VALL-E: Neural Codec Language Models for Zero-Shot Text-to-Speech.

  VALL-E treats TTS as conditional language modeling over neural codec tokens
  (from EnCodec). Given text and a 3-second audio prompt, VALL-E generates
  speech that preserves the speaker's voice characteristics.

  ## Architecture

  VALL-E uses a two-stage approach with separate models for coarse and fine tokens:

  ```
  Text tokens [batch, text_len]
  Audio prompt tokens [batch, num_codebooks, prompt_len]
        |
  +------------------------------------------+
  | AR Model (Autoregressive)                |
  |                                          |
  | Decoder-only transformer                 |
  | Generates codebook 0 (coarse tokens)     |
  | Left-to-right, causal attention          |
  +------------------------------------------+
        |
  Coarse tokens [batch, seq_len]  (codebook 0)
        |
  +------------------------------------------+
  | NAR Model (Non-Autoregressive)           |
  |                                          |
  | Bidirectional transformer                |
  | Generates codebooks 1-7 (fine tokens)    |
  | Processes all positions in parallel      |
  +------------------------------------------+
        |
  Fine tokens [batch, 7, seq_len]  (codebooks 1-7)
        |
  Full tokens [batch, 8, seq_len] -> EnCodec decoder -> waveform
  ```

  ## Two-Stage Generation

  1. **AR Stage**: Given text + audio prompt, autoregressively generate the
     coarse (codebook 0) tokens. This captures the overall prosody and content.

  2. **NAR Stage**: Given text + coarse tokens, predict fine (codebooks 1-7)
     tokens in parallel. Each codebook level is predicted conditioning on
     all previous levels. This adds acoustic detail.

  ## Zero-Shot Capability

  The 3-second audio prompt provides speaker characteristics (timbre, accent,
  speaking style). VALL-E learns to preserve these while generating new content
  from the text. No per-speaker fine-tuning is needed.

  ## Usage

      # Build full VALL-E
      {ar_model, nar_model} = VALLE.build(
        text_vocab_size: 256,
        audio_vocab_size: 1024,
        num_layers: 12,
        hidden_dim: 1024
      )

      # AR forward pass
      coarse_logits = VALLE.ar_forward(ar_fn, params, text_tokens, prompt_tokens)

      # NAR forward pass
      fine_logits = VALLE.nar_forward(nar_fn, params, text_tokens, coarse_tokens, codebook_idx: 1)

  ## References

  - Wang et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    (Microsoft, 2023) — https://arxiv.org/abs/2301.02111
  - VALL-E X (multilingual): https://arxiv.org/abs/2303.03926
  """

  import Nx.Defn

  @default_text_vocab_size 256
  @default_audio_vocab_size 1024
  @default_hidden_dim 1024
  @default_num_layers 12
  @default_num_heads 16
  @default_num_codebooks 8
  @default_dropout 0.1

  @typedoc "Options for `build/1` and related functions."
  @type build_opt ::
          {:text_vocab_size, pos_integer()}
          | {:audio_vocab_size, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_codebooks, pos_integer()}
          | {:dropout, float()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a complete VALL-E model (AR + NAR).

  ## Options

    - `:text_vocab_size` - Text vocabulary size (default: 256 for BPE/phonemes)
    - `:audio_vocab_size` - Audio codebook size (default: 1024, matches EnCodec)
    - `:hidden_dim` - Transformer hidden dimension (default: 1024)
    - `:num_layers` - Number of transformer layers (default: 12)
    - `:num_heads` - Number of attention heads (default: 16)
    - `:num_codebooks` - Number of EnCodec codebooks (default: 8)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns

    A tuple `{ar_model, nar_model}` of Axon models.
    - AR model: autoregressive for coarse token generation
    - NAR model: non-autoregressive for fine token generation
  """
  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    ar_model = build_ar(opts)
    nar_model = build_nar(opts)
    {ar_model, nar_model}
  end

  @doc """
  Build the autoregressive (AR) model for coarse token generation.

  The AR model is a decoder-only transformer that generates codebook 0 tokens
  autoregressively. It conditions on:
  - Text tokens (phonemes or BPE)
  - Audio prompt tokens (3 seconds of reference audio, all codebooks)

  ## Options

    - `:text_vocab_size` - Text vocabulary size (default: 256)
    - `:audio_vocab_size` - Audio codebook size (default: 1024)
    - `:hidden_dim` - Transformer dimension (default: 1024)
    - `:num_layers` - Number of decoder layers (default: 12)
    - `:num_heads` - Number of attention heads (default: 16)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns

    An Axon model with inputs:
    - "text_tokens": `[batch, text_len]`
    - "prompt_tokens": `[batch, num_codebooks, prompt_len]`
    - "audio_tokens": `[batch, audio_len]` (codebook 0 tokens being generated)

    Output: logits `[batch, total_len, audio_vocab_size]`
  """
  @spec build_ar([build_opt()]) :: Axon.t()
  def build_ar(opts \\ []) do
    text_vocab_size = Keyword.get(opts, :text_vocab_size, @default_text_vocab_size)
    audio_vocab_size = Keyword.get(opts, :audio_vocab_size, @default_audio_vocab_size)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_codebooks = Keyword.get(opts, :num_codebooks, @default_num_codebooks)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Inputs
    text_tokens = Axon.input("text_tokens", shape: {nil, nil})
    prompt_tokens = Axon.input("prompt_tokens", shape: {nil, num_codebooks, nil})
    audio_tokens = Axon.input("audio_tokens", shape: {nil, nil})

    # Text embedding
    text_embed =
      Axon.embedding(text_tokens, text_vocab_size, hidden_dim, name: "ar_text_embedding")

    # Audio prompt embedding: sum embeddings from all codebooks
    prompt_embed =
      embed_multi_codebook(
        prompt_tokens,
        audio_vocab_size,
        hidden_dim,
        num_codebooks,
        "ar_prompt"
      )

    # Audio tokens (codebook 0) embedding
    audio_embed =
      Axon.embedding(audio_tokens, audio_vocab_size, hidden_dim, name: "ar_audio_embedding")

    # Concatenate: [text, prompt, audio]
    combined =
      Axon.concatenate([text_embed, prompt_embed, audio_embed],
        axis: 1,
        name: "ar_concat"
      )

    # Add positional embeddings
    combined = add_positional_embedding(combined, hidden_dim, "ar_pos")

    # Transformer decoder layers (causal attention)
    x =
      Enum.reduce(1..num_layers, combined, fn layer_idx, acc ->
        decoder_block(acc, hidden_dim, num_heads, dropout, "ar_layer_#{layer_idx}", causal: true)
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "ar_final_norm")

    # Output projection to audio vocabulary
    Axon.dense(x, audio_vocab_size, name: "ar_output_proj")
  end

  @doc """
  Build the non-autoregressive (NAR) model for fine token generation.

  The NAR model generates codebooks 1-7 given the coarse tokens (codebook 0).
  It uses bidirectional attention since fine tokens are predicted in parallel.

  ## Options

    Same as `build_ar/1`.

  ## Returns

    An Axon model with inputs:
    - "text_tokens": `[batch, text_len]`
    - "coarse_tokens": `[batch, seq_len]` (codebook 0)
    - "prev_codebook_tokens": `[batch, seq_len]` (tokens from codebook i-1)
    - "codebook_idx": scalar indicating which codebook (1-7) to predict

    Output: logits `[batch, seq_len, audio_vocab_size]`
  """
  @spec build_nar([build_opt()]) :: Axon.t()
  def build_nar(opts \\ []) do
    text_vocab_size = Keyword.get(opts, :text_vocab_size, @default_text_vocab_size)
    audio_vocab_size = Keyword.get(opts, :audio_vocab_size, @default_audio_vocab_size)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_codebooks = Keyword.get(opts, :num_codebooks, @default_num_codebooks)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Inputs
    text_tokens = Axon.input("text_tokens", shape: {nil, nil})
    coarse_tokens = Axon.input("coarse_tokens", shape: {nil, nil})
    prev_tokens = Axon.input("prev_codebook_tokens", shape: {nil, nil})
    codebook_idx = Axon.input("codebook_idx", shape: {})

    # Text embedding
    text_embed =
      Axon.embedding(text_tokens, text_vocab_size, hidden_dim, name: "nar_text_embedding")

    # Coarse (codebook 0) embedding
    coarse_embed =
      Axon.embedding(coarse_tokens, audio_vocab_size, hidden_dim, name: "nar_coarse_embedding")

    # Previous codebook embedding (for iterative refinement)
    prev_embed =
      Axon.embedding(prev_tokens, audio_vocab_size, hidden_dim, name: "nar_prev_embedding")

    # Codebook-level embedding (which codebook we're predicting)
    codebook_embed =
      Axon.embedding(codebook_idx, num_codebooks, hidden_dim, name: "nar_codebook_embedding")

    # Broadcast codebook embedding to sequence length
    codebook_broadcast =
      Axon.layer(
        fn coarse, cb_emb, _opts ->
          seq_len = Nx.axis_size(coarse, 1)
          batch = Nx.axis_size(coarse, 0)
          Nx.broadcast(cb_emb, {batch, seq_len, Nx.axis_size(cb_emb, -1)})
        end,
        [coarse_embed, codebook_embed],
        name: "nar_codebook_broadcast"
      )

    # Combine: text + coarse + prev + codebook_level
    # For audio part, we sum coarse + prev + codebook embeddings
    audio_combined =
      Axon.add([coarse_embed, prev_embed, codebook_broadcast], name: "nar_audio_sum")

    # Concatenate text and audio
    combined =
      Axon.concatenate([text_embed, audio_combined], axis: 1, name: "nar_concat")

    # Add positional embeddings
    combined = add_positional_embedding(combined, hidden_dim, "nar_pos")

    # Transformer layers (bidirectional - no causal mask)
    x =
      Enum.reduce(1..num_layers, combined, fn layer_idx, acc ->
        decoder_block(acc, hidden_dim, num_heads, dropout, "nar_layer_#{layer_idx}",
          causal: false
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "nar_final_norm")

    # Output projection
    Axon.dense(x, audio_vocab_size, name: "nar_output_proj")
  end

  # ============================================================================
  # Building Blocks
  # ============================================================================

  # Embed multi-codebook audio (sum embeddings from each codebook)
  # Uses a shared embedding table and sums embeddings from all codebooks
  defp embed_multi_codebook(tokens, vocab_size, hidden_dim, num_codebooks, name) do
    # Create a shared embedding layer
    embedding_param =
      Axon.param("#{name}_embeddings", fn _ -> {vocab_size, hidden_dim} end,
        initializer: :glorot_uniform
      )

    Axon.layer(
      fn tokens_tensor, embeddings, _opts ->
        # tokens: [batch, num_codebooks, seq_len]
        {batch, _num_cb, seq_len} = Nx.shape(tokens_tensor)

        # Sum embeddings from all codebooks
        Enum.reduce(0..(num_codebooks - 1), Nx.broadcast(0.0, {batch, seq_len, hidden_dim}), fn i,
                                                                                                acc ->
          # Get tokens for codebook i: [batch, seq_len]
          cb_tokens = Nx.slice_along_axis(tokens_tensor, i, 1, axis: 1) |> Nx.squeeze(axes: [1])
          # Flatten for indexing
          flat_tokens = Nx.reshape(cb_tokens, {batch * seq_len})
          # Look up embeddings
          emb_flat = Nx.take(embeddings, flat_tokens, axis: 0)
          emb = Nx.reshape(emb_flat, {batch, seq_len, hidden_dim})
          Nx.add(acc, emb)
        end)
      end,
      [tokens, embedding_param],
      name: name,
      op_name: :multi_codebook_embed
    )
  end

  # Add sinusoidal positional embeddings
  defp add_positional_embedding(input, hidden_dim, name) do
    Axon.layer(
      fn x, opts ->
        dim = opts[:hidden_dim]
        seq_len = Nx.axis_size(x, 1)

        # Sinusoidal positional encoding
        pos = Nx.iota({seq_len, 1}, type: :f32)
        dim_idx = Nx.iota({1, dim}, type: :f32)

        # angle_rates = 1 / (10000 ^ (2i / d))
        # Using the formula: exp(-log(10000) * 2i / d)
        angle_rates =
          Nx.exp(
            Nx.multiply(
              Nx.negate(Nx.log(Nx.tensor(10_000.0))),
              Nx.divide(Nx.multiply(Nx.floor(Nx.divide(dim_idx, 2)), 2.0), dim)
            )
          )

        angles = Nx.multiply(pos, angle_rates)

        # Apply sin to even indices (0, 2, 4, ...), cos to odd indices (1, 3, 5, ...)
        even_mask =
          Nx.remainder(dim_idx, 2)
          |> Nx.equal(0)
          |> Nx.broadcast({seq_len, dim})

        pe = Nx.select(even_mask, Nx.sin(angles), Nx.cos(angles))

        Nx.add(x, pe)
      end,
      [input],
      name: name,
      hidden_dim: hidden_dim,
      op_name: :positional_encoding
    )
  end

  # Transformer decoder block
  defp decoder_block(input, hidden_dim, num_heads, dropout, name, opts) do
    causal = Keyword.get(opts, :causal, true)

    # Pre-norm attention
    normed = Axon.layer_norm(input, name: "#{name}_attn_norm")
    attn_out = multi_head_attention(normed, hidden_dim, num_heads, causal, "#{name}_attn")
    attn_out = Axon.dropout(attn_out, rate: dropout, name: "#{name}_attn_dropout")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # Pre-norm FFN
    normed_ffn = Axon.layer_norm(x, name: "#{name}_ffn_norm")
    ffn_out = feed_forward(normed_ffn, hidden_dim, dropout, "#{name}_ffn")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  # Multi-head self-attention
  defp multi_head_attention(input, hidden_dim, num_heads, causal, name) do
    head_dim = div(hidden_dim, num_heads)

    qkv = Axon.dense(input, hidden_dim * 3, name: "#{name}_qkv")

    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          {batch, seq_len, _} = Nx.shape(qkv_tensor)

          query = Nx.slice_along_axis(qkv_tensor, 0, hidden_dim, axis: 2)
          key = Nx.slice_along_axis(qkv_tensor, hidden_dim, hidden_dim, axis: 2)
          value = Nx.slice_along_axis(qkv_tensor, hidden_dim * 2, hidden_dim, axis: 2)

          # Reshape to heads
          query = reshape_to_heads(query, batch, seq_len, num_heads, head_dim)
          key = reshape_to_heads(key, batch, seq_len, num_heads, head_dim)
          value = reshape_to_heads(value, batch, seq_len, num_heads, head_dim)

          # Scaled dot-product attention
          scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(query)))
          scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1])
          scores = Nx.divide(scores, scale)

          # Apply causal mask if needed
          scores =
            if causal do
              mask =
                Edifice.Blocks.CausalMask.causal(seq_len)
                |> Nx.reshape({1, 1, seq_len, seq_len})
                |> Nx.broadcast(Nx.shape(scores))

              Nx.select(
                mask,
                scores,
                Nx.broadcast(Nx.tensor(-1.0e9, type: Nx.type(scores)), Nx.shape(scores))
              )
            else
              scores
            end

          # Softmax
          max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
          exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

          weights =
            Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-9))

          # Apply attention
          output = Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])

          # Reshape back
          output
          |> Nx.transpose(axes: [0, 2, 1, 3])
          |> Nx.reshape({batch, seq_len, num_heads * head_dim})
        end,
        name: "#{name}_compute"
      )

    Axon.dense(attended, hidden_dim, name: "#{name}_out_proj")
  end

  # Feed-forward network (SwiGLU variant)
  defp feed_forward(input, hidden_dim, dropout, name) do
    inner_dim = hidden_dim * 4

    gate = Axon.dense(input, inner_dim, name: "#{name}_gate")
    up = Axon.dense(input, inner_dim, name: "#{name}_up")

    gated =
      Axon.layer(
        fn g, u, _opts -> Nx.multiply(Nx.sigmoid(g), u) end,
        [gate, up],
        name: "#{name}_glu"
      )

    gated
    |> Axon.dropout(rate: dropout, name: "#{name}_dropout")
    |> Axon.dense(hidden_dim, name: "#{name}_down")
  end

  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # ============================================================================
  # Forward Pass Helpers
  # ============================================================================

  @doc """
  AR model forward pass for coarse token generation.

  ## Parameters

    - `ar_fn` - Compiled AR model prediction function
    - `params` - AR model parameters
    - `text_tokens` - Text/phoneme tokens `[batch, text_len]`
    - `prompt_tokens` - Audio prompt tokens `[batch, num_codebooks, prompt_len]`
    - `audio_tokens` - Generated audio tokens so far `[batch, audio_len]`

  ## Returns

    Logits `[batch, total_len, audio_vocab_size]` for next token prediction.
  """
  @spec ar_forward(function(), map(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          Nx.Tensor.t()
  def ar_forward(ar_fn, params, text_tokens, prompt_tokens, audio_tokens) do
    ar_fn.(params, %{
      "text_tokens" => text_tokens,
      "prompt_tokens" => prompt_tokens,
      "audio_tokens" => audio_tokens
    })
  end

  @doc """
  NAR model forward pass for fine token generation.

  ## Parameters

    - `nar_fn` - Compiled NAR model prediction function
    - `params` - NAR model parameters
    - `text_tokens` - Text tokens `[batch, text_len]`
    - `coarse_tokens` - Coarse (codebook 0) tokens `[batch, seq_len]`
    - `prev_tokens` - Previous codebook tokens `[batch, seq_len]`
    - `codebook_idx` - Which codebook to predict (1-7)

  ## Returns

    Logits `[batch, total_len, audio_vocab_size]`.
  """
  @spec nar_forward(function(), map(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), integer()) ::
          Nx.Tensor.t()
  def nar_forward(nar_fn, params, text_tokens, coarse_tokens, prev_tokens, codebook_idx) do
    nar_fn.(params, %{
      "text_tokens" => text_tokens,
      "coarse_tokens" => coarse_tokens,
      "prev_codebook_tokens" => prev_tokens,
      "codebook_idx" => Nx.tensor(codebook_idx)
    })
  end

  # ============================================================================
  # Loss Functions
  # ============================================================================

  @doc """
  Compute VALL-E combined loss (AR + NAR cross-entropy).

  ## Parameters

    - `ar_logits` - AR model logits `[batch, seq_len, vocab_size]`
    - `ar_targets` - Target coarse tokens `[batch, seq_len]`
    - `nar_logits` - NAR model logits `[batch, seq_len, vocab_size]`
    - `nar_targets` - Target fine tokens `[batch, seq_len]`

  ## Options

    - `:ar_weight` - Weight for AR loss (default: 1.0)
    - `:nar_weight` - Weight for NAR loss (default: 1.0)

  ## Returns

    Combined loss scalar.
  """
  @spec valle_loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def valle_loss(ar_logits, ar_targets, nar_logits, nar_targets, opts \\ []) do
    ar_weight = Keyword.get(opts, :ar_weight, 1.0)
    nar_weight = Keyword.get(opts, :nar_weight, 1.0)
    do_valle_loss(ar_logits, ar_targets, nar_logits, nar_targets, ar_weight, nar_weight)
  end

  defnp do_valle_loss(ar_logits, ar_targets, nar_logits, nar_targets, ar_weight, nar_weight) do
    ar_loss = cross_entropy_loss(ar_logits, ar_targets)
    nar_loss = cross_entropy_loss(nar_logits, nar_targets)
    ar_weight * ar_loss + nar_weight * nar_loss
  end

  @doc """
  Compute cross-entropy loss for language modeling.

  ## Parameters

    - `logits` - Predicted logits `[batch, seq_len, vocab_size]`
    - `targets` - Target token IDs `[batch, seq_len]`

  ## Returns

    Cross-entropy loss scalar.
  """
  @spec cross_entropy_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn cross_entropy_loss(logits, targets) do
    {batch, seq_len, vocab_size} = Nx.shape(logits)

    # Flatten for cross-entropy
    logits_flat = Nx.reshape(logits, {batch * seq_len, vocab_size})
    targets_flat = Nx.reshape(targets, {batch * seq_len})

    # Stable softmax + cross-entropy
    max_logits = Nx.reduce_max(logits_flat, axes: [-1], keep_axes: true)
    shifted = logits_flat - max_logits
    log_sum_exp = Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1]))

    # Gather logits for target classes
    target_logits =
      Nx.take_along_axis(
        logits_flat,
        Nx.new_axis(targets_flat, -1),
        axis: -1
      )
      |> Nx.squeeze(axes: [-1])

    # Cross-entropy = -target_logits + log_sum_exp + max
    ce = -target_logits + log_sum_exp + Nx.squeeze(max_logits)

    Nx.mean(ce)
  end

  @doc "Get the output vocabulary size."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :audio_vocab_size, @default_audio_vocab_size)
  end
end
