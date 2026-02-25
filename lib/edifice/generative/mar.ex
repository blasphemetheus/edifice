defmodule Edifice.Generative.MAR do
  @moduledoc """
  MAR: Masked Autoregressive Generation.

  MAR bridges autoregressive (AR) models and masked prediction models for
  iterative discrete token generation. Unlike pure AR models that generate
  tokens left-to-right, MAR predicts all masked positions in parallel and
  progressively unmasks tokens from most to least confident over K steps.

  ## Architecture

  ```
  Token Indices [batch, seq_len]
        |
        v
  Token Embedding + Sinusoidal Positional Encoding
        |
        v
  +--------------------------------------+
  | Transformer Encoder (bidirectional)  |
  |   N × (LayerNorm + MHA + FFN)        |
  |   No causal mask — all positions see |
  |   each other, conditioned on unmasked|
  +--------------------------------------+
        |
        v
  LayerNorm → Dense → Logits [batch, seq_len, vocab_size]
  ```

  ## Training

  At each training step:
  1. Sample mask ratio r ~ cosine schedule (biased toward moderate r)
  2. Randomly mask r fraction of tokens with [MASK] id
  3. Forward pass → logits for all positions
  4. Compute cross-entropy only on masked positions

  ```
  L = -1/|M| Σ_{i ∈ M} log p(y_i | context)
  ```

  ## Inference (Iterative Decoding)

  Starts fully masked; unmasking is driven by model confidence:

  1. Initialise: all tokens = [MASK]
  2. For step k = 1..K:
     a. Forward pass — predict logits at all masked positions
     b. Compute confidence = max softmax probability per masked token
     c. Unmask the `n_k` tokens with highest confidence
        (`n_k` increases each step so all are revealed by step K)
  3. Return the fully unmasked sequence

  ## Usage

      model = MAR.build(
        vocab_size: 8192,
        embed_dim: 256,
        num_layers: 6,
        num_heads: 8,
        seq_len: 256
      )

      # Training
      loss = MAR.mar_loss(logits, targets, mask)

      # Inference
      tokens = MAR.iterative_decode(model, params, seq_len: 256, vocab_size: 8192)

  ## References

  - Li et al., "Autoregressive Image Generation without Vector Quantization" (2024)
  - https://arxiv.org/abs/2406.11838
  """

  @mask_token_id 0
  @default_embed_dim 256
  @default_num_layers 6
  @default_num_heads 8
  @default_seq_len 256
  @default_dropout 0.1

  @doc """
  Build a MAR transformer model.

  ## Options

    - `:vocab_size` - Vocabulary size, including [MASK] token at id 0 (required)
    - `:embed_dim` - Embedding / model hidden dimension (default: 256)
    - `:num_layers` - Number of bidirectional encoder layers (default: 6)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:seq_len` - Sequence length for positional encoding (default: 256)
    - `:dropout` - Dropout rate applied after embedding and in each block (default: 0.1)

  ## Returns

    An Axon model taking token indices `[batch, seq_len]` (integer) and
    returning logits `[batch, seq_len, vocab_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:vocab_size, pos_integer()}
          | {:embed_dim, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:dropout, float()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    seq_len = Keyword.get(opts, :seq_len, @default_seq_len)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Input: token indices [batch, seq_len]
    input = Axon.input("tokens", shape: {nil, nil})

    # Token embedding: indices → [batch, seq_len, embed_dim]
    x = embed_tokens(input, vocab_size, embed_dim)

    # Add sinusoidal positional encoding
    x = add_positional_encoding(x, embed_dim, seq_len)

    # Optional dropout after embedding
    x = if dropout > 0, do: Axon.dropout(x, rate: dropout, name: "embed_dropout"), else: x

    # Bidirectional (non-causal) transformer encoder blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        encoder_block(acc,
          embed_dim: embed_dim,
          num_heads: num_heads,
          dropout: dropout,
          name: "layer_#{layer_idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Output projection → logits [batch, seq_len, vocab_size]
    Axon.dense(x, vocab_size, name: "lm_head")
  end

  @doc """
  MAR training loss: cross-entropy over masked positions only.

  ## Parameters

    - `logits` - Model output `[batch, seq_len, vocab_size]`
    - `targets` - Ground-truth token indices `[batch, seq_len]`
    - `mask` - Binary mask `[batch, seq_len]`; 1 = masked (predict), 0 = unmasked

  ## Returns

    Scalar loss (mean CE over masked tokens).
  """
  @spec mar_loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def mar_loss(logits, targets, mask) do
    batch_size = Nx.axis_size(logits, 0)
    seq_len = Nx.axis_size(logits, 1)
    vocab_size = Nx.axis_size(logits, 2)

    # Numerically-stable log-softmax [batch, seq_len, vocab_size]
    log_probs = log_softmax_last(logits)

    # Flatten to [batch*seq_len, vocab_size] and [batch*seq_len]
    lp_flat = Nx.reshape(log_probs, {batch_size * seq_len, vocab_size})
    tgt_flat = Nx.reshape(targets, {batch_size * seq_len})

    # Gather log-prob for each target token → [batch*seq_len]
    target_lp = gather_log_probs(lp_flat, tgt_flat)
    target_lp = Nx.reshape(target_lp, {batch_size, seq_len})

    # CE = -log_p; apply mask and average over masked positions
    ce = Nx.negate(target_lp)
    masked_ce = Nx.multiply(ce, Nx.as_type(mask, :f32))
    num_masked = Nx.add(Nx.sum(Nx.as_type(mask, :f32)), 1.0e-8)
    Nx.divide(Nx.sum(masked_ce), num_masked)
  end

  @doc """
  Sample a random masking ratio from the cosine schedule used in MAR training.

  Draws `u ~ Uniform[0,1]` and returns `1 - cos(π·u/2)`, which biases
  sampling toward moderate mask fractions (away from 0 and 1).

  ## Returns

    Float in `[0.0, 1.0]`.
  """
  @spec sample_mask_ratio() :: float()
  def sample_mask_ratio do
    u = :rand.uniform()
    1.0 - :math.cos(:math.pi() * u / 2.0)
  end

  @doc """
  Iterative masked decoding for MAR inference.

  Starts with all tokens masked, then unmaskes the most-confident tokens
  first over `num_steps` iterations.

  ## Parameters

    - `model` - Axon model from `build/1`
    - `params` - Initialised model state (from `Axon.init/2`)

  ## Options

    - `:num_steps` - Number of decoding iterations K (default: 8)
    - `:seq_len` - Sequence length (required)
    - `:vocab_size` - Vocabulary size (required)
    - `:mask_token_id` - ID used for [MASK] tokens (default: 0)
    - `:temperature` - Softmax temperature (default: 1.0)

  ## Returns

    Token indices tensor `[1, seq_len]`.
  """
  @spec iterative_decode(Axon.t(), term(), keyword()) :: Nx.Tensor.t()
  def iterative_decode(model, params, opts \\ []) do
    num_steps = Keyword.get(opts, :num_steps, 8)
    seq_len = Keyword.fetch!(opts, :seq_len)
    mask_token_id = Keyword.get(opts, :mask_token_id, @mask_token_id)
    temperature = Keyword.get(opts, :temperature, 1.0)

    {_init_fn, predict_fn} = Axon.build(model, mode: :inference)

    # Start fully masked: [1, seq_len]
    tokens = Nx.broadcast(mask_token_id, {1, seq_len}) |> Nx.as_type(:s64)
    # is_masked[i] = 1 if position i is still masked
    is_masked = Nx.broadcast(1, {1, seq_len}) |> Nx.as_type(:u8)

    {final_tokens, _} =
      Enum.reduce(1..num_steps, {tokens, is_masked}, fn step, {toks, masked} ->
        # Forward pass → logits [1, seq_len, vocab_size]
        logits = predict_fn.(params, %{"tokens" => toks})

        # Confidence = max softmax probability per position [seq_len]
        scaled = if temperature != 1.0, do: Nx.divide(logits, temperature), else: logits
        probs = softmax_last(scaled)
        confidence = probs |> Nx.reduce_max(axes: [2]) |> Nx.squeeze(axes: [0])

        # Argmax predictions at each position [seq_len]
        predicted = logits |> Nx.argmax(axis: 2) |> Nx.squeeze(axes: [0]) |> Nx.as_type(:s64)

        # How many tokens should remain masked after this step
        target_masked = round((1.0 - step / num_steps) * seq_len)

        # Compute updated mask: keep target_masked least-confident masked positions
        masked_flat = Nx.squeeze(masked, axes: [0])
        new_masked_flat = keep_least_confident(confidence, masked_flat, target_masked)

        # Unmask newly decided positions: update tokens for just-unmasked positions
        toks_flat = Nx.squeeze(toks, axes: [0])
        newly_unmasked = Nx.logical_and(masked_flat, Nx.logical_not(new_masked_flat))
        new_toks_flat = Nx.select(newly_unmasked, predicted, toks_flat)

        {Nx.new_axis(new_toks_flat, 0), Nx.new_axis(new_masked_flat, 0)}
      end)

    final_tokens
  end

  # ============================================================================
  # Model-building helpers
  # ============================================================================

  defp embed_tokens(input, vocab_size, embed_dim) do
    Axon.layer(
      &embed_tokens_impl/2,
      [input],
      name: "token_embed",
      vocab_size: vocab_size,
      embed_dim: embed_dim,
      op_name: :mar_token_embed
    )
  end

  defp embed_tokens_impl(indices, opts) do
    vocab_size = opts[:vocab_size]
    embed_dim = opts[:embed_dim]

    # One-hot lookup: [batch, seq_len] → [batch, seq_len, vocab_size] → embed
    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(indices, :s64), -1),
        Nx.iota({1, 1, vocab_size})
      )
      |> Nx.as_type(:f32)

    # Deterministic scaled projection (acts as a fixed embedding table here)
    scale = :math.sqrt(embed_dim)
    proj = Nx.divide(Nx.iota({vocab_size, embed_dim}, type: :f32), scale)
    Nx.dot(one_hot, proj)
  end

  defp add_positional_encoding(x, embed_dim, max_len) do
    Axon.layer(
      &positional_encoding_impl/2,
      [x],
      name: "pos_enc",
      embed_dim: embed_dim,
      max_len: max_len,
      op_name: :mar_pos_enc
    )
  end

  defp positional_encoding_impl(x, opts) do
    embed_dim = opts[:embed_dim]
    max_len = opts[:max_len]

    seq_len = min(Nx.axis_size(x, 1), max_len)
    half = div(embed_dim, 2)

    positions = Nx.iota({seq_len}, type: :f32)

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half}, type: :f32), max(half - 1, 1))
        )
      )

    angles = Nx.outer(positions, freqs)
    pe = Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: 1)
    pe = Nx.reshape(pe, {1, seq_len, embed_dim})

    Nx.add(x, pe)
  end

  defp encoder_block(input, opts) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    num_heads = Keyword.fetch!(opts, :num_heads)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.fetch!(opts, :name)
    mlp_dim = embed_dim * 4

    # Pre-norm → bidirectional multi-head self-attention → residual
    x_norm = Axon.layer_norm(input, name: "#{name}_attn_norm")
    attn_out = build_mha(x_norm, embed_dim, num_heads, "#{name}_attn")

    attn_out =
      if dropout > 0,
        do: Axon.dropout(attn_out, rate: dropout, name: "#{name}_attn_drop"),
        else: attn_out

    x = Axon.add(input, attn_out, name: "#{name}_attn_res")

    # Pre-norm → FFN → residual
    x_norm2 = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      x_norm2
      |> Axon.dense(mlp_dim, name: "#{name}_ffn_up")
      |> Axon.activation(:gelu, name: "#{name}_ffn_act")
      |> Axon.dense(embed_dim, name: "#{name}_ffn_down")

    ffn_out =
      if dropout > 0,
        do: Axon.dropout(ffn_out, rate: dropout, name: "#{name}_ffn_drop"),
        else: ffn_out

    Axon.add(x, ffn_out, name: "#{name}_ffn_res")
  end

  defp build_mha(input, embed_dim, num_heads, name) do
    head_dim = div(embed_dim, num_heads)

    q = Axon.dense(input, embed_dim, name: "#{name}_q")
    k = Axon.dense(input, embed_dim, name: "#{name}_k")
    v = Axon.dense(input, embed_dim, name: "#{name}_v")

    attn =
      Axon.layer(
        &bidirectional_attn_impl/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :mar_bidirectional_attn
      )

    Axon.dense(attn, embed_dim, name: "#{name}_out")
  end

  defp bidirectional_attn_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape for multi-head: [batch, seq, heads, head_dim] → [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    # Scores [batch, heads, seq, seq]
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Bidirectional: no causal mask
    max_s = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_s = Nx.exp(Nx.subtract(scores, max_s))
    weights = Nx.divide(exp_s, Nx.add(Nx.sum(exp_s, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # Numerical helpers (used both in training loss and iterative decode)
  # ============================================================================

  defp log_softmax_last(x) do
    max_x = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(x, max_x)
    log_sum = Nx.log(Nx.sum(Nx.exp(shifted), axes: [-1], keep_axes: true))
    Nx.subtract(shifted, log_sum)
  end

  defp softmax_last(x) do
    max_x = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(x, max_x)
    exp_x = Nx.exp(shifted)
    Nx.divide(exp_x, Nx.sum(exp_x, axes: [-1], keep_axes: true))
  end

  defp gather_log_probs(log_probs_flat, targets_flat) do
    vocab_size = Nx.axis_size(log_probs_flat, 1)

    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(targets_flat, :s64), 1),
        Nx.iota({1, vocab_size})
      )
      |> Nx.as_type(:f32)

    Nx.sum(Nx.multiply(log_probs_flat, one_hot), axes: [1])
  end

  # Keep the `k` currently-masked positions with lowest confidence still masked.
  # Positions with highest confidence get unmasked (they're safe to commit).
  defp keep_least_confident(confidence, is_masked, k) do
    seq_len = Nx.axis_size(confidence, 0)

    if k <= 0 do
      Nx.broadcast(0, {seq_len}) |> Nx.as_type(:u8)
    else
      conf_list = Nx.to_flat_list(confidence)
      mask_list = Nx.to_flat_list(is_masked)

      # Only consider currently-masked positions; sort by confidence ascending
      indexed_masked =
        conf_list
        |> Enum.with_index()
        |> Enum.filter(fn {_c, i} -> Enum.at(mask_list, i) == 1 end)
        |> Enum.sort_by(fn {c, _i} -> c end)

      # Keep the k least-confident positions masked
      keep_count = min(k, length(indexed_masked))

      keep_set =
        indexed_masked |> Enum.take(keep_count) |> Enum.map(fn {_, i} -> i end) |> MapSet.new()

      Nx.iota({seq_len})
      |> Nx.to_flat_list()
      |> Enum.map(fn i -> if MapSet.member?(keep_set, i), do: 1, else: 0 end)
      |> Nx.tensor(type: :u8)
    end
  end
end
