defmodule Edifice.Audio.SoundStorm do
  @moduledoc """
  SoundStorm: Efficient Parallel Audio Generation via masked prediction.

  SoundStorm generates audio *non-autoregressively* by iteratively refining
  masked neural codec tokens (e.g. EnCodec). Instead of predicting one token
  at a time left-to-right, SoundStorm masks out low-confidence tokens and
  predicts all of them in parallel, repeating for T refinement steps.
  A Conformer backbone operates on the flattened codec token sequence.

  ## Motivation

  Autoregressive audio generation (WaveNet, AudioLM) is slow because each
  token depends on all previous tokens. SoundStorm achieves ~100x speedup
  by predicting entire codebook layers in parallel. The coarse-to-fine
  hierarchy from neural codecs (codebook 0 = coarse, 1-7 = fine detail)
  allows progressive refinement.

  ## Architecture

  ```
  Input: codec tokens [batch, num_codebooks, seq_len]
         (codebook 0 may be provided as conditioning)
        |
  Flatten to [batch, num_codebooks * seq_len, hidden_dim]
        |
  +------------------------------------------+
  | Conformer Backbone (num_layers)          |
  |   - Self-attention (bidirectional)       |
  |   - Convolution module                   |
  |   - FFN with Macaron structure           |
  +------------------------------------------+
        |
  Project to [batch, num_codebooks * seq_len, codebook_size]
        |
  Unflatten to [batch, num_codebooks, seq_len, codebook_size]
        |
  Apply mask: only predict masked positions
        |
  Cosine schedule: mask_ratio decreases over T steps
  ```

  ## Iterative Refinement (Inference)

  1. Start with codebook 0 as conditioning (from text or audio prompt)
  2. Initialize codebooks 1-7 with [MASK] tokens
  3. For step t = 1..T:
     - Forward pass: get logits for all positions
     - Compute confidence (max prob) for masked positions
     - Unmask top-k% most confident predictions (cosine schedule)
  4. Return final tokens

  ## Usage

      model = SoundStorm.build(
        num_codebooks: 8,
        codebook_size: 1024,
        hidden_dim: 512,
        num_layers: 12
      )

      # One refinement step
      new_tokens = SoundStorm.soundstorm_step(model, params, tokens, mask, step: 5, total_steps: 16)

      # Full generation
      final_tokens = SoundStorm.generate(model, params, conditioning_tokens, num_steps: 16)

  ## References

  - Borsos et al., "SoundStorm: Efficient Parallel Audio Generation"
    (Google, 2023) — https://arxiv.org/abs/2305.09636
  - AudioLM: https://arxiv.org/abs/2209.03143
  - EnCodec: https://arxiv.org/abs/2210.13438
  """

  @default_num_codebooks 8
  @default_codebook_size 1024
  @default_hidden_dim 512
  @default_num_layers 12
  @default_num_heads 8
  @default_conv_kernel_size 31
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:num_codebooks, pos_integer()}
          | {:codebook_size, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:conv_kernel_size, pos_integer()}
          | {:dropout, float()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a SoundStorm model.

  ## Options

    - `:num_codebooks` - Number of codec codebooks (default: 8)
    - `:codebook_size` - Vocabulary size per codebook (default: 1024)
    - `:hidden_dim` - Conformer hidden dimension (default: 512)
    - `:num_layers` - Number of Conformer blocks (default: 12)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:conv_kernel_size` - Depthwise conv kernel size (default: 31)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns

    An Axon model that takes flattened codec tokens `[batch, num_codebooks * seq_len]`
    and outputs logits `[batch, num_codebooks * seq_len, codebook_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    num_codebooks = Keyword.get(opts, :num_codebooks, @default_num_codebooks)
    codebook_size = Keyword.get(opts, :codebook_size, @default_codebook_size)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    conv_kernel_size = Keyword.get(opts, :conv_kernel_size, @default_conv_kernel_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Input: flattened codec token IDs [batch, num_codebooks * seq_len]
    # We use nil for seq_len since it varies
    tokens = Axon.input("tokens", shape: {nil, nil})

    # Embed tokens: each codebook shares the same embedding
    # Shape after embedding: [batch, num_codebooks * seq_len, hidden_dim]
    embedded = Axon.embedding(tokens, codebook_size, hidden_dim, name: "token_embedding")

    # Add codebook positional embedding (which codebook 0-7 this position belongs to)
    # We need to know num_codebooks at runtime, so we use a layer
    embedded_with_pos =
      Axon.layer(
        fn emb, _opts ->
          add_codebook_position_embedding(emb, num_codebooks, hidden_dim)
        end,
        [embedded],
        name: "add_codebook_pos",
        op_name: :add_codebook_pos
      )

    # Conformer backbone
    x = conformer_backbone(embedded_with_pos, hidden_dim, num_heads, num_layers, conv_kernel_size, dropout)

    # Project to codebook logits
    Axon.dense(x, codebook_size, name: "output_projection")
  end

  # Add learnable codebook position embeddings
  defp add_codebook_position_embedding(emb, num_codebooks, hidden_dim) do
    {batch, total_len, _dim} = Nx.shape(emb)
    seq_len = div(total_len, num_codebooks)

    # Create codebook indices: [0,0,0,...,1,1,1,...,7,7,7,...] for interleaved
    # Or for flattened by codebook: [0,0,...0, 1,1,...1, ...]
    # SoundStorm uses codebook-major ordering: all of codebook 0, then all of codebook 1, etc.
    codebook_indices =
      Nx.iota({num_codebooks, 1})
      |> Nx.broadcast({num_codebooks, seq_len})
      |> Nx.reshape({total_len})

    # Simple learned embedding per codebook (initialized to zero for simplicity)
    # In a real implementation this would be a trainable parameter
    # Here we just add a scaled positional signal
    codebook_embed =
      codebook_indices
      |> Nx.as_type(:f32)
      |> Nx.divide(num_codebooks)
      |> Nx.new_axis(0)
      |> Nx.new_axis(2)
      |> Nx.broadcast({batch, total_len, hidden_dim})
      |> Nx.multiply(0.1)

    Nx.add(emb, codebook_embed)
  end

  # Simplified Conformer backbone (uses existing Conformer block structure)
  defp conformer_backbone(input, hidden_dim, num_heads, num_layers, conv_kernel_size, dropout) do
    Enum.reduce(1..num_layers, input, fn layer_idx, acc ->
      conformer_block(acc, hidden_dim, num_heads, conv_kernel_size, dropout, "conformer_#{layer_idx}")
    end)
    |> Axon.layer_norm(name: "final_norm")
  end

  # Single Conformer block (Macaron-style)
  defp conformer_block(input, hidden_dim, num_heads, conv_kernel_size, dropout, name) do
    # 1. First half-step FFN
    x = half_ffn(input, hidden_dim, dropout, "#{name}_ffn1")

    # 2. Multi-head self-attention
    x = mhsa_module(x, hidden_dim, num_heads, dropout, "#{name}_mhsa")

    # 3. Convolution module
    x = conv_module(x, hidden_dim, conv_kernel_size, dropout, "#{name}_conv")

    # 4. Second half-step FFN
    x = half_ffn(x, hidden_dim, dropout, "#{name}_ffn2")

    # 5. Final LayerNorm
    Axon.layer_norm(x, name: "#{name}_final_norm")
  end

  # Half-step FFN: norm -> FFN -> scale(0.5) -> residual
  defp half_ffn(input, hidden_dim, dropout, name) do
    inner_size = hidden_dim * 4

    ffn_out =
      input
      |> Axon.layer_norm(name: "#{name}_norm")
      |> Axon.dense(inner_size, name: "#{name}_up")
      |> Axon.activation(:silu, name: "#{name}_act")
      |> Axon.dropout(rate: dropout, name: "#{name}_dropout")
      |> Axon.dense(hidden_dim, name: "#{name}_down")

    scaled = Axon.nx(ffn_out, fn t -> Nx.multiply(t, 0.5) end, name: "#{name}_scale")
    Axon.add(input, scaled, name: "#{name}_residual")
  end

  # Multi-head self-attention module
  defp mhsa_module(input, hidden_dim, num_heads, dropout, name) do
    head_dim = div(hidden_dim, num_heads)
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    qkv = Axon.dense(normed, hidden_dim * 3, name: "#{name}_qkv")

    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          {batch, seq_len, _} = Nx.shape(qkv_tensor)

          query = Nx.slice_along_axis(qkv_tensor, 0, hidden_dim, axis: 2)
          key = Nx.slice_along_axis(qkv_tensor, hidden_dim, hidden_dim, axis: 2)
          value = Nx.slice_along_axis(qkv_tensor, hidden_dim * 2, hidden_dim, axis: 2)

          query = reshape_to_heads(query, batch, seq_len, num_heads, head_dim)
          key = reshape_to_heads(key, batch, seq_len, num_heads, head_dim)
          value = reshape_to_heads(value, batch, seq_len, num_heads, head_dim)

          scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(query)))
          scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1])
          scores = Nx.divide(scores, scale)

          # Bidirectional attention (no causal mask) for masked prediction
          max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
          exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
          weights = Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-9))

          output = Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])

          output
          |> Nx.transpose(axes: [0, 2, 1, 3])
          |> Nx.reshape({batch, seq_len, num_heads * head_dim})
        end,
        name: "#{name}_compute"
      )

    attn_out =
      attended
      |> Axon.dense(hidden_dim, name: "#{name}_out_proj")
      |> Axon.dropout(rate: dropout, name: "#{name}_dropout")

    Axon.add(input, attn_out, name: "#{name}_residual")
  end

  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Convolution module with GLU gating
  defp conv_module(input, hidden_dim, conv_kernel_size, dropout, name) do
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Pointwise expansion (2x for GLU)
    expanded = Axon.dense(normed, hidden_dim * 2, name: "#{name}_pw_up")

    # GLU: split and gate
    gated =
      Axon.nx(
        expanded,
        fn t ->
          a = Nx.slice_along_axis(t, 0, hidden_dim, axis: 2)
          b = Nx.slice_along_axis(t, hidden_dim, hidden_dim, axis: 2)
          Nx.multiply(a, Nx.sigmoid(b))
        end,
        name: "#{name}_glu"
      )

    # Depthwise conv with causal padding
    conv_out =
      Axon.conv(gated, hidden_dim,
        kernel_size: {conv_kernel_size},
        padding: [{conv_kernel_size - 1, 0}],
        feature_group_size: hidden_dim,
        name: "#{name}_dw_conv"
      )

    conv_normed = Axon.layer_norm(conv_out, name: "#{name}_conv_norm")
    activated = Axon.activation(conv_normed, :silu, name: "#{name}_act")

    down =
      activated
      |> Axon.dense(hidden_dim, name: "#{name}_pw_down")
      |> Axon.dropout(rate: dropout, name: "#{name}_conv_dropout")

    Axon.add(input, down, name: "#{name}_residual")
  end

  # ============================================================================
  # Inference Utilities
  # ============================================================================

  @doc """
  Perform one SoundStorm refinement step.

  Given current tokens and a mask indicating which positions to predict,
  runs the model and selectively unmasks the most confident predictions
  according to the cosine schedule.

  ## Parameters

    - `predict_fn` - Compiled prediction function from `Axon.build/2`
    - `params` - Model parameters
    - `tokens` - Current token tensor `[batch, num_codebooks * seq_len]`
    - `mask` - Boolean mask `[batch, num_codebooks * seq_len]`, true = predict
    - `step` - Current refinement step (1-indexed)
    - `total_steps` - Total number of refinement steps

  ## Returns

    Updated tokens tensor with some positions unmasked.
  """
  @spec soundstorm_step(
          (map(), map() -> Nx.Tensor.t()),
          map(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          pos_integer(),
          pos_integer()
        ) :: Nx.Tensor.t()
  def soundstorm_step(predict_fn, params, tokens, mask, step, total_steps) do
    # Get logits from model
    logits = predict_fn.(params, %{"tokens" => tokens})

    # Compute confidence (max probability after softmax)
    probs = softmax(logits)
    confidence = Nx.reduce_max(probs, axes: [-1])

    # Get predicted tokens
    predictions = Nx.argmax(probs, axis: -1)

    # Cosine schedule: mask_ratio decreases from 1.0 to 0.0 over steps
    mask_ratio = cosine_schedule(step, total_steps)

    # Count masked positions and determine how many to unmask
    {batch, seq_len} = Nx.shape(mask)
    mask_f32 = Nx.as_type(mask, :f32)
    num_masked = Nx.sum(mask_f32, axes: [-1], keep_axes: true)
    num_to_keep_masked = Nx.multiply(num_masked, mask_ratio) |> Nx.floor() |> Nx.as_type(:s64)

    # For each batch element, keep only the lowest-confidence positions masked
    # This is done by sorting confidences and thresholding
    new_tokens = unmask_confident(tokens, predictions, mask, confidence, num_to_keep_masked)

    new_tokens
  end

  @doc """
  Full SoundStorm generation loop.

  Starting from conditioning tokens (codebook 0), generates codebooks 1-7
  through iterative refinement.

  ## Parameters

    - `predict_fn` - Compiled prediction function
    - `params` - Model parameters
    - `conditioning_tokens` - Codebook 0 tokens `[batch, seq_len]`

  ## Options

    - `:num_steps` - Number of refinement steps (default: 16)
    - `:num_codebooks` - Number of codebooks (default: 8)
    - `:mask_token` - Token ID used for masking (default: 0)

  ## Returns

    Generated tokens `[batch, num_codebooks, seq_len]`.
  """
  @spec generate(
          (map(), map() -> Nx.Tensor.t()),
          map(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  def generate(predict_fn, params, conditioning_tokens, opts \\ []) do
    num_steps = Keyword.get(opts, :num_steps, 16)
    num_codebooks = Keyword.get(opts, :num_codebooks, @default_num_codebooks)
    mask_token = Keyword.get(opts, :mask_token, 0)

    {batch, seq_len} = Nx.shape(conditioning_tokens)

    # Initialize: codebook 0 = conditioning, codebooks 1-7 = mask_token
    initial_tokens =
      Nx.concatenate([
        conditioning_tokens,
        Nx.broadcast(Nx.tensor(mask_token), {batch, (num_codebooks - 1) * seq_len})
      ], axis: -1)

    # Initial mask: codebook 0 = false (keep), codebooks 1-7 = true (predict)
    initial_mask =
      Nx.concatenate([
        Nx.broadcast(Nx.tensor(0, type: :u8), {batch, seq_len}),
        Nx.broadcast(Nx.tensor(1, type: :u8), {batch, (num_codebooks - 1) * seq_len})
      ], axis: -1)
      |> Nx.as_type(:u8)
      |> Nx.equal(1)

    # Iterative refinement
    {final_tokens, _final_mask} =
      Enum.reduce(1..num_steps, {initial_tokens, initial_mask}, fn step, {tokens, mask} ->
        new_tokens = soundstorm_step(predict_fn, params, tokens, mask, step, num_steps)

        # Update mask: positions that were just unmasked are now false
        new_mask = Nx.logical_and(mask, Nx.equal(new_tokens, tokens))

        {new_tokens, new_mask}
      end)

    # Reshape to [batch, num_codebooks, seq_len]
    Nx.reshape(final_tokens, {batch, num_codebooks, seq_len})
  end

  @doc "Get output size (codebook_size for per-position predictions)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :codebook_size, @default_codebook_size)
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp softmax(logits) do
    max_logits = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    exp_logits = Nx.exp(Nx.subtract(logits, max_logits))
    Nx.divide(exp_logits, Nx.sum(exp_logits, axes: [-1], keep_axes: true))
  end

  # Cosine schedule: returns mask ratio (1.0 at step 1, 0.0 at step total_steps)
  defp cosine_schedule(step, total_steps) do
    # ratio = cos(pi * step / (2 * total_steps))
    progress = step / total_steps
    :math.cos(:math.pi() * progress / 2)
  end

  # Unmask the most confident predictions while keeping num_to_keep_masked masked
  defp unmask_confident(tokens, predictions, mask, confidence, num_to_keep_masked) do
    {batch, seq_len} = Nx.shape(tokens)

    # For simplicity, process each batch element
    # In production this would be vectorized
    Enum.reduce(0..(batch - 1), tokens, fn b, acc ->
      tokens_b = Nx.slice(acc, [b, 0], [1, seq_len]) |> Nx.squeeze(axes: [0])
      preds_b = Nx.slice(predictions, [b, 0], [1, seq_len]) |> Nx.squeeze(axes: [0])
      mask_b = Nx.slice(mask, [b, 0], [1, seq_len]) |> Nx.squeeze(axes: [0])
      conf_b = Nx.slice(confidence, [b, 0], [1, seq_len]) |> Nx.squeeze(axes: [0])
      keep_masked = num_to_keep_masked |> Nx.slice([b, 0], [1, 1]) |> Nx.squeeze() |> Nx.to_number()

      # Get masked indices sorted by confidence (ascending = least confident first)
      masked_conf = Nx.select(mask_b, conf_b, Nx.tensor(2.0))
      sorted_indices = Nx.argsort(masked_conf, direction: :asc)

      # The first `keep_masked` indices stay masked, rest get unmasked
      unmask_count = Nx.axis_size(mask_b, 0) - trunc(keep_masked)

      # Create unmask indicator
      position_in_sort = Nx.argsort(sorted_indices, direction: :asc)
      should_unmask = Nx.greater_equal(position_in_sort, trunc(keep_masked))
      should_unmask = Nx.logical_and(should_unmask, mask_b)

      # Update tokens where unmasking
      new_tokens_b = Nx.select(should_unmask, preds_b, tokens_b)

      # Put back into batch
      Nx.put_slice(acc, [b, 0], Nx.new_axis(new_tokens_b, 0))
    end)
  end
end
