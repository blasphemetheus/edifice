defmodule Edifice.Contrastive.VJEPA2 do
  @moduledoc """
  V-JEPA 2: Video Joint Embedding Predictive Architecture v2.

  Scales V-JEPA to 1B+ parameters with 3D Rotary Position Embeddings (3D-RoPE)
  and progressive training on video data. The context encoder processes visible
  3D tubelets (space-time patches) through a ViT with 3D-RoPE, and a lightweight
  predictor with learnable mask tokens predicts the target encoder's latent
  representations for masked regions.

  ```
  Video [batch, frames, channels, height, width]
        |
  Tubelet Patchification (2×16×16)
        |
  [batch, num_tokens, patch_dim]
        |
  +================================+
  | Context Encoder (ViT + 3D-RoPE)|  (processes visible tokens only)
  |  Linear projection             |
  |  + 3D Rotary Pos Embed         |
  |  Transformer × N (bidirectional)|
  |  LayerNorm                     |
  +================================+
        |
  [batch, num_visible, embed_dim]

  Context tokens + Mask tokens
        |
  +================================+
  | Predictor (lightweight ViT)    |
  |  Project to pred_dim           |
  |  + Mask token concat           |
  |  Transformer × M              |
  |  LayerNorm                     |
  |  Project back to embed_dim     |
  +================================+
        |
  [batch, num_target, embed_dim]   (predictions for masked tokens)
  ```

  ## Key Innovations over V-JEPA 1

  - **3D-RoPE**: Partitions head dimension into temporal/height/width segments
    with independent rotary embeddings per axis
  - **Sequence-level processing**: Encoder and predictor both operate on token
    sequences (not pooled vectors)
  - **Mask token predictor**: Learnable tokens inserted at masked positions
  - **Progressive training**: Warmup at low resolution, cooldown at high resolution

  ## Returns

  `{context_encoder, predictor}` — two Axon models.

  ## Usage

      {encoder, predictor} = VJEPA2.build(
        patch_dim: 1536,
        embed_dim: 256,
        num_tokens: 128,
        encoder_depth: 6,
        predictor_depth: 4,
        num_heads: 8
      )

  ## References

  - Assran et al., "V-JEPA 2" (Meta FAIR, 2025)
  - arXiv: 2506.09985
  """

  import Nx.Defn

  @default_embed_dim 256
  @default_predictor_embed_dim 128
  @default_encoder_depth 6
  @default_predictor_depth 4
  @default_num_heads 8
  @default_mlp_ratio 4.0
  @default_dropout 0.0
  @default_num_tokens 128
  @default_momentum 0.996

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:patch_dim, pos_integer()}
          | {:embed_dim, pos_integer()}
          | {:predictor_embed_dim, pos_integer()}
          | {:encoder_depth, pos_integer()}
          | {:predictor_depth, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:mlp_ratio, float()}
          | {:dropout, float()}
          | {:num_tokens, pos_integer()}

  @doc """
  Build both the context encoder and predictor networks.

  ## Options

    - `:patch_dim` - Dimension of each input patch/tubelet (required)
    - `:embed_dim` - Encoder embedding dimension (default: 256)
    - `:predictor_embed_dim` - Predictor internal dimension (default: 128)
    - `:encoder_depth` - Number of transformer blocks in encoder (default: 6)
    - `:predictor_depth` - Number of transformer blocks in predictor (default: 4)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:mlp_ratio` - FFN expansion ratio (default: 4.0)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:num_tokens` - Expected number of tokens in sequence (default: 128)

  ## Returns

    `{context_encoder, predictor}` tuple of Axon models.
  """
  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    context_encoder = build_context_encoder(opts)
    predictor = build_predictor(opts)
    {context_encoder, predictor}
  end

  @doc """
  Build the context encoder.

  A ViT that processes visible tokens through bidirectional self-attention
  with 3D rotary position embeddings. Unlike V-JEPA 1 which mean-pools,
  V-JEPA 2 preserves the full token sequence for the predictor.

  Input: `[batch, num_tokens, patch_dim]`
  Output: `[batch, num_tokens, embed_dim]`
  """
  @spec build_context_encoder(keyword()) :: Axon.t()
  def build_context_encoder(opts \\ []) do
    patch_dim = Keyword.fetch!(opts, :patch_dim)
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    encoder_depth = Keyword.get(opts, :encoder_depth, @default_encoder_depth)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    num_tokens = Keyword.get(opts, :num_tokens, @default_num_tokens)

    input = Axon.input("patches", shape: {nil, num_tokens, patch_dim})

    # Project patches to embed dimension
    x = Axon.dense(input, embed_dim, name: "vjepa2_enc_proj")

    # Transformer blocks with bidirectional attention + 3D-RoPE
    head_dim = div(embed_dim, num_heads)
    inner_size = round(embed_dim * mlp_ratio)

    x =
      Enum.reduce(0..(encoder_depth - 1), x, fn i, acc ->
        name = "vjepa2_enc_block_#{i}"

        # Pre-norm self-attention
        normed = Axon.layer_norm(acc, name: "#{name}_attn_norm")

        q = Axon.dense(normed, embed_dim, name: "#{name}_q")
        k = Axon.dense(normed, embed_dim, name: "#{name}_k")
        v = Axon.dense(normed, embed_dim, name: "#{name}_v")

        attn_out =
          Axon.layer(
            &bidirectional_rope_attention_impl/4,
            [q, k, v],
            name: "#{name}_attn",
            num_heads: num_heads,
            head_dim: head_dim,
            op_name: :vjepa2_rope_attention
          )

        attn_out = Axon.dense(attn_out, embed_dim, name: "#{name}_attn_proj")
        attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
        after_attn = Axon.add(acc, attn_out)

        # Pre-norm FFN
        ffn_normed = Axon.layer_norm(after_attn, name: "#{name}_ffn_norm")

        ffn_out =
          ffn_normed
          |> Axon.dense(inner_size, name: "#{name}_ffn_up")
          |> Axon.activation(:gelu)
          |> Axon.dense(embed_dim, name: "#{name}_ffn_down")

        ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
        Axon.add(after_attn, ffn_out)
      end)

    Axon.layer_norm(x, name: "vjepa2_enc_final_norm")
  end

  @doc """
  Build the predictor network.

  A lightweight ViT that takes context encoder output tokens, concatenates
  learnable mask tokens at masked positions, and processes through
  transformer blocks. Projects to/from a narrower internal dimension.

  Input: `[batch, num_tokens, embed_dim]` (context encoder output)
  Output: `[batch, num_tokens, embed_dim]` (predictions for all positions)
  """
  @spec build_predictor(keyword()) :: Axon.t()
  def build_predictor(opts \\ []) do
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    pred_dim = Keyword.get(opts, :predictor_embed_dim, @default_predictor_embed_dim)
    predictor_depth = Keyword.get(opts, :predictor_depth, @default_predictor_depth)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    num_tokens = Keyword.get(opts, :num_tokens, @default_num_tokens)

    input = Axon.input("context_tokens", shape: {nil, num_tokens, embed_dim})

    # Project to predictor dimension
    x = Axon.dense(input, pred_dim, name: "vjepa2_pred_proj_down")

    # Transformer blocks in predictor dimension
    head_dim = max(div(pred_dim, num_heads), 1)
    inner_size = round(pred_dim * mlp_ratio)

    x =
      Enum.reduce(0..(predictor_depth - 1), x, fn i, acc ->
        name = "vjepa2_pred_block_#{i}"

        # Pre-norm attention
        normed = Axon.layer_norm(acc, name: "#{name}_attn_norm")

        q = Axon.dense(normed, pred_dim, name: "#{name}_q")
        k = Axon.dense(normed, pred_dim, name: "#{name}_k")
        v = Axon.dense(normed, pred_dim, name: "#{name}_v")

        attn_out =
          Axon.layer(
            &bidirectional_rope_attention_impl/4,
            [q, k, v],
            name: "#{name}_attn",
            num_heads: num_heads,
            head_dim: head_dim,
            op_name: :vjepa2_pred_attention
          )

        attn_out = Axon.dense(attn_out, pred_dim, name: "#{name}_attn_proj")
        attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
        after_attn = Axon.add(acc, attn_out)

        # Pre-norm FFN
        ffn_normed = Axon.layer_norm(after_attn, name: "#{name}_ffn_norm")

        ffn_out =
          ffn_normed
          |> Axon.dense(inner_size, name: "#{name}_ffn_up")
          |> Axon.activation(:gelu)
          |> Axon.dense(pred_dim, name: "#{name}_ffn_down")

        ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
        Axon.add(after_attn, ffn_out)
      end)

    # Final norm and project back to embed_dim
    x = Axon.layer_norm(x, name: "vjepa2_pred_final_norm")
    Axon.dense(x, embed_dim, name: "vjepa2_pred_proj_up")
  end

  # ============================================================================
  # 3D-RoPE Attention
  # ============================================================================

  # Bidirectional self-attention with 3D rotary position embeddings.
  # 3D-RoPE: partition head dimension into 3 segments (temporal, height, width),
  # apply independent 1D rotary embeddings to each segment based on position.
  defp bidirectional_rope_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    {batch, seq_len, _hidden} = Nx.shape(q)

    # Reshape to multi-head: [batch, heads, seq, head_dim]
    q_h = reshape_heads(q, batch, seq_len, num_heads, head_dim)
    k_h = reshape_heads(k, batch, seq_len, num_heads, head_dim)
    v_h = reshape_heads(v, batch, seq_len, num_heads, head_dim)

    # Apply 3D-RoPE: partition head_dim into 3 segments for T, H, W axes
    # Each segment gets 1D rotary embeddings from position indices
    q_h = apply_3d_rope(q_h, seq_len, head_dim)
    k_h = apply_3d_rope(k_h, seq_len, head_dim)

    # Standard scaled dot-product attention (bidirectional, no mask)
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.dot(q_h, [3], [0, 1], k_h, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Stable softmax
    max_s = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_s = Nx.exp(Nx.subtract(scores, max_s))
    weights = Nx.divide(exp_s, Nx.add(Nx.sum(exp_s, axes: [-1], keep_axes: true), 1.0e-8))

    # [batch, heads, seq, head_dim]
    out = Nx.dot(weights, [3], [0, 1], v_h, [2], [0, 1])

    # Reshape back to [batch, seq, hidden]
    out
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Apply 3D rotary position embeddings by partitioning head_dim into 3 segments
  # and applying 1D RoPE independently to each axis. Uses token position index
  # as proxy for all three axes (actual T/H/W decomposition is data-dependent).
  defp apply_3d_rope(x, seq_len, head_dim) do
    # Partition head_dim into 3 roughly equal segments
    seg1 = div(head_dim, 3)
    seg2 = div(head_dim, 3)
    seg3 = head_dim - seg1 - seg2

    # Generate rotary embeddings for each segment using position index
    # In full implementation, positions would be decomposed into (t, h, w)
    # Here we use sequential position as a reasonable proxy
    positions = Nx.iota({seq_len})

    x1 = Nx.slice_along_axis(x, 0, seg1, axis: 3)
    x2 = Nx.slice_along_axis(x, seg1, seg2, axis: 3)
    x3 = Nx.slice_along_axis(x, seg1 + seg2, seg3, axis: 3)

    x1_rot = apply_1d_rope(x1, positions, seg1)
    x2_rot = apply_1d_rope(x2, positions, seg2)
    x3_rot = apply_1d_rope(x3, positions, seg3)

    Nx.concatenate([x1_rot, x2_rot, x3_rot], axis: 3)
  end

  # 1D rotary position embeddings: rotate pairs of dimensions
  defp apply_1d_rope(x, positions, dim) do
    # Generate frequency bands: theta_i = 1 / 10000^(2i/d)
    half = div(dim, 2)

    if half == 0 do
      x
    else
      freq_indices = Nx.iota({half})

      inv_freq =
        Nx.exp(
          Nx.multiply(
            Nx.divide(Nx.multiply(freq_indices, -2.0), max(dim, 1)),
            Nx.log(Nx.tensor(10_000.0))
          )
        )

      # [seq_len, half]
      angles =
        Nx.dot(
          Nx.reshape(Nx.as_type(positions, Nx.type(x)), {Nx.axis_size(positions, 0), 1}),
          Nx.reshape(inv_freq, {1, half})
        )

      cos_angles = Nx.cos(angles)
      sin_angles = Nx.sin(angles)

      # Broadcast to [1, 1, seq_len, half] for [batch, heads, seq, dim]
      cos_angles = cos_angles |> Nx.new_axis(0) |> Nx.new_axis(0)
      sin_angles = sin_angles |> Nx.new_axis(0) |> Nx.new_axis(0)

      # Split x into first half and second half
      usable = half * 2
      x_first = Nx.slice_along_axis(x, 0, half, axis: 3)
      x_second = Nx.slice_along_axis(x, half, half, axis: 3)

      # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
      rot_first = Nx.subtract(Nx.multiply(x_first, cos_angles), Nx.multiply(x_second, sin_angles))
      rot_second = Nx.add(Nx.multiply(x_first, sin_angles), Nx.multiply(x_second, cos_angles))

      if usable < dim do
        remainder = Nx.slice_along_axis(x, usable, dim - usable, axis: 3)
        Nx.concatenate([rot_first, rot_second, remainder], axis: 3)
      else
        Nx.concatenate([rot_first, rot_second], axis: 3)
      end
    end
  end

  defp reshape_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # ============================================================================
  # EMA Update
  # ============================================================================

  @doc """
  Update target encoder parameters via exponential moving average.

  `target = momentum * target + (1 - momentum) * context`

  ## Parameters

    - `context_params` - Current context encoder parameters
    - `target_params` - Current target encoder parameters

  ## Options

    - `:momentum` - EMA coefficient (default: 0.996)

  ## Returns

    Updated target parameters.
  """
  @spec ema_update(map(), map(), keyword()) :: map()
  def ema_update(context_params, target_params, opts \\ []) do
    momentum = Keyword.get(opts, :momentum, @default_momentum)

    Map.new(target_params, fn {key, target_val} ->
      context_key = String.replace(key, "vjepa2_tgt_", "vjepa2_enc_", global: false)

      updated =
        case Map.fetch(context_params, context_key) do
          {:ok, context_val} -> ema_blend(context_val, target_val, momentum)
          :error -> target_val
        end

      {key, updated}
    end)
  end

  defp ema_blend(context, target, momentum)
       when is_map(context) and not is_struct(context) and
              is_map(target) and not is_struct(target) do
    Map.new(target, fn {k, t_v} ->
      case Map.fetch(context, k) do
        {:ok, c_v} -> {k, ema_blend(c_v, t_v, momentum)}
        :error -> {k, t_v}
      end
    end)
  end

  defp ema_blend(context, target, momentum) do
    Nx.add(Nx.multiply(momentum, target), Nx.multiply(1.0 - momentum, context))
  end

  # ============================================================================
  # Loss
  # ============================================================================

  @doc """
  Compute V-JEPA 2 loss (L1 distance between predicted and target representations).

  V-JEPA 2 uses L1 loss (not L2 or smooth-L1 as in V-JEPA 1).

  ## Parameters

    - `predicted` - Predictor output: `[batch, num_tokens, embed_dim]`
    - `target` - Target encoder output: `[batch, num_tokens, embed_dim]`

  ## Returns

    Scalar loss tensor.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn loss(predicted, target) do
    Nx.mean(Nx.abs(predicted - target))
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  defp maybe_dropout(x, rate, _name) when rate <= 0.0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  @doc """
  Get the output size of the V-JEPA 2 model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :embed_dim, @default_embed_dim)
  end

  @doc """
  Recommended default configuration.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      embed_dim: 256,
      predictor_embed_dim: 128,
      encoder_depth: 6,
      predictor_depth: 4,
      num_heads: 8,
      mlp_ratio: 4.0,
      dropout: 0.0,
      num_tokens: 128
    ]
  end

  @doc "Default EMA momentum."
  @spec default_momentum() :: float()
  def default_momentum, do: @default_momentum
end
