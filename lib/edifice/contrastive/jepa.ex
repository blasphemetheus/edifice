defmodule Edifice.Contrastive.JEPA do
  @moduledoc """
  JEPA - Joint Embedding Predictive Architecture.

  Implements JEPA from "Self-Supervised Learning from Images with a Joint-Embedding
  Predictive Architecture" (Assran et al., CVPR 2023). JEPA predicts representations
  of masked regions rather than pixel values, learning more abstract features.

  ## Key Innovations

  - **Predict representations, not pixels**: Unlike MAE which reconstructs raw input,
    JEPA predicts the target encoder's representation of masked regions
  - **Asymmetric architecture**: A narrow predictor bridges context to target space
  - **EMA target**: Target encoder uses exponential moving average of context encoder
    weights (same pattern as BYOL, handled at training time)

  ## Architecture

  ```
  Input (with mask)
        |
        v
  +===================+
  | Context Encoder   |  (processes visible patches)
  |  Projection       |
  |  + Pos Embed      |
  |  Transformer x N  |
  |  LayerNorm        |
  |  Mean Pool        |
  +===================+
        |
        v
  [batch, embed_dim]   (context representation)

  Context Repr + Mask Tokens
        |
        v
  +===================+
  | Predictor         |  (narrow transformer)
  |  Project to       |
  |    predictor_dim  |
  |  + Pos Embed      |
  |  Concat mask tkns |
  |  Transformer x M  |
  |  LayerNorm        |
  |  Project back to  |
  |    embed_dim      |
  +===================+
        |
        v
  [batch, embed_dim]   (predicted target representation)
  ```

  The target encoder is architecturally identical to the context encoder
  with EMA-updated parameters (not part of the computational graph).

  ## Returns

  `{context_encoder, predictor}` — two Axon models.

  ## Usage

      {context_encoder, predictor} = JEPA.build(
        input_dim: 287,
        embed_dim: 256,
        predictor_embed_dim: 128,
        encoder_depth: 6,
        predictor_depth: 4
      )

      # After each training step, update target via EMA
      target_params = JEPA.ema_update(context_params, target_params, momentum: 0.996)

  ## References

  - "Self-Supervised Learning from Images with a Joint-Embedding Predictive
    Architecture" (Assran et al., CVPR 2023)
  - arXiv: https://arxiv.org/abs/2301.08243
  """

  import Nx.Defn

  @default_embed_dim 256
  @default_predictor_embed_dim 128
  @default_encoder_depth 6
  @default_predictor_depth 4
  @default_num_heads 8
  @default_mlp_ratio 4.0
  @default_dropout 0.1
  @default_momentum 0.996

  @doc """
  Build both the context encoder and predictor networks.

  ## Options
    - `:input_dim` - Input feature dimension (required)
    - `:embed_dim` - Encoder embedding dimension (default: 256)
    - `:predictor_embed_dim` - Predictor hidden dimension, narrower than encoder (default: 128)
    - `:encoder_depth` - Number of transformer blocks in encoder (default: 6)
    - `:predictor_depth` - Number of transformer blocks in predictor (default: 4)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:mlp_ratio` - FFN expansion ratio (default: 4.0)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    `{context_encoder, predictor}` tuple of Axon models.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:encoder_depth, pos_integer()}
          | {:input_dim, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_heads, pos_integer()}
          | {:predictor_depth, pos_integer()}
          | {:predictor_embed_dim, pos_integer()}

  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    context_encoder = build_context_encoder(opts)
    predictor = build_predictor(opts)
    {context_encoder, predictor}
  end

  @doc """
  Build the context encoder.

  Processes input features through a projection, positional embedding,
  and a stack of transformer blocks, then mean-pools to produce a
  fixed-size representation.

  ## Options
    - `:input_dim` - Input feature dimension (required)
    - `:embed_dim` - Output embedding dimension (default: 256)
    - `:encoder_depth` - Number of transformer blocks (default: 6)
    - `:num_heads` - Attention heads (default: 8)
    - `:mlp_ratio` - FFN expansion ratio (default: 4.0)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    Axon model mapping `[batch, input_dim]` to `[batch, embed_dim]`.
  """
  @spec build_context_encoder(keyword()) :: Axon.t()
  def build_context_encoder(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    encoder_depth = Keyword.get(opts, :encoder_depth, @default_encoder_depth)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    input = Axon.input("features", shape: {nil, input_dim})

    # Input projection to embed_dim
    x = Axon.dense(input, embed_dim, name: "ctx_enc_proj")
    x = Axon.activation(x, :gelu, name: "ctx_enc_proj_act")

    # Transformer blocks (operating on [batch, embed_dim] as single-token sequence)
    x =
      build_transformer_stack(
        x,
        encoder_depth,
        embed_dim,
        num_heads,
        mlp_ratio,
        dropout,
        "ctx_enc"
      )

    # Final normalization
    Axon.layer_norm(x, name: "ctx_enc_final_norm")
  end

  @doc """
  Build the predictor network.

  Takes context encoder output, projects to a narrower dimension,
  processes through transformer blocks, and projects back to embed_dim.

  ## Options
    - `:embed_dim` - Context encoder output dimension (default: 256)
    - `:predictor_embed_dim` - Predictor internal dimension (default: 128)
    - `:predictor_depth` - Number of transformer blocks (default: 4)
    - `:num_heads` - Attention heads (default: 8)
    - `:mlp_ratio` - FFN expansion ratio (default: 4.0)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    Axon model mapping `[batch, embed_dim]` to `[batch, embed_dim]`.
  """
  @spec build_predictor(keyword()) :: Axon.t()
  def build_predictor(opts \\ []) do
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    pred_dim = Keyword.get(opts, :predictor_embed_dim, @default_predictor_embed_dim)
    predictor_depth = Keyword.get(opts, :predictor_depth, @default_predictor_depth)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    input = Axon.input("context", shape: {nil, embed_dim})

    # Project context to predictor dimension
    x = Axon.dense(input, pred_dim, name: "pred_proj_down")
    x = Axon.activation(x, :gelu, name: "pred_proj_act")

    # Transformer blocks in predictor dimension
    x =
      build_transformer_stack(x, predictor_depth, pred_dim, num_heads, mlp_ratio, dropout, "pred")

    # Final norm and project back to embed_dim
    x = Axon.layer_norm(x, name: "pred_final_norm")
    Axon.dense(x, embed_dim, name: "pred_proj_up")
  end

  # Build a stack of transformer-style MLP blocks for flat [batch, dim] inputs
  # Since JEPA operates on pooled representations (not sequences), these are
  # simplified transformer blocks without positional encoding or causal masks
  defp build_transformer_stack(x, depth, dim, num_heads, mlp_ratio, dropout, prefix) do
    head_dim = max(div(dim, num_heads), 1)
    inner_size = round(dim * mlp_ratio)

    Enum.reduce(1..depth, x, fn layer_idx, acc ->
      name = "#{prefix}_block_#{layer_idx}"

      # Self-attention sublayer (simplified for flat vectors)
      attn_normed = Axon.layer_norm(acc, name: "#{name}_attn_norm")

      attn_out =
        build_flat_attention(attn_normed,
          dim: dim,
          num_heads: num_heads,
          head_dim: head_dim,
          name: "#{name}_attn"
        )

      attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
      after_attn = Axon.add(acc, attn_out, name: "#{name}_attn_residual")

      # FFN sublayer
      ffn_normed = Axon.layer_norm(after_attn, name: "#{name}_ffn_norm")

      ffn_out =
        ffn_normed
        |> Axon.dense(inner_size, name: "#{name}_ffn_up")
        |> Axon.activation(:gelu, name: "#{name}_ffn_act")
        |> Axon.dense(dim, name: "#{name}_ffn_down")

      ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
      Axon.add(after_attn, ffn_out, name: "#{name}_ffn_residual")
    end)
  end

  # Simplified multi-head attention for flat [batch, dim] inputs
  # Treats the input as a single token sequence
  defp build_flat_attention(input, opts) do
    dim = Keyword.fetch!(opts, :dim)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    name = Keyword.get(opts, :name, "flat_attn")

    # QKV projection
    qkv = Axon.dense(input, dim * 3, name: "#{name}_qkv")

    # Compute multi-head attention on single-token "sequence"
    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          {batch, _} = Nx.shape(qkv_tensor)

          query = Nx.slice_along_axis(qkv_tensor, 0, dim, axis: 1)
          key = Nx.slice_along_axis(qkv_tensor, dim, dim, axis: 1)
          value = Nx.slice_along_axis(qkv_tensor, dim * 2, dim, axis: 1)

          # Reshape: [batch, dim] -> [batch, num_heads, 1, head_dim]
          query = Nx.reshape(query, {batch, num_heads, 1, head_dim})
          key = Nx.reshape(key, {batch, num_heads, 1, head_dim})
          value = Nx.reshape(value, {batch, num_heads, 1, head_dim})

          # Attention (single token -> trivial softmax, but keeps the structure)
          scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(query)))
          scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1])
          scores = Nx.divide(scores, scale)

          weights =
            Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))

          weights =
            Nx.divide(weights, Nx.add(Nx.sum(weights, axes: [-1], keep_axes: true), 1.0e-8))

          output = Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])

          # [batch, num_heads, 1, head_dim] -> [batch, dim]
          Nx.reshape(output, {batch, num_heads * head_dim})
        end,
        name: "#{name}_compute"
      )

    # Output projection
    Axon.dense(attended, dim, name: "#{name}_out")
  end

  # ============================================================================
  # EMA Update (same pattern as BYOL)
  # ============================================================================

  @doc """
  Update target encoder parameters via exponential moving average.

  target_params = momentum * target_params + (1 - momentum) * context_params

  ## Parameters
    - `context_params` - Current context encoder parameters (map of tensors)
    - `target_params` - Current target encoder parameters (map of tensors)

  ## Options
    - `:momentum` - EMA momentum coefficient (default: 0.996)

  ## Returns
    Updated target parameters.
  """
  @spec ema_update(map(), map(), keyword()) :: map()
  def ema_update(context_params, target_params, opts \\ []) do
    momentum = Keyword.get(opts, :momentum, @default_momentum)

    Map.new(target_params, fn {key, target_val} ->
      # Map context encoder keys to target encoder keys
      context_key = String.replace(key, "tgt_enc_", "ctx_enc_", global: false)

      updated =
        case Map.fetch(context_params, context_key) do
          {:ok, context_val} ->
            ema_blend(context_val, target_val, momentum)

          :error ->
            target_val
        end

      {key, updated}
    end)
  end

  defp ema_blend(context, target, momentum)
       when is_map(context) and not is_struct(context) and is_map(target) and
              not is_struct(target) do
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
  Compute the JEPA loss (smooth L1 / Huber loss between predicted and target representations).

  ## Parameters
    - `predicted` - Predictor output: [batch, embed_dim]
    - `target` - Target encoder output: [batch, embed_dim]

  ## Returns
    Scalar loss tensor.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn loss(predicted, target) do
    diff = predicted - target
    abs_diff = Nx.abs(diff)

    # Smooth L1 (Huber): 0.5*x^2 if |x|<1 else |x|-0.5
    smooth_l1 =
      Nx.select(
        Nx.less(abs_diff, 1.0),
        0.5 * diff * diff,
        abs_diff - 0.5
      )

    Nx.mean(smooth_l1)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  @doc """
  Get the output size of the JEPA model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :embed_dim, @default_embed_dim)
  end

  @doc """
  Get recommended defaults.
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
      dropout: 0.1
    ]
  end

  @doc "Default EMA momentum."
  @spec default_momentum() :: float()
  def default_momentum, do: @default_momentum
end
