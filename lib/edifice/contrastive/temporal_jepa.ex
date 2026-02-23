defmodule Edifice.Contrastive.TemporalJEPA do
  @moduledoc """
  Temporal JEPA — Joint Embedding Predictive Architecture for sequences.

  Extends JEPA to temporal data (video frames, time series, trajectories).
  The context encoder processes visible timesteps with bidirectional attention,
  and the predictor estimates representations of masked timesteps.

  ## Architecture

  ```
  Visible timesteps [batch, seq_len, input_dim]
          |
  +========================+
  | Context Encoder        |
  |  Input projection      |
  |  + Positional embed    |
  |  Bidirectional Attn×N  |
  |  LayerNorm             |
  |  Mean Pool             |
  +========================+
          |
  [batch, embed_dim]  (context representation)

  Context Repr [batch, embed_dim]
          |
  +========================+
  | Predictor              |
  |  Project to pred_dim   |
  |  MLP blocks × M       |
  |  LayerNorm             |
  |  Project to embed_dim  |
  +========================+
          |
  [batch, embed_dim]  (predicted target representation)
  ```

  The target encoder is architecturally identical to the context encoder
  with EMA-updated parameters (not part of the computational graph).

  ## Returns

  `{context_encoder, predictor}` — two Axon models.

  ## Usage

      {context_encoder, predictor} = TemporalJEPA.build(
        input_dim: 128,
        embed_dim: 128,
        predictor_embed_dim: 64,
        seq_len: 60,
        mask_ratio: 0.5
      )

      # Training: encode visible frames, predict masked frame representations
      # Target encoder uses EMA of context encoder weights
      target_params = TemporalJEPA.ema_update(context_params, target_params, momentum: 0.996)

  ## References

  - Bardes et al., "V-JEPA: Latent Video Prediction for Visual Representation
    Learning" (Meta AI, 2024)
  - Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
    Predictive Architecture" (CVPR 2023)
  """

  import Nx.Defn

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}
  alias Edifice.Attention.{MultiHead}

  @default_embed_dim 128
  @default_predictor_embed_dim 64
  @default_encoder_depth 4
  @default_predictor_depth 2
  @default_num_heads 8
  @default_dropout 0.1
  @default_seq_len 60
  @default_momentum 0.996

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_dim, pos_integer()}
          | {:embed_dim, pos_integer()}
          | {:predictor_embed_dim, pos_integer()}
          | {:encoder_depth, pos_integer()}
          | {:predictor_depth, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:dropout, float()}
          | {:seq_len, pos_integer()}
          | {:mask_ratio, float()}

  @doc """
  Build both the context encoder and predictor networks.

  ## Options

    - `:input_dim` - Input feature dimension per timestep (required)
    - `:embed_dim` - Encoder embedding dimension (default: 128)
    - `:predictor_embed_dim` - Predictor internal dimension (default: 64)
    - `:encoder_depth` - Number of transformer blocks in encoder (default: 4)
    - `:predictor_depth` - Number of MLP blocks in predictor (default: 2)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` - Expected sequence length (default: 60)
    - `:mask_ratio` - Fraction of timesteps to mask (default: 0.5)

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
  Build the context encoder for temporal sequences.

  Processes visible timesteps through bidirectional self-attention
  (no causal mask) and mean-pools to a fixed-size representation.

  ## Returns

    Axon model: `[batch, seq_len, input_dim]` → `[batch, embed_dim]`
  """
  @spec build_context_encoder(keyword()) :: Axon.t()
  def build_context_encoder(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    encoder_depth = Keyword.get(opts, :encoder_depth, @default_encoder_depth)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    seq_len = Keyword.get(opts, :seq_len, @default_seq_len)

    ModelBuilder.build_sequence_model(
      embed_dim: input_dim,
      hidden_size: embed_dim,
      num_layers: encoder_depth,
      seq_len: seq_len,
      output_mode: :mean_pool,
      final_norm: true,
      block_builder: fn input, block_opts ->
        layer_idx = block_opts[:layer_idx]
        name = "tjepa_enc_block_#{layer_idx}"

        attn_fn = fn x, attn_name ->
          # Bidirectional attention (no causal mask)
          MultiHead.self_attention(x,
            hidden_size: embed_dim,
            num_heads: num_heads,
            causal: false,
            name: attn_name
          )
        end

        TransformerBlock.layer(input,
          attention_fn: attn_fn,
          hidden_size: embed_dim,
          norm: :layer_norm,
          dropout: dropout,
          name: name
        )
      end
    )
  end

  @doc """
  Build the predictor network.

  Takes the context encoder output (flat vector) and predicts the target
  encoder's representation of masked timesteps.

  ## Returns

    Axon model: `[batch, embed_dim]` → `[batch, embed_dim]`
  """
  @spec build_predictor(keyword()) :: Axon.t()
  def build_predictor(opts \\ []) do
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    pred_dim = Keyword.get(opts, :predictor_embed_dim, @default_predictor_embed_dim)
    predictor_depth = Keyword.get(opts, :predictor_depth, @default_predictor_depth)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    input = Axon.input("context", shape: {nil, embed_dim})

    # Project down to predictor dimension
    x = Axon.dense(input, pred_dim, name: "tjepa_pred_proj_down")
    x = Axon.activation(x, :gelu, name: "tjepa_pred_proj_act")

    # MLP blocks (flat [batch, pred_dim] → same as JEPA predictor)
    x =
      Enum.reduce(1..predictor_depth, x, fn layer_idx, acc ->
        name = "tjepa_pred_block_#{layer_idx}"

        normed = Axon.layer_norm(acc, name: "#{name}_norm")

        ffn_out =
          normed
          |> Axon.dense(pred_dim * 4, name: "#{name}_ffn_up")
          |> Axon.activation(:gelu, name: "#{name}_ffn_act")
          |> Axon.dense(pred_dim, name: "#{name}_ffn_down")

        ffn_out =
          if dropout > 0 do
            Axon.dropout(ffn_out, rate: dropout, name: "#{name}_dropout")
          else
            ffn_out
          end

        Axon.add(acc, ffn_out, name: "#{name}_residual")
      end)

    # Final norm and project back to embed_dim
    x = Axon.layer_norm(x, name: "tjepa_pred_final_norm")
    Axon.dense(x, embed_dim, name: "tjepa_pred_proj_up")
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

    # Map target encoder keys to context encoder keys
    Map.new(target_params, fn {key, target_val} ->
      context_key = String.replace(key, "tjepa_tgt_", "tjepa_enc_", global: false)

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
  Compute temporal JEPA loss (smooth L1 between predicted and target representations).

  ## Parameters

    - `predicted` - Predictor output: `[batch, embed_dim]`
    - `target` - Target encoder output: `[batch, embed_dim]`

  ## Returns

    Scalar loss tensor.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn loss(predicted, target) do
    diff = predicted - target
    abs_diff = Nx.abs(diff)

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

  @doc """
  Generate a temporal mask for masking timesteps.

  Returns a boolean tensor of shape `[seq_len]` where `true` means visible
  and `false` means masked.

  ## Parameters

    - `key` - PRNG key
    - `seq_len` - Number of timesteps
    - `mask_ratio` - Fraction to mask (default: 0.5)

  ## Returns

    `{visible_mask, key}` where `visible_mask` is `[seq_len]` boolean tensor.
  """
  @spec generate_temporal_mask(Nx.Tensor.t(), pos_integer(), float()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def generate_temporal_mask(key, seq_len, mask_ratio \\ 0.5) do
    {random, key} = Nx.Random.uniform(key, shape: {seq_len})
    visible_mask = Nx.greater(random, mask_ratio)
    {visible_mask, key}
  end

  @doc "Get the output size of the temporal JEPA model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :embed_dim, @default_embed_dim)
  end

  @doc "Default EMA momentum."
  @spec default_momentum() :: float()
  def default_momentum, do: @default_momentum
end
