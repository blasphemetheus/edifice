defmodule Edifice.Attention.Perceiver do
  @moduledoc """
  Perceiver IO: General-purpose architecture with learned latent array.

  Perceiver IO uses cross-attention to map arbitrary inputs to a fixed-size
  latent array, processes latents with self-attention, then optionally
  cross-attends back for structured output. This decouples compute from
  input size.

  ## Key Innovation: Latent Bottleneck

  Instead of self-attending over the full input (O(N^2)), Perceiver
  cross-attends inputs to a small learned latent array (M << N), then
  self-attends over latents (O(M^2)). Total: O(N*M + M^2).

  ```
  Input [batch, N, input_dim]     Latents [1, M, latent_dim] (learned)
        |                               |
        +-- Cross-Attention(L, Input) --+
                    |
              Latents' [batch, M, latent_dim]
                    |
              Self-Attention x num_layers
                    |
              Latents'' [batch, M, latent_dim]
                    |
              Pool -> [batch, latent_dim]
  ```

  ## Architecture

  ```
  Input [batch, seq_len, input_dim]
        |
        v
  +-------------------------------------+
  |  Cross-Attention                     |
  |  Q = Latent Array (learned, M x D)  |
  |  K, V = Input                       |
  |  -> Latents absorb input info       |
  +-------------------------------------+
        |
        v (repeat num_cross_layers)
  +-------------------------------------+
  |  Self-Attention Block                |
  |  LayerNorm -> Self-Attn -> Residual  |
  |  LayerNorm -> FFN -> Residual        |
  +-------------------------------------+
        | (repeat num_layers)
        v
  Mean pool over latents -> [batch, latent_dim]
  ```

  ## Complexity

  | Component | Standard Transformer | Perceiver |
  |-----------|---------------------|-----------|
  | Self-Attn | O(N^2) | O(M^2) |
  | Cross-Attn | - | O(N*M) |
  | Total | O(N^2) | O(N*M + M^2) |

  Where M = num_latents << N = input length.

  ## Usage

      model = Perceiver.build(
        input_dim: 287,
        latent_dim: 256,
        num_latents: 64,
        num_layers: 4,
        num_cross_layers: 1,
        num_heads: 4
      )

  ## References
  - Paper: "Perceiver IO: A General Architecture for Structured Inputs & Outputs"
    (Jaegle et al., DeepMind 2021)
  - Original: "Perceiver: General Perception with Iterative Attention" (2021)
  """

  require Axon

  alias Edifice.Blocks.FFN
  alias Edifice.Utils.FusedOps

  # Default hyperparameters
  @default_latent_dim 256
  @default_num_latents 64
  @default_num_layers 4
  @default_num_cross_layers 1
  @default_num_heads 4
  @default_dropout 0.1

  @doc """
  Build a Perceiver IO model for sequence processing.

  ## Options

    - `:input_dim` - Size of input embedding per timestep (required)
    - `:latent_dim` - Latent array dimension (default: 256)
    - `:num_latents` - Number of latent vectors M (default: 64)
    - `:num_layers` - Number of self-attention layers over latents (default: 4)
    - `:num_cross_layers` - Number of input cross-attention passes (default: 1)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns

    An Axon model that outputs [batch, latent_dim].
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    latent_dim = Keyword.get(opts, :latent_dim, @default_latent_dim)
    num_latents = Keyword.get(opts, :num_latents, @default_num_latents)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_cross_layers = Keyword.get(opts, :num_cross_layers, @default_num_cross_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Input: [batch, seq_len, input_dim] - arbitrary length
    input = Axon.input("state_sequence", shape: {nil, nil, input_dim})

    # Project input to latent_dim for cross-attention compatibility
    input_proj = Axon.dense(input, latent_dim, name: "input_projection")

    # Create learned latent array: [1, num_latents, latent_dim]
    # We use a parameter node that gets broadcast to batch size
    latent_init =
      Axon.param("latent_array", {1, num_latents, latent_dim}, initializer: :glorot_uniform)

    # Wrap latent_init so it can be combined with the input graph
    # Use Axon.layer to create the latent array broadcast to batch size
    latents =
      Axon.layer(
        &broadcast_latents/3,
        [input_proj, latent_init],
        name: "latent_broadcast",
        num_latents: num_latents,
        latent_dim: latent_dim,
        op_name: :broadcast_latents
      )

    # Iterative refinement: interleave cross-attention with self-attention blocks.
    # Each cross-attention pass is followed by a group of self-attention layers,
    # matching the original Perceiver paper's progressive refinement pattern.
    # With num_cross_layers=1, this is identical to Perceiver IO (1 cross + N self).
    self_layers_per_cross = max(div(num_layers, num_cross_layers), 1)

    latents =
      Enum.reduce(1..num_cross_layers, latents, fn cross_idx, acc ->
        # Cross-attention: latents attend to input
        after_cross =
          build_cross_attention_block(
            acc,
            input_proj,
            latent_dim: latent_dim,
            num_heads: num_heads,
            dropout: dropout,
            name: "cross_attn_#{cross_idx}"
          )

        # Self-attention blocks for this cross-attention group
        num_self =
          if cross_idx == num_cross_layers do
            # Last group gets any remaining self-attention layers
            num_layers - self_layers_per_cross * (num_cross_layers - 1)
          else
            self_layers_per_cross
          end

        start_idx = self_layers_per_cross * (cross_idx - 1) + 1

        Enum.reduce(start_idx..(start_idx + num_self - 1), after_cross, fn layer_idx, inner_acc ->
          build_self_attention_block(
            inner_acc,
            latent_dim: latent_dim,
            num_heads: num_heads,
            dropout: dropout,
            name: "self_attn_block_#{layer_idx}"
          )
        end)
      end)

    # Final layer norm
    latents = Axon.layer_norm(latents, name: "final_norm")

    # Mean pool over latent dimension: [batch, num_latents, latent_dim] -> [batch, latent_dim]
    Axon.nx(
      latents,
      fn tensor ->
        Nx.mean(tensor, axes: [1])
      end,
      name: "latent_pool"
    )
  end

  # Broadcast latent array to batch size
  defp broadcast_latents(input_proj, latent_array, _opts) do
    batch_size = Nx.axis_size(input_proj, 0)
    {1, num_latents, latent_dim} = Nx.shape(latent_array)
    Nx.broadcast(latent_array, {batch_size, num_latents, latent_dim})
  end

  @doc """
  Build a cross-attention block where latents attend to input.

  Structure: LayerNorm(latents) -> CrossAttn(Q=latents, KV=input) -> Residual
             -> LayerNorm -> FFN -> Residual
  """
  @spec build_cross_attention_block(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def build_cross_attention_block(latents, input_kv, opts) do
    latent_dim = Keyword.get(opts, :latent_dim, @default_latent_dim)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "cross_attn")

    head_dim = div(latent_dim, num_heads)

    # Pre-norm latents
    latents_normed = Axon.layer_norm(latents, name: "#{name}_q_norm")
    input_normed = Axon.layer_norm(input_kv, name: "#{name}_kv_norm")

    # Q from latents, K/V from input
    q_proj = Axon.dense(latents_normed, latent_dim, name: "#{name}_q_proj")
    k_proj = Axon.dense(input_normed, latent_dim, name: "#{name}_k_proj")
    v_proj = Axon.dense(input_normed, latent_dim, name: "#{name}_v_proj")

    # Cross-attention
    attn_out =
      Axon.layer(
        &cross_attention_impl/4,
        [q_proj, k_proj, v_proj],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :cross_attention
      )

    # Output projection + dropout
    attn_out = Axon.dense(attn_out, latent_dim, name: "#{name}_out_proj")

    attn_out =
      if dropout > 0 do
        Axon.dropout(attn_out, rate: dropout, name: "#{name}_dropout")
      else
        attn_out
      end

    # Residual
    after_attn = Axon.add(latents, attn_out, name: "#{name}_residual")

    # FFN
    ffn_normed = Axon.layer_norm(after_attn, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed, hidden_size: latent_dim, dropout: dropout, name: "#{name}_ffn")

    Axon.add(after_attn, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Build a self-attention block over latents.

  Structure: LayerNorm -> Self-Attention -> Residual -> LayerNorm -> FFN -> Residual
  """
  @spec build_self_attention_block(Axon.t(), keyword()) :: Axon.t()
  def build_self_attention_block(input, opts) do
    latent_dim = Keyword.get(opts, :latent_dim, @default_latent_dim)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "self_attn")

    head_dim = div(latent_dim, num_heads)

    # Pre-norm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Q, K, V from same input (self-attention)
    q_proj = Axon.dense(x, latent_dim, name: "#{name}_q_proj")
    k_proj = Axon.dense(x, latent_dim, name: "#{name}_k_proj")
    v_proj = Axon.dense(x, latent_dim, name: "#{name}_v_proj")

    # Self-attention (no causal mask - latents are unordered)
    attn_out =
      Axon.layer(
        &self_attention_impl/4,
        [q_proj, k_proj, v_proj],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :self_attention
      )

    # Output projection + dropout
    attn_out = Axon.dense(attn_out, latent_dim, name: "#{name}_out_proj")

    attn_out =
      if dropout > 0 do
        Axon.dropout(attn_out, rate: dropout, name: "#{name}_dropout")
      else
        attn_out
      end

    # Residual
    after_attn = Axon.add(input, attn_out, name: "#{name}_residual")

    # FFN
    ffn_normed = Axon.layer_norm(after_attn, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed, hidden_size: latent_dim, dropout: dropout, name: "#{name}_ffn")

    Axon.add(after_attn, ffn_out, name: "#{name}_ffn_residual")
  end

  # Cross-attention: Q from one source, K/V from another (no causal mask)
  defp cross_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_q = Nx.axis_size(q, 1)
    seq_kv = Nx.axis_size(k, 1)

    # Reshape for multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_q, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_kv, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_kv, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention (no mask for cross-attention)
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)
    weights = FusedOps.fused_softmax(scores)
    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, seq_q, num_heads * head_dim]
    output |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({batch, seq_q, num_heads * head_dim})
  end

  # Self-attention without causal mask (latents are unordered)
  defp self_attention_impl(q, k, v, opts) do
    cross_attention_impl(q, k, v, opts)
  end

  # FFN delegated to Edifice.Blocks.FFN

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Perceiver model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :latent_dim, @default_latent_dim)
  end

  @doc """
  Calculate approximate parameter count for a Perceiver model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    input_dim = Keyword.get(opts, :input_dim, 287)
    latent_dim = Keyword.get(opts, :latent_dim, @default_latent_dim)
    num_latents = Keyword.get(opts, :num_latents, @default_num_latents)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_cross_layers = Keyword.get(opts, :num_cross_layers, @default_num_cross_layers)

    inner_size = latent_dim * 4

    # Latent array
    latent_params = num_latents * latent_dim

    # Input projection
    input_proj = input_dim * latent_dim

    # Cross-attention per layer: Q + K + V + output + FFN
    cross_attn_params =
      latent_dim * latent_dim * 3 + latent_dim * latent_dim +
        (latent_dim * inner_size + inner_size * latent_dim)

    # Self-attention per layer: Q + K + V + output + FFN
    self_attn_params =
      latent_dim * latent_dim * 3 + latent_dim * latent_dim +
        (latent_dim * inner_size + inner_size * latent_dim)

    latent_params + input_proj +
      cross_attn_params * num_cross_layers +
      self_attn_params * num_layers
  end

  @doc """
  Recommended default configuration for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      latent_dim: 256,
      num_latents: 64,
      num_layers: 4,
      num_cross_layers: 1,
      num_heads: 4,
      dropout: 0.1
    ]
  end
end
