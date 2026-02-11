defmodule Edifice.Attention.GLA do
  @moduledoc """
  GLA: Gated Linear Attention with data-dependent gating.

  GLA combines the efficiency of linear attention (O(L) complexity) with
  data-dependent gating for improved expressiveness. It's particularly
  effective on short sequences (<2K tokens) where it can outperform
  FlashAttention-2.

  ## Key Innovation: Data-Dependent Gating

  Unlike standard linear attention which uses fixed feature maps, GLA
  computes gates from the input that control information flow:

  ```
  output[t] = gate[t] * (Q[t] @ cumsum(K[i] * V[i] / cumsum(K[i])))
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  +-------------------------------------+
  |  GLA Block                           |
  |                                      |
  |  Q, K, V projections                 |
  |         |                            |
  |  Data-dependent gating (G)           |
  |         |                            |
  |  Linear attention with gates         |
  |         |                            |
  |  Output projection                   |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  [batch, hidden_size]
  ```

  ## Complexity

  | Aspect | Standard Attention | GLA |
  |--------|-------------------|-----|
  | Time | O(L^2) | O(L) |
  | Space | O(L^2) | O(L) |
  | Hardware | FlashAttention needed | Native tensor ops |

  ## Usage

      model = GLA.build(
        embed_size: 287,
        hidden_size: 256,
        num_layers: 6,
        num_heads: 4
      )

  ## Reference

  - Paper: "Gated Linear Attention Transformers with Hardware-Efficient Training"
  - Implementation: flash-linear-attention (https://github.com/fla-org/flash-linear-attention)
  """

  require Axon

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 6
  @default_num_heads 4
  @default_head_dim 64
  @default_expand_factor 2
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a GLA model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of GLA blocks (default: 6)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:head_dim` - Dimension per head (default: 64)
    - `:expand_factor` - FFN expansion factor (default: 2)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project input to hidden dimension if different
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack GLA blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_gla_block(
          acc,
          hidden_size: hidden_size,
          num_heads: num_heads,
          head_dim: head_dim,
          expand_factor: expand_factor,
          dropout: dropout,
          name: "gla_block_#{layer_idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      x,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Build a single GLA block.

  Each block has:
  1. Gated Linear Attention layer
  2. Gated FFN (similar to GLU)
  """
  @spec build_gla_block(Axon.t(), keyword()) :: Axon.t()
  def build_gla_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "gla_block")

    # Gated Linear Attention
    x = build_gated_linear_attention(input,
      hidden_size: hidden_size,
      num_heads: num_heads,
      head_dim: head_dim,
      dropout: dropout,
      name: "#{name}_attention"
    )

    # Gated FFN
    build_gated_ffn(x,
      hidden_size: hidden_size,
      expand_factor: expand_factor,
      dropout: dropout,
      name: "#{name}_ffn"
    )
  end

  @doc """
  Build the Gated Linear Attention layer.

  Key components:
  1. Q, K, V, G projections (G = gate)
  2. Linear attention with data-dependent gating
  3. Output projection
  """
  @spec build_gated_linear_attention(Axon.t(), keyword()) :: Axon.t()
  def build_gated_linear_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "gla")

    attn_dim = num_heads * head_dim

    # Pre-LayerNorm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Q, K, V projections
    q_proj = Axon.dense(x, attn_dim, name: "#{name}_q_proj")
    k_proj = Axon.dense(x, attn_dim, name: "#{name}_k_proj")
    v_proj = Axon.dense(x, attn_dim, name: "#{name}_v_proj")

    # Gate projection (data-dependent gating)
    # This is the key innovation of GLA - gates computed from input
    g_proj = Axon.dense(x, attn_dim, name: "#{name}_g_proj")
    g_proj = Axon.activation(g_proj, :sigmoid, name: "#{name}_g_sigmoid")

    # Gated linear attention
    output = Axon.layer(
      &gated_linear_attention_impl/5,
      [q_proj, k_proj, v_proj, g_proj],
      name: "#{name}_gla",
      num_heads: num_heads,
      head_dim: head_dim,
      op_name: :gated_linear_attention
    )

    # Output projection
    output = Axon.dense(output, hidden_size, name: "#{name}_output")

    # Dropout
    output =
      if dropout > 0 do
        Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
      else
        output
      end

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  @doc """
  Build the Gated FFN layer (GLU-style).

  Uses gated structure: output = gate * up_proj
  """
  @spec build_gated_ffn(Axon.t(), keyword()) :: Axon.t()
  def build_gated_ffn(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "gated_ffn")

    inner_size = hidden_size * expand_factor

    # Pre-LayerNorm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Gate and up projection (GLU style)
    gate_proj = Axon.dense(x, inner_size, name: "#{name}_gate_proj")
    up_proj = Axon.dense(x, inner_size, name: "#{name}_up_proj")

    # SiLU gate * up
    gate = Axon.activation(gate_proj, :silu, name: "#{name}_silu")
    gated = Axon.multiply(gate, up_proj, name: "#{name}_gated")

    # Down projection
    output = Axon.dense(gated, hidden_size, name: "#{name}_down_proj")

    # Dropout
    output =
      if dropout > 0 do
        Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
      else
        output
      end

    # Residual connection
    Axon.add(input, output, name: "#{name}_residual")
  end

  # Gated Linear Attention implementation
  defp gated_linear_attention_impl(q, k, v, g, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    # q, k, v, g: [batch, seq_len, num_heads * head_dim]
    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to heads: [batch, seq_len, num_heads, head_dim]
    q = Nx.reshape(q, {batch, seq_len, num_heads, head_dim})
    k = Nx.reshape(k, {batch, seq_len, num_heads, head_dim})
    v = Nx.reshape(v, {batch, seq_len, num_heads, head_dim})
    g = Nx.reshape(g, {batch, seq_len, num_heads, head_dim})

    # Apply feature map to Q and K (ELU + 1 for positive features)
    # This is a common choice for linear attention
    q_feat = Nx.add(1.0, Nx.max(q, 0.0)) # Simple ReLU + 1
    k_feat = Nx.add(1.0, Nx.max(k, 0.0))

    # Causal linear attention with gating
    # For each position t:
    # numerator[t] = sum_{i<=t} (q[t] @ k[i]) * v[i] * g[i]
    # denominator[t] = sum_{i<=t} (q[t] @ k[i])
    # output[t] = g[t] * (numerator[t] / denominator[t])

    # Compute key-value outer products: [batch, seq_len, num_heads, head_dim, head_dim]
    # kv[t] = k[t] outer v[t]
    k_expanded = Nx.new_axis(k_feat, 4)  # [batch, seq_len, num_heads, head_dim, 1]
    v_expanded = Nx.new_axis(v, 3)       # [batch, seq_len, num_heads, 1, head_dim]
    kv = Nx.multiply(k_expanded, v_expanded)  # [batch, seq_len, num_heads, head_dim, head_dim]

    # Apply gate to kv
    g_expanded = Nx.new_axis(g, 4)  # [batch, seq_len, num_heads, head_dim, 1]
    kv_gated = Nx.multiply(kv, g_expanded)

    # Cumulative sum for causal attention
    kv_cumsum = Nx.cumulative_sum(kv_gated, axis: 1)  # [batch, seq_len, num_heads, head_dim, head_dim]
    k_cumsum = Nx.cumulative_sum(k_feat, axis: 1)     # [batch, seq_len, num_heads, head_dim]

    # Query attention: output = g * (q @ kv_cumsum) / (q @ k_cumsum)
    # q: [batch, seq_len, num_heads, head_dim]
    # kv_cumsum: [batch, seq_len, num_heads, head_dim, head_dim]

    # Numerator: q @ kv_cumsum -> [batch, seq_len, num_heads, head_dim]
    q_expanded = Nx.new_axis(q_feat, 3)  # [batch, seq_len, num_heads, 1, head_dim]
    numerator = Nx.sum(Nx.multiply(q_expanded, kv_cumsum), axes: [4])  # [batch, seq_len, num_heads, head_dim]

    # Denominator: q @ k_cumsum -> [batch, seq_len, num_heads]
    denominator = Nx.sum(Nx.multiply(q_feat, k_cumsum), axes: [3], keep_axes: true)  # [batch, seq_len, num_heads, 1]

    # Final output with gate
    eps = 1.0e-6
    output = Nx.multiply(g, Nx.divide(numerator, Nx.add(denominator, eps)))

    # Reshape back: [batch, seq_len, num_heads * head_dim]
    Nx.reshape(output, {batch, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a GLA model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a GLA model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    expand_factor = Keyword.get(opts, :expand_factor, @default_expand_factor)

    attn_dim = num_heads * head_dim
    inner_size = hidden_size * expand_factor

    # Per layer:
    # Attention:
    #   - Q, K, V, G projections: 4 * hidden * attn_dim
    #   - Output projection: attn_dim * hidden
    attention_params =
      4 * hidden_size * attn_dim +
      attn_dim * hidden_size

    # FFN (GLU style):
    #   - Gate, Up projections: 2 * hidden * inner
    #   - Down projection: inner * hidden
    ffn_params =
      2 * hidden_size * inner_size +
      inner_size * hidden_size

    per_layer = attention_params + ffn_params

    # Input projection
    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Recommended default configuration for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 6,
      num_heads: 4,
      head_dim: 64,
      expand_factor: 2,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
