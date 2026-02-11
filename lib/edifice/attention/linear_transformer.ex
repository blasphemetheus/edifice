defmodule Edifice.Attention.LinearTransformer do
  @moduledoc """
  Linear Transformer: Linear attention using kernel feature maps.

  Replaces softmax attention with a kernel-based linear attention mechanism,
  reducing complexity from O(N^2) to O(N) by avoiding explicit computation
  of the N x N attention matrix.

  ## Key Innovation: Kernel Feature Maps

  Standard attention computes: Attn(Q,K,V) = softmax(QK^T/sqrt(d)) * V

  Linear attention rewrites this using a feature map phi:
  ```
  Attn(Q,K,V) = phi(Q) * (phi(K)^T * V) / (phi(Q) * sum(phi(K)))
  ```

  By computing phi(K)^T * V first (a d x d matrix), we avoid the N x N
  attention matrix entirely. The feature map phi(x) = ELU(x) + 1 ensures
  non-negative attention weights.

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  +-------------------------------------+
  |  Linear Transformer Block            |
  |                                      |
  |  LayerNorm                           |
  |    -> Q, K, V projections            |
  |    -> phi(Q), phi(K) feature maps    |
  |    -> KV = phi(K)^T * V  [d x d]    |
  |    -> out = phi(Q) * KV   [N x d]   |
  |    -> normalize by phi(Q)*sum(K)     |
  |  -> Residual                         |
  |                                      |
  |  LayerNorm -> FFN -> Residual        |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Last timestep -> [batch, hidden_size]
  ```

  ## Complexity

  | Operation | Standard | Linear |
  |-----------|----------|--------|
  | Attention | O(N^2 * d) | O(N * d^2) |
  | Memory | O(N^2) | O(N * d) |
  | Best when | N < d | N > d |

  Linear attention is most beneficial when sequence length N exceeds
  the head dimension d.

  ## Usage

      model = LinearTransformer.build(
        embed_size: 287,
        hidden_size: 256,
        num_layers: 4,
        dropout: 0.1
      )

  ## References
  - Paper: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    (Katharopoulos et al., 2020)
  - Feature map: ELU+1 from the original paper
  """

  require Axon

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 4
  @default_num_heads 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a Linear Transformer model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of transformer blocks (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
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

    # Stack linear transformer blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_linear_attn_block(
          acc,
          hidden_size: hidden_size,
          num_heads: Keyword.get(opts, :num_heads, @default_num_heads),
          dropout: dropout,
          name: "linear_block_#{layer_idx}"
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
  Build a single Linear Transformer block.

  Each block has:
  1. LayerNorm -> Linear Attention -> Residual
  2. LayerNorm -> FFN -> Residual
  """
  @spec build_linear_attn_block(Axon.t(), keyword()) :: Axon.t()
  def build_linear_attn_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "linear_block")

    head_dim = div(hidden_size, num_heads)

    # 1. Linear attention branch
    attn_normed = Axon.layer_norm(input, name: "#{name}_attn_norm")

    # Q, K, V projections
    q_proj = Axon.dense(attn_normed, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(attn_normed, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(attn_normed, hidden_size, name: "#{name}_v_proj")

    # Apply linear attention
    attn_out = Axon.layer(
      &linear_attention_impl/4,
      [q_proj, k_proj, v_proj],
      name: "#{name}_linear_attn",
      num_heads: num_heads,
      head_dim: head_dim,
      op_name: :linear_attention
    )

    # Output projection + dropout
    attn_out = Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")

    attn_out =
      if dropout > 0 do
        Axon.dropout(attn_out, rate: dropout, name: "#{name}_attn_dropout")
      else
        attn_out
      end

    # Residual
    after_attn = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # 2. FFN branch
    ffn_normed = Axon.layer_norm(after_attn, name: "#{name}_ffn_norm")
    ffn_out = build_ffn(ffn_normed, hidden_size, "#{name}_ffn")

    ffn_out =
      if dropout > 0 do
        Axon.dropout(ffn_out, rate: dropout, name: "#{name}_ffn_dropout")
      else
        ffn_out
      end

    Axon.add(after_attn, ffn_out, name: "#{name}_ffn_residual")
  end

  # Linear attention implementation
  # Uses ELU+1 feature map: phi(x) = ELU(x) + 1
  # Computes: phi(Q) * (phi(K)^T * V) / (phi(Q) * sum(phi(K)))
  defp linear_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    eps = 1.0e-6

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape for multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Apply ELU+1 feature map for non-negative features
    # phi(x) = ELU(x) + 1 = max(x, 0) + exp(min(x, 0))
    q_feat = elu_plus_one(q)
    k_feat = elu_plus_one(k)

    # Causal linear attention using cumulative sums
    # For causal: at position t, we only use K/V up to position t
    # KV[t] = sum_{i<=t} phi(K[i])^T * V[i] (cumulative)
    # Z[t] = sum_{i<=t} phi(K[i]) (cumulative normalizer)

    # Compute K^T * V outer products: [batch, heads, seq, head_dim, head_dim]
    k_expanded = Nx.new_axis(k_feat, 4)    # [batch, heads, seq, head_dim, 1]
    v_expanded = Nx.new_axis(v, 3)          # [batch, heads, seq, 1, head_dim]
    kv = Nx.multiply(k_expanded, v_expanded) # [batch, heads, seq, head_dim, head_dim]

    # Cumulative sum for causal attention
    kv_cumsum = Nx.cumulative_sum(kv, axis: 2)   # [batch, heads, seq, head_dim, head_dim]
    k_cumsum = Nx.cumulative_sum(k_feat, axis: 2) # [batch, heads, seq, head_dim]

    # Compute output: phi(Q) @ KV_cumsum
    # q_feat: [batch, heads, seq, head_dim]
    # kv_cumsum: [batch, heads, seq, head_dim, head_dim]
    q_expanded = Nx.new_axis(q_feat, 3)  # [batch, heads, seq, 1, head_dim]
    numerator = Nx.sum(Nx.multiply(q_expanded, kv_cumsum), axes: [4])  # [batch, heads, seq, head_dim]

    # Normalizer: phi(Q) . sum(phi(K))
    denominator = Nx.sum(Nx.multiply(q_feat, k_cumsum), axes: [3], keep_axes: true)  # [batch, heads, seq, 1]

    # Normalized output
    output = Nx.divide(numerator, Nx.add(denominator, eps))

    # Reshape back: [batch, seq, num_heads * head_dim]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ELU+1 feature map: ensures non-negative attention weights
  # phi(x) = ELU(x) + 1 = max(x,0) + exp(min(x,0))
  defp elu_plus_one(x) do
    positive = Nx.max(x, 0.0)
    negative = Nx.exp(Nx.min(x, 0.0))
    Nx.add(positive, negative)
  end

  # Feed-forward network
  defp build_ffn(input, hidden_size, name) do
    inner_size = hidden_size * 4

    input
    |> Axon.dense(inner_size, name: "#{name}_up")
    |> Axon.activation(:gelu, name: "#{name}_gelu")
    |> Axon.dense(hidden_size, name: "#{name}_down")
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Linear Transformer model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a Linear Transformer model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    inner_size = hidden_size * 4

    # Per layer:
    # Linear attention: Q + K + V + output projections
    attn_params = hidden_size * hidden_size * 4

    # FFN: up + down
    ffn_params =
      hidden_size * inner_size +
      inner_size * hidden_size

    per_layer = attn_params + ffn_params

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
      num_layers: 4,
      num_heads: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
