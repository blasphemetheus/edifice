defmodule Edifice.Attention.SPLA do
  @moduledoc """
  SPLA: Block Sparse Plus Linear Attention.

  Combines block-sparse exact attention on high-relevance blocks with residual
  linear attention on the remaining "long tail" blocks for efficient long-context
  processing.

  ## Key Innovations

  - **2nd-order Taylor selection**: Scores blocks using both mean and covariance
    of key vectors, improving selection accuracy over 1st-order methods
  - **Residual linear attention (RLA)**: Instead of discarding unselected blocks,
    compresses them into a recurrent state via linear attention
  - **Subtraction trick**: `o_RLA = o_global - o_selected` avoids explicit
    computation on unselected blocks

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  +------------------------------------------+
  |          SPLA Attention Block             |
  |                                          |
  |  Q, K, V = Linear(input)                |
  |                                          |
  |  Block Partitioning (stride, window)     |
  |         |                                |
  |  2nd-Order Taylor Selection Metric       |
  |  score = exp(q*k_bar) * (1 + 0.5*q*C*q) |
  |         |                                |
  |  +------+-------+                       |
  |  |              |                        |
  |  v              v                        |
  |  Selected    Unselected                  |
  |  (softmax    (linear attention           |
  |   attention)  via subtraction)           |
  |  |              |                        |
  |  +------+-------+                       |
  |         v                                |
  |  o = o_sparse + o_RLA                    |
  +------------------------------------------+
        |
  Output [batch, seq_len, embed_dim]
  ```

  ## Complexity

  | Mechanism | Time | Space |
  |-----------|------|-------|
  | Full attention | O(n^2 d) | O(n^2) |
  | SPLA | O(n(k + d^2)) | O(nd + kw) |

  Where k = selected blocks, d = head dim, w = block window.

  ## Usage

      model = SPLA.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 8,
        head_dim: 32,
        block_stride: 16,
        block_window: 32,
        selection_ratio: 0.25,
        num_layers: 4
      )

  ## References

  - Paper: "SPLA: Block Sparse Plus Linear Attention for Long Context Modeling"
  - Authors: Bailin Wang, Dan Friedman, Tao Lei (January 2026)
  - ArXiv: 2601.22379
  """

  alias Edifice.Blocks.{FFN, ModelBuilder}

  @default_hidden_size 256
  @default_num_heads 8
  @default_head_dim 32
  @default_block_stride 16
  @default_block_window 32
  @default_selection_ratio 0.25
  @default_num_layers 4
  @default_dropout 0.1

  @doc """
  Build an SPLA model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:head_dim` - Dimension per head (default: 32)
    - `:block_stride` - Stride between block starts (default: 16)
    - `:block_window` - Block window size (default: 32)
    - `:selection_ratio` - Fraction of blocks to select for exact attention (default: 0.25)
    - `:num_layers` - Number of SPLA transformer blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` - Expected sequence length (default: 256)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:block_stride, pos_integer()}
          | {:block_window, pos_integer()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:selection_ratio, float()}
          | {:seq_len, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    seq_len = Keyword.get(opts, :seq_len, 256)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        seq_len: seq_len,
        block_builder: fn input, block_opts ->
          layer_idx = Keyword.get(block_opts, :layer_idx, 1)

          build_spla_block(input,
            hidden_size: hidden_size,
            num_heads: Keyword.get(opts, :num_heads, @default_num_heads),
            head_dim: Keyword.get(opts, :head_dim, @default_head_dim),
            block_stride: Keyword.get(opts, :block_stride, @default_block_stride),
            block_window: Keyword.get(opts, :block_window, @default_block_window),
            selection_ratio: Keyword.get(opts, :selection_ratio, @default_selection_ratio),
            dropout: dropout,
            name: "spla_block_#{layer_idx}"
          )
        end
      )
    )
  end

  @doc """
  Build a single SPLA transformer block.

  Pre-norm attention + FFN with residual connections.
  """
  @spec build_spla_block(Axon.t(), keyword()) :: Axon.t()
  def build_spla_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "spla_block")

    # 1. Attention sublayer: norm -> SPLA attention -> residual
    attn_normed = Axon.layer_norm(input, name: "#{name}_attn_norm")
    attn_out = build_spla_attention(attn_normed, opts)
    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # 2. FFN sublayer: norm -> FFN -> residual
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")
    ffn_out = FFN.layer(ffn_normed, hidden_size: hidden_size, name: "#{name}_ffn")
    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Build the SPLA attention layer.

  Combines block-sparse exact attention on selected blocks with residual
  linear attention on unselected blocks via the subtraction trick.
  """
  @spec build_spla_attention(Axon.t(), keyword()) :: Axon.t()
  def build_spla_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    block_stride = Keyword.get(opts, :block_stride, @default_block_stride)
    block_window = Keyword.get(opts, :block_window, @default_block_window)
    selection_ratio = Keyword.get(opts, :selection_ratio, @default_selection_ratio)
    name = Keyword.get(opts, :name, "spla_attn")

    inner_dim = num_heads * head_dim

    # Project Q, K, V
    q = Axon.dense(input, inner_dim, name: "#{name}_q_proj")
    k = Axon.dense(input, inner_dim, name: "#{name}_k_proj")
    v = Axon.dense(input, inner_dim, name: "#{name}_v_proj")

    # SPLA attention computation
    attn_out =
      Axon.layer(
        &spla_attention_fn/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        block_stride: block_stride,
        block_window: block_window,
        selection_ratio: selection_ratio
      )

    # Output projection
    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  # ============================================================================
  # Core SPLA Attention
  # ============================================================================

  defp spla_attention_fn(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    block_stride = opts[:block_stride]
    _block_window = opts[:block_window]
    selection_ratio = opts[:selection_ratio]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to [batch, seq_len, num_heads, head_dim]
    q = Nx.reshape(q, {batch, seq_len, num_heads, head_dim})
    k = Nx.reshape(k, {batch, seq_len, num_heads, head_dim})
    v = Nx.reshape(v, {batch, seq_len, num_heads, head_dim})

    # Transpose to [batch, num_heads, seq_len, head_dim]
    q = Nx.transpose(q, axes: [0, 2, 1, 3])
    k = Nx.transpose(k, axes: [0, 2, 1, 3])
    v = Nx.transpose(v, axes: [0, 2, 1, 3])

    # Scale queries
    scale = Nx.rsqrt(Nx.tensor(head_dim, type: :f32))
    q = Nx.multiply(q, scale)

    # --- Path 1: Global linear attention (over all tokens) ---
    # Use elu(x) + 1 as feature kernel for linear attention
    phi_q = Nx.add(Nx.max(q, Nx.tensor(0.0)), Nx.tensor(1.0))
    phi_k = Nx.add(Nx.max(k, Nx.tensor(0.0)), Nx.tensor(1.0))

    # S = phi(K)^T @ V: [batch, heads, head_dim, head_dim]
    s_global =
      Nx.dot(
        Nx.transpose(phi_k, axes: [0, 1, 3, 2]),
        [3],
        [0, 1],
        v,
        [2],
        [0, 1]
      )

    # z = sum(phi(K), dim=seq): [batch, heads, head_dim]
    z_global = Nx.sum(phi_k, axes: [2])

    # o_global = (phi(Q) @ S) / (phi(Q) @ z): [batch, heads, seq_len, head_dim]
    numer_global = Nx.dot(phi_q, [3], [0, 1], s_global, [2], [0, 1])
    denom_global = Nx.dot(phi_q, [3], [0, 1], Nx.new_axis(z_global, 3), [2], [0, 1])
    denom_global = Nx.max(denom_global, Nx.tensor(1.0e-6))
    o_global = Nx.divide(numer_global, denom_global)

    # --- Path 2: Block scoring and selection ---
    # Partition into blocks and compute statistics
    num_blocks = div(seq_len, block_stride)
    num_selected = max(round(num_blocks * selection_ratio), 1)

    # Reshape K into blocks: [batch, heads, num_blocks, block_stride, head_dim]
    # Note: Using block_stride (not block_window) for non-overlapping partitions
    k_blocks = Nx.reshape(k, {batch, num_heads, num_blocks, block_stride, head_dim})
    v_blocks = Nx.reshape(v, {batch, num_heads, num_blocks, block_stride, head_dim})

    # Block mean: [batch, heads, num_blocks, head_dim]
    k_bar = Nx.mean(k_blocks, axes: [3])

    # 2nd-order Taylor selection metric
    # score_i = sum_q exp(q * k_bar_i) * (1 + 0.5 * variance_correction)
    # Use mean Q across sequence for block scoring
    q_mean = Nx.mean(q, axes: [2])

    # First-order: dot(q_mean, k_bar) -> [batch, heads, num_blocks]
    first_order =
      Nx.dot(
        Nx.new_axis(q_mean, 2),
        [3],
        [0, 1],
        Nx.transpose(k_bar, axes: [0, 1, 3, 2]),
        [2],
        [0, 1]
      )
      |> Nx.squeeze(axes: [2])

    # Block variance (trace of covariance) for 2nd-order correction
    k_var = Nx.variance(k_blocks, axes: [3])
    # Trace: sum of variances across head_dim -> [batch, heads, num_blocks]
    var_trace = Nx.sum(k_var, axes: [3])

    # Score = exp(first_order) * (1 + 0.5 * var_trace)
    block_scores = Nx.multiply(Nx.exp(first_order), Nx.add(1.0, Nx.multiply(0.5, var_trace)))

    # Soft top-k selection: use sigmoid on centered scores for differentiability
    # (Axon static graph can't do dynamic gather, so we use soft masking)
    score_threshold = compute_soft_topk_threshold(block_scores, num_selected)
    # Steep sigmoid for near-binary selection
    selection_mask =
      Nx.sigmoid(Nx.multiply(10.0, Nx.subtract(block_scores, score_threshold)))

    # --- Path 3: Sparse exact attention on selected blocks ---
    # Weight blocks by selection mask
    # selection_mask: [batch, heads, num_blocks] -> expand for broadcasting
    mask_expanded = Nx.reshape(selection_mask, {batch, num_heads, num_blocks, 1, 1})
    k_selected = Nx.multiply(k_blocks, mask_expanded)
    v_selected = Nx.multiply(v_blocks, mask_expanded)

    # Flatten selected blocks back: [batch, heads, seq_len, head_dim]
    k_sel_flat = Nx.reshape(k_selected, {batch, num_heads, seq_len, head_dim})
    v_sel_flat = Nx.reshape(v_selected, {batch, num_heads, seq_len, head_dim})

    # Exact softmax attention on selected (masked) blocks
    # scores: [batch, heads, seq_len, seq_len]
    attn_scores =
      Nx.dot(q, [3], [0, 1], Nx.transpose(k_sel_flat, axes: [0, 1, 3, 2]), [2], [0, 1])

    # Apply causal mask
    causal = build_causal_mask(seq_len)
    attn_scores = Nx.add(attn_scores, causal)

    attn_weights =
      Nx.exp(Nx.subtract(attn_scores, Nx.reduce_max(attn_scores, axes: [3], keep_axes: true)))

    attn_weights =
      Nx.divide(attn_weights, Nx.add(Nx.sum(attn_weights, axes: [3], keep_axes: true), 1.0e-6))

    o_sparse = Nx.dot(attn_weights, [3], [0, 1], v_sel_flat, [2], [0, 1])

    # --- Linear attention on selected blocks (for subtraction) ---
    phi_k_sel = Nx.add(Nx.max(k_sel_flat, Nx.tensor(0.0)), Nx.tensor(1.0))

    s_selected =
      Nx.dot(
        Nx.transpose(phi_k_sel, axes: [0, 1, 3, 2]),
        [3],
        [0, 1],
        v_sel_flat,
        [2],
        [0, 1]
      )

    z_selected = Nx.sum(phi_k_sel, axes: [2])

    numer_selected = Nx.dot(phi_q, [3], [0, 1], s_selected, [2], [0, 1])
    denom_selected = Nx.dot(phi_q, [3], [0, 1], Nx.new_axis(z_selected, 3), [2], [0, 1])
    denom_selected = Nx.max(denom_selected, Nx.tensor(1.0e-6))
    o_linear_selected = Nx.divide(numer_selected, denom_selected)

    # --- Combine: o = o_sparse + (o_global - o_linear_selected) ---
    o_rla = Nx.subtract(o_global, o_linear_selected)
    output = Nx.add(o_sparse, o_rla)

    # Transpose back: [batch, heads, seq_len, head_dim] -> [batch, seq_len, heads*head_dim]
    output = Nx.transpose(output, axes: [0, 2, 1, 3])
    Nx.reshape(output, {batch, seq_len, num_heads * head_dim})
  end

  # Compute an approximate threshold for soft top-k selection.
  # Returns a threshold such that approximately num_selected blocks score above it.
  defp compute_soft_topk_threshold(scores, num_selected) do
    # Sort scores descending, pick the num_selected-th value as threshold
    sorted = Nx.sort(scores, axis: 2, direction: :desc)
    # Index the num_selected-th element (0-indexed: num_selected - 1)
    idx = max(num_selected - 1, 0)
    Nx.slice_along_axis(sorted, idx, 1, axis: 2)
  end

  defp build_causal_mask(seq_len) do
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    mask = Nx.select(Nx.greater_equal(rows, cols), Nx.tensor(0.0), Nx.tensor(-1.0e9))
    Nx.reshape(mask, {1, 1, seq_len, seq_len})
  end

  defp maybe_dropout(x, rate, name) when rate > 0, do: Axon.dropout(x, rate: rate, name: name)
  defp maybe_dropout(x, _rate, _name), do: x

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an SPLA model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
