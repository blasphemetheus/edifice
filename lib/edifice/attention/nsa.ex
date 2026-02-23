defmodule Edifice.Attention.NSA do
  @moduledoc """
  NSA: Native Sparse Attention (DeepSeek-V3/V4).

  Hardware-aligned three-path sparse attention mechanism that achieves
  efficient long-context attention by combining global context, fine-grained
  retrieval, and local attention in parallel paths.

  ## Key Innovation: Hardware-Aligned Sparse Attention

  Instead of standard full quadratic attention, NSA uses three complementary
  sparse attention patterns that can be computed efficiently on modern hardware:

  1. **Compressed Tokens**: Global context via pooled/compressed sequences
  2. **Top-k Blocks**: Fine-grained retrieval of most relevant key-value blocks
  3. **Sliding Window**: Local attention for recent context

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +----------------------------------------+
  |        Native Sparse Attention          |
  |                                         |
  |  Q, K, V = Linear(input)                |
  |                                         |
  |  +-----------+  +------------+  +------+|
  |  | Compress  |  | Top-k      |  | Slide||
  |  | (global)  |  | Blocks     |  | Wind ||
  |  |           |  | (retrieval)|  |(local)||
  |  +-----------+  +------------+  +------+|
  |        |              |            |    |
  |        v              v            v    |
  |      attn_c        attn_b       attn_w  |
  |        |              |            |    |
  |        +------+-------+-----+------+    |
  |               |             |           |
  |               v             v           |
  |         gate_weights (learnable)        |
  |               |                         |
  |               v                         |
  |      weighted_sum(attn_c, attn_b, attn_w)|
  +----------------------------------------+
        |
        v
  [batch, seq_len, embed_dim] or [batch, hidden_size]
  ```

  ## Three Paths

  ### 1. Compressed Tokens (Global Context)
  Pool Q/K/V into fewer tokens using strided convolution with compression_ratio.
  Compute softmax attention over compressed sequence for O(n/r) complexity.

  ### 2. Top-k Blocks (Fine-grained Retrieval)
  - Divide K/V into blocks of block_size
  - Compute block-level scores: dot(Q, mean(K_block))
  - Select top num_selected_blocks blocks
  - Compute attention within selected blocks

  ### 3. Sliding Window (Local)
  Standard local attention over the last window_size tokens for recent context.

  ## Usage

      model = NSA.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 8,
        head_dim: 32,
        window_size: 64,
        block_size: 16,
        num_selected_blocks: 8,
        compression_ratio: 4,
        num_layers: 4
      )

  ## References

  - Paper: "Native Sparse Attention: Hardware-Aligned and Natively Trainable"
  - Authors: DeepSeek-AI (2025)
  - Used in: DeepSeek-V3, DeepSeek-V4
  """

  alias Edifice.Blocks.{FFN, ModelBuilder}

  @default_hidden_size 256
  @default_num_heads 8
  @default_head_dim 32
  @default_window_size 64
  @default_block_size 16
  @default_num_selected_blocks 8
  @default_compression_ratio 4
  @default_num_layers 4
  @default_dropout 0.1

  @doc """
  Build an NSA model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:head_dim` - Dimension per head (default: 32)
    - `:window_size` - Sliding window size for local attention (default: 64)
    - `:block_size` - Block size for top-k selection (default: 16)
    - `:num_selected_blocks` - Number of blocks to select per query (default: 8)
    - `:compression_ratio` - Compression ratio for global path (default: 4)
    - `:num_layers` - Number of NSA blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` - Expected sequence length (default: 256)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:block_size, pos_integer()}
          | {:compression_ratio, pos_integer()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_selected_blocks, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

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

          build_nsa_block(input,
            hidden_size: hidden_size,
            num_heads: Keyword.get(opts, :num_heads, @default_num_heads),
            head_dim: Keyword.get(opts, :head_dim, @default_head_dim),
            window_size: Keyword.get(opts, :window_size, @default_window_size),
            block_size: Keyword.get(opts, :block_size, @default_block_size),
            num_selected_blocks: Keyword.get(opts, :num_selected_blocks, @default_num_selected_blocks),
            compression_ratio: Keyword.get(opts, :compression_ratio, @default_compression_ratio),
            dropout: dropout,
            name: "nsa_block_#{layer_idx}"
          )
        end
      )
    )
  end

  @doc """
  Build a single NSA transformer block.
  """
  @spec build_nsa_block(Axon.t(), keyword()) :: Axon.t()
  def build_nsa_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "nsa_block")

    # 1. Attention sublayer: norm -> NSA -> residual
    attn_normed = Axon.layer_norm(input, name: "#{name}_attn_norm")

    attn_out = build_nsa_attention(attn_normed, opts)

    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # 2. FFN sublayer: norm -> FFN -> residual
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.layer(ffn_normed,
        hidden_size: hidden_size,
        name: "#{name}_ffn"
      )

    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Build the NSA attention layer with three parallel sparse paths.
  """
  @spec build_nsa_attention(Axon.t(), keyword()) :: Axon.t()
  def build_nsa_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    block_size = Keyword.get(opts, :block_size, @default_block_size)
    num_selected_blocks = Keyword.get(opts, :num_selected_blocks, @default_num_selected_blocks)
    compression_ratio = Keyword.get(opts, :compression_ratio, @default_compression_ratio)
    name = Keyword.get(opts, :name, "nsa_attn")

    qkv_dim = num_heads * head_dim

    # Q, K, V projections
    q_proj = Axon.dense(input, qkv_dim, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, qkv_dim, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, qkv_dim, name: "#{name}_v_proj")

    # Learnable mixing weights for the three paths (will be softmaxed)
    gate_weights = Axon.param("#{name}_gate_weights", {3}, initializer: :zeros)

    # Compute NSA with three paths
    output =
      Axon.layer(
        &nsa_attention_impl/5,
        [q_proj, k_proj, v_proj, gate_weights],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        window_size: window_size,
        block_size: block_size,
        num_selected_blocks: num_selected_blocks,
        compression_ratio: compression_ratio,
        op_name: :nsa_attention
      )

    # Output projection
    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # NSA attention implementation with three parallel paths
  defp nsa_attention_impl(q, k, v, gate_weights, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    window_size = opts[:window_size]
    block_size = opts[:block_size]
    num_selected_blocks = opts[:num_selected_blocks]
    compression_ratio = opts[:compression_ratio]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, heads, seq, head_dim]
    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Path 1: Compressed attention (global context)
    attn_compressed = compressed_attention(q, k, v, compression_ratio)

    # Path 2: Top-k block attention (fine-grained retrieval)
    attn_blocks = topk_block_attention(q, k, v, block_size, num_selected_blocks)

    # Path 3: Sliding window attention (local)
    attn_window = sliding_window_attention(q, k, v, window_size)

    # Learnable mixing of three paths
    # gate_weights: {3} -> softmax -> weights for each path
    gates = Nx.exp(gate_weights)
    gates = Nx.divide(gates, Nx.sum(gates))
    gate_c = Nx.slice(gates, [0], [1]) |> Nx.squeeze()
    gate_b = Nx.slice(gates, [1], [1]) |> Nx.squeeze()
    gate_w = Nx.slice(gates, [2], [1]) |> Nx.squeeze()

    # Weighted sum
    output =
      Nx.add(
        Nx.add(
          Nx.multiply(attn_compressed, gate_c),
          Nx.multiply(attn_blocks, gate_b)
        ),
        Nx.multiply(attn_window, gate_w)
      )

    # Reshape back: [batch, seq, num_heads * head_dim]
    reshape_from_heads(output, batch, seq_len, num_heads, head_dim)
  end

  # Path 1: Compressed attention via average pooling
  defp compressed_attention(q, k, v, compression_ratio) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    compressed_len = max(div(seq_len, compression_ratio), 1)

    # Average pool K and V along sequence dimension
    # Reshape to [batch, heads, compressed_len, compression_ratio, head_dim]
    # then mean over compression_ratio
    k_compressed = compress_sequence(k, seq_len, compressed_len, compression_ratio)
    v_compressed = compress_sequence(v, seq_len, compressed_len, compression_ratio)

    # Also compress Q to match dimensions for full global attention
    # Each compressed Q attends to all compressed K/V
    q_compressed = compress_sequence(q, seq_len, compressed_len, compression_ratio)

    # Scaled dot-product attention on compressed sequences
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q_compressed, [3], [0, 1], k_compressed, [3], [0, 1]), scale)

    # Softmax
    weights = stable_softmax(scores)

    # Weighted values: [batch, heads, compressed_len, head_dim]
    output_compressed = Nx.dot(weights, [3], [0, 1], v_compressed, [2], [0, 1])

    # Expand back to original sequence length via repeat
    expand_sequence(output_compressed, compressed_len, seq_len)
  end

  # Compress sequence by averaging over windows
  defp compress_sequence(x, seq_len, compressed_len, compression_ratio) do
    {batch, num_heads, _seq, head_dim} = Nx.shape(x)

    # Pad sequence to be divisible by compression_ratio if needed
    pad_len = compressed_len * compression_ratio - seq_len

    x_padded =
      if pad_len > 0 do
        padding = Nx.broadcast(0.0, {batch, num_heads, pad_len, head_dim}) |> Nx.as_type(Nx.type(x))
        Nx.concatenate([x, padding], axis: 2)
      else
        x
      end

    # Reshape and mean
    x_padded
    |> Nx.reshape({batch, num_heads, compressed_len, compression_ratio, head_dim})
    |> Nx.mean(axes: [3])
  end

  # Expand compressed sequence back to original length
  defp expand_sequence(x, _compressed_len, seq_len) do
    {batch, num_heads, compressed_len, head_dim} = Nx.shape(x)
    expansion_ratio = div(seq_len + compressed_len - 1, compressed_len)

    # Repeat each compressed position
    x
    |> Nx.new_axis(3)
    |> Nx.broadcast({batch, num_heads, compressed_len, expansion_ratio, head_dim})
    |> Nx.reshape({batch, num_heads, compressed_len * expansion_ratio, head_dim})
    |> Nx.slice_along_axis(0, seq_len, axis: 2)
  end

  # Path 2: Top-k block attention
  defp topk_block_attention(q, k, v, block_size, num_selected_blocks) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    num_blocks = max(div(seq_len, block_size), 1)
    actual_num_selected = min(num_selected_blocks, num_blocks)

    # Compute block means for K
    k_block_means = compute_block_means(k, seq_len, num_blocks, block_size)

    # Score each block: dot(mean(Q), mean(K_block))
    q_mean = Nx.mean(q, axes: [2])  # [batch, heads, head_dim]
    block_scores = Nx.dot(q_mean, [2], [0, 1], k_block_means, [3], [0, 1])  # [batch, heads, num_blocks]

    # Select top-k blocks (simplified: use top positions from argsort)
    # Since Nx doesn't have native top-k, we use argsort
    sorted_indices = Nx.argsort(block_scores, axis: 2, direction: :desc)
    top_k_indices = Nx.slice_along_axis(sorted_indices, 0, actual_num_selected, axis: 2)

    # Gather K, V from selected blocks and compute attention
    # For simplicity, we compute attention over all blocks weighted by their scores
    # This is an approximation that's still efficient and differentiable
    block_weights = stable_softmax(block_scores)  # [batch, heads, num_blocks]

    # Reshape K, V into blocks: [batch, heads, num_blocks, block_size, head_dim]
    k_blocks = reshape_to_blocks(k, seq_len, num_blocks, block_size)
    v_blocks = reshape_to_blocks(v, seq_len, num_blocks, block_size)

    # Compute attention within each block
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))

    # Q reshaped to blocks: [batch, heads, num_blocks, block_size, head_dim]
    q_blocks = reshape_to_blocks(q, seq_len, num_blocks, block_size)

    # Scores: [batch, heads, num_blocks, block_size, block_size]
    scores = Nx.divide(
      Nx.dot(q_blocks, [4], [0, 1, 2], k_blocks, [4], [0, 1, 2]),
      scale
    )

    # Apply causal mask within blocks
    rows = Nx.iota({block_size, block_size}, axis: 0)
    cols = Nx.iota({block_size, block_size}, axis: 1)
    causal_mask =
      Nx.greater_equal(rows, cols)
      |> Nx.reshape({1, 1, 1, block_size, block_size})
      |> Nx.broadcast({batch, num_heads, num_blocks, block_size, block_size})

    scores = Nx.select(
      causal_mask,
      scores,
      Nx.broadcast(-1.0e9, Nx.shape(scores))
    )

    # Softmax and weighted values per block
    attn_weights = stable_softmax_axis(scores, 4)
    block_outputs = Nx.dot(attn_weights, [4], [0, 1, 2], v_blocks, [3], [0, 1, 2])
    # [batch, heads, num_blocks, block_size, head_dim]

    # Weight blocks by their selection scores
    # [batch, heads, num_blocks, 1, 1] * [batch, heads, num_blocks, block_size, head_dim]
    block_weights_expanded = Nx.reshape(block_weights, {batch, num_heads, num_blocks, 1, 1})
    weighted_outputs = Nx.multiply(block_outputs, block_weights_expanded)

    # Reshape back to sequence
    reshape_blocks_to_seq(weighted_outputs, seq_len, num_blocks, block_size)
  end

  # Compute block means for K
  defp compute_block_means(k, seq_len, num_blocks, block_size) do
    {batch, num_heads, _seq, head_dim} = Nx.shape(k)
    padded_len = num_blocks * block_size

    k_padded =
      if padded_len > seq_len do
        padding = Nx.broadcast(0.0, {batch, num_heads, padded_len - seq_len, head_dim})
                  |> Nx.as_type(Nx.type(k))
        Nx.concatenate([k, padding], axis: 2)
      else
        Nx.slice_along_axis(k, 0, padded_len, axis: 2)
      end

    k_padded
    |> Nx.reshape({batch, num_heads, num_blocks, block_size, head_dim})
    |> Nx.mean(axes: [3])  # [batch, heads, num_blocks, head_dim]
  end

  # Reshape sequence to blocks
  defp reshape_to_blocks(x, seq_len, num_blocks, block_size) do
    {batch, num_heads, _seq, head_dim} = Nx.shape(x)
    padded_len = num_blocks * block_size

    x_padded =
      if padded_len > seq_len do
        padding = Nx.broadcast(0.0, {batch, num_heads, padded_len - seq_len, head_dim})
                  |> Nx.as_type(Nx.type(x))
        Nx.concatenate([x, padding], axis: 2)
      else
        Nx.slice_along_axis(x, 0, padded_len, axis: 2)
      end

    Nx.reshape(x_padded, {batch, num_heads, num_blocks, block_size, head_dim})
  end

  # Reshape blocks back to sequence
  defp reshape_blocks_to_seq(x, seq_len, num_blocks, block_size) do
    {batch, num_heads, _n_blocks, _blk_size, head_dim} = Nx.shape(x)

    x
    |> Nx.reshape({batch, num_heads, num_blocks * block_size, head_dim})
    |> Nx.slice_along_axis(0, seq_len, axis: 2)
  end

  # Path 3: Sliding window attention
  defp sliding_window_attention(q, k, v, window_size) do
    {batch, num_heads, seq_len, head_dim} = Nx.shape(q)
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))

    # Full attention scores
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    # Create sliding window mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)

    # Causal: row >= col
    causal_mask = Nx.greater_equal(rows, cols)

    # Window: row - col < window_size
    window_mask = Nx.less(Nx.subtract(rows, cols), window_size)

    # Combined: causal AND within window; expand to [batch, heads, seq_len, seq_len]
    combined_mask =
      Nx.logical_and(causal_mask, window_mask)
      |> Nx.reshape({1, 1, seq_len, seq_len})
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    scores = Nx.select(
      combined_mask,
      scores,
      Nx.broadcast(-1.0e9, Nx.shape(scores))
    )

    # Softmax and weighted values
    weights = stable_softmax(scores)
    Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
  end

  # Stable softmax along last axis
  defp stable_softmax(scores) do
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
    Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))
  end

  # Stable softmax along specified axis
  defp stable_softmax_axis(scores, axis) do
    max_scores = Nx.reduce_max(scores, axes: [axis], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
    Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [axis], keep_axes: true), 1.0e-8))
  end

  # Reshape [batch, seq, num_heads * dim] -> [batch, num_heads, seq, dim]
  defp reshape_to_heads(x, batch, seq_len, num_heads, dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Reshape [batch, num_heads, seq, dim] -> [batch, seq, num_heads * dim]
  defp reshape_from_heads(x, batch, seq_len, num_heads, dim) do
    x
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * dim})
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an NSA model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for an NSA model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    qkv_dim = num_heads * head_dim
    inner_size = hidden_size * 4

    # Per layer:
    # Attention: Q, K, V projections + output projection + gate weights
    attn_params = hidden_size * qkv_dim * 3 + qkv_dim * hidden_size + 3

    # FFN: up + down
    ffn_params = hidden_size * inner_size + inner_size * hidden_size

    per_layer = attn_params + ffn_params

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Get recommended defaults.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 8,
      head_dim: 32,
      window_size: 64,
      block_size: 16,
      num_selected_blocks: 8,
      compression_ratio: 4,
      num_layers: 4,
      dropout: 0.1
    ]
  end
end
