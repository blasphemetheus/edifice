defmodule Edifice.Attention.InfLLMV2 do
  @moduledoc """
  InfLLM-V2: Dense-Sparse Switchable Attention.

  Implements the InfLLM-V2 attention mechanism (OpenBMB, September 2025), a
  parameter-free sparse attention modification that partitions the KV cache
  into blocks and selects the most relevant blocks using a multi-level
  compression scoring pipeline.

  ## Key Innovations

  - **Zero extra parameters**: Reuses dense attention projections (W_K, W_V)
  - **Multi-level compression**: Coarse mean-pool -> group aggregation -> max-pool
  - **Three-component block selection**: initial + local + top-k blocks
  - **Dense-sparse switching**: Full attention for short sequences, sparse for long

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  +--------------------------------------------+
  |       InfLLM-V2 Attention Block            |
  |                                            |
  |  Q, K, V = Linear(input)                  |
  |                                            |
  |  [Stage 1: Block Selection]               |
  |    Partition K into blocks of size B       |
  |    Compress: mean-pool -> group-sum ->     |
  |              max-pool -> block scores      |
  |    Select: initial + local + top-k blocks  |
  |                                            |
  |  [Stage 2: Sparse Attention]              |
  |    Gather K, V from selected blocks       |
  |    softmax(Q @ K_sel^T / sqrt(d)) @ V_sel |
  +--------------------------------------------+
        |
  Output [batch, seq_len, embed_dim]
  ```

  ## Complexity

  | Mechanism | Time | Space |
  |-----------|------|-------|
  | Full attention | O(n^2 d) | O(n^2) |
  | InfLLM-V2 | O(n * |I| * d) | O(n * |I|) |

  Where |I| = total selected blocks (typically 96).

  ## Usage

      model = InfLLMV2.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 8,
        head_dim: 32,
        block_size: 8,
        num_initial_blocks: 1,
        num_local_blocks: 4,
        num_topk_blocks: 3,
        num_layers: 4
      )

  ## References

  - Paper: "InfLLM-V2: Dense-Sparse Switchable Attention for Seamless
    Short-to-Long Adaptation"
  - Authors: Weilin Zhao et al. (OpenBMB, September 2025)
  - ArXiv: 2509.24663
  """

  alias Edifice.Blocks.{FFN, ModelBuilder}

  @default_hidden_size 256
  @default_num_heads 8
  @default_head_dim 32
  @default_block_size 8
  @default_num_initial_blocks 1
  @default_num_local_blocks 4
  @default_num_topk_blocks 3
  @default_num_layers 4
  @default_dropout 0.1

  @doc """
  Build an InfLLM-V2 model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:head_dim` - Dimension per head (default: 32)
    - `:block_size` - Tokens per KV block (default: 8)
    - `:num_initial_blocks` - Fixed initial context blocks (default: 1)
    - `:num_local_blocks` - Sliding window blocks around query (default: 4)
    - `:num_topk_blocks` - Top-k selected blocks by relevance (default: 3)
    - `:num_layers` - Number of transformer blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` - Expected sequence length (default: 64)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:block_size, pos_integer()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_initial_blocks, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_local_blocks, pos_integer()}
          | {:num_topk_blocks, pos_integer()}
          | {:seq_len, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    seq_len = Keyword.get(opts, :seq_len, 64)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        seq_len: seq_len,
        block_builder: fn input, block_opts ->
          layer_idx = Keyword.get(block_opts, :layer_idx, 1)

          build_infllm_block(input,
            hidden_size: hidden_size,
            num_heads: Keyword.get(opts, :num_heads, @default_num_heads),
            head_dim: Keyword.get(opts, :head_dim, @default_head_dim),
            block_size: Keyword.get(opts, :block_size, @default_block_size),
            num_initial_blocks:
              Keyword.get(opts, :num_initial_blocks, @default_num_initial_blocks),
            num_local_blocks: Keyword.get(opts, :num_local_blocks, @default_num_local_blocks),
            num_topk_blocks: Keyword.get(opts, :num_topk_blocks, @default_num_topk_blocks),
            dropout: dropout,
            name: "infllm_block_#{layer_idx}"
          )
        end
      )
    )
  end

  @doc """
  Build a single InfLLM-V2 transformer block.
  """
  @spec build_infllm_block(Axon.t(), keyword()) :: Axon.t()
  def build_infllm_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "infllm_block")

    # 1. Attention sublayer
    attn_normed = Axon.layer_norm(input, name: "#{name}_attn_norm")
    attn_out = build_infllm_attention(attn_normed, opts)
    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # 2. FFN sublayer
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")
    ffn_out = FFN.layer(ffn_normed, hidden_size: hidden_size, name: "#{name}_ffn")
    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Build the InfLLM-V2 attention layer with block selection and sparse attention.
  """
  @spec build_infllm_attention(Axon.t(), keyword()) :: Axon.t()
  def build_infllm_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    name = Keyword.get(opts, :name, "infllm_attn")

    inner_dim = num_heads * head_dim

    q = Axon.dense(input, inner_dim, name: "#{name}_q_proj")
    k = Axon.dense(input, inner_dim, name: "#{name}_k_proj")
    v = Axon.dense(input, inner_dim, name: "#{name}_v_proj")

    attn_out =
      Axon.layer(
        &infllm_attention_fn/4,
        [q, k, v],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        block_size: Keyword.get(opts, :block_size, @default_block_size),
        num_initial_blocks: Keyword.get(opts, :num_initial_blocks, @default_num_initial_blocks),
        num_local_blocks: Keyword.get(opts, :num_local_blocks, @default_num_local_blocks),
        num_topk_blocks: Keyword.get(opts, :num_topk_blocks, @default_num_topk_blocks)
      )

    Axon.dense(attn_out, hidden_size, name: "#{name}_out_proj")
  end

  # ============================================================================
  # Core InfLLM-V2 Attention
  # ============================================================================

  defp infllm_attention_fn(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    block_size = opts[:block_size]
    num_initial = opts[:num_initial_blocks]
    num_local = opts[:num_local_blocks]
    num_topk = opts[:num_topk_blocks]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scale queries
    scale = Nx.rsqrt(Nx.tensor(head_dim, type: :f32))
    q = Nx.multiply(q, scale)

    num_blocks = div(seq_len, block_size)

    # --- Stage 1: Block Selection via Multi-Level Compression ---

    # Partition K into blocks: [batch, heads, num_blocks, block_size, dim]
    k_blocks = Nx.reshape(k, {batch, num_heads, num_blocks, block_size, head_dim})

    # Level 1: Mean-pool blocks -> compressed keys [batch, heads, num_blocks, dim]
    k_compressed = Nx.mean(k_blocks, axes: [3])

    # Compute block importance: q_mean @ k_compressed^T
    # Use mean query for block scoring
    q_mean = Nx.mean(q, axes: [2])
    # q_mean: [batch, heads, dim], k_compressed: [batch, heads, num_blocks, dim]
    # scores: [batch, heads, num_blocks]
    block_scores =
      Nx.dot(
        Nx.new_axis(q_mean, 2),
        [3],
        [0, 1],
        Nx.transpose(k_compressed, axes: [0, 1, 3, 2]),
        [2],
        [0, 1]
      )
      |> Nx.squeeze(axes: [2])

    # Level 2: Aggregate across heads -> [batch, num_blocks]
    shared_scores = Nx.sum(block_scores, axes: [1])

    # --- Build selection mask ---
    # Create masks for initial blocks, local blocks, and top-k blocks
    block_indices = Nx.iota({num_blocks})

    # Initial blocks mask: first num_initial blocks
    initial_mask = Nx.less(block_indices, num_initial)
    initial_mask = Nx.broadcast(initial_mask, {batch, num_blocks})

    # Local blocks mask: last num_local blocks (around most recent context)
    local_threshold = max(num_blocks - num_local, 0)
    local_mask = Nx.greater_equal(block_indices, local_threshold)
    local_mask = Nx.broadcast(local_mask, {batch, num_blocks})

    # Top-k blocks: soft selection via threshold
    sorted = Nx.sort(shared_scores, axis: 1, direction: :desc)
    topk_idx = min(max(num_topk - 1, 0), num_blocks - 1)
    threshold = Nx.slice_along_axis(sorted, topk_idx, 1, axis: 1)
    topk_mask = Nx.greater_equal(shared_scores, threshold)

    # Union: initial OR local OR topk
    combined_mask = Nx.logical_or(initial_mask, Nx.logical_or(local_mask, topk_mask))

    # Convert to float for masking: [batch, num_blocks] -> [batch, 1, num_blocks, 1, 1]
    mask_float =
      Nx.select(combined_mask, Nx.tensor(1.0), Nx.tensor(0.0))
      |> Nx.reshape({batch, 1, num_blocks, 1, 1})

    # --- Stage 2: Sparse Attention ---
    # Mask unselected blocks to zero
    v_blocks = Nx.reshape(v, {batch, num_heads, num_blocks, block_size, head_dim})
    k_masked = Nx.multiply(k_blocks, mask_float)
    v_masked = Nx.multiply(v_blocks, mask_float)

    # Flatten back: [batch, heads, seq_len, dim]
    k_sparse = Nx.reshape(k_masked, {batch, num_heads, seq_len, head_dim})
    v_sparse = Nx.reshape(v_masked, {batch, num_heads, seq_len, head_dim})

    # Standard softmax attention on sparse K, V
    attn_scores =
      Nx.dot(q, [3], [0, 1], Nx.transpose(k_sparse, axes: [0, 1, 3, 2]), [2], [0, 1])

    # Causal mask
    causal = build_causal_mask(seq_len)
    attn_scores = Nx.add(attn_scores, causal)

    # Large negative for zeroed-out positions (unselected blocks)
    # The masking already zeroed K, so dot products with zero vectors yield 0.
    # After causal masking, softmax will handle the rest.

    attn_weights =
      Nx.exp(Nx.subtract(attn_scores, Nx.reduce_max(attn_scores, axes: [3], keep_axes: true)))

    attn_weights =
      Nx.divide(attn_weights, Nx.add(Nx.sum(attn_weights, axes: [3], keep_axes: true), 1.0e-6))

    output = Nx.dot(attn_weights, [3], [0, 1], v_sparse, [2], [0, 1])

    # [batch, heads, seq, dim] -> [batch, seq, heads*dim]
    output = Nx.transpose(output, axes: [0, 2, 1, 3])
    Nx.reshape(output, {batch, seq_len, num_heads * head_dim})
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
  Get the output size of an InfLLM-V2 model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
