defmodule Edifice.Attention.LightningAttention do
  @moduledoc """
  Lightning Attention — hybrid linear/softmax block attention.

  Splits the sequence into fixed-size blocks and uses two complementary
  attention mechanisms:

  - **Intra-block:** Standard softmax attention within each block (O(B²) per block)
  - **Inter-block:** Linear attention via cumulative KV state across blocks (O(B·d) per block)

  This achieves near-linear overall complexity while retaining the expressivity
  of softmax attention at the local level.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Input Projection to hidden_size
        |
  +--------------------------------------------+
  |  Lightning Attention Block (x num_layers)  |
  |                                            |
  |  LayerNorm -> Q,K,V projections           |
  |  Reshape to [batch, heads, blocks, B, d]  |
  |                                            |
  |  Intra-block: softmax(Q_b @ K_b^T) @ V_b |
  |  Inter-block: Q_b @ cumsum(K_j^T V_j)    |
  |  Output = intra + inter                   |
  |                                            |
  |  -> Residual                              |
  |  LayerNorm -> FFN -> Residual             |
  +--------------------------------------------+
        |
  Final LayerNorm
        |
  Last timestep -> [batch, hidden_size]
  ```

  ## Constraints

  `seq_len` must be divisible by `block_size`.

  ## Usage

      model = LightningAttention.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 8,
        num_layers: 4,
        block_size: 64
      )

  ## References

  - Qin et al., "Lightning Attention-2: A Free Lunch for Handling Unlimited
    Sequence Lengths in Large Language Models" (2024)
  """

  import Nx.Defn

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 8
  @default_num_layers 4
  @default_block_size 64
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:block_size, pos_integer()}
          | {:dropout, float()}

  @doc """
  Build a Lightning Attention model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:num_layers` - Number of Lightning Attention blocks (default: 4)
    - `:block_size` - Block size B for chunked attention (default: 64).
      `seq_len` must be divisible by this value.
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` / `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    block_size = Keyword.get(opts, :block_size, @default_block_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    head_dim = div(hidden_size, num_heads)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "lightning_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_lightning_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              head_dim: head_dim,
              block_size: block_size,
              name: attn_name
            )
          end

          TransformerBlock.layer(input,
            attention_fn: attn_fn,
            hidden_size: hidden_size,
            norm: :layer_norm,
            ffn_type: :standard,
            dropout: dropout,
            name: name
          )
        end
      )
    )
  end

  @doc """
  Build the lightning attention sublayer.

  This creates the core attention mechanism with both intra-block (softmax)
  and inter-block (linear) attention pathways.
  """
  @spec build_lightning_attention(Axon.t(), keyword()) :: Axon.t()
  def build_lightning_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    block_size = Keyword.fetch!(opts, :block_size)
    name = Keyword.get(opts, :name, "lightning_attn")

    # QKV projection
    qkv = Axon.dense(input, hidden_size * 3, name: "#{name}_qkv")

    # Lightning attention computation
    output =
      Axon.layer(
        &lightning_attention_impl/2,
        [qkv],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        block_size: block_size,
        op_name: :lightning_attention
      )

    # Output projection
    Axon.dense(output, hidden_size, name: "#{name}_out")
  end

  defnp lightning_attention_impl(qkv, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    block_size = opts[:block_size]
    hidden_size = num_heads * head_dim

    batch = Nx.axis_size(qkv, 0)
    seq_len = Nx.axis_size(qkv, 1)

    # Split QKV
    q = Nx.slice_along_axis(qkv, 0, hidden_size, axis: 2)
    k = Nx.slice_along_axis(qkv, hidden_size, hidden_size, axis: 2)
    v = Nx.slice_along_axis(qkv, hidden_size * 2, hidden_size, axis: 2)

    # Reshape to [batch, num_heads, seq_len, head_dim]
    q = Nx.reshape(q, {batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = Nx.reshape(k, {batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = Nx.reshape(v, {batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    num_blocks = div(seq_len, block_size)

    # Reshape to blocks: [batch, heads, num_blocks, block_size, head_dim]
    q_blocks = Nx.reshape(q, {batch, num_heads, num_blocks, block_size, head_dim})
    k_blocks = Nx.reshape(k, {batch, num_heads, num_blocks, block_size, head_dim})
    v_blocks = Nx.reshape(v, {batch, num_heads, num_blocks, block_size, head_dim})

    scale = Nx.rsqrt(Nx.as_type(head_dim, Nx.type(q)))

    # === Intra-block: standard softmax attention within each block ===
    # scores: [batch, heads, num_blocks, block_size, block_size]
    scores =
      Nx.dot(q_blocks, [4], [0, 1, 2], k_blocks, [4], [0, 1, 2])
      |> Nx.multiply(scale)

    # Causal mask within each block
    # Build as [block_size, block_size] then broadcast to scores shape
    row_idx = Nx.iota({block_size, block_size}, axis: 0)
    col_idx = Nx.iota({block_size, block_size}, axis: 1)
    causal_mask_2d = Nx.greater_equal(row_idx, col_idx)
    causal_mask = Nx.broadcast(causal_mask_2d, Nx.shape(scores), axes: [3, 4])

    scores =
      Nx.select(
        causal_mask,
        scores,
        Nx.broadcast(-1.0e9, Nx.shape(scores))
      )

    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(scores - max_scores)
    attn_weights = exp_scores / (Nx.sum(exp_scores, axes: [-1], keep_axes: true) + 1.0e-8)

    # intra: [batch, heads, num_blocks, block_size, head_dim]
    intra = Nx.dot(attn_weights, [4], [0, 1, 2], v_blocks, [3], [0, 1, 2])

    # === Inter-block: linear attention via cumulative KV state ===
    # Compute KV outer product per block: K^T @ V → [batch, heads, num_blocks, head_dim, head_dim]
    kv_per_block =
      Nx.dot(
        Nx.transpose(k_blocks, axes: [0, 1, 2, 4, 3]),
        [4],
        [0, 1, 2],
        v_blocks,
        [3],
        [0, 1, 2]
      )

    # Cumulative sum of KV states (shifted by 1 — block i uses states from blocks 0..i-1)
    cumulative_kv = Nx.cumulative_sum(kv_per_block, axis: 2)

    # Shift: block 0 sees no history, block i sees cumsum of blocks 0..i-1
    zeros =
      Nx.broadcast(
        Nx.tensor(0.0, type: Nx.type(cumulative_kv)),
        {batch, num_heads, 1, head_dim, head_dim}
      )

    # Take cumsum up to second-to-last block
    shifted_cumulative = Nx.slice_along_axis(cumulative_kv, 0, num_blocks - 1, axis: 2)
    shifted_kv = Nx.concatenate([zeros, shifted_cumulative], axis: 2)

    # inter: Q_block @ cumulative_KV → [batch, heads, num_blocks, block_size, head_dim]
    inter = Nx.dot(q_blocks, [4], [0, 1, 2], shifted_kv, [3], [0, 1, 2])

    # Normalize inter-block output
    # Compute normalizer from K cumsum
    k_sum_per_block = Nx.sum(k_blocks, axes: [3])
    # [batch, heads, num_blocks, head_dim]
    cumulative_k_sum = Nx.cumulative_sum(k_sum_per_block, axis: 2)

    shifted_k_zeros =
      Nx.broadcast(
        Nx.tensor(0.0, type: Nx.type(cumulative_k_sum)),
        {batch, num_heads, 1, head_dim}
      )

    shifted_k_prefix = Nx.slice_along_axis(cumulative_k_sum, 0, num_blocks - 1, axis: 2)
    shifted_k_sum = Nx.concatenate([shifted_k_zeros, shifted_k_prefix], axis: 2)

    # normalizer: [batch, heads, num_blocks, block_size]
    normalizer =
      Nx.dot(q_blocks, [4], [0, 1, 2], Nx.new_axis(shifted_k_sum, 4), [3], [0, 1, 2])
      |> Nx.squeeze(axes: [4])
      |> Nx.add(1.0e-8)

    inter_normalized = inter / Nx.new_axis(normalizer, 4)

    # === Combine ===
    combined = intra + inter_normalized

    # Reshape back: [batch, heads, num_blocks, block_size, head_dim] -> [batch, seq_len, hidden_size]
    combined
    |> Nx.reshape({batch, num_heads, seq_len, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  @doc "Get the output size of the model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
