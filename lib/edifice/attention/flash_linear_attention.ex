defmodule Edifice.Attention.FlashLinearAttention do
  @moduledoc """
  Flash Linear Attention — chunked linear attention with feature maps.

  Combines the efficiency of linear attention with block-wise computation
  for better hardware utilization, following the LightningAttention pattern
  but using explicit feature maps on Q and K.

  ## Key Innovation: Feature-Mapped Chunked Attention

  Unlike LightningAttention which uses raw QKV, FlashLinearAttention applies
  learnable feature maps (ELU+1, ReLU+eps, or identity) to Q and K before
  computing attention. This creates a true linear attention kernel while
  maintaining the chunked computation pattern for efficiency.

  - **Intra-chunk**: Quadratic attention on phi(Q), phi(K), V (causal masked)
  - **Inter-chunk**: Linear recurrence via cumulative `S_c = S_{c-1} + phi(K_c)^T @ V_c`

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  +---------------------------------------------------+
  |  Flash Linear Attention Block (x num_layers)       |
  |                                                     |
  |  LayerNorm → Q, K, V projections                   |
  |  phi(Q), phi(K) ← feature map (ELU+1/ReLU/id)     |
  |  Reshape to [batch, heads, chunks, chunk_size, d]  |
  |                                                     |
  |  Intra-chunk: phi(Q)·phi(K)^T · V (causal masked) |
  |  Inter-chunk: phi(Q) · cumsum(phi(K)^T · V)       |
  |  Output = intra + inter                            |
  |                                                     |
  |  → Residual → LayerNorm → FFN → Residual           |
  +---------------------------------------------------+
        |
  [batch, hidden_size]
  ```

  ## Feature Maps

  - `:elu` (default) — `1 + ELU(x)`: smooth, always positive, good gradients
  - `:relu` — `ReLU(x) + eps`: sparse but simple
  - `:identity` — `x`: no transformation (equivalent to raw linear attention)

  ## Constraints

  `seq_len` must be divisible by `chunk_size`.

  ## Usage

      model = FlashLinearAttention.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 4,
        chunk_size: 64,
        feature_map: :elu
      )

  ## References

  - "Flash Linear Attention" (Yang et al., 2024)
  - flash-linear-attention: https://github.com/fla-org/flash-linear-attention
  """

  import Nx.Defn

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 4
  @default_chunk_size 64
  @default_feature_map :elu
  @default_dropout 0.1

  @doc """
  Build a Flash Linear Attention model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of blocks (default: 4)
    - `:chunk_size` - Chunk size for block-wise attention (default: 64)
    - `:feature_map` - Feature map type: `:elu`, `:relu`, or `:identity` (default: `:elu`)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` / `:window_size` - Expected sequence length (default: 64)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:chunk_size, pos_integer()}
          | {:feature_map, :elu | :relu | :identity}
          | {:dropout, float()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    feature_map = Keyword.get(opts, :feature_map, @default_feature_map)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    head_dim = div(hidden_size, num_heads)

    # Default seq_len to chunk_size if not specified (must be divisible)
    opts =
      Keyword.put_new(opts, :seq_len, Keyword.get(opts, :window_size, chunk_size))

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "fla_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_flash_linear_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              head_dim: head_dim,
              chunk_size: chunk_size,
              feature_map: feature_map,
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

  defp build_flash_linear_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    chunk_size = Keyword.fetch!(opts, :chunk_size)
    feature_map = Keyword.fetch!(opts, :feature_map)
    name = Keyword.get(opts, :name, "fla")

    # QKV projection
    qkv = Axon.dense(input, hidden_size * 3, name: "#{name}_qkv")

    # Flash linear attention computation
    output =
      Axon.layer(
        &flash_linear_attention_impl/2,
        [qkv],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        chunk_size: chunk_size,
        feature_map: feature_map,
        op_name: :flash_linear_attention
      )

    Axon.dense(output, hidden_size, name: "#{name}_out")
  end

  defnp flash_linear_attention_impl(qkv, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    chunk_size = opts[:chunk_size]
    feature_map = opts[:feature_map]
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

    # Apply feature maps to Q and K
    q = apply_feature_map(q, feature_map)
    k = apply_feature_map(k, feature_map)

    num_chunks = div(seq_len, chunk_size)

    # Reshape to chunks: [batch, heads, num_chunks, chunk_size, head_dim]
    q_chunks = Nx.reshape(q, {batch, num_heads, num_chunks, chunk_size, head_dim})
    k_chunks = Nx.reshape(k, {batch, num_heads, num_chunks, chunk_size, head_dim})
    v_chunks = Nx.reshape(v, {batch, num_heads, num_chunks, chunk_size, head_dim})

    # === Intra-chunk: quadratic attention with causal mask ===
    # scores: [batch, heads, num_chunks, chunk_size, chunk_size]
    scores = Nx.dot(q_chunks, [4], [0, 1, 2], k_chunks, [4], [0, 1, 2])

    # Causal mask within each chunk
    row_idx = Nx.iota({chunk_size, chunk_size}, axis: 0)
    col_idx = Nx.iota({chunk_size, chunk_size}, axis: 1)
    causal_mask_2d = Nx.greater_equal(row_idx, col_idx)
    causal_mask = Nx.broadcast(causal_mask_2d, Nx.shape(scores), axes: [3, 4])

    scores =
      Nx.select(
        causal_mask,
        scores,
        Nx.broadcast(Nx.tensor(0.0, type: Nx.type(scores)), Nx.shape(scores))
      )

    # Normalize: divide by sum of attention weights
    score_sums = Nx.sum(scores, axes: [-1], keep_axes: true) |> Nx.add(1.0e-8)
    attn_weights = scores / score_sums

    # intra: [batch, heads, num_chunks, chunk_size, head_dim]
    intra = Nx.dot(attn_weights, [4], [0, 1, 2], v_chunks, [3], [0, 1, 2])

    # === Inter-chunk: linear recurrence via cumulative KV ===
    # KV per chunk: phi(K)^T @ V -> [batch, heads, num_chunks, head_dim, head_dim]
    kv_per_chunk =
      Nx.dot(
        Nx.transpose(k_chunks, axes: [0, 1, 2, 4, 3]),
        [4],
        [0, 1, 2],
        v_chunks,
        [3],
        [0, 1, 2]
      )

    # Cumulative sum (shifted by 1)
    cumulative_kv = Nx.cumulative_sum(kv_per_chunk, axis: 2)
    zeros = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(cumulative_kv)), {batch, num_heads, 1, head_dim, head_dim})
    shifted_cumulative = Nx.slice_along_axis(cumulative_kv, 0, num_chunks - 1, axis: 2)
    shifted_kv = Nx.concatenate([zeros, shifted_cumulative], axis: 2)

    # inter: Q_chunk @ cumulative_KV -> [batch, heads, num_chunks, chunk_size, head_dim]
    inter = Nx.dot(q_chunks, [4], [0, 1, 2], shifted_kv, [3], [0, 1, 2])

    # Normalize inter-chunk
    k_sum_per_chunk = Nx.sum(k_chunks, axes: [3])
    cumulative_k_sum = Nx.cumulative_sum(k_sum_per_chunk, axis: 2)
    shifted_k_zeros = Nx.broadcast(Nx.tensor(0.0, type: Nx.type(cumulative_k_sum)), {batch, num_heads, 1, head_dim})
    shifted_k_prefix = Nx.slice_along_axis(cumulative_k_sum, 0, num_chunks - 1, axis: 2)
    shifted_k_sum = Nx.concatenate([shifted_k_zeros, shifted_k_prefix], axis: 2)

    normalizer =
      Nx.dot(q_chunks, [4], [0, 1, 2], Nx.new_axis(shifted_k_sum, 4), [3], [0, 1, 2])
      |> Nx.squeeze(axes: [4])
      |> Nx.add(1.0e-8)

    inter_normalized = inter / Nx.new_axis(normalizer, 4)

    # === Combine ===
    combined = intra + inter_normalized

    # Reshape back: [batch, seq_len, hidden_size]
    combined
    |> Nx.reshape({batch, num_heads, seq_len, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Feature map implementations
  deftransformp apply_feature_map(x, feature_map) do
    case feature_map do
      :elu ->
        # 1 + ELU(x): smooth, always positive
        Nx.add(1.0, Nx.select(Nx.greater(x, 0.0), x, Nx.subtract(Nx.exp(x), 1.0)))

      :relu ->
        # ReLU(x) + eps: sparse positive features
        Nx.add(Nx.max(x, 0.0), 1.0e-6)

      :identity ->
        x
    end
  end

  @doc "Get the output size of the model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
