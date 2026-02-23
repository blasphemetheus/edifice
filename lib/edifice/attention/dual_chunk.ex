defmodule Edifice.Attention.DualChunk do
  @moduledoc """
  Dual Chunk Attention — context extension via intra-chunk and inter-chunk attention.

  DeepSeek's method for handling long sequences (used in Qwen2.5-128K): splits sequences
  into fixed-size chunks, computes standard attention within each chunk, then uses a
  separate mechanism for attending across chunk summaries. This reduces peak memory
  from O(seq^2) to O(chunk^2 + num_chunks^2).

  ## Key Innovation

  Instead of computing full quadratic attention over long sequences:
  1. **Intra-chunk attention**: Standard multi-head attention within each chunk (local patterns)
  2. **Inter-chunk attention**: Attention over chunk summaries (global context)
  3. **Combination**: Learnable blending of local and global representations

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Input projection to hidden_size
        |
  +----------------------------------------------+
  |   Dual Chunk Attention Block (x num_layers)  |
  |                                              |
  |   LayerNorm -> Dual Chunk Attention          |
  |     Split into chunks [batch, num_chunks, chunk_size, hidden]
  |     Intra-chunk: MultiHead per chunk         |
  |     Inter-chunk: Summarize -> Attend -> Expand
  |     Combine: gate * inter + (1-gate) * intra |
  |   -> Residual                                |
  |   LayerNorm -> FFN -> Residual               |
  +----------------------------------------------+
        |
  Final LayerNorm
        |
  Last timestep -> [batch, hidden_size]
  ```

  ## Memory Complexity

  For seq_len = N and chunk_size = C:
  - Standard attention: O(N^2)
  - Dual Chunk: O((N/C) * C^2 + (N/C)^2) = O(N*C + N^2/C^2)

  With C = sqrt(N), this becomes O(N^1.5) — subquadratic!

  ## Usage

      model = DualChunk.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 8,
        num_layers: 4,
        chunk_size: 64
      )

  ## Constraints

  `seq_len` must be divisible by `chunk_size`.

  ## References

  - DeepSeek context extension methods (2024)
  - Qwen2.5 long-context training (Alibaba, 2024)
  - "Efficient Long-Range Transformers" survey literature
  """

  import Nx.Defn

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 8
  @default_num_layers 4
  @default_chunk_size 64
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:chunk_size, pos_integer()}
          | {:dropout, float()}

  @doc """
  Build a Dual Chunk Attention model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 8)
    - `:num_layers` - Number of Dual Chunk Attention blocks (default: 4)
    - `:chunk_size` - Chunk size C for chunked attention (default: 64).
      `seq_len` must be divisible by this value.
    - `:dropout` - Dropout rate (default: 0.1)
    - `:seq_len` / `:window_size` - Expected sequence length (default: 64)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    chunk_size = Keyword.get(opts, :chunk_size, @default_chunk_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    head_dim = div(hidden_size, num_heads)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "dual_chunk_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_dual_chunk_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              head_dim: head_dim,
              chunk_size: chunk_size,
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
  Build the dual chunk attention sublayer.

  This creates the core attention mechanism with both intra-chunk and inter-chunk
  attention pathways.
  """
  @spec build_dual_chunk_attention(Axon.t(), keyword()) :: Axon.t()
  def build_dual_chunk_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    chunk_size = Keyword.fetch!(opts, :chunk_size)
    name = Keyword.get(opts, :name, "dual_chunk_attn")

    # QKV projection
    qkv = Axon.dense(input, hidden_size * 3, name: "#{name}_qkv")

    # Learnable gate for combining intra and inter chunk outputs
    gate = Axon.param("#{name}_gate", {num_heads, head_dim}, initializer: :zeros)

    # Dual chunk attention computation
    output =
      Axon.layer(
        &dual_chunk_attention_impl/3,
        [qkv, gate],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        chunk_size: chunk_size,
        op_name: :dual_chunk_attention
      )

    # Output projection
    Axon.dense(output, hidden_size, name: "#{name}_out")
  end

  defnp dual_chunk_attention_impl(qkv, gate, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    chunk_size = opts[:chunk_size]
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

    num_chunks = div(seq_len, chunk_size)

    # Reshape to chunks: [batch, heads, num_chunks, chunk_size, head_dim]
    q_chunks = Nx.reshape(q, {batch, num_heads, num_chunks, chunk_size, head_dim})
    k_chunks = Nx.reshape(k, {batch, num_heads, num_chunks, chunk_size, head_dim})
    v_chunks = Nx.reshape(v, {batch, num_heads, num_chunks, chunk_size, head_dim})

    # === Intra-chunk attention: standard softmax attention within each chunk ===
    intra_output = intra_chunk_attention(q_chunks, k_chunks, v_chunks, head_dim)

    # === Inter-chunk attention: attend over chunk summaries ===
    inter_output = inter_chunk_attention(q_chunks, k_chunks, v_chunks, head_dim)

    # === Combine outputs with learnable gate ===
    combined = combine_chunk_outputs(intra_output, inter_output, gate, batch, num_heads, num_chunks, chunk_size, head_dim)

    # Reshape back: [batch, heads, num_chunks, chunk_size, head_dim] -> [batch, seq_len, hidden_size]
    combined
    |> Nx.reshape({batch, num_heads, seq_len, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Apply standard attention within each chunk.
  # Takes Q, K, V tensors of shape [batch, heads, num_chunks, chunk_size, head_dim]
  # and computes independent attention for each chunk.
  defnp intra_chunk_attention(q_chunks, k_chunks, v_chunks, head_dim) do
    scale = Nx.rsqrt(Nx.as_type(head_dim, Nx.type(q_chunks)))

    # Attention scores within each chunk: [batch, heads, num_chunks, chunk_size, chunk_size]
    scores =
      Nx.dot(q_chunks, [4], [0, 1, 2], k_chunks, [4], [0, 1, 2])
      |> Nx.multiply(scale)

    # Causal mask within each chunk
    chunk_size = Nx.axis_size(q_chunks, 3)
    row_idx = Nx.iota({chunk_size, chunk_size}, axis: 0)
    col_idx = Nx.iota({chunk_size, chunk_size}, axis: 1)
    causal_mask_2d = Nx.greater_equal(row_idx, col_idx)
    causal_mask = Nx.broadcast(causal_mask_2d, Nx.shape(scores), axes: [3, 4])

    scores =
      Nx.select(
        causal_mask,
        scores,
        Nx.broadcast(-1.0e9, Nx.shape(scores))
      )

    # Numerically stable softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(scores - max_scores)
    attn_weights = exp_scores / (Nx.sum(exp_scores, axes: [-1], keep_axes: true) + 1.0e-8)

    # Output: [batch, heads, num_chunks, chunk_size, head_dim]
    Nx.dot(attn_weights, [4], [0, 1, 2], v_chunks, [3], [0, 1, 2])
  end

  # Apply attention over chunk summaries for global context.
  # Summarizes each chunk (mean pooling), applies attention across summaries,
  # then broadcasts back to full chunk size.
  defnp inter_chunk_attention(q_chunks, k_chunks, v_chunks, head_dim) do
    batch = Nx.axis_size(q_chunks, 0)
    num_heads = Nx.axis_size(q_chunks, 1)
    num_chunks = Nx.axis_size(q_chunks, 2)
    chunk_size = Nx.axis_size(q_chunks, 3)

    scale = Nx.rsqrt(Nx.as_type(head_dim, Nx.type(q_chunks)))

    # Summarize chunks via mean pooling: [batch, heads, num_chunks, head_dim]
    q_summary = Nx.mean(q_chunks, axes: [3])
    k_summary = Nx.mean(k_chunks, axes: [3])
    v_summary = Nx.mean(v_chunks, axes: [3])

    # Attention over chunk summaries: [batch, heads, num_chunks, num_chunks]
    scores =
      Nx.dot(q_summary, [3], [0, 1], k_summary, [3], [0, 1])
      |> Nx.multiply(scale)

    # Causal mask for inter-chunk attention (chunk i can only attend to chunks 0..i)
    row_idx = Nx.iota({num_chunks, num_chunks}, axis: 0)
    col_idx = Nx.iota({num_chunks, num_chunks}, axis: 1)
    causal_mask_2d = Nx.greater_equal(row_idx, col_idx)
    causal_mask = Nx.broadcast(causal_mask_2d, {batch, num_heads, num_chunks, num_chunks}, axes: [2, 3])

    scores =
      Nx.select(
        causal_mask,
        scores,
        Nx.broadcast(-1.0e9, {batch, num_heads, num_chunks, num_chunks})
      )

    # Softmax
    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(scores - max_scores)
    attn_weights = exp_scores / (Nx.sum(exp_scores, axes: [-1], keep_axes: true) + 1.0e-8)

    # Weighted sum of value summaries: [batch, heads, num_chunks, head_dim]
    inter_summary = Nx.dot(attn_weights, [3], [0, 1], v_summary, [2], [0, 1])

    # Broadcast back to chunk size: [batch, heads, num_chunks, chunk_size, head_dim]
    inter_summary
    |> Nx.new_axis(3)
    |> Nx.broadcast({batch, num_heads, num_chunks, chunk_size, head_dim})
  end

  # Combine intra-chunk and inter-chunk outputs with learnable gating.
  # Uses a sigmoid gate to blend local (intra) and global (inter) representations:
  # output = gate * inter + (1 - gate) * intra
  defnp combine_chunk_outputs(intra, inter, gate, batch, num_heads, num_chunks, chunk_size, head_dim) do
    # gate: [num_heads, head_dim] -> broadcast to match output shape
    gate_sigmoid = Nx.sigmoid(gate)
    gate_broadcast =
      gate_sigmoid
      |> Nx.reshape({1, num_heads, 1, 1, head_dim})
      |> Nx.broadcast({batch, num_heads, num_chunks, chunk_size, head_dim})

    # Gated combination: gate * inter + (1 - gate) * intra
    Nx.add(
      Nx.multiply(gate_broadcast, inter),
      Nx.multiply(Nx.subtract(1.0, gate_broadcast), intra)
    )
  end

  @doc "Get the output size of the model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
