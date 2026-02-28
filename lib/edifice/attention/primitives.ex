defmodule Edifice.Attention.Primitives do
  @moduledoc """
  Pure tensor attention computations.

  Low-level building blocks for attention mechanisms — scaled dot-product,
  chunked, and memory-efficient (online softmax) variants in both 3D
  `[batch, seq, dim]` and 4D `[batch, heads, seq, head_dim]` forms.

  These functions operate on raw `Nx` tensors with no Axon graph construction,
  making them composable across attention modules (MultiHead, GQA,
  InfiniAttention, etc.).

  ## 3D vs 4D Functions

  - **3D** (`scaled_dot_product_attention/4`, `chunked_attention/4`,
    `memory_efficient_attention/4`): Operate on `[batch, seq, dim]` tensors.
    Suitable for single-head attention or when heads are already merged.

  - **4D** (`multi_head_sdpa/3`, `multi_head_chunked_attention/3`,
    `multi_head_memory_efficient_attention/3`): Operate on
    `[batch, heads, seq, head_dim]` tensors. Used inside multi-head attention
    layers after Q/K/V have been reshaped to separate heads.

  ## Head Reshaping

  `reshape_to_heads/5` and `reshape_from_heads/5` convert between the merged
  `[batch, seq, hidden]` and per-head `[batch, heads, seq, head_dim]` layouts.
  """

  alias Edifice.Utils.FusedOps

  # ============================================================================
  # 3D Attention (batch, seq, dim)
  # ============================================================================

  @doc """
  Scaled dot-product attention.

  Computes: softmax(QK^T / sqrt(d_k)) * V

  ## Parameters
    - `query` - Query tensor [batch, seq_q, dim]
    - `key` - Key tensor [batch, seq_k, dim]
    - `value` - Value tensor [batch, seq_k, dim]
    - `opts` - Options including :mask for causal/window masking

  ## Returns
    Attention output [batch, seq_q, dim]
  """
  @spec scaled_dot_product_attention(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def scaled_dot_product_attention(query, key, value, opts \\ []) do
    mask = opts[:mask]

    d_k = Nx.axis_size(key, -1)
    scale = Nx.sqrt(d_k) |> Nx.as_type(Nx.type(query))

    # Batched dot product: QK^T
    scores = Nx.dot(query, [2], [0], key, [2], [0])
    scores = Nx.divide(scores, scale)

    # Apply mask if provided
    scores =
      if mask do
        # Broadcast mask to match scores shape if needed
        mask =
          if tuple_size(Nx.shape(mask)) == 2 do
            # Mask is [seq, seq], need to broadcast to [batch, seq, seq]
            Nx.broadcast(Nx.new_axis(mask, 0), Nx.shape(scores))
          else
            mask
          end

        Nx.select(
          mask,
          scores,
          Nx.broadcast(-1.0e9, Nx.shape(scores))
        )
      else
        scores
      end

    # Numerically stable softmax (fused for better performance)
    weights = FusedOps.fused_softmax(scores)

    # Batched: weights @ value
    Nx.dot(weights, [2], [0], value, [1], [0])
  end

  @doc """
  Chunked attention for reduced peak memory usage.

  Processes query in chunks, computing attention for each chunk against all keys.
  This reduces peak memory from O(seq^2) to O(seq x chunk_size) while producing
  identical results to standard attention.

  ## Parameters
    - `query` - Query tensor [batch, seq_q, dim]
    - `key` - Key tensor [batch, seq_k, dim]
    - `value` - Value tensor [batch, seq_k, dim]
    - `opts` - Options:
      - `:chunk_size` - Size of query chunks (default: 32)
      - `:mask` - Attention mask (will be chunked automatically)

  ## Returns
    Attention output [batch, seq_q, dim] - identical to scaled_dot_product_attention
  """
  @spec chunked_attention(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def chunked_attention(query, key, value, opts \\ []) do
    chunk_size = Keyword.get(opts, :chunk_size, 32)
    mask = Keyword.get(opts, :mask)

    {batch, seq_q, dim} = Nx.shape(query)
    seq_k = Nx.axis_size(key, 1)
    d_k = dim
    scale = Nx.sqrt(d_k) |> Nx.as_type(Nx.type(query))

    # Calculate number of chunks
    num_chunks = div(seq_q + chunk_size - 1, chunk_size)

    # Process each chunk
    chunk_results =
      for chunk_idx <- 0..(num_chunks - 1) do
        # Calculate chunk boundaries
        start_idx = chunk_idx * chunk_size
        # Handle last chunk which may be smaller
        actual_chunk_size = min(chunk_size, seq_q - start_idx)

        # Extract query chunk: [batch, chunk_size, dim]
        q_chunk = Nx.slice_along_axis(query, start_idx, actual_chunk_size, axis: 1)

        # Compute attention scores for this chunk against all keys
        # scores: [batch, chunk_size, seq_k]
        scores = Nx.dot(q_chunk, [2], [0], key, [2], [0])
        scores = Nx.divide(scores, scale)

        # Apply mask for this chunk if provided
        scores =
          if mask do
            # Extract mask chunk: [chunk_size, seq_k] or [batch, chunk_size, seq_k]
            chunk_mask =
              case Nx.shape(mask) do
                {^seq_q, ^seq_k} ->
                  # 2D mask: [seq_q, seq_k] -> slice and broadcast
                  Nx.slice_along_axis(mask, start_idx, actual_chunk_size, axis: 0)
                  |> Nx.new_axis(0)
                  |> Nx.broadcast({batch, actual_chunk_size, seq_k})

                {^batch, ^seq_q, ^seq_k} ->
                  # 3D mask: [batch, seq_q, seq_k] -> slice
                  Nx.slice_along_axis(mask, start_idx, actual_chunk_size, axis: 1)

                _ ->
                  # Unexpected shape, try to broadcast
                  Nx.broadcast(mask, Nx.shape(scores))
              end

            Nx.select(
              chunk_mask,
              scores,
              Nx.broadcast(-1.0e9, Nx.shape(scores))
            )
          else
            scores
          end

        # Softmax over keys (last axis)
        weights = FusedOps.fused_softmax(scores)

        # Compute output: [batch, chunk_size, dim]
        Nx.dot(weights, [2], [0], value, [1], [0])
      end

    # Concatenate all chunks along sequence axis
    Nx.concatenate(chunk_results, axis: 1)
  end

  @doc """
  Memory-efficient attention using online softmax normalization.

  Achieves true O(n) memory by processing key/value in chunks with online
  softmax, never materializing the full attention matrix. Based on the
  algorithm from "Self-attention Does Not Need O(n^2) Memory" (Rabe & Staats, 2021).

  ## Parameters
    - `query` - Query tensor [batch, seq_q, dim]
    - `key` - Key tensor [batch, seq_k, dim]
    - `value` - Value tensor [batch, seq_k, dim]
    - `opts` - Options:
      - `:chunk_size` - Size of K/V chunks (default: 32)
      - `:causal` - Use causal masking (default: false)

  ## Returns
    Attention output [batch, seq_q, dim]
  """
  @spec memory_efficient_attention(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def memory_efficient_attention(query, key, value, opts \\ []) do
    chunk_size = Keyword.get(opts, :chunk_size, 32)
    causal = Keyword.get(opts, :causal, false)

    {batch, seq_q, dim} = Nx.shape(query)
    seq_k = Nx.axis_size(key, 1)
    scale = Nx.sqrt(dim) |> Nx.as_type(Nx.type(query))

    # Number of K/V chunks
    num_kv_chunks = div(seq_k + chunk_size - 1, chunk_size)

    # Initialize accumulators for online softmax
    tensor_type = Nx.type(query)
    neg_inf = Nx.Constants.neg_infinity() |> Nx.as_type(tensor_type)

    init_output = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, seq_q, dim})
    init_max = Nx.broadcast(neg_inf, {batch, seq_q})
    init_sum = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, seq_q})

    # Process each K/V chunk with online softmax
    {final_output, _final_max, final_sum} =
      Enum.reduce(0..(num_kv_chunks - 1), {init_output, init_max, init_sum}, fn chunk_idx,
                                                                                {acc_output,
                                                                                 acc_max, acc_sum} ->
        # Extract K/V chunk
        k_start = chunk_idx * chunk_size
        actual_chunk_size = min(chunk_size, seq_k - k_start)

        k_chunk = Nx.slice_along_axis(key, k_start, actual_chunk_size, axis: 1)
        v_chunk = Nx.slice_along_axis(value, k_start, actual_chunk_size, axis: 1)

        # Compute attention scores for this chunk: [batch, seq_q, chunk_size]
        scores = Nx.dot(query, [2], [0], k_chunk, [2], [0])
        scores = Nx.divide(scores, scale)

        # Apply causal masking if needed
        scores =
          if causal do
            # Create causal mask for this chunk
            q_positions = Nx.iota({seq_q, 1}, axis: 0)
            k_positions = Nx.iota({1, actual_chunk_size}, axis: 1) |> Nx.add(k_start)

            # Valid if key_pos <= query_pos
            causal_mask = Nx.greater_equal(q_positions, k_positions)

            causal_mask =
              Nx.broadcast(Nx.new_axis(causal_mask, 0), {batch, seq_q, actual_chunk_size})

            Nx.select(causal_mask, scores, Nx.broadcast(neg_inf, Nx.shape(scores)))
          else
            scores
          end

        # Online softmax update (numerically stable)
        # 1. Find new max
        chunk_max = Nx.reduce_max(scores, axes: [-1])
        new_max = Nx.max(acc_max, chunk_max)

        # 2. Rescale old accumulator to new max
        old_scale = Nx.exp(Nx.subtract(acc_max, new_max))

        # 3. Compute exp(scores - new_max) for this chunk
        exp_scores = Nx.exp(Nx.subtract(scores, Nx.new_axis(new_max, -1)))

        # 4. Sum of exp for this chunk
        chunk_sum = Nx.sum(exp_scores, axes: [-1])

        # 5. Update running sum
        new_sum = Nx.add(Nx.multiply(acc_sum, old_scale), chunk_sum)

        # 6. Compute weighted values for this chunk
        chunk_output = Nx.dot(exp_scores, [2], [0], v_chunk, [1], [0])

        # 7. Update output (rescale old output and add new)
        new_output =
          Nx.add(
            Nx.multiply(acc_output, Nx.new_axis(old_scale, -1)),
            chunk_output
          )

        {new_output, new_max, new_sum}
      end)

    # Final normalization: divide accumulated output by total sum
    Nx.divide(final_output, Nx.new_axis(final_sum, -1))
  end

  # ============================================================================
  # 4D Multi-Head Attention (batch, heads, seq, head_dim)
  # ============================================================================

  @doc """
  Multi-head scaled dot-product attention.

  Operates on 4D tensors where heads are a separate dimension, using batched
  dot products that contract over both batch and head axes simultaneously.

  ## Parameters
    - `query` - [batch, heads, seq_q, head_dim]
    - `key` - [batch, heads, seq_k, head_dim]
    - `value` - [batch, heads, seq_k, head_dim]
    - `opts` - Options:
      - `:mask` - Attention mask [seq, seq] or [batch, heads, seq, seq]

  ## Returns
    [batch, heads, seq_q, head_dim]
  """
  @spec multi_head_sdpa(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def multi_head_sdpa(query, key, value, opts \\ []) do
    mask = opts[:mask]

    d_k = Nx.axis_size(key, -1)
    scale = Nx.sqrt(d_k) |> Nx.as_type(Nx.type(query))

    # QK^T: [batch, heads, seq_q, seq_k]
    scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Apply mask if provided (mask is [seq, seq] — broadcast to [batch, heads, seq, seq])
    scores =
      if mask do
        mask_shape = Nx.shape(scores)

        mask =
          case tuple_size(Nx.shape(mask)) do
            2 ->
              # [seq, seq] -> broadcast to [batch, heads, seq, seq]
              mask |> Nx.new_axis(0) |> Nx.new_axis(0) |> Nx.broadcast(mask_shape)

            _ ->
              Nx.broadcast(mask, mask_shape)
          end

        Nx.select(mask, scores, Nx.broadcast(-1.0e9, mask_shape))
      else
        scores
      end

    weights = FusedOps.fused_softmax(scores)

    # weights @ V: [batch, heads, seq_q, head_dim]
    Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])
  end

  @doc """
  Multi-head chunked attention.

  4D variant of `chunked_attention/4` — processes query in chunks to reduce
  peak memory while producing identical results to `multi_head_sdpa/3`.

  ## Parameters
    - `query` - [batch, heads, seq_q, head_dim]
    - `key` - [batch, heads, seq_k, head_dim]
    - `value` - [batch, heads, seq_k, head_dim]
    - `opts` - Options:
      - `:chunk_size` - Query chunk size (default: 32)
      - `:mask` - Attention mask [seq, seq]

  ## Returns
    [batch, heads, seq_q, head_dim]
  """
  @spec multi_head_chunked_attention(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def multi_head_chunked_attention(query, key, value, opts \\ []) do
    chunk_size = Keyword.get(opts, :chunk_size, 32)
    mask = Keyword.get(opts, :mask)

    seq_q = Nx.axis_size(query, 2)
    seq_k = Nx.axis_size(key, 2)
    d_k = Nx.axis_size(key, 3)
    scale = Nx.sqrt(d_k) |> Nx.as_type(Nx.type(query))

    num_chunks = div(seq_q + chunk_size - 1, chunk_size)

    chunk_results =
      for chunk_idx <- 0..(num_chunks - 1) do
        start_idx = chunk_idx * chunk_size
        actual_chunk_size = min(chunk_size, seq_q - start_idx)

        q_chunk = Nx.slice_along_axis(query, start_idx, actual_chunk_size, axis: 2)

        # scores: [batch, heads, chunk_size, seq_k]
        scores = Nx.dot(q_chunk, [3], [0, 1], key, [3], [0, 1])
        scores = Nx.divide(scores, scale)

        scores =
          if mask do
            batch = Nx.axis_size(query, 0)
            heads = Nx.axis_size(query, 1)

            chunk_mask =
              case Nx.shape(mask) do
                {^seq_q, ^seq_k} ->
                  Nx.slice_along_axis(mask, start_idx, actual_chunk_size, axis: 0)
                  |> Nx.new_axis(0)
                  |> Nx.new_axis(0)
                  |> Nx.broadcast({batch, heads, actual_chunk_size, seq_k})

                _ ->
                  Nx.broadcast(mask, Nx.shape(scores))
              end

            Nx.select(chunk_mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))
          else
            scores
          end

        weights = FusedOps.fused_softmax(scores)
        Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])
      end

    Nx.concatenate(chunk_results, axis: 2)
  end

  @doc """
  Multi-head memory-efficient attention using online softmax.

  4D variant of `memory_efficient_attention/4` — achieves O(n) memory by
  processing K/V in chunks with running softmax normalization.

  ## Parameters
    - `query` - [batch, heads, seq_q, head_dim]
    - `key` - [batch, heads, seq_k, head_dim]
    - `value` - [batch, heads, seq_k, head_dim]
    - `opts` - Options:
      - `:chunk_size` - K/V chunk size (default: 32)
      - `:causal` - Use causal masking (default: false)

  ## Returns
    [batch, heads, seq_q, head_dim]
  """
  @spec multi_head_memory_efficient_attention(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  def multi_head_memory_efficient_attention(query, key, value, opts \\ []) do
    chunk_size = Keyword.get(opts, :chunk_size, 32)
    causal = Keyword.get(opts, :causal, false)

    {batch, heads, seq_q, dim} = Nx.shape(query)
    seq_k = Nx.axis_size(key, 2)
    scale = Nx.sqrt(dim) |> Nx.as_type(Nx.type(query))

    num_kv_chunks = div(seq_k + chunk_size - 1, chunk_size)

    tensor_type = Nx.type(query)
    neg_inf = Nx.Constants.neg_infinity() |> Nx.as_type(tensor_type)

    init_output = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, heads, seq_q, dim})
    init_max = Nx.broadcast(neg_inf, {batch, heads, seq_q})
    init_sum = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, heads, seq_q})

    {final_output, _final_max, final_sum} =
      Enum.reduce(0..(num_kv_chunks - 1), {init_output, init_max, init_sum}, fn chunk_idx,
                                                                                {acc_output,
                                                                                 acc_max, acc_sum} ->
        k_start = chunk_idx * chunk_size
        actual_chunk_size = min(chunk_size, seq_k - k_start)

        k_chunk = Nx.slice_along_axis(key, k_start, actual_chunk_size, axis: 2)
        v_chunk = Nx.slice_along_axis(value, k_start, actual_chunk_size, axis: 2)

        # scores: [batch, heads, seq_q, chunk_size]
        scores = Nx.dot(query, [3], [0, 1], k_chunk, [3], [0, 1])
        scores = Nx.divide(scores, scale)

        scores =
          if causal do
            q_positions = Nx.iota({seq_q, 1}, axis: 0)
            k_positions = Nx.iota({1, actual_chunk_size}, axis: 1) |> Nx.add(k_start)
            causal_mask = Nx.greater_equal(q_positions, k_positions)

            causal_mask =
              Nx.broadcast(
                causal_mask |> Nx.new_axis(0) |> Nx.new_axis(0),
                {batch, heads, seq_q, actual_chunk_size}
              )

            Nx.select(causal_mask, scores, Nx.broadcast(neg_inf, Nx.shape(scores)))
          else
            scores
          end

        chunk_max = Nx.reduce_max(scores, axes: [-1])
        new_max = Nx.max(acc_max, chunk_max)
        old_scale_factor = Nx.exp(Nx.subtract(acc_max, new_max))
        exp_scores = Nx.exp(Nx.subtract(scores, Nx.new_axis(new_max, -1)))
        chunk_sum = Nx.sum(exp_scores, axes: [-1])
        new_sum = Nx.add(Nx.multiply(acc_sum, old_scale_factor), chunk_sum)
        chunk_output = Nx.dot(exp_scores, [3], [0, 1], v_chunk, [2], [0, 1])

        new_output =
          Nx.add(
            Nx.multiply(acc_output, Nx.new_axis(old_scale_factor, -1)),
            chunk_output
          )

        {new_output, new_max, new_sum}
      end)

    Nx.divide(final_output, Nx.new_axis(final_sum, -1))
  end

  # ============================================================================
  # Head Reshaping Utilities
  # ============================================================================

  @doc """
  Reshape from merged to per-head layout.

  `[batch, seq, num_heads * head_dim]` -> `[batch, num_heads, seq, head_dim]`
  """
  @spec reshape_to_heads(
          Nx.Tensor.t(),
          non_neg_integer(),
          non_neg_integer(),
          pos_integer(),
          pos_integer()
        ) ::
          Nx.Tensor.t()
  def reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  @doc """
  Reshape from per-head to merged layout.

  `[batch, num_heads, seq, head_dim]` -> `[batch, seq, num_heads * head_dim]`
  """
  @spec reshape_from_heads(
          Nx.Tensor.t(),
          non_neg_integer(),
          non_neg_integer(),
          pos_integer(),
          pos_integer()
        ) ::
          Nx.Tensor.t()
  def reshape_from_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # QK Normalization
  # ============================================================================

  @doc """
  Apply layer normalization to Q or K tensors.

  QK LayerNorm normalizes across the feature dimension to prevent
  attention score explosion in deep networks.
  """
  @spec qk_layer_norm(Nx.Tensor.t()) :: Nx.Tensor.t()
  def qk_layer_norm(tensor) do
    # Normalize across the last axis (feature dimension)
    mean = Nx.mean(tensor, axes: [-1], keep_axes: true)
    variance = Nx.variance(tensor, axes: [-1], keep_axes: true)
    eps = 1.0e-6

    Nx.divide(Nx.subtract(tensor, mean), Nx.sqrt(Nx.add(variance, eps)))
  end

  # ============================================================================
  # Positional Encoding
  # ============================================================================

  @doc """
  Add sinusoidal positional encoding to input.

  Uses sin for position encoding — compatible with Axon's JIT compilation.
  Each position gets a unique encoding based on sine waves at different
  frequencies across the embedding dimensions.

  This is a thin Axon wrapper provided here as a utility since it's used
  across multiple attention model builders.
  """
  @spec add_positional_encoding(Axon.t(), keyword()) :: Axon.t()
  def add_positional_encoding(input, opts \\ []) do
    name = Keyword.get(opts, :name, "pos_enc")
    scale = Keyword.get(opts, :scale, 0.01)

    Axon.nx(
      input,
      fn tensor ->
        # Get actual shape from tensor
        shape = Nx.shape(tensor)
        seq_len = elem(shape, 1)
        embed_dim = elem(shape, 2)

        # Position indices [1, seq_len, 1] - will broadcast to [batch, seq_len, embed_dim]
        pos = Nx.iota({1, seq_len, 1}, axis: 1) |> Nx.as_type(:f32)

        # Dimension indices [1, 1, embed_dim] - different frequency per dimension
        dim = Nx.iota({1, 1, embed_dim}, axis: 2) |> Nx.as_type(:f32)

        # Compute angles: combine position and dimension
        freq = Nx.divide(1.0, Nx.add(1.0, Nx.multiply(dim, scale)))

        # angles shape: [1, seq_len, embed_dim] via broadcasting
        angles = Nx.multiply(pos, freq)

        # Apply sin to create positional encoding
        pe = Nx.sin(angles)

        # Add to input (broadcasts over batch dimension)
        Nx.add(tensor, pe)
      end,
      name: name
    )
  end
end
