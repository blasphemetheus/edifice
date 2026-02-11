defmodule Edifice.Attention.MultiHead do
  @moduledoc """
  Temporal attention mechanisms for sequence processing.

  Provides two main architectures:

  ## Option C: Sliding Window Attention

  Efficient attention that only looks at the last K timesteps:

  ```
  Step t-5   Step t-4   Step t-3   Step t-2   Step t-1   Step t
     |         |         |         |         |         |
     +---------+---------+---------+---------+---------+
                         Attend to window
                               |
                               v
                        Current Output
  ```

  O(K^2) complexity instead of O(N^2) - practical for real-time.

  ## Option B: Hybrid LSTM + Attention

  LSTM compresses temporal information, then attention finds long-range patterns:

  ```
  Frames -> LSTM -> [h1, h2, ..., hN] -> Self-Attention -> Output
  ```

  Best of both worlds:
  - LSTM captures local sequential patterns
  - Attention finds sparse long-range dependencies

  ## Why Attention Helps Temporal Processing

  1. **Direct timestep access**: "What happened exactly 12 steps ago?"
  2. **Learned relevance**: Model decides which past timesteps matter
  3. **Parallel training**: Unlike LSTM, attention can process all timesteps simultaneously
  4. **Interpretable**: Attention weights show what the model focuses on

  ## Usage

      # Sliding window model
      model = MultiHead.build_sliding_window(
        embed_size: 1024,
        window_size: 60,
        num_heads: 4,
        head_dim: 64
      )

      # Hybrid LSTM + Attention
      model = MultiHead.build_hybrid(
        embed_size: 1024,
        lstm_hidden: 256,
        num_heads: 4,
        head_dim: 64
      )
  """

  require Axon

  alias Edifice.Utils.FusedOps
  alias Edifice.Recurrent

  # Default hyperparameters
  @default_num_heads 4
  @default_head_dim 64
  @default_window_size 60
  @default_dropout 0.1

  # ============================================================================
  # Core Attention Mechanisms
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

  ## Memory Comparison
    For seq_len=128, batch=32, dim=256:
    - Standard: 32 x 128 x 128 x 4 bytes = 2MB peak for scores
    - Chunked (chunk=32): 32 x 32 x 128 x 4 bytes = 512KB peak (4x reduction)
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
    # Note: We use Nx.concatenate at the end rather than building a list
    # because XLA needs static shapes during tracing
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
  softmax, never materializing the full attention matrix. This is based on
  the algorithm from "Self-attention Does Not Need O(n^2) Memory" (Rabe & Staats, 2021).

  ## Algorithm

  Instead of computing the full attention matrix, we:
  1. Process K/V in chunks
  2. For each chunk, compute partial attention scores
  3. Use online softmax to combine results: track running max and sum
  4. Update output with proper normalization

  ## Parameters
    - `query` - Query tensor [batch, seq_q, dim]
    - `key` - Key tensor [batch, seq_k, dim]
    - `value` - Value tensor [batch, seq_k, dim]
    - `opts` - Options:
      - `:chunk_size` - Size of K/V chunks (default: 32)
      - `:causal` - Use causal masking (default: false)

  ## Returns
    Attention output [batch, seq_q, dim]

  ## Memory Comparison
    For seq_len=128, batch=32, dim=256:
    - Standard: 32 x 128 x 128 x 4 bytes = 2MB (full attention matrix)
    - Memory-efficient: 32 x 128 x 32 x 4 bytes = 512KB (one chunk at a time)

  ## Notes
    - Slightly slower than standard attention due to online softmax overhead
    - Output may have minor numerical differences (< 1e-5) due to different
      summation order in softmax
    - Causal masking is applied per-chunk
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
    # We track:
    # - acc_output: unnormalized weighted sum of values (will divide by exp(lse) at end)
    # - acc_max: running max of scores (for numerical stability)
    # - acc_sum: running sum of exp(scores - max) (for normalization)
    tensor_type = Nx.type(query)
    neg_inf = Nx.Constants.neg_infinity() |> Nx.as_type(tensor_type)

    init_output = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, seq_q, dim})
    init_max = Nx.broadcast(neg_inf, {batch, seq_q})
    init_sum = Nx.broadcast(Nx.tensor(0.0, type: tensor_type), {batch, seq_q})

    # Process each K/V chunk with online softmax
    {final_output, _final_max, final_sum} =
      Enum.reduce(0..(num_kv_chunks - 1), {init_output, init_max, init_sum}, fn chunk_idx,
                                                                                {acc_output,
                                                                                 acc_max,
                                                                                 acc_sum} ->
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

  @doc """
  Create a causal (autoregressive) attention mask.

  Each position can only attend to itself and previous positions.
  """
  @spec causal_mask(non_neg_integer()) :: Nx.Tensor.t()
  def causal_mask(seq_len) do
    # Lower triangular matrix of 1s
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    Nx.greater_equal(rows, cols)
  end

  @doc """
  Create a sliding window attention mask.

  Each position can only attend to positions within the window.
  """
  @spec window_mask(non_neg_integer(), non_neg_integer()) :: Nx.Tensor.t()
  def window_mask(seq_len, window_size) do
    # Each position attends to [max(0, i - window + 1), i]
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)

    # Valid if: col <= row (causal) AND col >= row - window + 1
    causal = Nx.greater_equal(rows, cols)
    in_window = Nx.greater_equal(cols, Nx.subtract(rows, window_size - 1))

    Nx.logical_and(causal, in_window)
  end

  # ============================================================================
  # Multi-Head Attention Layer
  # ============================================================================

  @doc """
  Build a multi-head self-attention Axon layer.

  Properly reshapes Q, K, V to `[batch, num_heads, seq, head_dim]` so each
  head computes its own independent attention pattern, then reshapes back.

  ## Options
    - `:hidden_dim` - Hidden dimension = num_heads * head_dim (default: 256)
    - `:num_heads` - Number of attention heads (default: 1)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:causal` - Use causal masking (default: true)
    - `:qk_layernorm` - Normalize Q and K before attention (stabilizes training, default: false)
    - `:chunked` - Use chunked attention for lower memory (default: false)
    - `:memory_efficient` - Use memory-efficient attention with online softmax for true O(n) memory (default: false)
    - `:chunk_size` - Chunk size for chunked/memory-efficient attention (default: 32)
    - `:name` - Layer name prefix
  """
  @spec self_attention(Axon.t(), keyword()) :: Axon.t()
  def self_attention(input, opts \\ []) do
    hidden_dim = Keyword.get(opts, :hidden_dim, 256)
    num_heads = Keyword.get(opts, :num_heads, 1)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    causal = Keyword.get(opts, :causal, true)
    qk_layernorm = Keyword.get(opts, :qk_layernorm, false)
    chunked = Keyword.get(opts, :chunked, false)
    memory_efficient = Keyword.get(opts, :memory_efficient, false)
    chunk_size = Keyword.get(opts, :chunk_size, 32)
    name = Keyword.get(opts, :name, "self_attn")

    head_dim = div(hidden_dim, num_heads)

    # Project to Q, K, V and concatenate for single layer call
    qkv = Axon.dense(input, hidden_dim * 3, name: "#{name}_qkv")

    # Apply attention in a single Axon.nx call
    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          {batch, seq_len, _} = Nx.shape(qkv_tensor)

          # Split into Q, K, V: each [batch, seq, hidden_dim]
          query = Nx.slice_along_axis(qkv_tensor, 0, hidden_dim, axis: 2)
          key = Nx.slice_along_axis(qkv_tensor, hidden_dim, hidden_dim, axis: 2)
          value = Nx.slice_along_axis(qkv_tensor, hidden_dim * 2, hidden_dim, axis: 2)

          # Reshape to multi-head: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
          query = reshape_to_heads(query, batch, seq_len, num_heads, head_dim)
          key = reshape_to_heads(key, batch, seq_len, num_heads, head_dim)
          value = reshape_to_heads(value, batch, seq_len, num_heads, head_dim)

          # QK LayerNorm: normalize Q and K per head to prevent attention explosion
          {query, key} =
            if qk_layernorm do
              {qk_layer_norm(query), qk_layer_norm(key)}
            else
              {query, key}
            end

          # Compute attention per head
          # Q, K, V are [batch, heads, seq, head_dim]
          output =
            cond do
              memory_efficient ->
                multi_head_memory_efficient_attention(query, key, value,
                  chunk_size: chunk_size,
                  causal: causal
                )

              chunked ->
                mask = if causal, do: causal_mask(seq_len), else: nil

                multi_head_chunked_attention(query, key, value,
                  mask: mask,
                  chunk_size: chunk_size
                )

              true ->
                mask = if causal, do: causal_mask(seq_len), else: nil
                multi_head_sdpa(query, key, value, mask: mask)
            end

          # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden_dim]
          reshape_from_heads(output, batch, seq_len, num_heads, head_dim)
        end,
        name: "#{name}_compute"
      )

    # Output projection
    attended
    |> Axon.dense(hidden_dim, name: "#{name}_output")
    |> Axon.dropout(rate: dropout, name: "#{name}_dropout")
  end

  @doc """
  Build multi-head attention with configurable heads and head dimension.

  Computes `hidden_dim = num_heads * head_dim` and delegates to `self_attention/2`
  with proper multi-head reshaping.
  """
  @spec multi_head_attention(Axon.t(), keyword()) :: Axon.t()
  def multi_head_attention(input, opts \\ []) do
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    hidden_dim = num_heads * head_dim

    opts =
      opts
      |> Keyword.put(:hidden_dim, hidden_dim)
      |> Keyword.put(:num_heads, num_heads)

    self_attention(input, opts)
  end

  @doc """
  Build a sliding window attention layer.

  More efficient than full attention - O(K^2) per position instead of O(N^2).

  ## Options
    - `:window_size` - Attention window size (default: 60)
    - `:hidden_dim` - Hidden dimension (default: 256)
    - `:mask` - Pre-computed attention mask (recommended for efficient compilation)
    - `:qk_layernorm` - Normalize Q and K before attention (stabilizes training, default: false)
    - `:chunked` - Use chunked attention for lower memory (default: false)
    - `:memory_efficient` - Use memory-efficient attention with online softmax for true O(n) memory (default: false)
    - `:chunk_size` - Chunk size for chunked/memory-efficient attention (default: 32)
    - `:name` - Layer name prefix
  """
  @spec sliding_window_attention(Axon.t(), keyword()) :: Axon.t()
  def sliding_window_attention(input, opts \\ []) do
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    precomputed_mask = Keyword.get(opts, :mask, nil)
    qk_layernorm = Keyword.get(opts, :qk_layernorm, false)
    chunked = Keyword.get(opts, :chunked, false)
    memory_efficient = Keyword.get(opts, :memory_efficient, false)
    chunk_size = Keyword.get(opts, :chunk_size, 32)
    name = Keyword.get(opts, :name, "window_attn")

    hidden_dim = num_heads * head_dim

    # Project to Q, K, V in single dense layer
    qkv = Axon.dense(input, hidden_dim * 3, name: "#{name}_qkv")

    # Apply windowed attention with pre-computed mask (captured from outer scope)
    # This avoids dynamic mask creation inside Axon.nx which causes XLA issues
    Axon.nx(
      qkv,
      fn qkv_tensor ->
        {batch, seq_len, _} = Nx.shape(qkv_tensor)

        # Split into Q, K, V: each [batch, seq, hidden_dim]
        query = Nx.slice_along_axis(qkv_tensor, 0, hidden_dim, axis: 2)
        key = Nx.slice_along_axis(qkv_tensor, hidden_dim, hidden_dim, axis: 2)
        value = Nx.slice_along_axis(qkv_tensor, hidden_dim * 2, hidden_dim, axis: 2)

        # Reshape to multi-head: [batch, seq, hidden] -> [batch, heads, seq, head_dim]
        query = reshape_to_heads(query, batch, seq_len, num_heads, head_dim)
        key = reshape_to_heads(key, batch, seq_len, num_heads, head_dim)
        value = reshape_to_heads(value, batch, seq_len, num_heads, head_dim)

        # QK LayerNorm: normalize Q and K per head to prevent attention explosion
        {query, key} =
          if qk_layernorm do
            {qk_layer_norm(query), qk_layer_norm(key)}
          else
            {query, key}
          end

        # Compute attention per head
        output =
          cond do
            memory_efficient ->
              multi_head_memory_efficient_attention(query, key, value,
                chunk_size: chunk_size,
                causal: true
              )

            chunked ->
              mask =
                if precomputed_mask != nil,
                  do: precomputed_mask,
                  else: window_mask(seq_len, window_size)

              multi_head_chunked_attention(query, key, value,
                mask: mask,
                chunk_size: chunk_size
              )

            true ->
              mask =
                if precomputed_mask != nil,
                  do: precomputed_mask,
                  else: window_mask(seq_len, window_size)

              multi_head_sdpa(query, key, value, mask: mask)
          end

        # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden_dim]
        reshape_from_heads(output, batch, seq_len, num_heads, head_dim)
      end,
      name: "#{name}_compute"
    )
  end

  @doc """
  Apply layer normalization to Q or K tensors.

  QK LayerNorm normalizes across the feature dimension to prevent
  attention score explosion in deep networks.
  """
  @spec qk_layer_norm(Nx.Tensor.t()) :: Nx.Tensor.t()
  def qk_layer_norm(tensor) do
    # Normalize across the last axis (feature dimension)
    # Mean and variance over last axis
    mean = Nx.mean(tensor, axes: [-1], keep_axes: true)
    variance = Nx.variance(tensor, axes: [-1], keep_axes: true)
    eps = 1.0e-6

    # Normalize
    Nx.divide(Nx.subtract(tensor, mean), Nx.sqrt(Nx.add(variance, eps)))
  end

  # ============================================================================
  # Positional Encoding
  # ============================================================================

  @doc """
  Add sinusoidal positional encoding to input.

  Uses sin for position encoding - compatible with Axon's JIT compilation.
  Each position gets a unique encoding based on sine waves at different
  frequencies across the embedding dimensions.
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
        # Higher dimensions get lower frequencies (standard positional encoding behavior)
        # angle = pos * exp(-dim * log(10000) / embed_dim)
        # Simplified: angle = pos / (1 + dim * scale_factor)
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

  # ============================================================================
  # Standard Build (Registry Entry Point)
  # ============================================================================

  @doc """
  Build a multi-head attention model.

  This is the standard entry point used by `Edifice.build(:attention, opts)`.
  Delegates to `build_sliding_window/1` which provides efficient attention
  that only attends to recent timesteps.

  For a hybrid LSTM + attention model, use `build_hybrid/1` directly.

  ## Options
    - `:embed_size` - Input embedding size (required)
    - `:hidden_size` - Total hidden dimension; overrides num_heads * head_dim (optional)
    - `:window_size` - Attention window / sequence length (default: 60)
    - `:num_heads` - Attention heads (default: 4)
    - `:head_dim` - Dimension per head (default: 64)
    - `:num_layers` - Number of attention layers (default: 2)
    - `:ffn_dim` - Feed-forward dimension (default: 256)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    Model that outputs [batch, hidden_dim] from last position.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    # Map hidden_size to head_dim if provided (for registry API consistency)
    opts =
      case Keyword.fetch(opts, :hidden_size) do
        {:ok, hidden_size} ->
          num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
          head_dim = div(hidden_size, num_heads)
          opts |> Keyword.put(:head_dim, head_dim) |> Keyword.delete(:hidden_size)

        :error ->
          opts
      end

    build_sliding_window(opts)
  end

  # ============================================================================
  # Option C: Sliding Window Attention Model
  # ============================================================================

  @doc """
  Build a complete sliding window attention model.

  Efficient for real-time inference - only attends to recent timesteps.

  ## Options
    - `:embed_size` - Input embedding size (required)
    - `:window_size` - Attention window (default: 60)
    - `:num_heads` - Attention heads (default: 4)
    - `:head_dim` - Dimension per head (default: 64)
    - `:num_layers` - Number of attention layers (default: 2)
    - `:ffn_dim` - Feed-forward dimension (default: 256)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    Model that outputs [batch, hidden_dim] from last position.
  """
  @spec build_sliding_window(keyword()) :: Axon.t()
  def build_sliding_window(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    num_layers = Keyword.get(opts, :num_layers, 2)
    ffn_dim = Keyword.get(opts, :ffn_dim, 256)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    hidden_dim = num_heads * head_dim

    # Sequence length configuration:
    # - :seq_len option: Explicit sequence length for the input
    # - Defaults to window_size for training efficiency (concrete shape = fast JIT)
    # - Set to nil for dynamic sequence length (slower JIT, more flexible)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Pre-compute attention mask if seq_len is concrete
    # This avoids creating masks inside Axon.nx which causes XLA compilation issues
    # IMPORTANT: Convert mask to BinaryBackend to avoid EXLA/Defn.Expr mismatch
    # when the mask is captured in Axon.nx closures during JIT compilation
    {precomputed_mask, input_seq_dim} =
      if seq_len do
        mask = window_mask(seq_len, window_size) |> Nx.backend_copy(Nx.BinaryBackend)
        {mask, seq_len}
      else
        # Dynamic - mask computed at runtime (slow path)
        {nil, nil}
      end

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project to hidden dimension
    x = Axon.dense(input, hidden_dim, name: "input_proj")

    # Add positional encoding
    x = add_positional_encoding(x, name: "pos_encoding")

    # Stack of attention + FFN layers
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        # Sliding window attention with pre-computed mask for efficient compilation
        attended =
          sliding_window_attention(acc,
            window_size: window_size,
            num_heads: num_heads,
            head_dim: head_dim,
            mask: precomputed_mask,
            name: "layer_#{layer_idx}_attn"
          )

        # Residual + LayerNorm
        acc = Axon.add(acc, attended, name: "layer_#{layer_idx}_residual1")
        acc = Axon.layer_norm(acc, name: "layer_#{layer_idx}_norm1")

        # Feed-forward network
        ffn =
          acc
          |> Axon.dense(ffn_dim, name: "layer_#{layer_idx}_ffn1")
          |> Axon.gelu()
          |> Axon.dense(hidden_dim, name: "layer_#{layer_idx}_ffn2")
          |> Axon.dropout(rate: dropout)

        # Residual + LayerNorm
        acc = Axon.add(acc, ffn, name: "layer_#{layer_idx}_residual2")
        Axon.layer_norm(acc, name: "layer_#{layer_idx}_norm2")
      end)

    # Take last position output: [batch, seq, hidden] -> [batch, hidden]
    # Use concrete seq_len if available for efficient compilation
    Axon.nx(
      x,
      fn tensor ->
        last_idx =
          if seq_len do
            seq_len - 1
          else
            Nx.axis_size(tensor, 1) - 1
          end

        Nx.slice_along_axis(tensor, last_idx, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_position"
    )
  end

  # ============================================================================
  # Option B: Hybrid LSTM + Attention
  # ============================================================================

  @doc """
  Build a hybrid LSTM + Attention model.

  LSTM captures local sequential patterns, attention finds long-range dependencies.

  ## Architecture
  ```
  Frames -> LSTM -> Hidden States -> Self-Attention -> Output
  ```

  ## Options
    - `:embed_size` - Input embedding size (required)
    - `:lstm_hidden` - LSTM hidden size (default: 256)
    - `:lstm_layers` - Number of LSTM layers (default: 1)
    - `:num_heads` - Attention heads (default: 4)
    - `:head_dim` - Dimension per head (default: 64)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    Model that outputs [batch, hidden_dim] combining LSTM and attention.
  """
  @spec build_hybrid(keyword()) :: Axon.t()
  def build_hybrid(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    lstm_hidden = Keyword.get(opts, :lstm_hidden, 256)
    lstm_layers = Keyword.get(opts, :lstm_layers, 1)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)

    hidden_dim = num_heads * head_dim

    # Sequence length configuration (same as build_sliding_window)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # LSTM backbone (returns all timesteps)
    lstm_output =
      Recurrent.build_backbone(input,
        hidden_size: lstm_hidden,
        num_layers: lstm_layers,
        cell_type: :lstm,
        dropout: dropout,
        # Need all timesteps for attention
        return_sequences: true
      )

    # Project LSTM output to attention dimension
    x = Axon.dense(lstm_output, hidden_dim, name: "lstm_to_attn_proj")

    # Self-attention over LSTM hidden states
    attended =
      multi_head_attention(x,
        num_heads: num_heads,
        head_dim: head_dim,
        dropout: dropout,
        causal: true,
        name: "hybrid_attn"
      )

    # Residual connection
    x = Axon.add(x, attended, name: "attn_residual")
    x = Axon.layer_norm(x, name: "attn_norm")

    # Take last position - use concrete seq_len if available
    Axon.nx(
      x,
      fn tensor ->
        last_idx =
          if seq_len do
            seq_len - 1
          else
            Nx.axis_size(tensor, 1) - 1
          end

        Nx.slice_along_axis(tensor, last_idx, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "hybrid_last"
    )
  end

  @doc """
  Build hybrid model with additional MLP layers on top.

  Good for policy/value heads that need more non-linearity.
  """
  @spec build_hybrid_mlp(keyword()) :: Axon.t()
  def build_hybrid_mlp(opts \\ []) do
    mlp_sizes = Keyword.get(opts, :mlp_sizes, [256, 256])
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Build base hybrid model
    hybrid = build_hybrid(opts)

    # Add MLP layers
    mlp_sizes
    |> Enum.with_index()
    |> Enum.reduce(hybrid, fn {size, idx}, acc ->
      acc
      |> Axon.dense(size, name: "hybrid_mlp_#{idx}")
      |> Axon.relu()
      |> Axon.dropout(rate: dropout)
    end)
  end

  # ============================================================================
  # Multi-Head Attention Helpers
  # ============================================================================

  # Reshape [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Reshape [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
  defp reshape_from_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # Multi-head scaled dot-product attention
  # Q, K, V: [batch, heads, seq, head_dim]
  # Returns: [batch, heads, seq, head_dim]
  defp multi_head_sdpa(query, key, value, opts) do
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

  # Multi-head chunked attention (processes query in chunks)
  # Q, K, V: [batch, heads, seq, head_dim]
  defp multi_head_chunked_attention(query, key, value, opts) do
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

  # Multi-head memory-efficient attention (online softmax)
  # Q, K, V: [batch, heads, seq, head_dim]
  defp multi_head_memory_efficient_attention(query, key, value, opts) do
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
                                                                                 acc_max,
                                                                                 acc_sum} ->
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
  # Utility Functions
  # ============================================================================

  @doc """
  Get the output dimension for a model configuration.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    num_heads * head_dim
  end

  @doc """
  Recommended default configuration for temporal sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      window_size: 60,
      num_heads: 4,
      # 256 total dim
      head_dim: 64,
      num_layers: 2,
      dropout: 0.1
    ]
  end
end
