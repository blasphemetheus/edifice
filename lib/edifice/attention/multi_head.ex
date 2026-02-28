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
        embed_dim: 1024,
        window_size: 60,
        num_heads: 4,
        head_dim: 64
      )

      # Hybrid LSTM + Attention
      model = MultiHead.build_hybrid(
        embed_dim: 1024,
        lstm_hidden: 256,
        num_heads: 4,
        head_dim: 64
      )
  """

  alias Edifice.Attention.Primitives
  alias Edifice.Recurrent

  # Default hyperparameters
  @default_num_heads 4
  @default_head_dim 64
  @default_window_size 60
  @default_dropout 0.1

  # ============================================================================
  # Delegated Primitives (backward compatibility)
  # ============================================================================

  @doc "Delegates to `Edifice.Attention.Primitives.scaled_dot_product_attention/4`."
  defdelegate scaled_dot_product_attention(query, key, value, opts \\ []), to: Primitives

  @doc "Delegates to `Edifice.Attention.Primitives.chunked_attention/4`."
  defdelegate chunked_attention(query, key, value, opts \\ []), to: Primitives

  @doc "Delegates to `Edifice.Attention.Primitives.memory_efficient_attention/4`."
  defdelegate memory_efficient_attention(query, key, value, opts \\ []), to: Primitives

  @doc "Delegates to `Edifice.Attention.Primitives.qk_layer_norm/1`."
  defdelegate qk_layer_norm(tensor), to: Primitives

  @doc "Delegates to `Edifice.Attention.Primitives.add_positional_encoding/2`."
  defdelegate add_positional_encoding(input, opts \\ []), to: Primitives

  @doc "Delegates to `Edifice.Blocks.CausalMask.causal/1`."
  defdelegate causal_mask(seq_len), to: Edifice.Blocks.CausalMask, as: :causal

  @doc "Delegates to `Edifice.Blocks.CausalMask.window/2`."
  defdelegate window_mask(seq_len, window_size), to: Edifice.Blocks.CausalMask, as: :window

  # ============================================================================
  # Multi-Head Attention Layer
  # ============================================================================

  @doc """
  Build a multi-head self-attention Axon layer.

  Properly reshapes Q, K, V to `[batch, num_heads, seq, head_dim]` so each
  head computes its own independent attention pattern, then reshapes back.

  ## Options
    - `:hidden_size` - Hidden dimension = num_heads * head_dim (default: 256)
    - `:num_heads` - Number of attention heads (default: 1)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:causal` - Use causal masking (default: true)
    - `:qk_layernorm` - Normalize Q and K before attention (stabilizes training, default: false)
    - `:rope` - Apply Rotary Position Embedding to Q and K (default: false)
    - `:chunked` - Use chunked attention for lower memory (default: false)
    - `:memory_efficient` - Use memory-efficient attention with online softmax for true O(n) memory (default: false)
    - `:chunk_size` - Chunk size for chunked/memory-efficient attention (default: 32)
    - `:name` - Layer name prefix
  """
  @spec self_attention(Axon.t(), keyword()) :: Axon.t()
  def self_attention(input, opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, 256)
    num_heads = Keyword.get(opts, :num_heads, 1)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    causal = Keyword.get(opts, :causal, true)
    qk_layernorm = Keyword.get(opts, :qk_layernorm, false)
    use_rope = Keyword.get(opts, :rope, false)
    chunked = Keyword.get(opts, :chunked, false)
    memory_efficient = Keyword.get(opts, :memory_efficient, false)
    chunk_size = Keyword.get(opts, :chunk_size, 32)
    name = Keyword.get(opts, :name, "self_attn")

    head_dim = div(hidden_size, num_heads)

    # Project to Q, K, V and concatenate for single layer call
    qkv = Axon.dense(input, hidden_size * 3, name: "#{name}_qkv")

    # Apply attention in a single Axon.nx call
    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          {batch, seq_len, _} = Nx.shape(qkv_tensor)

          # Split into Q, K, V: each [batch, seq, hidden_size]
          query = Nx.slice_along_axis(qkv_tensor, 0, hidden_size, axis: 2)
          key = Nx.slice_along_axis(qkv_tensor, hidden_size, hidden_size, axis: 2)
          value = Nx.slice_along_axis(qkv_tensor, hidden_size * 2, hidden_size, axis: 2)

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

          # Apply RoPE if enabled: rotate Q and K per head
          # Q, K are [batch, heads, seq, head_dim]
          {query, key} =
            if use_rope do
              Edifice.Blocks.RoPE.apply_rotary_4d(query, key)
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

          # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden_size]
          reshape_from_heads(output, batch, seq_len, num_heads, head_dim)
        end,
        name: "#{name}_compute"
      )

    # Output projection
    attended
    |> Axon.dense(hidden_size, name: "#{name}_output")
    |> Axon.dropout(rate: dropout, name: "#{name}_dropout")
  end

  @doc """
  Build multi-head attention with configurable heads and head dimension.

  Computes `hidden_size = num_heads * head_dim` and delegates to `self_attention/2`
  with proper multi-head reshaping.
  """
  @spec multi_head_attention(Axon.t(), keyword()) :: Axon.t()
  def multi_head_attention(input, opts \\ []) do
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    hidden_size = num_heads * head_dim

    opts =
      opts
      |> Keyword.put(:hidden_size, hidden_size)
      |> Keyword.put(:num_heads, num_heads)

    self_attention(input, opts)
  end

  @doc """
  Build a sliding window attention layer.

  More efficient than full attention - O(K^2) per position instead of O(N^2).

  ## Options
    - `:window_size` - Attention window size (default: 60)
    - `:hidden_size` - Hidden dimension (default: 256)
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

    hidden_size = num_heads * head_dim

    # Project to Q, K, V in single dense layer
    qkv = Axon.dense(input, hidden_size * 3, name: "#{name}_qkv")

    # Apply windowed attention with pre-computed mask (captured from outer scope)
    # This avoids dynamic mask creation inside Axon.nx which causes XLA issues
    Axon.nx(
      qkv,
      fn qkv_tensor ->
        {batch, seq_len, _} = Nx.shape(qkv_tensor)

        # Split into Q, K, V: each [batch, seq, hidden_size]
        query = Nx.slice_along_axis(qkv_tensor, 0, hidden_size, axis: 2)
        key = Nx.slice_along_axis(qkv_tensor, hidden_size, hidden_size, axis: 2)
        value = Nx.slice_along_axis(qkv_tensor, hidden_size * 2, hidden_size, axis: 2)

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

        # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden_size]
        reshape_from_heads(output, batch, seq_len, num_heads, head_dim)
      end,
      name: "#{name}_compute"
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
    - `:embed_dim` - Input embedding size (required)
    - `:hidden_size` - Total hidden dimension; overrides num_heads * head_dim (optional)
    - `:window_size` - Attention window / sequence length (default: 60)
    - `:num_heads` - Attention heads (default: 4)
    - `:head_dim` - Dimension per head (default: 64)
    - `:num_layers` - Number of attention layers (default: 2)
    - `:ffn_dim` - Feed-forward dimension (default: 256)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    Model that outputs [batch, hidden_size] from last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt :: {:num_heads, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
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
    - `:embed_dim` - Input embedding size (required)
    - `:window_size` - Attention window (default: 60)
    - `:num_heads` - Attention heads (default: 4)
    - `:head_dim` - Dimension per head (default: 64)
    - `:num_layers` - Number of attention layers (default: 2)
    - `:ffn_dim` - Feed-forward dimension (default: 256)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    Model that outputs [batch, hidden_size] from last position.
  """
  @spec build_sliding_window(keyword()) :: Axon.t()
  def build_sliding_window(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    num_layers = Keyword.get(opts, :num_layers, 2)
    ffn_dim = Keyword.get(opts, :ffn_dim, 256)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    hidden_size = num_heads * head_dim

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

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project to hidden dimension
    x = Axon.dense(input, hidden_size, name: "input_proj")

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
          |> Axon.dense(hidden_size, name: "layer_#{layer_idx}_ffn2")
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
    - `:embed_dim` - Input embedding size (required)
    - `:lstm_hidden` - LSTM hidden size (default: 256)
    - `:lstm_layers` - Number of LSTM layers (default: 1)
    - `:num_heads` - Attention heads (default: 4)
    - `:head_dim` - Dimension per head (default: 64)
    - `:dropout` - Dropout rate (default: 0.1)

  ## Returns
    Model that outputs [batch, hidden_size] combining LSTM and attention.
  """
  @spec build_hybrid(keyword()) :: Axon.t()
  def build_hybrid(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    lstm_hidden = Keyword.get(opts, :lstm_hidden, 256)
    lstm_layers = Keyword.get(opts, :lstm_layers, 1)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, @default_head_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)

    hidden_size = num_heads * head_dim

    # Sequence length configuration (same as build_sliding_window)
    seq_len = Keyword.get(opts, :seq_len, window_size)
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

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
    x = Axon.dense(lstm_output, hidden_size, name: "lstm_to_attn_proj")

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
  # Private Helpers (delegate to Primitives)
  # ============================================================================

  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim),
    do: Primitives.reshape_to_heads(x, batch, seq_len, num_heads, head_dim)

  defp reshape_from_heads(x, batch, seq_len, num_heads, head_dim),
    do: Primitives.reshape_from_heads(x, batch, seq_len, num_heads, head_dim)

  defp multi_head_sdpa(query, key, value, opts),
    do: Primitives.multi_head_sdpa(query, key, value, opts)

  defp multi_head_chunked_attention(query, key, value, opts),
    do: Primitives.multi_head_chunked_attention(query, key, value, opts)

  defp multi_head_memory_efficient_attention(query, key, value, opts),
    do: Primitives.multi_head_memory_efficient_attention(query, key, value, opts)

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
