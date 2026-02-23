defmodule Edifice.Blocks.KVCache do
  @moduledoc """
  KV Cache: Inference-time Key-Value Caching for Autoregressive Decoding.

  Provides utilities for caching key and value projections during
  autoregressive generation. Without caching, each new token requires
  re-computing K/V for all previous tokens (O(n²) per step). With
  caching, only the new token's K/V are computed and appended (O(n)).

  ## How It Works

  ```
  Step 1: prompt "The cat"
    K_cache = [K_the, K_cat]
    V_cache = [V_the, V_cat]

  Step 2: generate "sat"
    Compute K_sat, V_sat only for new token
    K_cache = [K_the, K_cat, K_sat]    (append)
    V_cache = [V_the, V_cat, V_sat]    (append)
    Attend: Q_sat × K_cache^T → weights → V_cache

  Step 3: generate "on"
    K_cache = [K_the, K_cat, K_sat, K_on]
    ...
  ```

  ## Usage

      # Initialize empty cache for a model
      cache = KVCache.init(batch_size: 2, num_layers: 4, num_heads: 4, head_dim: 64)

      # During generation loop:
      {new_k, new_v} = compute_kv(token)
      {cache, full_k, full_v} = KVCache.update(cache, layer_idx, new_k, new_v)
      output = attend(q, full_k, full_v)

      # Get current sequence length
      len = KVCache.seq_length(cache, layer_idx)

  ## Design

  The cache is a simple map of `{layer_idx => {k_tensor, v_tensor}}`.
  Tensors grow along the sequence dimension (axis 2 for head-first layout,
  axis 1 for seq-first layout). This module assumes head-first layout:
  `[batch, num_heads, seq_len, head_dim]`.
  """

  @typedoc "A KV cache: map from layer index to {k_tensor, v_tensor}."
  @type t :: %{non_neg_integer() => {Nx.Tensor.t(), Nx.Tensor.t()}}

  @typedoc "Options for `init/1`."
  @type init_opt ::
          {:batch_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:head_dim, pos_integer()}
          | {:max_seq_len, pos_integer()}
          | {:type, Nx.Type.t()}

  # ============================================================================
  # Initialization
  # ============================================================================

  @doc """
  Initialize an empty KV cache.

  Creates pre-allocated zero tensors for each layer. Using pre-allocated
  tensors with a position pointer is more efficient than repeated
  concatenation on accelerators.

  ## Options

    - `:batch_size` - Batch size (required)
    - `:num_layers` - Number of transformer layers (required)
    - `:num_heads` - Number of attention heads (required)
    - `:head_dim` - Dimension per head (required)
    - `:max_seq_len` - Maximum sequence length to pre-allocate (default: 2048)
    - `:type` - Numeric type (default: `:f32`)

  ## Returns

    A map `%{cache: %{layer => {k, v}}, position: 0, max_seq_len: n}`.
  """
  @spec init([init_opt()]) :: map()
  def init(opts) do
    batch_size = Keyword.fetch!(opts, :batch_size)
    num_layers = Keyword.fetch!(opts, :num_layers)
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    max_seq_len = Keyword.get(opts, :max_seq_len, 2048)
    type = Keyword.get(opts, :type, :f32)

    cache =
      for layer <- 0..(num_layers - 1), into: %{} do
        shape = {batch_size, num_heads, max_seq_len, head_dim}
        k = Nx.broadcast(Nx.tensor(0.0, type: type), shape)
        v = Nx.broadcast(Nx.tensor(0.0, type: type), shape)
        {layer, {k, v}}
      end

    %{
      cache: cache,
      position: 0,
      max_seq_len: max_seq_len,
      num_layers: num_layers
    }
  end

  # ============================================================================
  # Cache Update
  # ============================================================================

  @doc """
  Append new K/V entries to the cache for a given layer.

  Takes new K/V tensors of shape `[batch, num_heads, new_len, head_dim]`
  and writes them into the pre-allocated cache at the current position.

  ## Parameters

    - `state` - The cache state from `init/1` or a previous `update/4`
    - `layer_idx` - Which transformer layer (0-indexed)
    - `new_k` - New key tensor `[batch, num_heads, new_len, head_dim]`
    - `new_v` - New value tensor `[batch, num_heads, new_len, head_dim]`

  ## Returns

    `{updated_state, cached_k, cached_v}` where cached_k/v contain all
    entries up to and including the new ones (sliced from pre-allocated buffer).
  """
  @spec update(map(), non_neg_integer(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          {map(), Nx.Tensor.t(), Nx.Tensor.t()}
  def update(state, layer_idx, new_k, new_v) do
    %{cache: cache, position: pos, max_seq_len: max_len} = state
    {k_buf, v_buf} = Map.fetch!(cache, layer_idx)

    new_len = Nx.axis_size(new_k, 2)
    end_pos = pos + new_len

    # Write new entries into the pre-allocated buffer
    k_buf = write_to_buffer(k_buf, new_k, pos)
    v_buf = write_to_buffer(v_buf, new_v, pos)

    updated_cache = Map.put(cache, layer_idx, {k_buf, v_buf})
    updated_state = %{state | cache: updated_cache, position: min(end_pos, max_len)}

    # Return the valid portion of the cache
    cached_k = Nx.slice_along_axis(k_buf, 0, end_pos, axis: 2)
    cached_v = Nx.slice_along_axis(v_buf, 0, end_pos, axis: 2)

    {updated_state, cached_k, cached_v}
  end

  # ============================================================================
  # Queries
  # ============================================================================

  @doc """
  Get the current sequence length stored in the cache.
  """
  @spec seq_length(map()) :: non_neg_integer()
  def seq_length(%{position: pos}), do: pos

  @doc """
  Get the cached K/V tensors for a layer (valid portion only).

  Returns `{k, v}` sliced to the current position.
  """
  @spec get(map(), non_neg_integer()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def get(%{cache: cache, position: pos}, layer_idx) do
    {k_buf, v_buf} = Map.fetch!(cache, layer_idx)
    k = Nx.slice_along_axis(k_buf, 0, max(pos, 1), axis: 2)
    v = Nx.slice_along_axis(v_buf, 0, max(pos, 1), axis: 2)
    {k, v}
  end

  @doc """
  Reset the cache to empty (reuse the allocated buffers).
  """
  @spec reset(map()) :: map()
  def reset(state) do
    %{state | position: 0}
  end

  # ============================================================================
  # Attention Wrapper
  # ============================================================================

  @doc """
  Build a cached attention function.

  Returns a function that, given Q, K, V for the new tokens and a cache state,
  computes attention using the full cached K/V history.

  ## Options

    - `:num_heads` - Number of attention heads (required)
    - `:head_dim` - Dimension per head (required)
    - `:layer_idx` - Layer index for cache lookup (required)

  ## Returns

    A function `fn(q, k, v, cache_state) -> {output, updated_cache_state}`.
  """
  @spec build_cached_attention(keyword()) :: function()
  def build_cached_attention(opts) do
    num_heads = Keyword.fetch!(opts, :num_heads)
    head_dim = Keyword.fetch!(opts, :head_dim)
    layer_idx = Keyword.fetch!(opts, :layer_idx)

    fn q, new_k, new_v, cache_state ->
      # Update cache with new K/V
      {cache_state, full_k, full_v} = update(cache_state, layer_idx, new_k, new_v)

      # Standard scaled dot-product attention
      scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
      scores = Nx.dot(q, [3], [0, 1], full_k, [3], [0, 1]) |> Nx.divide(scale)

      # Softmax
      max_s = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
      exp_s = Nx.exp(Nx.subtract(scores, max_s))
      weights = Nx.divide(exp_s, Nx.add(Nx.sum(exp_s, axes: [-1], keep_axes: true), 1.0e-8))

      output = Nx.dot(weights, [3], [0, 1], full_v, [2], [0, 1])

      batch = Nx.axis_size(q, 0)
      q_len = Nx.axis_size(q, 2)

      output =
        output
        |> Nx.transpose(axes: [0, 2, 1, 3])
        |> Nx.reshape({batch, q_len, num_heads * head_dim})

      {output, cache_state}
    end
  end

  # ============================================================================
  # Private Helpers
  # ============================================================================

  defp write_to_buffer(buffer, new_data, position) do
    Nx.put_slice(buffer, [0, 0, position, 0], new_data)
  end
end
