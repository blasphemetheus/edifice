defmodule Edifice.Memory.MemoryLayer do
  @moduledoc """
  Memory Layers: sparse key-value lookup replacing dense FFN.

  Instead of a standard FFN (all parameters activated per token), a Memory Layer
  stores millions of key-value pairs and retrieves only the top-k nearest matches
  per query. Product-key quantization makes search efficient: keys are split into
  two halves, each searched independently in O(sqrt(N)), then combined.

  ```
  Input [batch, seq_len, dim]
        |
  +------------------------------------------+
  | Memory Layer Block (x num_layers)        |
  |                                          |
  | LayerNorm + Self-Attention + Residual    |
  |                                          |
  | LayerNorm + Memory Lookup:               |
  |   q = W_q(x)            [dim -> key_dim] |
  |   q1, q2 = split(q)     [key_dim/2 each] |
  |   s1 = K1 @ q1          [sqrt(N) scores] |
  |   s2 = K2 @ q2          [sqrt(N) scores] |
  |   TopK candidates from s1 + s2           |
  |   Softmax attention over selected values |
  | + Gated residual                         |
  +------------------------------------------+
        |
  [batch, hidden_size]
  ```

  ## Key Innovation

  Decouples parameter count from compute: a model can store millions of
  key-value pairs but only activate top-k per token. Product-key search
  reduces complexity from O(N) to O(2*sqrt(N) + k^2).

  ## Usage

      model = MemoryLayer.build(
        embed_dim: 256,
        hidden_size: 256,
        memory_size: 1024,
        top_k: 32,
        num_layers: 4
      )

  ## Reference

  - Berges et al., "Memory Layers at Scale" (Meta, ICLR 2025)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 4
  @default_memory_size 1024
  @default_key_dim 64
  @default_top_k 32
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:memory_size, pos_integer()}
          | {:key_dim, pos_integer()}
          | {:top_k, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Memory Layer model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Model hidden dimension (default: 256)
    - `:num_heads` - Attention heads for self-attention layers (default: 4)
    - `:num_layers` - Number of transformer + memory blocks (default: 4)
    - `:memory_size` - Number of key-value pairs in memory (default: 1024)
    - `:key_dim` - Key embedding dimension (default: 64)
    - `:top_k` - Number of retrieved entries per query (default: 32)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    memory_size = Keyword.get(opts, :memory_size, @default_memory_size)
    key_dim = Keyword.get(opts, :key_dim, @default_key_dim)
    top_k = Keyword.get(opts, :top_k, @default_top_k)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "mem_block_#{layer_idx}"

          # Self-attention sublayer (standard transformer)
          attn_fn = fn x, attn_name ->
            Edifice.Attention.MultiHead.self_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              name: attn_name
            )
          end

          after_attn =
            TransformerBlock.layer(input,
              attention_fn: attn_fn,
              hidden_size: hidden_size,
              dropout: dropout,
              name: "#{name}_attn"
            )

          # Memory lookup sublayer (replaces FFN)
          build_memory_sublayer(after_attn,
            hidden_size: hidden_size,
            memory_size: memory_size,
            key_dim: key_dim,
            top_k: top_k,
            name: "#{name}_mem"
          )
        end
      )
    )
  end

  # Memory lookup sublayer with gated residual
  defp build_memory_sublayer(input, opts) do
    hidden_size = opts[:hidden_size]
    memory_size = opts[:memory_size]
    key_dim = opts[:key_dim]
    top_k = opts[:top_k]
    name = opts[:name]

    # Product key sizes: sqrt(memory_size)
    half_keys = round(:math.sqrt(memory_size))

    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Query projection
    query = Axon.dense(normed, key_dim, name: "#{name}_query_proj")

    # Product key parameters (two sets of half-keys)
    keys1 =
      Axon.param("#{name}_keys1", {half_keys, div(key_dim, 2)}, initializer: :glorot_uniform)

    keys2 =
      Axon.param("#{name}_keys2", {half_keys, div(key_dim, 2)}, initializer: :glorot_uniform)

    # Values: [memory_size, hidden_size]
    # For tractability, we use half_keys^2 capped at memory_size
    actual_memory = min(half_keys * half_keys, memory_size)

    values =
      Axon.param("#{name}_values", {actual_memory, hidden_size}, initializer: :glorot_uniform)

    # Gate for gated residual
    gate_proj = Axon.dense(normed, hidden_size, name: "#{name}_gate")

    # Memory lookup computation
    mem_out =
      Axon.layer(
        &memory_lookup_impl/6,
        [query, keys1, keys2, values, gate_proj],
        name: "#{name}_lookup",
        top_k: top_k,
        half_keys: half_keys,
        hidden_size: hidden_size,
        op_name: :memory_lookup
      )

    # Residual
    Axon.add(input, mem_out)
  end

  # Product-key memory lookup with gating
  defp memory_lookup_impl(query, keys1, keys2, values, gate, opts) do
    top_k = opts[:top_k]
    half_keys = opts[:half_keys]
    {batch, seq_len, key_dim} = Nx.shape(query)
    half_dim = div(key_dim, 2)

    # Split query into two halves
    q1 = Nx.slice_along_axis(query, 0, half_dim, axis: 2)
    q2 = Nx.slice_along_axis(query, half_dim, half_dim, axis: 2)

    # Compute scores for each half: [batch, seq, half_keys]
    # q1: [batch, seq, half_dim], keys1: [half_keys, half_dim]
    s1 =
      Nx.dot(
        Nx.reshape(q1, {batch * seq_len, half_dim}),
        [1],
        keys1,
        [1]
      )
      |> Nx.reshape({batch, seq_len, half_keys})

    s2 =
      Nx.dot(
        Nx.reshape(q2, {batch * seq_len, half_dim}),
        [1],
        keys2,
        [1]
      )
      |> Nx.reshape({batch, seq_len, half_keys})

    # Top-k from each half
    k_per_half = min(top_k, half_keys)
    {_top_vals1, top_idx1} = Nx.top_k(s1, k: k_per_half)
    {_top_vals2, top_idx2} = Nx.top_k(s2, k: k_per_half)

    # Get scores for top indices
    top_s1 = Nx.take_along_axis(s1, top_idx1, axis: 2)
    top_s2 = Nx.take_along_axis(s2, top_idx2, axis: 2)

    # Combined scores: s1[i] + s2[j] for all (i,j) pairs
    # [batch, seq, k] + [batch, seq, k] -> [batch, seq, k*k]
    s1_expanded = Nx.new_axis(top_s1, 3) |> Nx.broadcast({batch, seq_len, k_per_half, k_per_half})

    s2_expanded =
      Nx.new_axis(top_s2, 2) |> Nx.broadcast({batch, seq_len, k_per_half, k_per_half})

    combined_scores = Nx.add(s1_expanded, s2_expanded)
    combined_scores = Nx.reshape(combined_scores, {batch, seq_len, k_per_half * k_per_half})

    # Top-k from combined
    final_k = min(top_k, k_per_half * k_per_half)
    {top_scores, top_combined_idx} = Nx.top_k(combined_scores, k: final_k)

    # Convert combined indices back to (i1, i2) -> linear value index
    pair_i1 = Nx.quotient(top_combined_idx, k_per_half)
    pair_i2 = Nx.remainder(top_combined_idx, k_per_half)

    # Map back to original key indices
    # Gather from top_idx1 using pair_i1, top_idx2 using pair_i2
    orig_i1 = Nx.take_along_axis(top_idx1, pair_i1, axis: 2)
    orig_i2 = Nx.take_along_axis(top_idx2, pair_i2, axis: 2)

    # Linear value index: i1 * half_keys + i2
    value_indices = Nx.add(Nx.multiply(orig_i1, half_keys), orig_i2)

    # Gather values: [batch, seq, final_k, hidden_size]
    hidden_size = Nx.axis_size(values, 1)

    flat_indices = Nx.reshape(value_indices, {:auto})

    # Clamp indices to valid range
    max_idx = Nx.axis_size(values, 0) - 1
    flat_indices = Nx.min(flat_indices, max_idx)

    gathered = Nx.take(values, flat_indices, axis: 0)
    gathered = Nx.reshape(gathered, {batch, seq_len, final_k, hidden_size})

    # Softmax attention over retrieved values
    weights =
      Nx.exp(Nx.subtract(top_scores, Nx.reduce_max(top_scores, axes: [-1], keep_axes: true)))

    weights = Nx.divide(weights, Nx.add(Nx.sum(weights, axes: [-1], keep_axes: true), 1.0e-8))

    # Weighted sum: [batch, seq, hidden_size]
    mem_out = Nx.sum(Nx.multiply(Nx.new_axis(weights, -1), gathered), axes: [2])

    # Apply SiLU gate
    Nx.multiply(mem_out, Nx.sigmoid(gate) |> Nx.multiply(gate))
  end

  @doc """
  Get the output dimension for a model configuration.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Recommended default configuration.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 4,
      num_layers: 4,
      memory_size: 1024,
      key_dim: 64,
      top_k: 32,
      dropout: 0.1
    ]
  end
end
