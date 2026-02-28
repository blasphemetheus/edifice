defmodule Edifice.Attention.MoBA do
  @moduledoc """
  MoBA: Mixture of Block Attention.

  Applies MoE-style block-level routing to KV cache, where each query selects
  the top-k most relevant KV blocks to attend to. This provides sub-quadratic
  attention with graceful degradation to full attention when topk >= num_blocks.

  ## Key Idea

  The KV sequence is partitioned into blocks of size B. For each query, block
  relevance is scored using the dot product of the query with each block's
  mean key vector (parameter-free gating). The top-k blocks are selected and
  standard softmax attention is computed only over tokens in selected blocks.

  ```
  Input [batch, seq_len, hidden_size]
        |
  +------------------------------------------+
  | MoBA Attention                           |
  | 1. Partition K into blocks of size B     |
  | 2. Score: s_i(q) = <q, mean(K_block_i)> |
  | 3. Apply causal block mask               |
  | 4. Select top-k blocks per query         |
  | 5. Standard attention on selected blocks |
  +------------------------------------------+
        |
  [batch, seq_len, hidden_size]
  ```

  ## Usage

      model = MoBA.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 4,
        block_size: 8,
        topk: 2
      )

  ## Reference

  - Li et al., "MoBA: Mixture of Block Attention for Long-Context LLMs" (Moonshot AI, 2025)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 4
  @default_dropout 0.1
  @default_block_size 8
  @default_topk 2

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:block_size, pos_integer()}
          | {:topk, pos_integer()}
          | {:causal, boolean()}
          | {:seq_len, pos_integer()}

  @doc """
  Build a MoBA attention model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Model hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of transformer layers (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:block_size` - KV block size for routing (default: 8)
    - `:topk` - Number of blocks to select per query (default: 2)
    - `:causal` - Whether to use causal masking (default: true)
    - `:seq_len` - Expected sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.

  ## Examples

      iex> model = Edifice.Attention.MoBA.build(embed_dim: 32, hidden_size: 16, num_heads: 2, num_layers: 1, block_size: 4, topk: 2)
      iex> %Axon{} = model
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    block_size = Keyword.get(opts, :block_size, @default_block_size)
    topk = Keyword.get(opts, :topk, @default_topk)
    causal = Keyword.get(opts, :causal, true)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "moba_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            moba_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              block_size: block_size,
              topk: topk,
              causal: causal,
              name: attn_name
            )
          end

          TransformerBlock.layer(input,
            attention_fn: attn_fn,
            hidden_size: hidden_size,
            dropout: dropout,
            name: name
          )
        end
      )
    )
  end

  defp moba_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    block_size = Keyword.get(opts, :block_size, @default_block_size)
    topk = Keyword.get(opts, :topk, @default_topk)
    causal = Keyword.get(opts, :causal, true)
    name = Keyword.get(opts, :name, "moba_attn")

    head_dim = div(hidden_size, num_heads)

    # Project to Q, K, V
    qkv = Axon.dense(input, hidden_size * 3, name: "#{name}_qkv")

    # MoBA attention computation
    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          {batch, seq_len, _} = Nx.shape(qkv_tensor)

          # Split Q, K, V
          query = Nx.slice_along_axis(qkv_tensor, 0, hidden_size, axis: 2)
          key = Nx.slice_along_axis(qkv_tensor, hidden_size, hidden_size, axis: 2)
          value = Nx.slice_along_axis(qkv_tensor, hidden_size * 2, hidden_size, axis: 2)

          # Reshape to multi-head: [batch, heads, seq, head_dim]
          query = reshape_heads(query, batch, seq_len, num_heads, head_dim)
          key = reshape_heads(key, batch, seq_len, num_heads, head_dim)
          value = reshape_heads(value, batch, seq_len, num_heads, head_dim)

          # Build block-level attention mask
          # Compute block means for K: [batch, heads, num_blocks, head_dim]
          num_blocks = max(div(seq_len, block_size), 1)
          actual_topk = min(topk, num_blocks)

          k_block_means = compute_block_means(key, seq_len, num_blocks, block_size)

          # Gate scores: [batch, heads, seq, num_blocks]
          gate = Nx.dot(query, [3], [0, 1], k_block_means, [3], [0, 1])

          # Causal block mask: query at pos p can only attend to blocks 0..floor(p/B)
          gate =
            if causal do
              apply_causal_block_mask(gate, seq_len, num_blocks, block_size)
            else
              gate
            end

          # Soft top-k: apply softmax to gate scores, zero out low-scoring blocks
          # Use threshold-based selection (differentiable approximation)
          gate_selected = soft_topk_mask(gate, actual_topk)

          # Expand block-level mask to token-level: [batch, heads, seq, seq]
          token_mask = expand_block_mask(gate_selected, seq_len, num_blocks, block_size)

          # Apply causal token mask
          token_mask =
            if causal do
              causal = causal_mask(seq_len)
              Nx.multiply(token_mask, causal)
            else
              token_mask
            end

          # Standard attention with mask
          scale = :math.sqrt(head_dim)
          scores = Nx.divide(Nx.dot(query, [3], [0, 1], key, [3], [0, 1]), scale)

          # Apply mask: set unselected positions to -inf
          scores =
            Nx.select(
              Nx.greater(token_mask, 0.0),
              scores,
              Nx.broadcast(-1.0e9, Nx.shape(scores))
            )

          # Softmax and weighted values
          weights = softmax_last_axis(scores)
          attn_out = Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])

          # Reshape back: [batch, seq, hidden_size]
          unshape_heads(attn_out, batch, seq_len, num_heads, head_dim)
        end,
        name: "#{name}_compute"
      )

    # Output projection
    Axon.dense(attended, hidden_size, name: "#{name}_out_proj")
  end

  defp reshape_heads(tensor, batch, seq_len, num_heads, head_dim) do
    tensor
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defp unshape_heads(tensor, batch, seq_len, num_heads, head_dim) do
    tensor
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  defp compute_block_means(k, seq_len, num_blocks, block_size) do
    {batch, num_heads, _seq, head_dim} = Nx.shape(k)
    padded_len = num_blocks * block_size

    k_padded =
      if padded_len > seq_len do
        padding =
          Nx.broadcast(0.0, {batch, num_heads, padded_len - seq_len, head_dim})
          |> Nx.as_type(Nx.type(k))

        Nx.concatenate([k, padding], axis: 2)
      else
        Nx.slice_along_axis(k, 0, padded_len, axis: 2)
      end

    k_padded
    |> Nx.reshape({batch, num_heads, num_blocks, block_size, head_dim})
    |> Nx.mean(axes: [3])
  end

  defp apply_causal_block_mask(gate, seq_len, num_blocks, block_size) do
    # Query at position p can only attend to blocks 0..floor(p/B)
    q_positions = Nx.iota({seq_len, 1}, axis: 0)

    # mask[q, b] = 1 if block_start(b) <= q, i.e. b * block_size <= q
    block_starts = Nx.multiply(Nx.iota({1, num_blocks}, axis: 1), block_size)
    mask = Nx.greater_equal(q_positions, block_starts)

    # Broadcast mask to gate shape: [batch, heads, seq_len, num_blocks]
    mask = Nx.broadcast(Nx.reshape(mask, {1, 1, seq_len, num_blocks}), Nx.shape(gate))

    neg_inf = Nx.broadcast(-1.0e9, Nx.shape(gate))
    Nx.select(mask, gate, neg_inf)
  end

  defp soft_topk_mask(gate, topk) do
    # Sort gate scores descending, find the k-th value as threshold
    sorted = Nx.sort(gate, axis: 3, direction: :desc)
    # threshold: the topk-th largest value
    threshold = Nx.slice_along_axis(sorted, topk - 1, 1, axis: 3)

    # Mask: 1 if gate >= threshold, 0 otherwise
    Nx.select(
      Nx.greater_equal(gate, threshold),
      Nx.broadcast(1.0, Nx.shape(gate)),
      Nx.broadcast(0.0, Nx.shape(gate))
    )
  end

  defp expand_block_mask(block_mask, seq_len, num_blocks, block_size) do
    # block_mask: [batch, heads, seq_len, num_blocks]
    # Expand to: [batch, heads, seq_len, seq_len]
    {batch, num_heads, _seq, _nblocks} = Nx.shape(block_mask)

    # For each KV position j, its block index is floor(j / block_size)
    kv_block_idx = Nx.divide(Nx.iota({seq_len}), block_size) |> Nx.as_type(:s32)

    # Gather block mask values for each KV position
    # block_mask[:, :, :, kv_block_idx[j]] for each j
    # Use Nx.take to gather along the block dimension
    # Reshape block_mask for gathering: [batch * heads * seq_len, num_blocks]
    flat_mask = Nx.reshape(block_mask, {batch * num_heads * seq_len, num_blocks})

    # For each of the batch*heads*seq query positions, gather the block scores
    # for each kv position's block
    # kv_block_idx: [seq_len] -> gather from flat_mask along axis 1
    token_mask = Nx.take(flat_mask, kv_block_idx, axis: 1)

    Nx.reshape(token_mask, {batch, num_heads, seq_len, seq_len})
  end

  defp causal_mask(seq_len) do
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    mask = Nx.select(Nx.greater_equal(rows, cols), 1.0, 0.0)
    Nx.reshape(mask, {1, 1, seq_len, seq_len})
  end

  defp softmax_last_axis(tensor) do
    max_val = Nx.reduce_max(tensor, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(tensor, max_val)
    exp_vals = Nx.exp(shifted)
    sum_exp = Nx.sum(exp_vals, axes: [-1], keep_axes: true)
    Nx.divide(exp_vals, sum_exp)
  end

  @doc """
  Get the output size of a MoBA model.
  """
  @spec output_size(keyword()) :: pos_integer()
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
      block_size: 8,
      topk: 2,
      dropout: 0.1,
      causal: true
    ]
  end
end
