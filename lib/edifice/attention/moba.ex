defmodule Edifice.Attention.MoBA do
  @moduledoc """
  MoBA: Mixture of Block Attention.

  Applies MoE-style block-level routing to KV cache, where each query selects
  the top-k most relevant KV blocks to attend to. Token attention gathers only
  selected blocks; routing still uses O(T² / block_size) scores. Selecting all
  blocks recovers full attention. This is a portable Nx implementation, not a
  fused sparse CUDA kernel.

  ## Key Idea

  The KV sequence is partitioned into blocks of size B. For each query, block
  relevance is scored using the dot product of the query with each block's
  mean key vector (parameter-free gating). The top-k blocks are selected and
  standard softmax attention is computed only over tokens in selected blocks.
  In causal mode the current block is always included in the top-k budget;
  only completed preceding blocks compete for the remaining slots. Exact top-k
  indices keep ties from exceeding the budget. Partial final blocks are retained.
  Corrected routing changes outputs from checkpoints trained with the old mask.

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

  alias Edifice.Attention.SparseAttention
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

    SparseAttention.validate_positive!(
      block_size: block_size,
      topk: topk,
      num_heads: num_heads,
      hidden_size: hidden_size
    )

    unless rem(hidden_size, num_heads) == 0,
      do: raise(ArgumentError, "hidden_size must be divisible by num_heads")

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

          selected = SparseAttention.route(query, key, block_size, topk, causal)

          attn_out =
            SparseAttention.selected(
              query,
              key,
              value,
              selected,
              block_size,
              causal
            )

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
