defmodule Edifice.Blocks.CrossAttention do
  @moduledoc """
  Cross-Attention Layer.

  Standard encoder-decoder attention where queries come from one sequence
  and keys/values come from another. Used in U-Net conditioning, Perceiver,
  CLIP text-image alignment, and sequence-to-sequence models.

  ## API Variants

  **Shared KV source** — `layer/3` projects K and V from the same input:

      output = CrossAttention.layer(queries, context,
        hidden_size: 256, num_heads: 4
      )

  **Separate KV sources** — `layer/4` when keys and values come from
  different tensors (e.g., SAM2 adds PE to keys but not values):

      output = CrossAttention.layer(queries, keys_with_pe, values,
        hidden_size: 256, num_heads: 4
      )

  ## References
  - "Attention Is All You Need" (Vaswani et al., 2017)
  """
  alias Edifice.Blocks.SDPA

  @doc """
  Build a cross-attention Axon layer.

  ## Parameters
    - `query_input` - Query sequence Axon node [batch, seq_q, dim_q]
    - `kv_input` - Key-value sequence Axon node [batch, seq_kv, dim_kv]

  ## Options
    - `:hidden_size` - Hidden dimension for Q, K, V projections (required)
    - `:num_heads` - Number of attention heads (default: 1)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:name` - Layer name prefix (default: "cross_attn")

  ## Examples

      iex> q = Axon.input("q", shape: {nil, 4, 32})
      iex> kv = Axon.input("kv", shape: {nil, 8, 32})
      iex> output = Edifice.Blocks.CrossAttention.layer(q, kv, hidden_size: 32, num_heads: 4)
      iex> %Axon{} = output
  """
  @spec layer(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def layer(query_input, kv_input, opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, 1)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.get(opts, :name, "cross_attn")

    head_dim = div(hidden_size, num_heads)

    # Project queries from query input
    query_proj = Axon.dense(query_input, hidden_size, name: "#{name}_q_proj")

    # Project keys and values from kv input
    key_proj = Axon.dense(kv_input, hidden_size, name: "#{name}_k_proj")
    value_proj = Axon.dense(kv_input, hidden_size, name: "#{name}_v_proj")

    # Compute multi-head cross-attention
    attended =
      Axon.layer(
        fn q, k, v, _opts ->
          SDPA.compute(q, k, v, num_heads, head_dim)
        end,
        [query_proj, key_proj, value_proj],
        name: "#{name}_compute",
        op_name: :cross_attention
      )

    # Output projection
    output = Axon.dense(attended, hidden_size, name: "#{name}_out_proj")

    if dropout > 0 do
      Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
    else
      output
    end
  end

  @doc """
  Build a cross-attention layer with separate key and value sources.

  Use when keys and values come from different tensors (e.g., positional
  encoding added to keys but not values).

  ## Parameters
    - `query_input` - Query sequence Axon node [batch, seq_q, dim_q]
    - `key_input` - Key sequence Axon node [batch, seq_kv, dim_k]
    - `value_input` - Value sequence Axon node [batch, seq_kv, dim_v]

  ## Options
    Same as `layer/3`.
  """
  @spec layer(Axon.t(), Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def layer(query_input, key_input, value_input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, 1)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.get(opts, :name, "cross_attn")

    head_dim = div(hidden_size, num_heads)

    query_proj = Axon.dense(query_input, hidden_size, name: "#{name}_q_proj")
    key_proj = Axon.dense(key_input, hidden_size, name: "#{name}_k_proj")
    value_proj = Axon.dense(value_input, hidden_size, name: "#{name}_v_proj")

    attended =
      Axon.layer(
        fn q, k, v, _opts ->
          SDPA.compute(q, k, v, num_heads, head_dim)
        end,
        [query_proj, key_proj, value_proj],
        name: "#{name}_compute",
        op_name: :cross_attention
      )

    output = Axon.dense(attended, hidden_size, name: "#{name}_out_proj")

    if dropout > 0 do
      Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
    else
      output
    end
  end
end
