defmodule Edifice.Blocks.CrossAttention do
  @moduledoc """
  Cross-Attention Layer.

  Standard encoder-decoder attention where queries come from one sequence
  and keys/values come from another. Used in U-Net conditioning, Perceiver,
  CLIP text-image alignment, and sequence-to-sequence models.

  ## Architecture

  ```
  Query source [batch, seq_q, dim_q]    KV source [batch, seq_kv, dim_kv]
        |                                      |
     Dense Wq                           Dense Wk, Wv
        |                                  |      |
        Q                                  K      V
        |                                  |      |
        +----- Attention(Q, K, V) ---------+------+
                      |
               Dense Wo (output projection)
                      |
  Output [batch, seq_q, hidden_dim]
  ```

  ## Usage

      output = CrossAttention.layer(queries, context,
        hidden_dim: 256,
        num_heads: 4,
        name: "cross_attn"
      )

  ## References
  - "Attention Is All You Need" (Vaswani et al., 2017)
  """

  require Axon
  alias Edifice.Utils.FusedOps

  @doc """
  Build a cross-attention Axon layer.

  ## Parameters
    - `query_input` - Query sequence Axon node [batch, seq_q, dim_q]
    - `kv_input` - Key-value sequence Axon node [batch, seq_kv, dim_kv]

  ## Options
    - `:hidden_dim` - Hidden dimension for Q, K, V projections (required)
    - `:num_heads` - Number of attention heads (default: 1)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:name` - Layer name prefix (default: "cross_attn")
  """
  @spec layer(Axon.t(), Axon.t(), keyword()) :: Axon.t()
  def layer(query_input, kv_input, opts \\ []) do
    hidden_dim = Keyword.fetch!(opts, :hidden_dim)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.get(opts, :name, "cross_attn")

    # Project queries from query input
    query_proj = Axon.dense(query_input, hidden_dim, name: "#{name}_q_proj")

    # Project keys and values from kv input
    key_proj = Axon.dense(kv_input, hidden_dim, name: "#{name}_k_proj")
    value_proj = Axon.dense(kv_input, hidden_dim, name: "#{name}_v_proj")

    # Compute cross-attention
    attended = Axon.layer(
      &cross_attention_impl/4,
      [query_proj, key_proj, value_proj],
      name: "#{name}_compute",
      op_name: :cross_attention
    )

    # Output projection
    output = Axon.dense(attended, hidden_dim, name: "#{name}_out_proj")

    if dropout > 0 do
      Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
    else
      output
    end
  end

  defp cross_attention_impl(query, key, value, _opts) do
    d_k = Nx.axis_size(key, -1)
    scale = Nx.sqrt(d_k) |> Nx.as_type(Nx.type(query))

    # scores: [batch, seq_q, seq_kv]
    scores = Nx.dot(query, [2], [0], key, [2], [0])
    scores = Nx.divide(scores, scale)

    # softmax over keys
    weights = FusedOps.fused_softmax(scores)

    # weighted sum of values: [batch, seq_q, hidden_dim]
    Nx.dot(weights, [2], [0], value, [1], [0])
  end
end
