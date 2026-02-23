defmodule Edifice.Attention.GatedAttention do
  @moduledoc """
  Gated Attention: learned gating over attention output.

  Applies a learnable sigmoid gate to attention output:

  ```
  output = sigmoid(g) * Attention(Q, K, V)
  ```

  Where `g` is a learned gate vector (one scalar per hidden dimension).
  This allows the model to selectively suppress or amplify attention outputs
  per feature dimension.

  ## Key Innovation

  Standard attention outputs are weighted sums that can be noisy. The gate
  learns which dimensions of the attention output are reliable/useful and
  which should be dampened. This is similar to gating in LSTMs/GRUs but
  applied to attention.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  +------------------------------+
  |  Gated Attention Block       |
  |                              |
  |  Q, K, V projections         |
  |         |                    |
  |  Standard attention          |
  |         |                    |
  |  sigmoid(g) * attn_out       |
  |         |                    |
  |  Output projection           |
  +------------------------------+
        |
  [batch, seq_len, hidden_size]
  ```

  ## Usage

      model = GatedAttention.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 6
      )

  ## Reference

  - "Gated Attention Networks" (NeurIPS 2025 Best Paper)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}
  alias Edifice.Utils.FusedOps

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 6
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Gated Attention model.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of transformer blocks (default: 6)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "gated_attn_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_gated_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
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

  @doc """
  Build the gated attention layer.

  Projects to Q, K, V, computes standard attention, then applies
  learned sigmoid gate to the output.
  """
  @spec build_gated_attention(Axon.t(), keyword()) :: Axon.t()
  def build_gated_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    name = Keyword.get(opts, :name, "gated_attn")

    head_dim = div(hidden_size, num_heads)

    # Q, K, V projections
    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # Gate parameter: learned vector of size hidden_size
    gate = Axon.param("#{name}_gate", {hidden_size}, initializer: :zeros)

    output =
      Axon.layer(
        &gated_attention_impl/5,
        [q_proj, k_proj, v_proj, gate],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :gated_attention
      )

    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # Gated attention implementation
  # Q, K, V: [batch, seq, hidden], gate: [hidden]
  defp gated_attention_impl(q, k, v, gate, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    {batch, seq_len, _} = Nx.shape(q)

    # Reshape to multi-head: [batch, heads, seq, head_dim]
    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Scale factor
    scale = Nx.sqrt(head_dim) |> Nx.as_type(Nx.type(q))

    # Attention scores: [batch, heads, seq, seq]
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Causal mask
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal_mask = Nx.greater_equal(rows, cols)

    causal_mask =
      causal_mask
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    scores = Nx.select(causal_mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))
    attn_weights = FusedOps.fused_softmax(scores)

    # Attention output: [batch, heads, seq, head_dim]
    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, seq, hidden]
    attn_out = reshape_from_heads(attn_out, batch, seq_len, num_heads, head_dim)

    # Apply learned gate: sigmoid(g) * attn_out
    # gate is [hidden], broadcast to [batch, seq, hidden]
    gate_activated = Nx.sigmoid(gate)
    Nx.multiply(gate_activated, attn_out)
  end

  # Reshape [batch, seq, hidden] -> [batch, heads, seq, head_dim]
  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Reshape [batch, heads, seq, head_dim] -> [batch, seq, hidden]
  defp reshape_from_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
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
      num_layers: 6,
      dropout: 0.1
    ]
  end
end
