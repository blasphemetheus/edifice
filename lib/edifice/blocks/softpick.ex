defmodule Edifice.Blocks.Softpick do
  @moduledoc """
  Softpick: non-saturating, naturally sparse normalization.

  Softpick normalizes inputs by dividing by the total absolute magnitude:

  ```
  Softpick(x)_i = x_i / (1 + sum_j(|x_j|))
  ```

  ## Key Properties

  - **Non-saturating**: Unlike softmax, gradients don't vanish for large inputs
  - **Naturally sparse**: Outputs preserve sign and relative magnitudes
  - **Bounded**: Output magnitudes are always < 1 (divided by 1 + sum)
  - **Simple**: No exponentials, just absolute values and division

  ## Comparison with Softmax

  | Property | Softmax | Softpick |
  |----------|---------|----------|
  | Output range | (0, 1) | (-1, 1) |
  | Sum of outputs | 1 | varies |
  | Preserves sign | No | Yes |
  | Saturation | Yes (exp) | No |
  | Sparsity | Low (sum=1) | Natural |

  ## Use Cases

  - Attention alternatives where sign matters
  - Routing in mixture-of-experts
  - Feature selection where sparsity is desired
  - Any normalization where you want bounded outputs without saturation

  ## Usage as Nx Function

      # Direct computation
      normalized = Softpick.compute(logits)

  ## Usage in Axon Model

      model = Softpick.build(embed_dim: 256, hidden_size: 256)

  ## Reference

  - "Beyond Softmax: Sparse and Non-Saturating Attention" (2025)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 6
  @default_dropout 0.1

  @doc """
  Apply Softpick normalization to a tensor.

  ## Parameters

    - `x` - Input tensor of any shape
    - `opts` - Options:
      - `:axis` - Axis to normalize over (default: -1, last axis)

  ## Returns

    Normalized tensor: x_i / (1 + sum(|x_j|)) over the specified axis.
  """
  @spec compute(Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def compute(x, opts \\ []) do
    axis = Keyword.get(opts, :axis, -1)

    # Softpick(x)_i = x_i / (1 + sum(|x_j|))
    abs_sum = Nx.sum(Nx.abs(x), axes: [axis], keep_axes: true)
    Nx.divide(x, Nx.add(1.0, abs_sum))
  end

  @doc """
  Create a Softpick Axon layer.

  ## Options

    - `:name` - Layer name prefix (default: "softpick")
    - `:axis` - Axis to normalize over (default: -1)

  ## Returns

    An Axon layer that applies Softpick normalization.
  """
  @spec layer(Axon.t(), keyword()) :: Axon.t()
  def layer(input, opts \\ []) do
    name = Keyword.get(opts, :name, "softpick")
    axis = Keyword.get(opts, :axis, -1)

    Axon.nx(
      input,
      fn x -> compute(x, axis: axis) end,
      name: name
    )
  end

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a transformer model using Softpick instead of softmax in attention.

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
          name = "softpick_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_softpick_attention(x,
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
  Build attention layer using Softpick instead of softmax.
  """
  @spec build_softpick_attention(Axon.t(), keyword()) :: Axon.t()
  def build_softpick_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    name = Keyword.get(opts, :name, "softpick_attn")

    head_dim = div(hidden_size, num_heads)

    # Q, K, V projections
    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    output =
      Axon.layer(
        &softpick_attention_impl/4,
        [q_proj, k_proj, v_proj],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :softpick_attention
      )

    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # Softpick attention implementation
  defp softpick_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    {batch, seq_len, _} = Nx.shape(q)

    # Reshape to multi-head
    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Scale
    scale = Nx.sqrt(head_dim) |> Nx.as_type(Nx.type(q))

    # Attention scores
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Causal mask - set future positions to 0 (not -inf since we don't use softmax)
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    causal_mask = Nx.greater_equal(rows, cols)

    causal_mask =
      causal_mask
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

    scores = Nx.select(causal_mask, scores, Nx.broadcast(0.0, Nx.shape(scores)))

    # Apply Softpick instead of softmax (normalize over last axis = keys)
    attn_weights = compute(scores, axis: -1)

    # Attention output
    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back
    reshape_from_heads(attn_out, batch, seq_len, num_heads, head_dim)
  end

  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

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
end
