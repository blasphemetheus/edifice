defmodule Edifice.Blocks.SSMax do
  @moduledoc """
  Scalable-Softmax (SSMax): sequence-length-aware softmax.

  SSMax adjusts softmax temperature based on sequence length to maintain
  consistent attention sharpness across different context sizes:

  ```
  SSMax(x)_i = exp(x_i - s*log(n)) / sum_j(exp(x_j - s*log(n)))
  ```

  Where:
  - `n` is the sequence length
  - `s` is a learnable scalar (default initialization: 1.0)

  ## Key Innovation

  Standard softmax becomes increasingly uniform as sequence length grows
  (more tokens to distribute attention over). SSMax learns to compensate:
  - `s > 0`: Sharper attention for longer sequences
  - `s = 0`: Standard softmax behavior
  - `s < 0`: Softer attention for longer sequences

  ## Usage as Drop-in Softmax Replacement

      # In attention computation
      scores = Nx.dot(q, Nx.transpose(k))
      scores = Nx.divide(scores, scale)
      attn_weights = SSMax.compute(scores, s_param, seq_len)

  ## Usage in Axon Model

      model = SSMax.build(embed_dim: 256, hidden_size: 256)

  ## Reference

  - "Scalable-Softmax Is All You Need" (2025)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}
  alias Edifice.Attention.MultiHead

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 6
  @default_dropout 0.1

  @doc """
  Apply SSMax to logits tensor.

  ## Parameters

    - `logits` - Attention scores [batch, ..., seq_len]
    - `s` - Learnable scaling parameter (scalar)
    - `seq_len` - Sequence length (integer)

  ## Returns

    Normalized attention weights with same shape as logits.
  """
  @spec compute(Nx.Tensor.t(), Nx.Tensor.t() | float(), pos_integer()) :: Nx.Tensor.t()
  def compute(logits, s, seq_len) do
    # SSMax(x)_i = exp(x_i - s*log(n)) / sum(exp(x_j - s*log(n)))
    # This is equivalent to softmax with a sequence-length-dependent shift
    log_n = Nx.log(seq_len) |> Nx.as_type(Nx.type(logits))
    shift = Nx.multiply(s, log_n)

    # Subtract shift from all logits (for numerical stability, combine with max subtraction)
    shifted_logits = Nx.subtract(logits, shift)

    # Standard numerically-stable softmax on shifted logits
    max_val = Nx.reduce_max(shifted_logits, axes: [-1], keep_axes: true)
    exp_logits = Nx.exp(Nx.subtract(shifted_logits, max_val))
    sum_exp = Nx.sum(exp_logits, axes: [-1], keep_axes: true)

    Nx.divide(exp_logits, sum_exp)
  end

  @doc """
  Create an SSMax Axon layer that learns the scaling parameter.

  ## Options

    - `:name` - Layer name prefix (default: "ssmax")
    - `:init_s` - Initial value for s parameter (default: 1.0)

  ## Returns

    An Axon layer that applies SSMax to input logits.
  """
  @spec layer(Axon.t(), keyword()) :: Axon.t()
  def layer(input, opts \\ []) do
    name = Keyword.get(opts, :name, "ssmax")
    init_s = Keyword.get(opts, :init_s, 1.0)

    # Learnable s parameter
    s_param =
      Axon.param("#{name}_s", {},
        initializer: fn _, _ -> Nx.tensor(init_s) end
      )

    Axon.layer(
      fn logits, s, _opts ->
        seq_len = Nx.axis_size(logits, -1)
        compute(logits, s, seq_len)
      end,
      [input, s_param],
      name: "#{name}_apply",
      op_name: :ssmax
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
  Build a transformer model using SSMax instead of standard softmax.

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
          name = "ssmax_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_ssmax_attention(x,
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
  Build attention layer using SSMax instead of softmax.
  """
  @spec build_ssmax_attention(Axon.t(), keyword()) :: Axon.t()
  def build_ssmax_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    name = Keyword.get(opts, :name, "ssmax_attn")

    head_dim = div(hidden_size, num_heads)

    # Q, K, V projections
    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # Learnable s parameter for SSMax
    s_param =
      Axon.param("#{name}_s", {},
        initializer: fn _, _ -> Nx.tensor(1.0) end
      )

    output =
      Axon.layer(
        &ssmax_attention_impl/5,
        [q_proj, k_proj, v_proj, s_param],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :ssmax_attention
      )

    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  # SSMax attention implementation
  defp ssmax_attention_impl(q, k, v, s, opts) do
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

    # Apply SSMax instead of regular softmax
    attn_weights = compute(scores, s, seq_len)

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
