defmodule Edifice.Attention.SigmoidAttention do
  @moduledoc """
  Sigmoid Self-Attention: drop-in softmax replacement using normalized sigmoid.

  Replaces the softmax normalization in standard attention with element-wise
  sigmoid plus a sequence-length-dependent bias:

  ```
  SigmoidAttn(X) = sigmoid(QK^T / sqrt(d) + b) * V
  ```

  Where `b = -log(n)` and `n` is the sequence length. This bias ensures finite
  output as sequence length grows and makes each attention weight approximately
  `1/n` at initialization (matching softmax's uniform initialization).

  ## Key Properties

  - **No token competition**: Each attention weight is independent (sigmoid is
    element-wise), unlike softmax where increasing one weight decreases others.
    This enables better gradient flow and parallelization.
  - **Sequence-length normalization**: The `b = -log(n)` bias prevents attention
    scores from growing unbounded with sequence length.
  - **FlashSigmoid compatible**: Eliminates the need for row-wise max/sum
    tracking in tiled attention, yielding ~17% kernel speedup on H100.
  - **Universal approximation**: Sigmoid attention transformers are universal
    function approximators with improved Lipschitz regularity bounds that
    depend on average (not maximum) input norms.

  ## Stabilization

  Includes optional LayerScale (initialized at 1e-4) and QK-norm for stable
  training, as recommended by the paper.

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  +------------------------------------------+
  |  Sigmoid Attention Block (x num_layers)  |
  |                                          |
  |  LayerNorm                               |
  |  Q, K projections (+ optional QK-norm)   |
  |  V projection                            |
  |  scores = QK^T / sqrt(d) - log(n)       |
  |  weights = sigmoid(scores) * causal_mask |
  |  output = weights @ V                    |
  |  Optional LayerScale(1e-4)               |
  |  + Residual                              |
  |                                          |
  |  LayerNorm + FFN + Residual              |
  +------------------------------------------+
        |
  [batch, hidden_size]
  ```

  ## Usage

      model = SigmoidAttention.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 6,
        layer_scale: true,
        qk_norm: true
      )

  ## References

  - Ramapuram et al., "Theory, Analysis, and Best Practices for Sigmoid
    Self-Attention" (ICLR 2025) https://arxiv.org/abs/2409.04431
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 6
  @default_dropout 0.1
  @default_layer_scale true
  @default_layer_scale_init 1.0e-4
  @default_qk_norm false

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}
          | {:layer_scale, boolean()}
          | {:layer_scale_init, float()}
          | {:qk_norm, boolean()}

  @doc """
  Build a transformer model using Sigmoid Self-Attention.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of transformer blocks (default: 6)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)
    - `:layer_scale` - Enable LayerScale on attention output (default: true)
    - `:layer_scale_init` - LayerScale initial value (default: 1.0e-4)
    - `:qk_norm` - Apply LayerNorm to Q and K before attention (default: false)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    layer_scale = Keyword.get(opts, :layer_scale, @default_layer_scale)
    layer_scale_init = Keyword.get(opts, :layer_scale_init, @default_layer_scale_init)
    qk_norm = Keyword.get(opts, :qk_norm, @default_qk_norm)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "sig_attn_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            build_sigmoid_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
              layer_scale: layer_scale,
              layer_scale_init: layer_scale_init,
              qk_norm: qk_norm,
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
  Build the sigmoid attention layer.

  Projects to Q, K, V, computes `sigmoid(QK^T/sqrt(d) - log(n)) * V` with
  causal masking, optional QK-norm, and optional LayerScale.
  """
  @spec build_sigmoid_attention(Axon.t(), keyword()) :: Axon.t()
  def build_sigmoid_attention(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    layer_scale = Keyword.get(opts, :layer_scale, @default_layer_scale)
    layer_scale_init = Keyword.get(opts, :layer_scale_init, @default_layer_scale_init)
    qk_norm = Keyword.get(opts, :qk_norm, @default_qk_norm)
    name = Keyword.get(opts, :name, "sig_attn")

    head_dim = div(hidden_size, num_heads)

    # Q, K, V projections
    q_proj = Axon.dense(input, hidden_size, name: "#{name}_q_proj")
    k_proj = Axon.dense(input, hidden_size, name: "#{name}_k_proj")
    v_proj = Axon.dense(input, hidden_size, name: "#{name}_v_proj")

    # Optional QK-norm: LayerNorm on Q and K for training stability
    {q_proj, k_proj} =
      if qk_norm do
        {
          Axon.layer_norm(q_proj, name: "#{name}_q_norm"),
          Axon.layer_norm(k_proj, name: "#{name}_k_norm")
        }
      else
        {q_proj, k_proj}
      end

    # Core sigmoid attention
    output =
      Axon.layer(
        &sigmoid_attention_impl/4,
        [q_proj, k_proj, v_proj],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :sigmoid_attention
      )

    # Optional LayerScale: learnable per-dimension scaling initialized small
    output =
      if layer_scale do
        ls_param =
          Axon.param("#{name}_ls", {hidden_size},
            initializer: fn _, _ -> Nx.broadcast(Nx.tensor(layer_scale_init), {hidden_size}) end
          )

        Axon.layer(
          fn x, gamma, _opts ->
            Nx.multiply(x, gamma)
          end,
          [output, ls_param],
          name: "#{name}_layer_scale",
          op_name: :layer_scale
        )
      else
        output
      end

    Axon.dense(output, hidden_size, name: "#{name}_out_proj")
  end

  @doc """
  Apply sigmoid attention to pre-computed score logits.

  Computes `sigmoid(logits - log(n))` where `n` is the key sequence length.
  This is the core normalization function, usable as a standalone drop-in
  replacement for softmax in any attention computation.

  ## Parameters

    - `logits` - Attention scores `[..., seq_q, seq_k]`

  ## Returns

    Attention weights with same shape as logits, values in `(0, 1)`.
  """
  @spec compute(Nx.Tensor.t()) :: Nx.Tensor.t()
  def compute(logits) do
    # b = -log(n) where n is the key dimension (last axis)
    n = Nx.axis_size(logits, -1)
    bias = Nx.negate(Nx.log(Nx.tensor(n, type: Nx.type(logits))))
    Nx.sigmoid(Nx.add(logits, bias))
  end

  # Sigmoid attention implementation
  # Q, K, V: [batch, seq, hidden]
  defp sigmoid_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    {batch, seq_len, _} = Nx.shape(q)

    # Reshape to multi-head: [batch, heads, seq, head_dim]
    q = reshape_to_heads(q, batch, seq_len, num_heads, head_dim)
    k = reshape_to_heads(k, batch, seq_len, num_heads, head_dim)
    v = reshape_to_heads(v, batch, seq_len, num_heads, head_dim)

    # Scale factor
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))

    # Attention scores: [batch, heads, seq, seq]
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.divide(scores, scale)

    # Sequence-length-dependent bias: b = -log(n)
    bias = Nx.negate(Nx.log(Nx.tensor(seq_len, type: Nx.type(scores))))
    scores = Nx.add(scores, bias)

    # Sigmoid instead of softmax — the key innovation
    attn_weights = Nx.sigmoid(scores)

    # Causal mask: zero out future positions by multiplying with binary mask
    # (sigmoid is element-wise, so multiplying by 0 gives exact zero)
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)

    causal_mask =
      Nx.greater_equal(rows, cols)
      |> Nx.new_axis(0)
      |> Nx.new_axis(0)
      |> Nx.broadcast({batch, num_heads, seq_len, seq_len})
      |> Nx.as_type(Nx.type(attn_weights))

    attn_weights = Nx.multiply(attn_weights, causal_mask)

    # Attention output: [batch, heads, seq, head_dim]
    attn_out = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Reshape back: [batch, seq, hidden]
    reshape_from_heads(attn_out, batch, seq_len, num_heads, head_dim)
  end

  # Reshape [batch, seq, hidden] -> [batch, heads, seq, head_dim]
  @spec reshape_to_heads(
          Nx.Tensor.t(),
          pos_integer(),
          pos_integer(),
          pos_integer(),
          pos_integer()
        ) ::
          Nx.Tensor.t()
  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Reshape [batch, heads, seq, head_dim] -> [batch, seq, hidden]
  @spec reshape_from_heads(
          Nx.Tensor.t(),
          pos_integer(),
          pos_integer(),
          pos_integer(),
          pos_integer()
        ) :: Nx.Tensor.t()
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
      dropout: 0.1,
      layer_scale: true,
      qk_norm: false
    ]
  end
end
