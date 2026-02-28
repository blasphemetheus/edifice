defmodule Edifice.Attention.LASER do
  @moduledoc """
  LASER: Attention with Exponential Transformation.

  Replaces standard softmax attention output with a log-exponential transform
  that produces larger gradient signals. Standard attention suffers from
  vanishing gradients when attention weights are very small (~80% < 1e-3),
  because the softmax Jacobian scales with attention probability magnitude.

  LASER decouples gradient flow from attention weight magnitude by computing:

      O = log(softmax(QK^T) @ exp(V))

  This yields a Log-Weighted-Sum-Exp structure that acts as a differentiable
  max over values, with gradients that don't vanish even when attention
  weights are tiny.

  ```
  Input [batch, seq_len, hidden_size]
        |
  +------------------------------------------+
  | LASER Attention                          |
  | 1. Project Q, K, V                      |
  | 2. m = max(V, dim=seq)  [stop gradient] |
  | 3. V_hat = V - m                        |
  | 4. A = softmax(QK^T/sqrt(d)) @ exp(V_hat)|
  | 5. O = log(A) + m                       |
  +------------------------------------------+
        |
  [batch, seq_len, hidden_size]
  ```

  ## Usage

      model = LASER.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 4
      )

  ## Reference

  - Duvvuri & Dhillon, "LASER: Attention with Exponential Transformation" (ICML 2025)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 4
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:causal, boolean()}
          | {:seq_len, pos_integer()}

  @doc """
  Build a LASER attention model.

  ## Options

    - `:embed_dim` - Input embedding dimension (required)
    - `:hidden_size` - Model hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of transformer layers (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:causal` - Whether to use causal masking (default: true)
    - `:seq_len` - Expected sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]`.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    causal = Keyword.get(opts, :causal, true)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          layer_idx = block_opts[:layer_idx]
          name = "laser_block_#{layer_idx}"

          attn_fn = fn x, attn_name ->
            laser_attention(x,
              hidden_size: hidden_size,
              num_heads: num_heads,
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

  @doc """
  LASER self-attention layer.

  Computes `log(softmax(QK^T / sqrt(d)) @ exp(V))` using the LWSE trick
  for numerical stability.

  ## Options

    - `:hidden_size` - Hidden dimension (required)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:causal` - Use causal masking (default: true)
    - `:name` - Layer name prefix
  """
  @spec self_attention(Axon.t(), keyword()) :: Axon.t()
  def self_attention(input, opts \\ []) do
    laser_attention(input, opts)
  end

  defp laser_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    causal = Keyword.get(opts, :causal, true)
    name = Keyword.get(opts, :name, "laser_attn")

    head_dim = div(hidden_size, num_heads)

    # Project to Q, K, V
    qkv = Axon.dense(input, hidden_size * 3, name: "#{name}_qkv")

    # LASER attention computation
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
          query = reshape_to_heads(query, batch, seq_len, num_heads, head_dim)
          key = reshape_to_heads(key, batch, seq_len, num_heads, head_dim)
          value = reshape_to_heads(value, batch, seq_len, num_heads, head_dim)

          # LWSE trick: subtract max(V) per column for numerical stability
          # m: [batch, heads, 1, head_dim]
          v_max = Nx.reduce_max(value, axes: [2], keep_axes: true)

          # V_hat = V - m (shifted values)
          v_hat = Nx.subtract(value, v_max)

          # exp(V_hat): element-wise exp of shifted values
          exp_v = Nx.exp(v_hat)

          # Standard attention: softmax(QK^T / sqrt(d)) @ exp(V_hat)
          scale = :math.sqrt(head_dim)
          scores = Nx.divide(Nx.dot(query, [3], [0, 1], key, [3], [0, 1]), scale)

          # Causal mask
          scores =
            if causal do
              mask = causal_mask(seq_len)
              Nx.add(scores, mask)
            else
              scores
            end

          # Softmax over key dimension
          weights = softmax_last_axis(scores)

          # Weighted sum of exp(V_hat)
          # weights: [batch, heads, seq_q, seq_k], exp_v: [batch, heads, seq_k, head_dim]
          attn_out = Nx.dot(weights, [3], [0, 1], exp_v, [2], [0, 1])

          # Log + add back max: log(attn_out) + m
          # v_max is [batch, heads, 1, head_dim], broadcasts with [batch, heads, seq, head_dim]
          output = Nx.add(Nx.log(Nx.max(attn_out, 1.0e-7)), v_max)

          # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden_size]
          reshape_from_heads(output, batch, seq_len, num_heads, head_dim)
        end,
        name: "#{name}_compute"
      )

    # Output projection
    Axon.dense(attended, hidden_size, name: "#{name}_out_proj")
  end

  defp reshape_to_heads(tensor, batch, seq_len, num_heads, head_dim) do
    tensor
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defp reshape_from_heads(tensor, batch, seq_len, num_heads, head_dim) do
    tensor
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  defp causal_mask(seq_len) do
    # Lower triangular mask: 0 for valid, -1e9 for invalid
    rows = Nx.iota({seq_len, seq_len}, axis: 0)
    cols = Nx.iota({seq_len, seq_len}, axis: 1)
    mask = Nx.select(Nx.greater_equal(rows, cols), 0.0, -1.0e9)
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
  Get the output size of a LASER model.
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
      dropout: 0.1,
      causal: true
    ]
  end
end
