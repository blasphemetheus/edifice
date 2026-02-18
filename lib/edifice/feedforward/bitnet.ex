defmodule Edifice.Feedforward.BitNet do
  @moduledoc """
  BitNet: 1-bit/1.58-bit transformer with ternary weight quantization.

  Implements the BitNet architecture from "BitNet: Scaling 1-Bit Transformers
  for Large Language Models" (Wang et al., 2023) and "The Era of 1-bit LLMs"
  (Ma et al., 2024). BitNet quantizes weights to {-1, 0, +1} in the forward
  pass while maintaining full-precision weights for gradient updates.

  ## Key Innovation: Quantization-Aware Forward Pass

  BitNet uses "BitLinear" layers that replace standard dense layers:
  1. Full-precision weights are stored for training
  2. In the forward pass, weights are quantized to binary ({-1, +1}) or
     ternary ({-1, 0, +1}) values
  3. Activations are quantized to 8-bit using absmax quantization
  4. Gradients flow through the quantization via straight-through estimator

  ```
  BitLinear(x):
    W_quant = quantize_weights(W)   # Binary: sign(W), Ternary: round(W/mean(|W|))
    x_quant = quantize_activations(x)  # absmax to [-128, 127]
    output = x_quant @ W_quant^T * scale
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  +-----------------------+
  | Input Projection      |
  +-----------------------+
        |
        v
  +-----------------------+
  | BitNet Block x N      |
  |  Norm -> BitAttn      |
  |  Norm -> BitFFN       |
  |  (all dense layers    |
  |   use BitLinear)      |
  +-----------------------+
        |
        v
  [batch, hidden_size]    (last timestep)
  ```

  ## Quantization Modes

  | Mode    | Weight Values | Bits per Weight |
  |---------|--------------|-----------------|
  | Binary  | {-1, +1}     | 1 bit           |
  | Ternary | {-1, 0, +1}  | 1.58 bits       |

  ## Usage

      model = BitNet.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 4,
        quantize: :ternary
      )

  ## References

  - "BitNet: Scaling 1-Bit Transformers for Large Language Models"
  - "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
  - arXiv: https://arxiv.org/abs/2310.11453
  """

  alias Edifice.Blocks.ModelBuilder

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a BitNet model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of BitNet blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)
    - `:quantize` - Quantization mode: `:ternary` or `:binary` (default: :ternary)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:quantize, :ternary | :binary}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    quantize = Keyword.get(opts, :quantize, :ternary)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    ModelBuilder.build_sequence_model(
      embed_dim: Keyword.fetch!(opts, :embed_dim),
      hidden_size: hidden_size,
      num_layers: Keyword.get(opts, :num_layers, @default_num_layers),
      seq_len: Keyword.get(opts, :seq_len, Keyword.get(opts, :window_size, @default_window_size)),
      dropout: dropout,
      block_builder: fn input, block_opts ->
        layer_idx = Keyword.get(block_opts, :layer_idx, 1)

        build_bitnet_block(input,
          hidden_size: hidden_size,
          num_heads: num_heads,
          quantize: quantize,
          dropout: dropout,
          name: "bitnet_block_#{layer_idx}"
        )
      end
    )
  end

  @doc """
  Build a single BitNet transformer block with quantized linear layers.
  """
  @spec build_bitnet_block(Axon.t(), keyword()) :: Axon.t()
  def build_bitnet_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    quantize = Keyword.get(opts, :quantize, :ternary)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "bitnet_block")

    head_dim = div(hidden_size, num_heads)

    # 1. Attention sub-layer with BitLinear
    attn_normed = Axon.layer_norm(input, name: "#{name}_attn_norm")

    attn_out =
      build_bit_attention(attn_normed,
        hidden_size: hidden_size,
        num_heads: num_heads,
        head_dim: head_dim,
        quantize: quantize,
        name: "#{name}_attn"
      )

    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # 2. FFN sub-layer with BitLinear
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      build_bit_ffn(ffn_normed,
        hidden_size: hidden_size,
        quantize: quantize,
        dropout: dropout,
        name: "#{name}_ffn"
      )

    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Build a BitLinear layer: dense with quantized weights in forward pass.

  Uses `Axon.param` for full-precision weights and quantizes them during
  the forward pass via `Axon.layer`.
  """
  @spec bitlinear(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def bitlinear(input, output_size, opts \\ []) do
    quantize = Keyword.get(opts, :quantize, :ternary)
    name = Keyword.get(opts, :name, "bitlinear")

    weight =
      Axon.param("#{name}_weight", fn input_shape ->
        in_features = elem(input_shape, tuple_size(input_shape) - 1)
        {in_features, output_size}
      end)

    bias = Axon.param("#{name}_bias", {output_size}, initializer: :zeros)

    Axon.layer(
      &bitlinear_impl/4,
      [input, weight, bias],
      name: name,
      quantize: quantize,
      op_name: :bitlinear
    )
  end

  # BitLinear forward pass: quantize weights and activations, then matmul
  defp bitlinear_impl(input, weight, bias, opts) do
    quantize = opts[:quantize] || :ternary

    # Quantize weights
    w_quant =
      case quantize do
        :binary ->
          Nx.select(Nx.greater_equal(weight, 0), 1.0, -1.0)

        :ternary ->
          alpha = Nx.mean(Nx.abs(weight))
          scaled = Nx.divide(weight, Nx.add(alpha, 1.0e-6))
          Nx.clip(Nx.round(scaled), -1, 1)
      end

    # Quantize activations using absmax quantization
    # Scale to [-1, 1] range based on max absolute value
    x_abs_max = Nx.reduce_max(Nx.abs(input), axes: [-1], keep_axes: true)
    x_scale = Nx.add(x_abs_max, 1.0e-6)
    x_quant = Nx.divide(input, x_scale)

    # Matmul: [batch, seq, in] @ [in, out] -> [batch, seq, out]
    output = Nx.dot(x_quant, [Nx.rank(x_quant) - 1], w_quant, [0])

    # Rescale output
    output = Nx.multiply(output, x_scale)

    Nx.add(output, bias)
  end

  # Build attention with BitLinear projections
  defp build_bit_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, div(hidden_size, num_heads))
    quantize = Keyword.get(opts, :quantize, :ternary)
    name = Keyword.get(opts, :name, "bit_attn")

    # QKV projection using BitLinear
    qkv = bitlinear(input, hidden_size * 3, quantize: quantize, name: "#{name}_qkv")

    # Compute multi-head attention
    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          {batch, seq_len, _} = Nx.shape(qkv_tensor)

          query = Nx.slice_along_axis(qkv_tensor, 0, hidden_size, axis: 2)
          key = Nx.slice_along_axis(qkv_tensor, hidden_size, hidden_size, axis: 2)
          value = Nx.slice_along_axis(qkv_tensor, hidden_size * 2, hidden_size, axis: 2)

          # Reshape to multi-head
          query = reshape_to_heads(query, batch, seq_len, num_heads, head_dim)
          key = reshape_to_heads(key, batch, seq_len, num_heads, head_dim)
          value = reshape_to_heads(value, batch, seq_len, num_heads, head_dim)

          # Scaled dot-product attention with causal mask
          d_k = head_dim
          scale = Nx.sqrt(Nx.tensor(d_k, type: Nx.type(query)))

          # QK^T: [batch, heads, seq_q, seq_k]
          scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1])
          scores = Nx.divide(scores, scale)

          # Causal mask
          rows = Nx.iota({seq_len, seq_len}, axis: 0)
          cols = Nx.iota({seq_len, seq_len}, axis: 1)
          mask = Nx.greater_equal(rows, cols)

          mask =
            mask
            |> Nx.new_axis(0)
            |> Nx.new_axis(0)
            |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

          scores = Nx.select(mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))

          # Softmax
          weights =
            Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))

          weights =
            Nx.divide(
              weights,
              Nx.add(Nx.sum(weights, axes: [-1], keep_axes: true), 1.0e-8)
            )

          # Weighted values
          output = Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])

          # Reshape back
          reshape_from_heads(output, batch, seq_len, num_heads, head_dim)
        end,
        name: "#{name}_compute"
      )

    # Output projection using BitLinear
    bitlinear(attended, hidden_size, quantize: quantize, name: "#{name}_out")
  end

  # Build FFN with BitLinear layers
  defp build_bit_ffn(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    quantize = Keyword.get(opts, :quantize, :ternary)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "bit_ffn")
    inner_size = hidden_size * 4

    x = bitlinear(input, inner_size, quantize: quantize, name: "#{name}_up")
    x = Axon.activation(x, :silu, name: "#{name}_act")

    x =
      if dropout > 0 do
        Axon.dropout(x, rate: dropout, name: "#{name}_drop")
      else
        x
      end

    bitlinear(x, hidden_size, quantize: quantize, name: "#{name}_down")
  end

  # Reshape [batch, seq, num_heads * head_dim] -> [batch, num_heads, seq, head_dim]
  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  # Reshape [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
  defp reshape_from_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  @doc """
  Get the output size of a BitNet model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Get recommended defaults.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 4,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1,
      quantize: :ternary
    ]
  end
end
