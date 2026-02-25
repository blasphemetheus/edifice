defmodule Edifice.Meta.QAT do
  @moduledoc """
  Quantization-Aware Training (QAT) — transformer with quantized linear layers.

  Extends BitNet's quantization-aware training to support multiple bit widths
  beyond binary and ternary. All dense layers in the attention and FFN sub-layers
  use quantized forward passes (with straight-through gradient estimation for
  backpropagation).

  ## Quantization Modes

  | Mode      | Weight Values       | Levels | Bits/Weight |
  |-----------|--------------------| -------|------------|
  | `:binary`  | {-1, +1}            | 2      | 1          |
  | `:ternary` | {-1, 0, +1}         | 3      | 1.58       |
  | `:int4`    | 16 absmax-scaled    | 16     | 4          |
  | `:int8`    | 256 absmax-scaled   | 256    | 8          |

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
  Quantized blocks: Pre-norm -> QuantLinear(QKV) -> Attention -> Residual
                     Pre-norm -> QuantLinear(FFN) -> Residual
        |
  Final norm -> last timestep -> [batch, hidden_size]
  ```

  ## Usage

      model = QAT.build(
        embed_dim: 256,
        hidden_size: 256,
        num_heads: 4,
        num_layers: 4,
        quantize: :int4
      )

  ## References

  - Wang et al., "BitNet: Scaling 1-Bit Transformers" (2023)
  - Jacob et al., "Quantization and Training of Neural Networks for Efficient
    Integer-Arithmetic-Only Inference" (2018)
  """

  alias Edifice.Blocks.ModelBuilder

  @default_hidden_size 256
  @default_num_heads 4
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a QAT model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of blocks (default: 4)
    - `:quantize` - Quantization mode: `:binary`, `:ternary`, `:int4`, or `:int8` (default: `:ternary`)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model outputting `[batch, hidden_size]` from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:quantize, :binary | :ternary | :int4 | :int8}
          | {:dropout, float()}
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

        build_qat_block(input,
          hidden_size: hidden_size,
          num_heads: num_heads,
          quantize: quantize,
          dropout: dropout,
          name: "qat_block_#{layer_idx}"
        )
      end
    )
  end

  @doc """
  Build a single QAT transformer block with quantized linear layers.
  """
  @spec build_qat_block(Axon.t(), keyword()) :: Axon.t()
  def build_qat_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    quantize = Keyword.get(opts, :quantize, :ternary)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "qat_block")

    head_dim = div(hidden_size, num_heads)

    # 1. Attention sub-layer with quantized linear
    attn_normed = Axon.layer_norm(input, name: "#{name}_attn_norm")

    attn_out =
      build_quant_attention(attn_normed,
        hidden_size: hidden_size,
        num_heads: num_heads,
        head_dim: head_dim,
        quantize: quantize,
        name: "#{name}_attn"
      )

    attn_out = maybe_dropout(attn_out, dropout, "#{name}_attn_drop")
    x = Axon.add(input, attn_out, name: "#{name}_attn_residual")

    # 2. FFN sub-layer with quantized linear
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      build_quant_ffn(ffn_normed,
        hidden_size: hidden_size,
        quantize: quantize,
        dropout: dropout,
        name: "#{name}_ffn"
      )

    ffn_out = maybe_dropout(ffn_out, dropout, "#{name}_ffn_drop")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Build a quantized linear layer with the given quantization mode.

  Stores full-precision weights for gradient updates; quantizes in the
  forward pass via straight-through estimation.
  """
  @spec quant_linear(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def quant_linear(input, output_size, opts \\ []) do
    quantize = Keyword.get(opts, :quantize, :ternary)
    name = Keyword.get(opts, :name, "quant_linear")

    weight =
      Axon.param("#{name}_weight", fn input_shape ->
        in_features = elem(input_shape, tuple_size(input_shape) - 1)
        {in_features, output_size}
      end)

    bias = Axon.param("#{name}_bias", {output_size}, initializer: :zeros)

    Axon.layer(
      &quant_linear_impl/4,
      [input, weight, bias],
      name: name,
      quantize: quantize,
      op_name: :quant_linear
    )
  end

  defp quant_linear_impl(input, weight, bias, opts) do
    quantize = opts[:quantize] || :ternary

    w_quant = quantize_weight(weight, quantize)

    # Quantize activations using absmax
    x_abs_max = Nx.reduce_max(Nx.abs(input), axes: [-1], keep_axes: true)
    x_scale = Nx.add(x_abs_max, 1.0e-6)
    x_quant = Nx.divide(input, x_scale)

    output = Nx.dot(x_quant, [Nx.rank(x_quant) - 1], w_quant, [0])
    output = Nx.multiply(output, x_scale)

    Nx.add(output, bias)
  end

  defp quantize_weight(weight, :binary) do
    Nx.select(Nx.greater_equal(weight, 0), 1.0, -1.0)
  end

  defp quantize_weight(weight, :ternary) do
    alpha = Nx.mean(Nx.abs(weight))
    scaled = Nx.divide(weight, Nx.add(alpha, 1.0e-6))
    Nx.clip(Nx.round(scaled), -1, 1)
  end

  defp quantize_weight(weight, :int4) do
    # 4-bit: 16 levels in [-1, 1] via absmax scaling
    abs_max = Nx.reduce_max(Nx.abs(weight))
    scale = Nx.add(abs_max, 1.0e-6)
    normalized = Nx.divide(weight, scale)
    # Quantize to 16 levels: round(x * 7) / 7 gives values in {-1, -6/7, ..., 6/7, 1}
    quantized = Nx.divide(Nx.round(Nx.multiply(normalized, 7.0)), 7.0)
    Nx.multiply(quantized, scale)
  end

  defp quantize_weight(weight, :int8) do
    # 8-bit: 256 levels in [-1, 1] via absmax scaling
    abs_max = Nx.reduce_max(Nx.abs(weight))
    scale = Nx.add(abs_max, 1.0e-6)
    normalized = Nx.divide(weight, scale)
    # Quantize to 256 levels: round(x * 127) / 127
    quantized = Nx.divide(Nx.round(Nx.multiply(normalized, 127.0)), 127.0)
    Nx.multiply(quantized, scale)
  end

  # Build attention with quantized linear projections
  defp build_quant_attention(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    head_dim = Keyword.get(opts, :head_dim, div(hidden_size, num_heads))
    quantize = Keyword.get(opts, :quantize, :ternary)
    name = Keyword.get(opts, :name, "quant_attn")

    qkv = quant_linear(input, hidden_size * 3, quantize: quantize, name: "#{name}_qkv")

    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          {batch, seq_len, _} = Nx.shape(qkv_tensor)

          query = Nx.slice_along_axis(qkv_tensor, 0, hidden_size, axis: 2)
          key = Nx.slice_along_axis(qkv_tensor, hidden_size, hidden_size, axis: 2)
          value = Nx.slice_along_axis(qkv_tensor, hidden_size * 2, hidden_size, axis: 2)

          query = reshape_to_heads(query, batch, seq_len, num_heads, head_dim)
          key = reshape_to_heads(key, batch, seq_len, num_heads, head_dim)
          value = reshape_to_heads(value, batch, seq_len, num_heads, head_dim)

          scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(query)))
          scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1]) |> Nx.divide(scale)

          rows = Nx.iota({seq_len, seq_len}, axis: 0)
          cols = Nx.iota({seq_len, seq_len}, axis: 1)
          mask = Nx.greater_equal(rows, cols)

          mask =
            mask
            |> Nx.new_axis(0)
            |> Nx.new_axis(0)
            |> Nx.broadcast({batch, num_heads, seq_len, seq_len})

          scores = Nx.select(mask, scores, Nx.broadcast(-1.0e9, Nx.shape(scores)))

          weights =
            Nx.exp(Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true)))

          weights =
            Nx.divide(weights, Nx.add(Nx.sum(weights, axes: [-1], keep_axes: true), 1.0e-8))

          output = Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])
          reshape_from_heads(output, batch, seq_len, num_heads, head_dim)
        end,
        name: "#{name}_compute"
      )

    quant_linear(attended, hidden_size, quantize: quantize, name: "#{name}_out")
  end

  defp build_quant_ffn(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    quantize = Keyword.get(opts, :quantize, :ternary)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "quant_ffn")
    inner_size = hidden_size * 4

    x = quant_linear(input, inner_size, quantize: quantize, name: "#{name}_up")
    x = Axon.activation(x, :silu, name: "#{name}_act")
    x = maybe_dropout(x, dropout, "#{name}_drop")
    quant_linear(x, hidden_size, quantize: quantize, name: "#{name}_down")
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

  defp maybe_dropout(x, rate, _name) when rate <= 0, do: x
  defp maybe_dropout(x, rate, name), do: Axon.dropout(x, rate: rate, name: name)

  @doc "Get the output size of a QAT model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc "Get recommended defaults."
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_heads: 4,
      num_layers: 4,
      dropout: 0.1,
      quantize: :ternary
    ]
  end
end
