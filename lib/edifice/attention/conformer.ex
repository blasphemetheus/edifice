defmodule Edifice.Attention.Conformer do
  @moduledoc """
  Conformer: convolution-augmented transformer for audio/speech processing.

  The Conformer combines self-attention with convolution to capture both global
  and local patterns. It uses a Macaron-style architecture with two half-step
  feed-forward modules sandwiching the attention and convolution modules.

  ## Architecture (Macaron Block)

  ```
  Input [batch, seq_len, hidden_size]
        |
  +------------------------------------------------+
  |   Conformer Block (x num_layers)               |
  |                                                |
  |   1. Half-FFN: norm -> FFN -> scale(0.5)       |
  |      -> residual                               |
  |   2. MHSA: norm -> self_attention -> residual  |
  |   3. Conv module:                              |
  |      norm -> pointwise_up -> GLU               |
  |      -> depthwise_conv -> norm -> act           |
  |      -> pointwise_down -> residual             |
  |   4. Half-FFN: norm -> FFN -> scale(0.5)       |
  |      -> residual                               |
  |   5. Final LayerNorm                           |
  +------------------------------------------------+
        |
  Final LayerNorm
        |
  Last timestep -> [batch, hidden_size]
  ```

  ## Usage

      model = Conformer.build(
        embed_dim: 287,
        hidden_size: 256,
        num_heads: 4,
        conv_kernel_size: 31,
        num_layers: 4
      )

  ## References
  - "Conformer: Convolution-augmented Transformer for Speech Recognition"
    (Gulati et al., 2020)
  """

  alias Edifice.Blocks.ModelBuilder

  @default_hidden_size 256
  @default_num_heads 4
  @default_conv_kernel_size 31
  @default_num_layers 4
  @default_dropout 0.1

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:conv_kernel_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Conformer model.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:conv_kernel_size` - Kernel size for depthwise convolution (default: 31)
    - `:num_layers` - Number of Conformer blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs `[batch, hidden_size]` from the last position.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    conv_kernel_size = Keyword.get(opts, :conv_kernel_size, @default_conv_kernel_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          build_conformer_block(input,
            hidden_size: hidden_size,
            num_heads: num_heads,
            conv_kernel_size: conv_kernel_size,
            dropout: dropout,
            name: "conformer_block_#{block_opts[:layer_idx]}"
          )
        end
      )
    )
  end

  @doc """
  Build a single Conformer block with the Macaron structure.
  """
  @spec build_conformer_block(Axon.t(), keyword()) :: Axon.t()
  def build_conformer_block(input, opts) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    conv_kernel_size = Keyword.get(opts, :conv_kernel_size, @default_conv_kernel_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "conformer")

    # 1. First half-step FFN
    x = half_ffn(input, hidden_size, dropout, "#{name}_ffn1")

    # 2. Multi-head self-attention module
    x = mhsa_module(x, hidden_size, num_heads, dropout, "#{name}_mhsa")

    # 3. Convolution module
    x = conv_module(x, hidden_size, conv_kernel_size, dropout, "#{name}_conv")

    # 4. Second half-step FFN
    x = half_ffn(x, hidden_size, dropout, "#{name}_ffn2")

    # 5. Final layer norm
    Axon.layer_norm(x, name: "#{name}_final_norm")
  end

  # Half-step FFN: norm -> FFN -> scale(0.5) -> residual
  defp half_ffn(input, hidden_size, dropout, name) do
    inner_size = hidden_size * 4

    ffn_out =
      input
      |> Axon.layer_norm(name: "#{name}_norm")
      |> Axon.dense(inner_size, name: "#{name}_up")
      |> Axon.activation(:silu, name: "#{name}_act")
      |> Axon.dropout(rate: dropout, name: "#{name}_dropout")
      |> Axon.dense(hidden_size, name: "#{name}_down")

    # Scale by 0.5
    scaled =
      Axon.nx(ffn_out, fn t -> Nx.multiply(t, 0.5) end, name: "#{name}_scale")

    Axon.add(input, scaled, name: "#{name}_residual")
  end

  # Multi-head self-attention module: norm -> MHSA -> dropout -> residual
  defp mhsa_module(input, hidden_size, num_heads, dropout, name) do
    head_dim = div(hidden_size, num_heads)

    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # QKV projection
    qkv = Axon.dense(normed, hidden_size * 3, name: "#{name}_qkv")

    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          {batch, seq_len, _} = Nx.shape(qkv_tensor)

          query = Nx.slice_along_axis(qkv_tensor, 0, hidden_size, axis: 2)
          key = Nx.slice_along_axis(qkv_tensor, hidden_size, hidden_size, axis: 2)
          value = Nx.slice_along_axis(qkv_tensor, hidden_size * 2, hidden_size, axis: 2)

          # Reshape to heads: [batch, heads, seq, head_dim]
          query = reshape_to_heads(query, batch, seq_len, num_heads, head_dim)
          key = reshape_to_heads(key, batch, seq_len, num_heads, head_dim)
          value = reshape_to_heads(value, batch, seq_len, num_heads, head_dim)

          # Scaled dot-product attention
          scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(query)))
          scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1])
          scores = Nx.divide(scores, scale)

          # Softmax (no causal mask for Conformer - bidirectional)
          max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
          exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

          weights =
            Nx.divide(
              exp_scores,
              Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-9)
            )

          output = Nx.dot(weights, [3], [0, 1], value, [2], [0, 1])

          # Reshape back
          output
          |> Nx.transpose(axes: [0, 2, 1, 3])
          |> Nx.reshape({batch, seq_len, num_heads * head_dim})
        end,
        name: "#{name}_compute"
      )

    attn_out =
      attended
      |> Axon.dense(hidden_size, name: "#{name}_out_proj")
      |> Axon.dropout(rate: dropout, name: "#{name}_dropout")

    Axon.add(input, attn_out, name: "#{name}_residual")
  end

  # Convolution module: norm -> pointwise_up -> GLU -> depthwise_conv -> norm -> act -> pointwise_down -> residual
  defp conv_module(input, hidden_size, conv_kernel_size, dropout, name) do
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Pointwise expansion (2x for GLU split)
    expanded = Axon.dense(normed, hidden_size * 2, name: "#{name}_pw_up")

    # GLU: split in half and apply sigmoid gating
    gated =
      Axon.nx(
        expanded,
        fn t ->
          a = Nx.slice_along_axis(t, 0, hidden_size, axis: 2)
          b = Nx.slice_along_axis(t, hidden_size, hidden_size, axis: 2)
          Nx.multiply(a, Nx.sigmoid(b))
        end,
        name: "#{name}_glu"
      )

    # Depthwise convolution with causal padding
    conv_out =
      Axon.conv(gated, hidden_size,
        kernel_size: {conv_kernel_size},
        padding: [{conv_kernel_size - 1, 0}],
        feature_group_size: hidden_size,
        name: "#{name}_dw_conv"
      )

    # Batch norm (use layer norm for simplicity / sequence models)
    conv_normed = Axon.layer_norm(conv_out, name: "#{name}_conv_norm")

    # Activation
    activated = Axon.activation(conv_normed, :silu, name: "#{name}_act")

    # Pointwise down-projection
    down =
      activated
      |> Axon.dense(hidden_size, name: "#{name}_pw_down")
      |> Axon.dropout(rate: dropout, name: "#{name}_conv_dropout")

    Axon.add(input, down, name: "#{name}_residual")
  end

  defp reshape_to_heads(x, batch, seq_len, num_heads, head_dim) do
    x
    |> Nx.reshape({batch, seq_len, num_heads, head_dim})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  @doc """
  Get the output size of a Conformer model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end
end
