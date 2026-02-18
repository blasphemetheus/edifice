defmodule Edifice.SSM.StripedHyena do
  @moduledoc """
  Striped Hyena: interleaved Hyena long convolution and gated convolution layers.

  Implements the Striped Hyena architecture from "StripedHyena: Moving Beyond
  Transformers with Hybrid Signal Processing Models" (Together AI, 2023).
  Striped Hyena alternates between Hyena long convolution blocks (for global
  context) and gated depthwise convolution blocks (for local patterns).

  ## Key Innovation: Striped Block Pattern

  Instead of using only Hyena blocks, Striped Hyena interleaves two block types:
  - **Even layers**: Hyena long convolution blocks (sub-quadratic global mixing)
  - **Odd layers**: Gated depthwise convolution blocks (efficient local mixing)

  This striped pattern achieves better efficiency while maintaining the
  expressivity of pure Hyena models.

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
  | Layer 1 (Hyena)       |
  |  LongConv + Gating    |
  +-----------------------+
        |
  +-----------------------+
  | Layer 2 (GatedConv)   |
  |  DepthwiseConv + Gate |
  +-----------------------+
        |
  +-----------------------+
  | Layer 3 (Hyena)       |
  |  ...repeating pattern |
  +-----------------------+
        |
        v
  [batch, hidden_size]    (last timestep)
  ```

  ## Gated Conv Block

  ```
  norm(x) -> dense(2*H) -> split(x_val, x_gate)
    -> DepthwiseConv(x_val) * sigmoid(x_gate)
    -> dense(H) -> residual
    -> FFN -> residual
  ```

  ## Usage

      model = StripedHyena.build(
        embed_dim: 287,
        hidden_size: 256,
        order: 2,
        num_layers: 4
      )

  ## Reference

  - Paper: "StripedHyena: Moving Beyond Transformers with Hybrid Signal Processing Models"
  - Blog: https://www.together.ai/blog/stripedhyena-7b
  """

  alias Edifice.Blocks.{DepthwiseConv, FFN}
  alias Edifice.SSM.Hyena

  @default_hidden_size 256
  @default_order 2
  @default_filter_size 64
  @default_conv_kernel_size 7
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a Striped Hyena model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:order` - Hyena gating order (default: 2)
    - `:filter_size` - Implicit filter MLP hidden size (default: 64)
    - `:conv_kernel_size` - Kernel size for gated conv blocks (default: 7)
    - `:num_layers` - Total number of layers (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:conv_kernel_size, pos_integer()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:filter_size, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:order, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Striped pattern: alternate Hyena and gated conv blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        if rem(layer_idx, 2) == 1 do
          # Odd layers: Hyena block (long convolution + gating)
          Hyena.build_hyena_block(acc,
            hidden_size: hidden_size,
            order: Keyword.get(opts, :order, @default_order),
            filter_size: Keyword.get(opts, :filter_size, @default_filter_size),
            dropout: dropout,
            seq_len: seq_len,
            name: "hyena_block_#{layer_idx}"
          )
        else
          # Even layers: Gated depthwise conv block
          build_gated_conv_block(acc,
            hidden_size: hidden_size,
            conv_kernel_size: Keyword.get(opts, :conv_kernel_size, @default_conv_kernel_size),
            dropout: dropout,
            name: "gated_conv_block_#{layer_idx}"
          )
        end
      end)

    x = Axon.layer_norm(x, name: "final_norm")

    Axon.nx(
      x,
      fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Build a gated depthwise convolution block.

  Architecture: norm -> dense(2*H) -> split -> DWConv(val) * sigmoid(gate) -> dense -> residual -> FFN -> residual
  """
  @spec build_gated_conv_block(Axon.t(), keyword()) :: Axon.t()
  def build_gated_conv_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    conv_kernel_size = Keyword.get(opts, :conv_kernel_size, @default_conv_kernel_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "gated_conv_block")

    # Pre-norm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Project to 2*hidden for split into value and gate paths
    x = Axon.dense(x, hidden_size * 2, name: "#{name}_proj")

    # Split: value path and gate path
    val =
      Axon.nx(
        x,
        fn tensor ->
          Nx.slice_along_axis(tensor, 0, hidden_size, axis: 2)
        end,
        name: "#{name}_split_val"
      )

    gate =
      Axon.nx(
        x,
        fn tensor ->
          Nx.slice_along_axis(tensor, hidden_size, hidden_size, axis: 2)
        end,
        name: "#{name}_split_gate"
      )

    # Apply depthwise conv to value path
    val = DepthwiseConv.layer(val, hidden_size, conv_kernel_size, name: "#{name}_dw_conv")

    # Gate: sigmoid(gate) * conv(val)
    gate = Axon.sigmoid(gate, name: "#{name}_gate_sigmoid")
    gated = Axon.multiply(val, gate, name: "#{name}_gate_mul")

    # Output projection
    out = Axon.dense(gated, hidden_size, name: "#{name}_out_proj")

    out =
      if dropout > 0 do
        Axon.dropout(out, rate: dropout, name: "#{name}_drop")
      else
        out
      end

    # Residual connection
    x = Axon.add(input, out, name: "#{name}_residual")

    # FFN sub-layer
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")

    ffn_out =
      FFN.gated_layer(ffn_normed,
        hidden_size: hidden_size,
        inner_size: hidden_size * 4,
        activation: :silu,
        dropout: dropout,
        name: "#{name}_ffn"
      )

    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  @doc """
  Get the output size of a Striped Hyena model.
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
      order: 2,
      filter_size: 64,
      conv_kernel_size: 7,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
