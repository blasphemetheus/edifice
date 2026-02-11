defmodule Edifice.SSM.Hyena do
  @moduledoc """
  Hyena: Sub-quadratic attention alternative via long convolutions and gating.

  Implements the Hyena Hierarchy from "Hyena Hierarchy: Towards Larger
  Convolutional Language Models" (Poli et al., ICML 2023). Hyena replaces
  attention with a hierarchy of long convolutions and element-wise gating,
  achieving sub-quadratic complexity in sequence length.

  ## Key Innovation: Implicit Long Convolution + Gating

  Instead of attention's O(L^2) pairwise interactions, Hyena uses:
  1. A learned implicit filter (small MLP) that generates long convolution kernels
  2. Element-wise gating for non-linearity
  3. Multiple "orders" of this operation for expressivity

  ```
  Order 2 Hyena:
    v, x1, x2 = linear_projections(input)  # 3 projections
    y = v
    y = long_conv(y, filter_1) * x1        # First order
    y = long_conv(y, filter_2) * x2        # Second order
    output = linear(y)
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  +-----------------------+
  | Input Projection      |
  +-----------------------+
        |
        v
  +-----------------------+
  | Hyena Block x N       |
  |  ShortConv(input)     |
  |  Split: v, x1, x2    |
  |  y = v                |
  |  y = LongConv(y)*x1   |  <- Implicit filter via MLP
  |  y = LongConv(y)*x2   |
  |  OutProj + Residual   |
  |  FFN                  |
  +-----------------------+
        |
        v
  [batch, hidden_size]    (last timestep)
  ```

  ## Complexity

  | Operation | Attention | Hyena |
  |-----------|-----------|-------|
  | Training  | O(L^2)    | O(L log L) via FFT |
  | Inference | O(L^2)    | O(L) with recurrence |

  ## Usage

      model = Hyena.build(
        embed_size: 287,
        hidden_size: 256,
        order: 2,
        filter_size: 64,
        num_layers: 4
      )

  ## Reference

  - Paper: "Hyena Hierarchy: Towards Larger Convolutional Language Models"
  - arXiv: https://arxiv.org/abs/2302.10866
  """

  require Axon

  @default_hidden_size 256
  @default_order 2
  @default_filter_size 64
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a Hyena model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:order` - Number of gating levels (default: 2)
    - `:filter_size` - Implicit filter MLP hidden size (default: 64)
    - `:num_layers` - Number of Hyena blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_hyena_block(acc,
          hidden_size: hidden_size,
          order: Keyword.get(opts, :order, @default_order),
          filter_size: Keyword.get(opts, :filter_size, @default_filter_size),
          dropout: dropout,
          seq_len: seq_len,
          name: "hyena_block_#{layer_idx}"
        )
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
  Build a single Hyena block with implicit long convolution and gating.
  """
  @spec build_hyena_block(Axon.t(), keyword()) :: Axon.t()
  def build_hyena_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    order = Keyword.get(opts, :order, @default_order)
    filter_size = Keyword.get(opts, :filter_size, @default_filter_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    seq_len = Keyword.get(opts, :seq_len, @default_window_size)
    name = Keyword.get(opts, :name, "hyena_block")

    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Short depthwise convolution (captures very local patterns)
    x =
      Axon.conv(x, hidden_size,
        kernel_size: {3},
        padding: [{2, 0}],
        feature_group_size: hidden_size,
        name: "#{name}_short_dw_conv"
      )

    # Project to (order + 1) * hidden_size for v + x_1..x_order
    num_projections = order + 1
    projections = Axon.dense(x, hidden_size * num_projections, name: "#{name}_proj")

    # Split projections: v, x_1, x_2, ...
    splits =
      for i <- 0..(num_projections - 1) do
        Axon.nx(
          projections,
          fn tensor ->
            Nx.slice_along_axis(tensor, i * hidden_size, hidden_size, axis: 2)
          end,
          name: "#{name}_split_#{i}"
        )
      end

    [v | gates] = splits

    # Precompute positional encoding for filter MLP
    # Normalized positions [0, 1/L, 2/L, ..., (L-1)/L] as constant
    positions =
      Nx.iota({1, seq_len, 1}, axis: 1, type: :f32)
      |> Nx.divide(max(seq_len - 1, 1))

    positions_node = Axon.constant(positions)

    # Apply order rounds of long convolution + gating
    y =
      Enum.with_index(gates)
      |> Enum.reduce(v, fn {gate_i, idx}, acc ->
        # Build implicit filter via MLP: positions -> filter kernel
        filter =
          positions_node
          |> Axon.dense(filter_size, name: "#{name}_filter#{idx}_dense1")
          |> Axon.nx(&Nx.sin/1, name: "#{name}_filter#{idx}_sin1")
          |> Axon.dense(filter_size, name: "#{name}_filter#{idx}_dense2")
          |> Axon.nx(&Nx.sin/1, name: "#{name}_filter#{idx}_sin2")
          |> Axon.dense(hidden_size, name: "#{name}_filter#{idx}_dense3")

        # Causal long convolution: convolve signal with learned filter
        conv_out =
          Axon.layer(
            &causal_long_conv_impl/3,
            [acc, filter],
            name: "#{name}_long_conv_#{idx}",
            hidden_size: hidden_size,
            op_name: :causal_long_conv
          )

        # Element-wise gating
        Axon.multiply(conv_out, gate_i, name: "#{name}_gate_#{idx}")
      end)

    # Output projection
    out = Axon.dense(y, hidden_size, name: "#{name}_out_proj")

    out =
      if dropout > 0 do
        Axon.dropout(out, rate: dropout, name: "#{name}_drop")
      else
        out
      end

    x = Axon.add(input, out, name: "#{name}_residual")

    # FFN
    build_ffn_block(x,
      hidden_size: hidden_size,
      dropout: dropout,
      name: "#{name}_ffn"
    )
  end

  # Causal long convolution using Nx.conv with dynamic kernel
  # signal: [batch, seq_len, hidden_size]
  # filter: [1, seq_len, hidden_size] (from filter MLP, batch=1 since positions are constant)
  defp causal_long_conv_impl(signal, filter, _opts) do
    hidden_size = Nx.axis_size(signal, 2)
    seq_len = Nx.axis_size(signal, 1)

    # Transpose signal: [batch, hidden_size, seq_len] for Nx.conv
    signal_t = Nx.transpose(signal, axes: [0, 2, 1])

    # Squeeze filter batch dim and build depthwise conv kernel
    # filter: [1, seq_len, hidden_size] -> [seq_len, hidden_size]
    h = Nx.squeeze(filter, axes: [0])

    # Reverse for convolution (Nx.conv does cross-correlation, need flip for true convolution)
    h_rev = Nx.reverse(h, axes: [0])

    # Reshape to depthwise kernel: [hidden_size, 1, seq_len]
    kernel =
      h_rev
      |> Nx.transpose(axes: [1, 0])
      |> Nx.reshape({hidden_size, 1, seq_len})

    # Causal depthwise conv: pad left by (seq_len - 1) for causal
    result =
      Nx.conv(signal_t, kernel,
        padding: [{seq_len - 1, 0}],
        feature_group_size: hidden_size
      )

    # Transpose back: [batch, hidden_size, seq_len] -> [batch, seq_len, hidden_size]
    Nx.transpose(result, axes: [0, 2, 1])
  end

  defp build_ffn_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "ffn")
    inner_size = hidden_size * 4

    x = Axon.layer_norm(input, name: "#{name}_norm")
    gate = Axon.dense(x, inner_size, name: "#{name}_gate")
    gate = Axon.activation(gate, :silu, name: "#{name}_silu")
    up = Axon.dense(x, inner_size, name: "#{name}_up")
    gated = Axon.multiply(gate, up, name: "#{name}_gated")
    x = Axon.dense(gated, hidden_size, name: "#{name}_down")

    x =
      if dropout > 0 do
        Axon.dropout(x, rate: dropout, name: "#{name}_drop")
      else
        x
      end

    Axon.add(input, x, name: "#{name}_residual")
  end

  @doc """
  Get the output size of a Hyena model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a Hyena model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    order = Keyword.get(opts, :order, @default_order)
    filter_size = Keyword.get(opts, :filter_size, @default_filter_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    inner_size = hidden_size * 4

    # Projections: (order+1) * hidden * hidden
    proj_params = (order + 1) * hidden_size * hidden_size
    # Short depthwise conv: hidden * 3 (kernel_size=3, depthwise)
    short_conv = hidden_size * 3
    # Filter MLPs: order * (1 * filter + filter * filter + filter * hidden)
    filter_params =
      order * (1 * filter_size + filter_size * filter_size + filter_size * hidden_size)

    # Output proj
    out_proj = hidden_size * hidden_size
    # FFN
    ffn_params = 2 * hidden_size * inner_size + inner_size * hidden_size
    per_layer = proj_params + short_conv + filter_params + out_proj + ffn_params
    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + per_layer * num_layers
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
      num_layers: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
