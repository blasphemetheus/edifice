defmodule Edifice.SSM.H3 do
  @moduledoc """
  H3: Hungry Hungry Hippos.

  Implements the H3 architecture from "Hungry Hungry Hippos: Towards Language
  Modeling with State Space Models" (Fu et al., ICLR 2023). H3 combines two
  SSM layers with a short convolution and multiplicative gating to close the
  gap between SSMs and Transformers on language modeling.

  ## Key Innovation: Two-SSM + Short Conv

  H3 interleaves two types of SSMs with multiplicative interaction:

  ```
  Branch 1 (Shift SSM): Captures local dependencies via diagonal SSM
  Branch 2 (Diag SSM):  Captures broader patterns via diagonal SSM
  Short Conv:           Models very local (1-4 token) patterns

  Output = ShortConv(ShiftSSM(x) * DiagSSM(x))
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
  | H3 Block x N          |
  |  +-- ShiftSSM(x) --+  |
  |  |                  |  |
  |  +-- DiagSSM(x) ---+  |
  |  |                  |  |
  |  +--- multiply ----+  |
  |  |                     |
  |  ShortConv + OutProj   |
  |  Residual + FFN        |
  +-----------------------+
        |
        v
  [batch, hidden_size]    (last timestep)
  ```

  ## Usage

      model = H3.build(
        embed_size: 287,
        hidden_size: 256,
        state_size: 64,
        conv_size: 4,
        num_layers: 4
      )

  ## Reference

  - Paper: "Hungry Hungry Hippos: Towards Language Modeling with State Space Models"
  - arXiv: https://arxiv.org/abs/2212.14052
  """

  require Axon

  @default_hidden_size 256
  @default_state_size 64
  @default_conv_size 4
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build an H3 model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:state_size` - SSM state dimension N (default: 64)
    - `:conv_size` - Short convolution kernel size (default: 4)
    - `:num_layers` - Number of H3 blocks (default: 4)
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
        build_h3_block(acc,
          hidden_size: hidden_size,
          state_size: Keyword.get(opts, :state_size, @default_state_size),
          conv_size: Keyword.get(opts, :conv_size, @default_conv_size),
          dropout: dropout,
          name: "h3_block_#{layer_idx}"
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
  Build a single H3 block: two SSMs multiplied + short conv + FFN.
  """
  @spec build_h3_block(Axon.t(), keyword()) :: Axon.t()
  def build_h3_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    conv_size = Keyword.get(opts, :conv_size, @default_conv_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "h3_block")

    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Branch 1: Shift SSM
    shift_b = Axon.dense(x, state_size, name: "#{name}_shift_b")
    shift_c = Axon.dense(x, state_size, name: "#{name}_shift_c")

    shift_out = Axon.layer(
      &diagonal_ssm_impl/4,
      [x, shift_b, shift_c],
      name: "#{name}_shift_ssm",
      hidden_size: hidden_size,
      state_size: state_size,
      dt_init: 0.01,
      op_name: :shift_ssm
    )

    # Branch 2: Diagonal SSM
    diag_b = Axon.dense(x, state_size, name: "#{name}_diag_b")
    diag_c = Axon.dense(x, state_size, name: "#{name}_diag_c")

    diag_out = Axon.layer(
      &diagonal_ssm_impl/4,
      [x, diag_b, diag_c],
      name: "#{name}_diag_ssm",
      hidden_size: hidden_size,
      state_size: state_size,
      dt_init: 0.001,
      op_name: :diag_ssm
    )

    # Multiplicative interaction
    combined = Axon.multiply(shift_out, diag_out, name: "#{name}_mul_gate")

    # Short convolution for very local patterns
    conv_out = build_short_conv(combined, hidden_size, conv_size, "#{name}_short_conv")

    # Output projection
    proj = Axon.dense(conv_out, hidden_size, name: "#{name}_out_proj")

    proj =
      if dropout > 0 do
        Axon.dropout(proj, rate: dropout, name: "#{name}_drop")
      else
        proj
      end

    x = Axon.add(input, proj, name: "#{name}_residual")

    # FFN sub-layer
    build_ffn_block(x,
      hidden_size: hidden_size,
      dropout: dropout,
      name: "#{name}_ffn"
    )
  end

  # Diagonal SSM implementation shared by both branches
  defp diagonal_ssm_impl(x, b, c, opts) do
    hidden_size = opts[:hidden_size]
    state_size = opts[:state_size]
    dt_init = opts[:dt_init] || 0.01

    batch = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)

    a_diag = Nx.negate(Nx.add(Nx.iota({state_size}, type: :f32), 1.0))

    a_bar = Nx.exp(Nx.multiply(dt_init, a_diag))
    a_bar = Nx.broadcast(a_bar, {batch, seq_len, state_size})

    b_bar = Nx.multiply(dt_init, b)
    bu = Nx.multiply(b_bar, Nx.mean(Nx.reshape(x, {batch, seq_len, hidden_size, 1}), axes: [2]))

    log_a = Nx.log(Nx.add(Nx.abs(a_bar), 1.0e-10))
    log_a_cumsum = Nx.cumulative_sum(log_a, axis: 1)
    a_cumprod = Nx.exp(log_a_cumsum)

    eps = 1.0e-10
    bu_normalized = Nx.divide(bu, Nx.add(a_cumprod, eps))
    bu_cumsum = Nx.cumulative_sum(bu_normalized, axis: 1)
    h = Nx.multiply(a_cumprod, bu_cumsum)

    y = Nx.multiply(c, h)
    y_summed = Nx.sum(y, axes: [2])
    y_expanded = Nx.new_axis(y_summed, 2)
    Nx.broadcast(y_expanded, {batch, seq_len, hidden_size})
  end

  # Short causal convolution
  defp build_short_conv(input, channels, kernel_size, name) do
    Axon.nx(
      input,
      fn x ->
        batch = Nx.axis_size(x, 0)
        ch = Nx.axis_size(x, 2)
        padding = kernel_size - 1
        pad_shape = {batch, padding, ch}
        padded = Nx.concatenate([Nx.broadcast(0.0, pad_shape), x], axis: 1)

        Nx.window_mean(padded, {1, kernel_size, 1}, strides: [1, 1, 1], padding: :valid)
      end,
      name: "#{name}_causal"
    )
    |> Axon.dense(channels, name: "#{name}_proj")
    |> Axon.activation(:silu, name: "#{name}_silu")
  end

  defp build_ffn_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "ffn")
    inner_size = hidden_size * 4

    x = Axon.layer_norm(input, name: "#{name}_norm")
    x = Axon.dense(x, inner_size, name: "#{name}_up")
    x = Axon.activation(x, :gelu, name: "#{name}_gelu")
    x = Axon.dense(x, hidden_size, name: "#{name}_down")

    x =
      if dropout > 0 do
        Axon.dropout(x, rate: dropout, name: "#{name}_drop")
      else
        x
      end

    Axon.add(input, x, name: "#{name}_residual")
  end

  @doc """
  Get the output size of an H3 model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for an H3 model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    inner_size = hidden_size * 4

    # Two SSMs (B+C projections each) + short conv proj + out proj
    ssm_params = 4 * hidden_size * state_size
    conv_proj = hidden_size * hidden_size
    out_proj = hidden_size * hidden_size
    ffn_params = hidden_size * inner_size + inner_size * hidden_size
    per_layer = ssm_params + conv_proj + out_proj + ffn_params
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
      state_size: 64,
      conv_size: 4,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
