defmodule Edifice.SSM.S4 do
  @moduledoc """
  S4: Structured State Spaces for Sequences.

  Implements the S4 architecture from "Efficiently Modeling Long Sequences with
  Structured State Spaces" (Gu et al., ICLR 2022). S4 introduced the key idea
  of using HiPPO-initialized diagonal state matrices for stable long-range
  sequence modeling.

  ## Key Innovation: HiPPO Initialization

  The state matrix A is initialized using the HiPPO framework, which produces
  matrices that optimally compress continuous signals into finite-dimensional
  state. The diagonal parameterization enables efficient parallel computation:

  ```
  Continuous SSM:
    x'(t) = A x(t) + B u(t)
    y(t)  = C x(t) + D u(t)

  Discretization (ZOH):
    A_bar = exp(dt * A)
    B_bar = (A_bar - I) * A^{-1} * B   (simplified to dt * B for diagonal A)
    h[t]  = A_bar * h[t-1] + B_bar * u[t]
    y[t]  = C * h[t]
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
  | S4 Block x N          |
  |  LayerNorm            |
  |  SSM (HiPPO A)        |
  |  Dropout + Residual   |
  |  FFN Block            |
  +-----------------------+
        |
        v
  +-----------------------+
  | Final LayerNorm       |
  +-----------------------+
        |
        v
  [batch, hidden_size]    (last timestep)
  ```

  ## Usage

      model = S4.build(
        embed_size: 287,
        hidden_size: 256,
        state_size: 64,
        num_layers: 4
      )

  ## Reference

  - Paper: "Efficiently Modeling Long Sequences with Structured State Spaces"
  - arXiv: https://arxiv.org/abs/2111.00396
  """

  require Axon

  alias Edifice.SSM.Common

  @default_hidden_size 256
  @default_state_size 64
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build an S4 model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:state_size` - SSM state dimension N (default: 64)
    - `:num_layers` - Number of S4 blocks (default: 4)
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
        build_s4_block(acc,
          hidden_size: hidden_size,
          state_size: Keyword.get(opts, :state_size, @default_state_size),
          dropout: dropout,
          name: "s4_block_#{layer_idx}"
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
  Build a single S4 block (LayerNorm -> SSM -> dropout -> residual + FFN).
  """
  @spec build_s4_block(Axon.t(), keyword()) :: Axon.t()
  def build_s4_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "s4_block")

    # SSM sub-layer with pre-norm
    x = Axon.layer_norm(input, name: "#{name}_ssm_norm")

    # B and C projections (input-dependent, per-timestep)
    b_proj = Axon.dense(x, state_size, name: "#{name}_b_proj")
    c_proj = Axon.dense(x, state_size, name: "#{name}_c_proj")

    # dt projection: input-dependent discretization step (like Mamba/S6)
    # Dense -> softplus in the SSM callback ensures positive dt
    dt_proj = Axon.dense(x, hidden_size, name: "#{name}_dt_proj")

    # HiPPO-LegS diagonal A: A_n = -(n + 1/2), passed as constant
    a_init = Nx.negate(Nx.add(Nx.iota({state_size}, type: :f32), 0.5))
    a_node = Axon.constant(a_init)

    # SSM with proper discretization and parallel scan
    ssm_out =
      Axon.layer(
        &hippo_ssm_impl/6,
        [x, b_proj, c_proj, dt_proj, a_node],
        name: "#{name}_ssm",
        state_size: state_size,
        op_name: :s4_ssm
      )

    ssm_out =
      if dropout > 0 do
        Axon.dropout(ssm_out, rate: dropout, name: "#{name}_ssm_drop")
      else
        ssm_out
      end

    x = Axon.add(input, ssm_out, name: "#{name}_ssm_residual")

    # FFN sub-layer with pre-norm
    build_ffn_block(x,
      hidden_size: hidden_size,
      dropout: dropout,
      name: "#{name}_ffn"
    )
  end

  # S4 SSM with HiPPO-initialized A, input-dependent dt, and Blelloch parallel scan
  defp hippo_ssm_impl(x, b, c, dt_logit, a_diag, opts) do
    state_size = opts[:state_size]

    # Softplus for dt to ensure positive values
    dt = Nx.log1p(Nx.exp(dt_logit))
    # Clamp to reasonable range for numerical stability
    dt = Nx.clip(dt, Common.dt_min(), Common.dt_max())

    # Discretize: A_bar = exp(dt * A), B_bar = dt * B
    # dt: [batch, seq_len, hidden_size]
    # a_diag: [state_size] (negative values)
    # -> a_bar: [batch, seq_len, hidden_size, state_size]
    dt_expanded = Nx.new_axis(dt, 3)
    a_expanded = Nx.reshape(a_diag, {1, 1, 1, state_size})
    a_bar = Nx.exp(Nx.multiply(dt_expanded, a_expanded))

    # B discretization: B_bar = dt_mean * B
    # b: [batch, seq_len, state_size]
    b_expanded = Nx.new_axis(b, 2)
    dt_mean = Nx.mean(dt, axes: [2], keep_axes: true)
    dt_for_b = Nx.new_axis(dt_mean, 3)
    b_bar = Nx.multiply(dt_for_b, b_expanded)

    # Per-channel input contribution: B_bar * x
    # x: [batch, seq_len, hidden_size] -> [batch, seq_len, hidden_size, 1]
    x_expanded = Nx.new_axis(x, 3)
    # bx: [batch, seq_len, hidden_size, state_size]
    bx = Nx.multiply(b_bar, x_expanded)

    # Parallel scan: h[t] = a_bar[t] * h[t-1] + bx[t]
    h = Common.blelloch_scan(a_bar, bx)

    # Output: y = C * h, summed over state dimension
    # c: [batch, seq_len, state_size] -> [batch, seq_len, 1, state_size]
    Common.compute_ssm_output(h, c)
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
  Get the output size of an S4 model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for an S4 model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    inner_size = hidden_size * 4

    # B_proj + C_proj + dt_proj
    ssm_params = 2 * hidden_size * state_size + hidden_size * hidden_size
    ffn_params = 2 * hidden_size * inner_size + inner_size * hidden_size
    per_layer = ssm_params + ffn_params
    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Get recommended defaults for real-time sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      state_size: 64,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
