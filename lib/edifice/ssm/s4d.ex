defmodule Edifice.SSM.S4D do
  @moduledoc """
  S4D: S4 with Diagonal State Matrix.

  A simplified variant of S4 where the state matrix A is purely diagonal,
  removing the need for the DPLR (Diagonal Plus Low-Rank) decomposition.
  S4D serves as the bridge between the original S4 (complex HiPPO matrices)
  and modern SSMs like S5 and Mamba.

  ## Key Simplification

  Original S4 uses DPLR decomposition of HiPPO:
  ```
  A = V * diag(Lambda) * V^{-1} + P * Q^T
  ```

  S4D directly uses diagonal A:
  ```
  A = diag(a_1, a_2, ..., a_N)    (real or complex)
  ```

  This dramatically simplifies implementation while maintaining strong
  performance on most benchmarks.

  ## Architecture

  Identical to S4 but with simpler diagonal-only A parameterization.
  Each block: LayerNorm -> Diagonal SSM -> Dropout -> Residual -> FFN.

  ## Comparison

  | Aspect | S4 | S4D |
  |--------|-----|-----|
  | A matrix | DPLR (HiPPO) | Pure diagonal |
  | Implementation | Complex | Simple |
  | Performance | Strong | Nearly identical |
  | Parameters | More | Fewer |

  ## Usage

      model = S4D.build(
        embed_size: 287,
        hidden_size: 256,
        state_size: 64,
        num_layers: 4
      )

  ## Reference

  - Paper: "On the Parameterization and Initialization of Diagonal State Space Models"
  - arXiv: https://arxiv.org/abs/2206.11893
  """

  require Axon

  @default_hidden_size 256
  @default_state_size 64
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build an S4D model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:state_size` - SSM state dimension N (default: 64)
    - `:num_layers` - Number of S4D blocks (default: 4)
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
        build_s4d_block(acc,
          hidden_size: hidden_size,
          state_size: Keyword.get(opts, :state_size, @default_state_size),
          dropout: dropout,
          name: "s4d_block_#{layer_idx}"
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
  Build a single S4D block.
  """
  @spec build_s4d_block(Axon.t(), keyword()) :: Axon.t()
  def build_s4d_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "s4d_block")

    # SSM sub-layer
    x = Axon.layer_norm(input, name: "#{name}_ssm_norm")

    b_proj = Axon.dense(x, state_size, name: "#{name}_b_proj")
    c_proj = Axon.dense(x, state_size, name: "#{name}_c_proj")

    ssm_out = Axon.layer(
      &diagonal_ssm_impl/4,
      [x, b_proj, c_proj],
      name: "#{name}_ssm",
      hidden_size: hidden_size,
      state_size: state_size,
      op_name: :s4d_ssm
    )

    ssm_out =
      if dropout > 0 do
        Axon.dropout(ssm_out, rate: dropout, name: "#{name}_ssm_drop")
      else
        ssm_out
      end

    x = Axon.add(input, ssm_out, name: "#{name}_ssm_residual")

    # FFN sub-layer
    build_ffn_block(x,
      hidden_size: hidden_size,
      dropout: dropout,
      name: "#{name}_ffn"
    )
  end

  # Diagonal SSM: purely diagonal A with learnable-like initialization
  # Simpler than S4's HiPPO -- uses evenly-spaced negative reals
  defp diagonal_ssm_impl(x, b, c, opts) do
    hidden_size = opts[:hidden_size]
    state_size = opts[:state_size]

    batch = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)

    # Diagonal A: evenly spaced in [-1, -N] (S4D-Lin initialization)
    a_diag = Nx.negate(Nx.add(Nx.iota({state_size}, type: :f32), 1.0))

    # Discretization
    dt = 0.01
    a_bar = Nx.exp(Nx.multiply(dt, a_diag))
    a_bar = Nx.broadcast(a_bar, {batch, seq_len, state_size})

    b_bar = Nx.multiply(dt, b)
    bu = Nx.multiply(b_bar, Nx.mean(Nx.reshape(x, {batch, seq_len, hidden_size, 1}), axes: [2]))

    # Parallel scan via cumulative operations
    log_a = Nx.log(Nx.add(Nx.abs(a_bar), 1.0e-10))
    log_a_cumsum = Nx.cumulative_sum(log_a, axis: 1)
    a_cumprod = Nx.exp(log_a_cumsum)

    eps = 1.0e-10
    bu_normalized = Nx.divide(bu, Nx.add(a_cumprod, eps))
    bu_cumsum = Nx.cumulative_sum(bu_normalized, axis: 1)
    h = Nx.multiply(a_cumprod, bu_cumsum)

    # Output
    y = Nx.multiply(c, h)
    y_summed = Nx.sum(y, axes: [2])
    y_expanded = Nx.new_axis(y_summed, 2)
    Nx.broadcast(y_expanded, {batch, seq_len, hidden_size})
  end

  defp build_ffn_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "ffn")

    inner_size = hidden_size * 4

    x = Axon.layer_norm(input, name: "#{name}_norm")

    # GLU-style FFN
    gate = Axon.dense(x, inner_size, name: "#{name}_gate")
    gate = Axon.activation(gate, :silu, name: "#{name}_gate_silu")
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
  Get the output size of an S4D model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for an S4D model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    inner_size = hidden_size * 4

    ssm_params = 2 * hidden_size * state_size
    ffn_params = 2 * hidden_size * inner_size + inner_size * hidden_size
    per_layer = ssm_params + ffn_params
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
      num_layers: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
