defmodule Edifice.SSM.BiMamba do
  @moduledoc """
  BiMamba: Bidirectional Mamba for non-causal sequence modeling.

  Extends Mamba with a backward pass for tasks where future context is
  available (e.g., classification, fill-in-the-blank, offline sequence
  analysis). Processes the sequence in both directions and combines
  the outputs.

  ## Key Innovation: Bidirectional SSM

  Standard Mamba is causal (left-to-right only). BiMamba runs two
  parallel SSMs:

  ```
  Forward:   h_f[t] = A_f * h_f[t-1] + B_f * x[t]    (t = 1..L)
  Backward:  h_b[t] = A_b * h_b[t+1] + B_b * x[t]    (t = L..1)
  Output:    y[t] = project(concat(h_f[t], h_b[t]))
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
  | BiMamba Block x N     |
  |  LayerNorm            |
  |  +-- Forward SSM --+  |
  |  |                  |  |
  |  +-- Backward SSM -+  |
  |  |                  |  |
  |  +--- combine -----+  |
  |  Projection + Residual|
  |  FFN                  |
  +-----------------------+
        |
        v
  [batch, hidden_size]    (last timestep)
  ```

  ## Use Cases

  BiMamba is suited for offline tasks where the full sequence is available:
  - Replay analysis (post-game)
  - Sequence classification
  - Bidirectional feature extraction

  For real-time inference (causal), use `Edifice.SSM.Mamba` instead.

  ## Usage

      model = BiMamba.build(
        embed_size: 287,
        hidden_size: 256,
        state_size: 16,
        num_layers: 4
      )

  ## Reference

  - Concept based on bidirectional extensions to Mamba (multiple concurrent works)
  - Original Mamba: https://arxiv.org/abs/2312.00752
  """

  require Axon

  @default_hidden_size 256
  @default_state_size 16
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

  @doc """
  Build a BiMamba model for bidirectional sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:state_size` - SSM state dimension N (default: 16)
    - `:num_layers` - Number of BiMamba blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)
    - `:combine` - How to merge directions: :add or :concat (default: :add)

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
    combine = Keyword.get(opts, :combine, :add)

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
        build_bimamba_block(acc,
          hidden_size: hidden_size,
          state_size: Keyword.get(opts, :state_size, @default_state_size),
          dropout: dropout,
          combine: combine,
          name: "bimamba_block_#{layer_idx}"
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
  Build a single BiMamba block with forward and backward SSMs.
  """
  @spec build_bimamba_block(Axon.t(), keyword()) :: Axon.t()
  def build_bimamba_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    combine = Keyword.get(opts, :combine, :add)
    name = Keyword.get(opts, :name, "bimamba_block")

    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Forward SSM
    fwd_b = Axon.dense(x, state_size, name: "#{name}_fwd_b")
    fwd_c = Axon.dense(x, state_size, name: "#{name}_fwd_c")

    fwd_out = Axon.layer(
      &forward_ssm_impl/4,
      [x, fwd_b, fwd_c],
      name: "#{name}_fwd_ssm",
      hidden_size: hidden_size,
      state_size: state_size,
      op_name: :forward_ssm
    )

    # Backward SSM (reverse input, run SSM, reverse output)
    bwd_b = Axon.dense(x, state_size, name: "#{name}_bwd_b")
    bwd_c = Axon.dense(x, state_size, name: "#{name}_bwd_c")

    bwd_out = Axon.layer(
      &backward_ssm_impl/4,
      [x, bwd_b, bwd_c],
      name: "#{name}_bwd_ssm",
      hidden_size: hidden_size,
      state_size: state_size,
      op_name: :backward_ssm
    )

    # Combine forward and backward
    combined =
      case combine do
        :concat ->
          cat = Axon.concatenate([fwd_out, bwd_out], axis: 2, name: "#{name}_concat")
          Axon.dense(cat, hidden_size, name: "#{name}_combine_proj")

        _add ->
          Axon.add(fwd_out, bwd_out, name: "#{name}_combine_add")
      end

    # Output projection
    proj = Axon.dense(combined, hidden_size, name: "#{name}_out_proj")

    proj =
      if dropout > 0 do
        Axon.dropout(proj, rate: dropout, name: "#{name}_drop")
      else
        proj
      end

    x = Axon.add(input, proj, name: "#{name}_residual")

    # FFN
    build_ffn_block(x,
      hidden_size: hidden_size,
      dropout: dropout,
      name: "#{name}_ffn"
    )
  end

  # Forward SSM: standard left-to-right scan
  defp forward_ssm_impl(x, b, c, opts) do
    run_directional_ssm(x, b, c, opts, :forward)
  end

  # Backward SSM: reverse -> scan -> reverse
  defp backward_ssm_impl(x, b, c, opts) do
    run_directional_ssm(x, b, c, opts, :backward)
  end

  defp run_directional_ssm(x, b, c, opts, direction) do
    hidden_size = opts[:hidden_size]
    state_size = opts[:state_size]

    batch = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)

    # Optionally reverse for backward pass
    {x_dir, b_dir, c_dir} =
      case direction do
        :backward ->
          {Nx.reverse(x, axes: [1]), Nx.reverse(b, axes: [1]), Nx.reverse(c, axes: [1])}

        :forward ->
          {x, b, c}
      end

    a_diag = Nx.negate(Nx.add(Nx.iota({state_size}, type: :f32), 1.0))
    dt = 0.01

    a_bar = Nx.exp(Nx.multiply(dt, a_diag))
    a_bar = Nx.broadcast(a_bar, {batch, seq_len, state_size})

    b_bar = Nx.multiply(dt, b_dir)
    bu = Nx.multiply(b_bar, Nx.mean(Nx.reshape(x_dir, {batch, seq_len, hidden_size, 1}), axes: [2]))

    log_a = Nx.log(Nx.add(Nx.abs(a_bar), 1.0e-10))
    log_a_cumsum = Nx.cumulative_sum(log_a, axis: 1)
    a_cumprod = Nx.exp(log_a_cumsum)

    eps = 1.0e-10
    bu_normalized = Nx.divide(bu, Nx.add(a_cumprod, eps))
    bu_cumsum = Nx.cumulative_sum(bu_normalized, axis: 1)
    h = Nx.multiply(a_cumprod, bu_cumsum)

    y = Nx.multiply(c_dir, h)
    y_summed = Nx.sum(y, axes: [2])
    y_expanded = Nx.new_axis(y_summed, 2)
    y_out = Nx.broadcast(y_expanded, {batch, seq_len, hidden_size})

    # Reverse output back for backward direction
    case direction do
      :backward -> Nx.reverse(y_out, axes: [1])
      :forward -> y_out
    end
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
  Get the output size of a BiMamba model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a BiMamba model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    state_size = Keyword.get(opts, :state_size, @default_state_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    inner_size = hidden_size * 4

    # Two SSMs (fwd + bwd), each with B + C projections
    ssm_params = 4 * hidden_size * state_size
    out_proj = hidden_size * hidden_size
    ffn_params = 2 * hidden_size * inner_size + inner_size * hidden_size
    per_layer = ssm_params + out_proj + ffn_params
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
      state_size: 16,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1,
      combine: :add
    ]
  end
end
