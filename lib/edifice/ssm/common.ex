defmodule Edifice.SSM.Common do
  @moduledoc """
  Shared components for all Mamba architecture variants.

  This module contains the common building blocks used across Mamba variants:
  - Default hyperparameters
  - Model structure builders (input projection, layer stacking, last timestep)
  - Block structure (normalization, projections, gating)
  - Depthwise convolution
  - SSM parameter projections
  - SSM discretization
  - Utility functions

  ## Mamba Variants

  All variants share the same architecture, differing only in the scan algorithm:

  | Variant | Scan Algorithm | Notes |
  |---------|---------------|-------|
  | `Mamba` | Blelloch | Work-efficient O(L) work, O(log L) depth |
  | `MambaHillisSteele` | Hillis-Steele | O(L log L) work, more parallelism |
  | `MambaCumsum` | Cumsum-based | Experimental log-space approach |
  | `MambaSSD` | SSD chunked | Mamba-2's matmul approach |
  ## See Also

  - `Edifice.SSM.Mamba` - Main Mamba implementation
  """

  require Axon

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension D"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default SSM state dimension N"
  @spec default_state_size() :: pos_integer()
  def default_state_size, do: 16

  @doc "Default expansion factor E"
  @spec default_expand_factor() :: pos_integer()
  def default_expand_factor, do: 2

  @doc "Default convolution kernel size"
  @spec default_conv_size() :: pos_integer()
  def default_conv_size, do: 4

  @doc "Default number of Mamba blocks"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 2

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.0

  @doc "Minimum delta for numerical stability"
  @spec dt_min() :: float()
  def dt_min, do: 0.001

  @doc "Maximum delta"
  @spec dt_max() :: float()
  def dt_max, do: 0.1

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build the common Mamba model structure.

  This handles:
  - Input projection (if embed_dim != hidden_size)
  - Layer stacking with residual connections and dropout
  - Last timestep extraction

  The caller provides a `block_builder` function that constructs each Mamba block.

  ## Parameters

  - `opts` - Model options (embed_dim, hidden_size, num_layers, dropout, etc.)
  - `block_builder` - Function `(input, opts) -> Axon.t()` that builds one block

  ## Returns

  An Axon model that outputs `[batch, hidden_size]`.
  """
  @spec build_model(keyword(), (Axon.t(), keyword() -> Axon.t())) :: Axon.t()
  def build_model(opts, block_builder) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project input to hidden dimension if different
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack Mamba blocks
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        block = block_builder.(acc, Keyword.put(opts, :layer_idx, layer_idx))

        # Add residual connection + optional dropout
        residual = Axon.add(acc, block, name: "residual_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(residual, rate: dropout, name: "dropout_#{layer_idx}")
        else
          residual
        end
      end)

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      output,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # Block Building
  # ============================================================================

  @doc """
  Build the common Mamba block structure.

  This handles everything except the SSM scan itself:
  - Layer normalization
  - Input projection (to 2x inner_size for x/z branches)
  - X/Z branch splitting
  - Depthwise convolution + SiLU on X branch
  - SiLU gating on Z branch
  - Gated multiplication
  - Output projection

  The caller provides an `ssm_builder` function that constructs the SSM layer.

  ## Parameters

  - `input` - Input Axon node
  - `opts` - Block options (hidden_size, state_size, expand_factor, conv_size, name)
  - `ssm_builder` - Function `(x_activated, ssm_opts) -> Axon.t()` that builds SSM

  ## Returns

  An Axon node representing the block output.
  """
  @spec build_block(Axon.t(), keyword(), (Axon.t(), keyword() -> Axon.t())) :: Axon.t()
  def build_block(input, opts, ssm_builder) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    state_size = Keyword.get(opts, :state_size, default_state_size())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())
    conv_size = Keyword.get(opts, :conv_size, default_conv_size())
    name = Keyword.get(opts, :name, "mamba_block")

    # Inner dimension (expanded)
    inner_size = hidden_size * expand_factor

    # Compute dt_rank (controls complexity of delta computation)
    dt_rank = max(div(hidden_size, 16), 1)

    # Input normalization (RMSNorm in original)
    normalized = Axon.layer_norm(input, name: "#{name}_norm")

    # Project to 2x inner_size (for x and z branches)
    xz = Axon.dense(normalized, inner_size * 2, name: "#{name}_in_proj")

    # Split into x (SSM path) and z (gating path)
    x_branch =
      Axon.nx(
        xz,
        fn tensor ->
          Nx.slice_along_axis(tensor, 0, inner_size, axis: 2)
        end,
        name: "#{name}_x_split"
      )

    z_branch =
      Axon.nx(
        xz,
        fn tensor ->
          Nx.slice_along_axis(tensor, inner_size, inner_size, axis: 2)
        end,
        name: "#{name}_z_split"
      )

    # X branch: Depthwise Conv1D -> SiLU -> SSM
    x_conv = build_depthwise_conv1d(x_branch, inner_size, conv_size, "#{name}_conv")
    x_activated = Axon.activation(x_conv, :silu, name: "#{name}_conv_silu")

    # SSM layer (provided by caller)
    ssm_opts = [
      hidden_size: inner_size,
      state_size: state_size,
      dt_rank: dt_rank,
      name: "#{name}_ssm"
    ]

    # Merge any extra opts (like chunk_size for SSD)
    ssm_opts = Keyword.merge(ssm_opts, Keyword.take(opts, [:chunk_size, :scan_algo]))

    x_ssm = ssm_builder.(x_activated, ssm_opts)

    # Z branch: SiLU activation (gating)
    z_activated = Axon.activation(z_branch, :silu, name: "#{name}_gate_silu")

    # Multiply x_ssm * z (gated output)
    gated = Axon.multiply(x_ssm, z_activated, name: "#{name}_gated")

    # Project back to hidden_size
    Axon.dense(gated, hidden_size, name: "#{name}_out_proj")
  end

  # ============================================================================
  # Depthwise Convolution
  # ============================================================================

  @doc """
  Build a depthwise separable 1D convolution layer.

  True Mamba uses learned depthwise convolution, not mean pooling.
  This approximates depthwise conv behavior for SSM input processing.

  ## Parameters

  - `input` - Input Axon node `[batch, seq_len, channels]`
  - `channels` - Number of output channels
  - `kernel_size` - Convolution kernel size
  - `name` - Layer name prefix
  """
  @spec build_depthwise_conv1d(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  def build_depthwise_conv1d(input, channels, kernel_size, name) do
    # Real depthwise 1D convolution with causal padding
    # Each channel has its own learned kernel (feature_group_size = channels)
    # Causal: pad (kernel_size - 1) on the left, 0 on the right
    Axon.conv(input, channels,
      kernel_size: {kernel_size},
      padding: [{kernel_size - 1, 0}],
      feature_group_size: channels,
      name: "#{name}_dw_conv"
    )
  end

  # ============================================================================
  # SSM Parameter Projections
  # ============================================================================

  @doc """
  Build the SSM parameter projections (B, C, dt).

  These are the "selective" parameters that make Mamba input-dependent:
  - B: Input matrix `[batch, seq_len, state_size]`
  - C: Output matrix `[batch, seq_len, state_size]`
  - dt: Discretization step `[batch, seq_len, hidden_size]`

  ## Parameters

  - `input` - Input Axon node `[batch, seq_len, hidden_size]`
  - `opts` - Options (hidden_size, state_size, dt_rank, name)

  ## Returns

  Tuple of `{b_matrix, c_matrix, dt_proj}` Axon nodes.
  """
  @spec build_ssm_projections(Axon.t(), keyword()) :: {Axon.t(), Axon.t(), Axon.t()}
  def build_ssm_projections(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    state_size = Keyword.get(opts, :state_size, default_state_size())
    dt_rank = Keyword.get(opts, :dt_rank, max(div(hidden_size, 16), 1))
    name = Keyword.get(opts, :name, "ssm")

    # B and C projections: [batch, seq_len, state_size] each
    bc_proj = Axon.dense(input, state_size * 2, name: "#{name}_bc_proj")

    b_matrix =
      Axon.nx(
        bc_proj,
        fn tensor ->
          Nx.slice_along_axis(tensor, 0, state_size, axis: 2)
        end,
        name: "#{name}_B"
      )

    c_matrix =
      Axon.nx(
        bc_proj,
        fn tensor ->
          Nx.slice_along_axis(tensor, state_size, state_size, axis: 2)
        end,
        name: "#{name}_C"
      )

    # Delta (Δ) projection through low-rank bottleneck
    # Δ controls the discretization step - how much to update state
    dt_proj =
      input
      |> Axon.dense(dt_rank, name: "#{name}_dt_rank")
      |> Axon.dense(hidden_size, name: "#{name}_dt_proj")
      |> Axon.activation(:softplus, name: "#{name}_dt_softplus")

    {b_matrix, c_matrix, dt_proj}
  end

  # ============================================================================
  # SSM Discretization
  # ============================================================================

  @doc """
  Discretize the SSM parameters for the scan.

  Converts continuous-time SSM to discrete-time:
  - A_bar = exp(Δ * A)
  - B_bar = Δ * B
  - Bx = B_bar * x

  ## Parameters

  - `x` - Input tensor `[batch, seq_len, hidden_size]`
  - `b` - B matrix `[batch, seq_len, state_size]`
  - `dt` - Delta tensor `[batch, seq_len, hidden_size]`
  - `state_size` - SSM state dimension

  ## Returns

  Tuple of `{a_bar, bx}` where:
  - `a_bar`: `[batch, seq_len, hidden_size, state_size]` - decay factors
  - `bx`: `[batch, seq_len, hidden_size, state_size]` - input contributions
  """
  @spec discretize_ssm(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), pos_integer()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def discretize_ssm(x, b, dt, state_size) do
    # Clamp dt to reasonable range for numerical stability
    dt = Nx.clip(dt, dt_min(), dt_max())

    # A matrix: fixed negative diagonal for stability
    # In Mamba, A is initialized as -exp(uniform(log(1), log(16)))
    # For simplicity, use fixed -1 to -state_size diagonal
    # a_diag: [state_size] negative values
    a_diag = Nx.negate(Nx.add(Nx.iota({state_size}), 1.0))

    # Discretize A: A_bar = exp(Δ * A)
    # dt: [batch, seq_len, hidden_size]
    # a_diag: [state_size]
    # We need per-channel discretization: [batch, seq_len, hidden_size, state_size]
    # [batch, seq_len, hidden_size, 1]
    dt_expanded = Nx.new_axis(dt, 3)
    # [1, 1, 1, state_size]
    a_expanded = Nx.reshape(a_diag, {1, 1, 1, state_size})

    # A_bar = exp(dt * A): [batch, seq_len, hidden_size, state_size]
    a_bar = Nx.exp(Nx.multiply(dt_expanded, a_expanded))

    # Discretize B: B_bar = Δ * B
    # b: [batch, seq_len, state_size]
    # dt: [batch, seq_len, hidden_size]
    # B_bar: [batch, seq_len, hidden_size, state_size]
    # [batch, seq_len, 1, state_size]
    b_expanded = Nx.new_axis(b, 2)
    # [batch, seq_len, 1]
    dt_mean = Nx.mean(dt, axes: [2], keep_axes: true)
    # [batch, seq_len, 1, 1]
    dt_for_b = Nx.new_axis(dt_mean, 3)
    # [batch, seq_len, 1, state_size]
    b_bar = Nx.multiply(dt_for_b, b_expanded)

    # x contribution: B_bar * x
    # x: [batch, seq_len, hidden_size]
    # [batch, seq_len, hidden_size, 1]
    x_expanded = Nx.new_axis(x, 3)
    # [batch, seq_len, hidden_size, state_size]
    bx = Nx.multiply(b_bar, x_expanded)

    {a_bar, bx}
  end

  @doc """
  Compute the SSM output from hidden states.

  y[t] = C[t] * h[t]

  ## Parameters

  - `h` - Hidden states `[batch, seq_len, hidden_size, state_size]`
  - `c` - C matrix `[batch, seq_len, state_size]`

  ## Returns

  Output tensor `[batch, seq_len, hidden_size]`
  """
  @spec compute_ssm_output(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def compute_ssm_output(h, c) do
    # c: [batch, seq_len, state_size]
    # h: [batch, seq_len, hidden_size, state_size]
    # [batch, seq_len, 1, state_size]
    c_expanded = Nx.new_axis(c, 2)

    # y = sum over state_size of (c * h)
    # [batch, seq_len, hidden_size]
    Nx.sum(Nx.multiply(c_expanded, h), axes: [3])
  end

  # ============================================================================
  # Scan Algorithms
  # ============================================================================

  @doc """
  Sequential scan for short sequences or fallback.

  Computes h[t] = a[t] * h[t-1] + b[t] for all t.

  ## Parameters

  - `a` - Decay factors `[batch, seq_len, hidden_size, state_size]`
  - `b` - Input contributions `[batch, seq_len, hidden_size, state_size]`

  ## Returns

  Hidden states `[batch, seq_len, hidden_size, state_size]`
  """
  @spec sequential_scan(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def sequential_scan(a, b) do
    # a, b: [batch, seq_len, hidden_size, state_size]
    seq_len = Nx.axis_size(a, 1)

    # Initialize h[0] = b[0] (h[-1] = 0)
    h0 = Nx.slice_along_axis(b, 0, 1, axis: 1)

    # Sequential recurrence for remaining timesteps
    {_, h_list} =
      Enum.reduce(1..(seq_len - 1), {h0, [Nx.squeeze(h0, axes: [1])]}, fn t, {h_prev, acc} ->
        a_t = Nx.slice_along_axis(a, t, 1, axis: 1)
        b_t = Nx.slice_along_axis(b, t, 1, axis: 1)

        h_t = Nx.add(Nx.multiply(a_t, h_prev), b_t)
        {h_t, [Nx.squeeze(h_t, axes: [1]) | acc]}
      end)

    # Stack results: [batch, seq_len, hidden_size, state_size]
    h_list
    |> Enum.reverse()
    |> Nx.stack(axis: 1)
  end

  @doc """
  Blelloch parallel scan (work-efficient O(L) work, O(log L) depth).

  Uses Enum.reduce for the loop - this lets XLA JIT each level efficiently.

  ## Parameters

  - `a` - Decay factors `[batch, seq_len, hidden_size, state_size]`
  - `b` - Input contributions `[batch, seq_len, hidden_size, state_size]`

  ## Returns

  Hidden states `[batch, seq_len, hidden_size, state_size]`
  """
  @spec blelloch_scan(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def blelloch_scan(a, b) do
    seq_len = Nx.axis_size(a, 1)
    log_len = ceil(:math.log2(seq_len))

    # Up-sweep: compute partial products/sums at each level
    {_a_reduced, b_reduced} =
      Enum.reduce(0..(log_len - 1), {a, b}, fn level, {a_curr, b_curr} ->
        stride = round(:math.pow(2, level))

        if stride >= seq_len do
          {a_curr, b_curr}
        else
          # Shift tensors for combining
          # For a: use 1.0 padding (identity for multiplication)
          # For b: use 0.0 padding (identity for addition after multiply)
          a_shifted =
            Nx.pad(
              Nx.slice_along_axis(a_curr, 0, seq_len - stride, axis: 1),
              1.0,
              [{0, 0, 0}, {stride, 0, 0}, {0, 0, 0}, {0, 0, 0}]
            )

          b_shifted =
            Nx.pad(
              Nx.slice_along_axis(b_curr, 0, seq_len - stride, axis: 1),
              0.0,
              [{0, 0, 0}, {stride, 0, 0}, {0, 0, 0}, {0, 0, 0}]
            )

          # Combine using associative operator:
          # (a1, b1) ⊗ (a2, b2) = (a1*a2, a1*b2 + b1)
          a_new = Nx.multiply(a_curr, a_shifted)
          b_new = Nx.add(Nx.multiply(a_curr, b_shifted), b_curr)

          {a_new, b_new}
        end
      end)

    b_reduced
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Mamba model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end

  @doc """
  Calculate approximate parameter count for a Mamba model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    state_size = Keyword.get(opts, :state_size, default_state_size())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    conv_size = Keyword.get(opts, :conv_size, default_conv_size())

    inner_size = hidden_size * expand_factor
    dt_rank = max(div(hidden_size, 16), 1)

    # Per layer:
    # - Input projection: hidden * (2 * inner)
    # - Conv kernel: conv_size * inner
    # - BC projection: inner * (2 * state)
    # - DT projection: inner * dt_rank + dt_rank * inner
    # - Output projection: inner * hidden
    per_layer =
      hidden_size * (2 * inner_size) +
        conv_size * inner_size +
        inner_size * (2 * state_size) +
        inner_size * dt_rank + dt_rank * inner_size +
        inner_size * hidden_size

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Get recommended defaults for real-time sequence processing (60fps).
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      state_size: 16,
      expand_factor: 2,
      conv_size: 4,
      num_layers: 2,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
