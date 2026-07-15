defmodule Edifice.SSM.Mamba do
  @moduledoc """
  Mamba: True Selective State Space Model with optimized parallel scan.

  Implements the Mamba architecture from "Mamba: Linear-Time Sequence Modeling
  with Selective State Spaces" (Gu & Dao, 2023).

  ## Key Innovation: Parallel Associative Scan

  The SSM recurrence h[t] = A * h[t-1] + B * x[t] seems sequential, but can be
  parallelized using associativity:

  ```
  Define: (a, b) ⊗ (c, d) = (a*c, a*d + b)

  Then the scan:
    h[0] = B[0] * x[0]
    h[1] = A[1] * h[0] + B[1] * x[1]
    h[2] = A[2] * h[1] + B[2] * x[2]
    ...

  Can be computed in O(log L) parallel time using prefix scan.
  ```

  ## Selective Mechanism

  Unlike linear time-invariant SSMs, Mamba makes A, B, C input-dependent:
  - Δ (discretization step) controls how much to update state
  - B (input matrix) projects input to state space
  - C (output matrix) projects state to output
  - These are computed from the input, enabling selective focus

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        │
        ▼
  ┌─────────────────────────────────────┐
  │         Mamba Block                  │
  │                                      │
  │  ┌──── Linear (expand) ────┐        │
  │  │           │              │        │
  │  │   DepthwiseConv + SiLU   │        │
  │  │           │              │        │
  │  │   Parallel Scan SSM  Linear+SiLU  │
  │  │           │              │        │
  │  └───────── multiply ───────┘        │
  │               │                      │
  │         Linear (project)             │
  └─────────────────────────────────────┘
        │
        ▼ (repeat for num_layers)
  ```

  ## Usage

      # Build Mamba backbone
      model = Mamba.build(
        embed_dim: 287,
        hidden_size: 256,
        state_size: 16,
        num_layers: 2,
        expand_factor: 2
      )

  ## References
  - Paper: https://arxiv.org/abs/2312.00752
  - Original code: https://github.com/state-spaces/mamba
  """

  alias Edifice.SSM.Common

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:state_size, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:conv_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:dropout, float()}
          | {:window_size, pos_integer()}

  @doc """
  Build a Mamba model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension D (default: 256)
    - `:state_size` - SSM state dimension N (default: 16)
    - `:expand_factor` - Expansion factor E for inner dim (default: 2)
    - `:conv_size` - 1D convolution kernel size (default: 4)
    - `:num_layers` - Number of Mamba blocks (default: 2)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.

  ## Examples

      iex> model = Edifice.SSM.Mamba.build(embed_dim: 32, hidden_size: 16, state_size: 4)
      iex> %Axon{} = model
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    Common.build_model(opts, &build_mamba_block/2)
  end

  @doc """
  Build a single Mamba block with parallel scan SSM.

  ## Options
    - `:hidden_size` - Internal dimension D
    - `:state_size` - SSM state dimension N
    - `:expand_factor` - Expansion factor E
    - `:conv_size` - Convolution kernel size
    - `:name` - Layer name prefix
  """
  @spec build_mamba_block(Axon.t(), keyword()) :: Axon.t()
  def build_mamba_block(input, opts \\ []) do
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = Keyword.get(opts, :name, "mamba_block_#{layer_idx}")
    opts = Keyword.put(opts, :name, name)

    Common.build_block(input, opts, &build_selective_ssm_parallel/2)
  end

  @doc """
  Build depthwise separable 1D convolution layer.
  """
  @spec build_depthwise_conv1d(Axon.t(), pos_integer(), pos_integer(), String.t()) :: Axon.t()
  defdelegate build_depthwise_conv1d(input, channels, kernel_size, name), to: Common

  @doc """
  Build the Selective SSM with parallel associative scan.

  This is the core of Mamba: an SSM where A, B, C, Δ are input-dependent,
  computed efficiently using parallel scan.

  The discretized SSM equations:
  - A_bar = exp(Δ * A)
  - B_bar = Δ * B
  - h[t] = A_bar * h[t-1] + B_bar * x[t]
  - y[t] = C * h[t]
  """
  @spec build_selective_ssm_parallel(Axon.t(), keyword()) :: Axon.t()
  def build_selective_ssm_parallel(input, opts \\ []) do
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    hidden_size = Keyword.get(opts, :hidden_size, Common.default_hidden_size())
    name = Keyword.get(opts, :name, "ssm")

    # Build parameter projections using Common
    {b_matrix, c_matrix, dt_proj} = Common.build_ssm_projections(input, opts)

    # Apply the parallel scan SSM
    Axon.layer(
      &parallel_scan_ssm_impl/5,
      [input, b_matrix, c_matrix, dt_proj],
      name: name,
      state_size: state_size,
      hidden_size: hidden_size,
      op_name: :parallel_scan_ssm
    )
  end

  # Parallel scan SSM implementation
  # This is the core algorithm that makes Mamba O(L) efficient.
  # Tries the fused selective scan CUDA kernel first (handles discretization
  # internally), falls back to Blelloch/sequential scan.
  defp parallel_scan_ssm_impl(x, b, c, dt, opts) do
    state_size = opts[:state_size]
    hidden_size = Nx.axis_size(x, 2)

    if Edifice.CUDA.FusedScan.custom_call_available?() do
      # Fused kernel handles discretization + scan + output in one pass.
      # Build A matrix: [hidden_size, state_size] — fixed negative diagonal
      a_diag = Nx.negate(Nx.add(Nx.iota({state_size}), 1.0))
      a_matrix = Nx.broadcast(Nx.reshape(a_diag, {1, state_size}), {hidden_size, state_size})

      Edifice.CUDA.FusedScan.selective_scan(x, dt, a_matrix, b, c)
    else
      seq_len = Nx.axis_size(x, 1)

      # Discretize SSM parameters
      {a_bar, bx} = Common.discretize_ssm(x, b, dt, state_size)

      # Parallel scan: compute all h[t] in O(log L) parallel time
      h =
        if seq_len <= 32 do
          Common.sequential_scan(a_bar, bx)
        else
          Common.blelloch_scan(a_bar, bx)
        end

      # Compute output: y[t] = C[t] * h[t]
      Common.compute_ssm_output(h, c)
    end
  end

  # ============================================================================
  # Stateful step contract (Edifice.Stateful)
  # ============================================================================

  @behaviour Edifice.Stateful

  alias Edifice.Stateful.Ops

  @doc """
  Initial recurrent state for O(1) stepping:
  `%{h: [batch, num_layers, inner, state_size], conv: [batch, num_layers,
  conv_size - 1, inner]}` zeros, where `inner = hidden_size * expand_factor`.

  Zero `h` matches `Common.sequential_scan/2`'s implicit `h[-1] = 0`; the
  zero conv ring buffer reproduces the forward's causal left-padding.

  Options: `:batch_size` (default 1) plus the same shape options as
  `build/1` (`:hidden_size`, `:state_size`, `:expand_factor`, `:conv_size`,
  `:num_layers`).
  """
  @impl Edifice.Stateful
  def init_state(_params, opts \\ []), do: Common.init_block_state(opts)

  @doc """
  Advance one frame: `[batch, embed_dim]` in, `{[batch, hidden], state}` out.

  Matches the CPU (non-fused) forward path — `Common.discretize_ssm/4` +
  scan — exactly; equivalence at every prefix length is pinned by
  `test/edifice/stateful/step_equivalence_test.exs`, including one case
  crossing the Blelloch-scan branch (`seq_len > 32`).
  """
  @impl Edifice.Stateful
  def step(params, %{h: h, conv: conv}, frame) do
    params = Ops.unwrap_params(params)
    num_layers = Nx.axis_size(h, 1)

    x =
      case params do
        %{"input_projection" => proj} -> Ops.dense(frame, proj)
        _ -> frame
      end

    {x, new_hs, new_convs} =
      Enum.reduce(1..num_layers, {x, [], []}, fn i, {acc, hs, convs} ->
        name = "mamba_block_#{i}"

        h_i = h |> Nx.slice_along_axis(i - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
        conv_i = conv |> Nx.slice_along_axis(i - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])

        {block_out, %{h: h_new, conv: conv_new}} =
          Common.step_block(params, name, %{h: h_i, conv: conv_i}, acc, &step_selective_ssm/4)

        # Residual around the whole block, mirroring Common.build_model/2
        {Nx.add(acc, block_out), [h_new | hs], [conv_new | convs]}
      end)

    new_state = %{
      h: new_hs |> Enum.reverse() |> Nx.stack(axis: 1),
      conv: new_convs |> Enum.reverse() |> Nx.stack(axis: 1)
    }

    {x, new_state}
  end

  @doc """
  One step of the selective SSM: B/C/dt projections for the current frame,
  then the discretized recurrence via `Common.step_discretized_ssm/5`.
  """
  @spec step_selective_ssm(map(), String.t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def step_selective_ssm(params, name, x, h) do
    state_size = Nx.axis_size(h, 2)

    bc = Ops.dense(x, Ops.layer_params!(params, "#{name}_bc_proj"))
    b = Nx.slice_along_axis(bc, 0, state_size, axis: 1)
    c = Nx.slice_along_axis(bc, state_size, state_size, axis: 1)

    dt =
      x
      |> Ops.dense(Ops.layer_params!(params, "#{name}_dt_rank"))
      |> Ops.dense(Ops.layer_params!(params, "#{name}_dt_proj"))
      |> Ops.softplus()

    Common.step_discretized_ssm(x, b, c, dt, h)
  end

  # ============================================================================
  # Utilities (delegated to Common)
  # ============================================================================

  @doc """
  Get the output size of a Mamba model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  defdelegate output_size(opts \\ []), to: Common

  @doc """
  Calculate approximate parameter count for a Mamba model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  defdelegate param_count(opts), to: Common

  @doc """
  Get recommended defaults for real-time sequence processing (60fps).
  """
  @spec recommended_defaults() :: keyword()
  defdelegate recommended_defaults(), to: Common
end
