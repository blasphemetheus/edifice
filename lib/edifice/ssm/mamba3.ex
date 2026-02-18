defmodule Edifice.SSM.Mamba3 do
  @moduledoc """
  Mamba-3: Advanced Selective State Space Model with complex state dynamics.

  Extends the Mamba architecture with three key innovations from "Mamba-3:
  Advancing State Space Models" for improved expressiveness and efficiency.

  ## Key Innovations

  ### 1. Complex-Valued State Dynamics
  State dimensions are paired and rotated by data-dependent angles (theta),
  similar to how RoPE encodes position. Since Nx has no native complex
  support, this is implemented as real-valued 2x2 rotation matrices on
  paired dimensions:

  ```
  [h_{2i}  ]     [cos(θ)  -sin(θ)] [h_{2i}  ]
  [h_{2i+1}]  =  [sin(θ)   cos(θ)] [h_{2i+1}]  * decay + input
  ```

  ### 2. Generalized Trapezoidal Discretization
  Instead of Euler discretization, uses a weighted blend of current and
  previous inputs controlled by a data-dependent lambda:

  ```
  h_t = A_bar * h_{t-1} + λ * dt * B_t * x_t + (1-λ) * dt * A_bar * B_{t-1} * x_{t-1}
  ```

  This reduces discretization error and improves long-range modeling.

  ### 3. MIMO Rank-r Updates
  Replaces the rank-1 outer product B * x^T with a rank-r product
  B_r @ X_r^T, increasing arithmetic intensity for better hardware
  utilization on modern GPUs/TPUs.

  ## Architecture

  Same gated block structure as Mamba, with the enhanced SSM core:

  ```
  Input [batch, seq_len, embed_dim]
        │
        ▼
  ┌─────────────────────────────────────┐
  │         Mamba-3 Block               │
  │  ┌──── Linear (expand) ────┐        │
  │  │           │              │        │
  │  │   DepthwiseConv + SiLU   │        │
  │  │           │              │        │
  │  │   Complex SSM + Trap.  Linear+SiLU│
  │  │   + MIMO rank-r          │        │
  │  │           │              │        │
  │  └───────── multiply ───────┘        │
  │               │                      │
  │         Linear (project)             │
  └─────────────────────────────────────┘
        │
        ▼ (repeat for num_layers)
  ```

  ## Usage

      model = Mamba3.build(
        embed_dim: 287,
        hidden_size: 256,
        state_size: 16,
        num_layers: 2,
        rank: 4,
        complex: true
      )

  ## References

  - "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)
  - "Transformers are SSMs: Generalized Models and Efficient Algorithms" (Dao & Gu, 2024)
  """

  alias Edifice.SSM.Common

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:complex, boolean()}
          | {:conv_size, pos_integer()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:rank, pos_integer()}
          | {:state_size, pos_integer()}
          | {:window_size, pos_integer()}

  @default_rank 4

  @doc """
  Build a Mamba-3 model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension D (default: 256)
    - `:state_size` - SSM state dimension N (default: 16)
    - `:expand_factor` - Expansion factor E for inner dim (default: 2)
    - `:conv_size` - 1D convolution kernel size (default: 4)
    - `:num_layers` - Number of Mamba-3 blocks (default: 2)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:window_size` - Expected sequence length (default: 60)
    - `:rank` - MIMO rank for input updates (default: 4)
    - `:complex` - Enable complex-valued state dynamics (default: true)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    Common.build_model(opts, &build_mamba3_block/2)
  end

  @doc """
  Build a single Mamba-3 block with enhanced SSM.
  """
  @spec build_mamba3_block(Axon.t(), keyword()) :: Axon.t()
  def build_mamba3_block(input, opts \\ []) do
    layer_idx = Keyword.get(opts, :layer_idx, 1)
    name = Keyword.get(opts, :name, "mamba3_block_#{layer_idx}")
    opts = Keyword.put(opts, :name, name)

    Common.build_block(input, opts, &build_mamba3_ssm/2)
  end

  @doc """
  Build the Mamba-3 SSM with complex dynamics, trapezoidal discretization,
  and MIMO rank-r updates.
  """
  @spec build_mamba3_ssm(Axon.t(), keyword()) :: Axon.t()
  def build_mamba3_ssm(input, opts \\ []) do
    state_size = Keyword.get(opts, :state_size, Common.default_state_size())
    hidden_size = Keyword.get(opts, :hidden_size, Common.default_hidden_size())
    rank = Keyword.get(opts, :rank, @default_rank)
    complex = Keyword.get(opts, :complex, true)
    name = Keyword.get(opts, :name, "ssm")
    dt_rank = Keyword.get(opts, :dt_rank, max(div(hidden_size, 16), 1))

    # Standard B, C, dt projections
    {b_matrix, c_matrix, dt_proj} = Common.build_ssm_projections(input, opts)

    # MIMO rank-r: project input to rank-r representations
    # B_r: [batch, seq, state_size * rank] and X_r: [batch, seq, hidden_size * rank]
    b_rank = Axon.dense(input, state_size * rank, name: "#{name}_b_rank")
    x_rank = Axon.dense(input, hidden_size * rank, name: "#{name}_x_rank")

    # Trapezoidal lambda: data-dependent blending weight
    lambda_proj =
      input
      |> Axon.dense(dt_rank, name: "#{name}_lambda_rank")
      |> Axon.dense(hidden_size, name: "#{name}_lambda_proj")
      |> Axon.activation(:sigmoid, name: "#{name}_lambda_sigmoid")

    # Complex state: theta projection for rotation angles
    theta =
      if complex do
        input
        |> Axon.dense(dt_rank, name: "#{name}_theta_rank")
        |> Axon.dense(div(state_size, 2), name: "#{name}_theta_proj")
      else
        nil
      end

    layer_opts = [
      name: name,
      state_size: state_size,
      hidden_size: hidden_size,
      rank: rank,
      complex: complex,
      op_name: :mamba3_ssm
    ]

    if complex do
      # 8 inputs + opts = arity 9
      Axon.layer(
        &mamba3_ssm_complex_impl/9,
        [input, c_matrix, dt_proj, b_rank, x_rank, lambda_proj, theta, b_matrix],
        layer_opts
      )
    else
      # 7 inputs + opts = arity 8
      Axon.layer(
        &mamba3_ssm_real_impl/8,
        [input, c_matrix, dt_proj, b_rank, x_rank, lambda_proj, b_matrix],
        layer_opts
      )
    end
  end

  # Complex variant: 8 inputs + opts
  defp mamba3_ssm_complex_impl(x, c, dt, b_rank, x_rank, lambda_val, theta, _b, opts) do
    mamba3_ssm_core(x, c, dt, b_rank, x_rank, lambda_val, theta, opts)
  end

  # Real variant: 7 inputs + opts
  defp mamba3_ssm_real_impl(x, c, dt, b_rank, x_rank, lambda_val, _b, opts) do
    mamba3_ssm_core(x, c, dt, b_rank, x_rank, lambda_val, nil, opts)
  end

  # Shared SSM core implementation
  defp mamba3_ssm_core(x, c, dt, b_rank, x_rank, lambda_val, theta, opts) do
    state_size = opts[:state_size]
    hidden_size = opts[:hidden_size]
    rank = opts[:rank]

    seq_len = Nx.axis_size(x, 1)

    # Clamp dt for stability
    dt = Nx.clip(dt, Common.dt_min(), Common.dt_max())

    # Build A decay factors
    a_diag = Nx.negate(Nx.add(Nx.iota({state_size}), 1.0))
    dt_expanded = Nx.new_axis(dt, 3)
    a_expanded = Nx.reshape(a_diag, {1, 1, 1, state_size})

    # A_bar = exp(dt * A): [batch, seq_len, hidden_size, state_size]
    a_bar = Nx.exp(Nx.multiply(dt_expanded, a_expanded))

    # Apply complex rotation if enabled
    a_bar =
      if theta != nil do
        apply_complex_rotation(a_bar, theta, state_size)
      else
        a_bar
      end

    # MIMO rank-r input contribution: B_r @ X_r^T approximation
    # b_rank: [batch, seq, state_size * rank]
    # x_rank: [batch, seq, hidden_size * rank]
    # Compute rank-r outer product sum
    bx = compute_mimo_input(b_rank, x_rank, dt, state_size, hidden_size, rank)

    # Trapezoidal discretization: blend current and previous input contributions
    # bx_trap = lambda * bx_t + (1-lambda) * a_bar * bx_{t-1}
    bx = apply_trapezoidal(bx, a_bar, lambda_val, seq_len)

    # Parallel scan
    h =
      if seq_len <= 32 do
        Common.sequential_scan(a_bar, bx)
      else
        Common.blelloch_scan(a_bar, bx)
      end

    # Output: y[t] = C[t] * h[t]
    Common.compute_ssm_output(h, c)
  end

  # Apply complex-valued rotation to A_bar using paired state dimensions
  # a_bar: [batch, seq, hidden, state_size]
  # theta: [batch, seq, state_size/2]
  defp apply_complex_rotation(a_bar, theta, state_size) do
    half_state = div(state_size, 2)

    # theta: [batch, seq, half_state] -> expand for hidden dim
    # [batch, seq, 1, half_state]
    theta_expanded = Nx.new_axis(theta, 2)
    cos_theta = Nx.cos(theta_expanded)
    sin_theta = Nx.sin(theta_expanded)

    # Split a_bar into even/odd state dimension pairs
    # Even dims: [batch, seq, hidden, half_state]
    a_even = Nx.slice_along_axis(a_bar, 0, half_state, axis: 3)
    a_odd = Nx.slice_along_axis(a_bar, half_state, half_state, axis: 3)

    # Apply rotation: [cos -sin; sin cos] * [even; odd]
    a_even_rot = Nx.subtract(Nx.multiply(a_even, cos_theta), Nx.multiply(a_odd, sin_theta))
    a_odd_rot = Nx.add(Nx.multiply(a_even, sin_theta), Nx.multiply(a_odd, cos_theta))

    Nx.concatenate([a_even_rot, a_odd_rot], axis: 3)
  end

  # Compute MIMO rank-r input contribution
  # Returns [batch, seq, hidden_size, state_size]
  defp compute_mimo_input(b_rank, x_rank, dt, state_size, hidden_size, rank) do
    batch = Nx.axis_size(b_rank, 0)
    seq_len = Nx.axis_size(b_rank, 1)

    # Reshape to rank groups
    # b_rank: [batch, seq, state_size, rank]
    b_r = Nx.reshape(b_rank, {batch, seq_len, state_size, rank})
    # x_rank: [batch, seq, hidden_size, rank]
    x_r = Nx.reshape(x_rank, {batch, seq_len, hidden_size, rank})

    # Rank-r outer product: sum over rank dimension of B_r[:,:,:,r] (x) X_r[:,:,:,r]
    # Result: [batch, seq, hidden_size, state_size]
    # B_r^T @ X_r summed over rank: contract on rank axis
    # x_r: [batch, seq, hidden, rank], b_r: [batch, seq, state, rank]
    # dot on rank: result is [batch, seq, hidden, state]
    bx = Nx.dot(x_r, [3], [0, 1], b_r, [3], [0, 1])

    # Scale by dt (mean across hidden for B scaling)
    dt_mean = Nx.mean(dt, axes: [2], keep_axes: true)
    dt_scale = Nx.new_axis(dt_mean, 3)
    Nx.multiply(bx, dt_scale)
  end

  # Apply trapezoidal discretization blending
  # bx: [batch, seq, hidden, state], a_bar: same, lambda: [batch, seq, hidden]
  defp apply_trapezoidal(bx, a_bar, lambda_val, seq_len) do
    # lambda: [batch, seq, hidden, 1] for broadcasting
    lambda_expanded = Nx.new_axis(lambda_val, 3)

    # Shift bx by 1 for previous timestep (zero-pad at start)
    bx_prev =
      if seq_len > 1 do
        prev = Nx.slice_along_axis(bx, 0, seq_len - 1, axis: 1)

        Nx.pad(prev, 0.0, [
          {0, 0, 0},
          {1, 0, 0},
          {0, 0, 0},
          {0, 0, 0}
        ])
      else
        Nx.broadcast(0.0, Nx.shape(bx))
      end

    # Trapezoidal: lambda * bx_current + (1 - lambda) * a_bar * bx_previous
    current_term = Nx.multiply(lambda_expanded, bx)
    previous_term = Nx.multiply(Nx.subtract(1.0, lambda_expanded), Nx.multiply(a_bar, bx_prev))

    Nx.add(current_term, previous_term)
  end

  # ============================================================================
  # Utilities (delegated to Common)
  # ============================================================================

  @doc """
  Get the output size of a Mamba-3 model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  defdelegate output_size(opts \\ []), to: Common

  @doc """
  Get recommended defaults for Mamba-3.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    Keyword.merge(Common.recommended_defaults(),
      rank: 4,
      complex: true
    )
  end
end
