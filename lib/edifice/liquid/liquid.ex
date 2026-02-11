defmodule Edifice.Liquid do
  @moduledoc """
  Liquid Neural Networks (LNN) - Continuous-time adaptive neural networks.

  LNNs use differential equations to model temporal dynamics, enabling
  continuous adaptation during inference. Uses the ODE solvers from
  `Edifice.Utils.ODESolver` including adaptive Dormand-Prince 4/5.

  ## Key Innovation

  Unlike traditional RNNs with discrete state updates, LNNs model the
  hidden state as evolving according to an ODE:

      dx/dt = -x/tau + f(x, I, theta)/tau

  Where:
  - tau is a learnable time constant (controls decay rate)
  - x is the hidden state
  - I is the input
  - f is a neural network

  ## Available Solvers

  | Solver | Order | Adaptive | Speed | Accuracy |
  |--------|-------|----------|-------|----------|
  | `:euler` | 1 | No | Fastest | Low |
  | `:midpoint` | 2 | No | Fast | Medium |
  | `:rk4` | 4 | No | Medium | Good |
  | `:dopri5` | 4/5 | Yes | Slower | Best |

  See `Edifice.Utils.ODESolver` for implementation details.

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  +-------------------------------------+
  |  LTC Block                           |
  |                                      |
  |  For each timestep:                  |
  |  1. Compute time constant tau(input) |
  |  2. Compute activation f(x, input)   |
  |  3. Integrate: dx/dt = -x/tau + f/tau|
  |                                      |
  |  (Optional: multiple sub-steps)      |
  |                                      |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  [batch, hidden_size]
  ```

  ## Use Case

  LNNs are particularly suited for real-time sequence processing because:
  - Can adapt to changing input patterns during inference
  - Robust to distributional drift (different data distributions)
  - Continuous dynamics model smooth transitions

  ## Reference

  - Paper: "Liquid Time-constant Networks" (AAAI 2021)
  - Company: Liquid AI (MIT spin-off, $250M from AMD)
  """

  require Axon

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60
  @default_integration_steps 1
  @default_solver :rk4

  @doc """
  Build a Liquid Neural Network model.

  ## Options

    - `:embed_size` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of LTC layers (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)
    - `:integration_steps` - ODE sub-steps per frame (default: 1)
    - `:solver` - ODE solver: `:euler`, `:midpoint`, `:rk4`, `:dopri5` (default: :rk4)

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

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project input to hidden dimension if different
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack LTC layers
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_ltc_layer(
          acc,
          hidden_size: hidden_size,
          dropout: dropout,
          integration_steps: Keyword.get(opts, :integration_steps, @default_integration_steps),
          solver: Keyword.get(opts, :solver, @default_solver),
          name: "ltc_layer_#{layer_idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      x,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Build a single LTC (Liquid Time-Constant) layer.

  Each layer processes the sequence through a continuous-time cell.
  """
  @spec build_ltc_layer(Axon.t(), keyword()) :: Axon.t()
  def build_ltc_layer(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    integration_steps = Keyword.get(opts, :integration_steps, @default_integration_steps)
    solver = Keyword.get(opts, :solver, @default_solver)
    name = Keyword.get(opts, :name, "ltc_layer")

    # Pre-LayerNorm
    x = Axon.layer_norm(input, name: "#{name}_norm")

    # Time constant network: tau = sigmoid(W_tau * x + b_tau) * (tau_max - tau_min) + tau_min
    # Higher tau = slower decay, lower tau = faster adaptation
    tau_proj = Axon.dense(x, hidden_size, name: "#{name}_tau")

    # Activation network: f(x, input)
    f_proj = Axon.dense(x, hidden_size * 2, name: "#{name}_f_hidden")
    f_proj = Axon.activation(f_proj, :silu, name: "#{name}_f_silu")
    f_proj = Axon.dense(f_proj, hidden_size, name: "#{name}_f_out")

    # Apply LTC dynamics
    ltc_output =
      Axon.layer(
        &ltc_impl/4,
        [x, tau_proj, f_proj],
        name: "#{name}_ltc",
        hidden_size: hidden_size,
        integration_steps: integration_steps,
        solver: solver,
        op_name: :ltc_cell
      )

    # Dropout
    ltc_output =
      if dropout > 0 do
        Axon.dropout(ltc_output, rate: dropout, name: "#{name}_dropout")
      else
        ltc_output
      end

    # Residual connection
    Axon.add(input, ltc_output, name: "#{name}_residual")
  end

  # LTC cell implementation with ODE integration
  defp ltc_impl(x, tau_proj, f_proj, opts) do
    hidden_size = opts[:hidden_size]
    integration_steps = opts[:integration_steps] || 1
    solver = opts[:solver] || :euler

    # x: [batch, seq_len, hidden_size]
    # tau_proj: [batch, seq_len, hidden_size]
    # f_proj: [batch, seq_len, hidden_size]

    batch = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)

    # Compute time constants: tau = softplus(tau_proj) + 0.1 (ensure positive)
    # Scale to reasonable range [0.1, 10.0]
    tau = Nx.add(softplus(tau_proj), 0.1)

    # dt per integration step (normalized to 1.0 total per frame)
    dt = 1.0 / integration_steps

    # Process sequence with ODE integration
    # Initialize hidden state as zeros
    h0 = Nx.broadcast(0.0, {batch, hidden_size})

    # Sequential processing through time
    # Using Enum.reduce for clarity (XLA will optimize)
    {final_outputs, _} =
      Enum.reduce(0..(seq_len - 1), {[], h0}, fn t, {outputs, h} ->
        # Extract current timestep values
        x_t = Nx.slice_along_axis(x, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        tau_t = Nx.slice_along_axis(tau, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        f_t = Nx.slice_along_axis(f_proj, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # Integrate ODE for this timestep
        h_new = integrate_ode(h, x_t, tau_t, f_t, dt, integration_steps, solver)

        {outputs ++ [Nx.new_axis(h_new, 1)], h_new}
      end)

    # Concatenate outputs: [batch, seq_len, hidden_size]
    Nx.concatenate(final_outputs, axis: 1)
  end

  # Integrate the ODE using the ODESolver module
  # LTC ODE: dx/dt = (activation - x) / tau
  defp integrate_ode(h, _x, tau, activation, _dt, steps, solver) do
    alias Edifice.Utils.ODESolver
    ODESolver.solve_ltc(h, activation, tau, solver: solver, steps: steps)
  end

  # Softplus activation: log(1 + exp(x))
  defp softplus(x) do
    # Numerically stable version
    Nx.log(Nx.add(1.0, Nx.exp(Nx.min(x, 20.0))))
  end

  # ============================================================================
  # Feed-forward network for deeper processing
  # ============================================================================

  @doc """
  Build a Liquid model with interleaved FFN layers.

  This variant adds feed-forward networks between LTC layers for more
  expressive power, similar to Transformer blocks.
  """
  @spec build_with_ffn(keyword()) :: Axon.t()
  def build_with_ffn(opts \\ []) do
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
        # LTC layer
        ltc_out =
          build_ltc_layer(
            acc,
            hidden_size: hidden_size,
            dropout: dropout,
            integration_steps: Keyword.get(opts, :integration_steps, @default_integration_steps),
            solver: Keyword.get(opts, :solver, @default_solver),
            name: "ltc_layer_#{layer_idx}"
          )

        # FFN layer (SwiGLU style)
        build_ffn(ltc_out,
          hidden_size: hidden_size,
          dropout: dropout,
          name: "ffn_#{layer_idx}"
        )
      end)

    x = Axon.layer_norm(x, name: "final_norm")

    Axon.nx(
      x,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  defp build_ffn(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "ffn")

    inner_size = hidden_size * 4

    x = Axon.layer_norm(input, name: "#{name}_norm")

    # SwiGLU: gate * up
    gate_proj = Axon.dense(x, inner_size, name: "#{name}_gate")
    up_proj = Axon.dense(x, inner_size, name: "#{name}_up")

    gate = Axon.activation(gate_proj, :silu, name: "#{name}_silu")
    gated = Axon.multiply(gate, up_proj, name: "#{name}_gated")

    x = Axon.dense(gated, hidden_size, name: "#{name}_down")

    x =
      if dropout > 0 do
        Axon.dropout(x, rate: dropout, name: "#{name}_dropout")
      else
        x
      end

    Axon.add(input, x, name: "#{name}_residual")
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a Liquid model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for a Liquid model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    # Per layer:
    # - tau projection: hidden * hidden
    # - f network: hidden * (hidden * 2) + (hidden * 2) * hidden
    # tau
    # f hidden
    # f out
    per_layer =
      hidden_size * hidden_size +
        hidden_size * (hidden_size * 2) +
        hidden_size * 2 * hidden_size

    # Input projection
    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Get recommended defaults for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1,
      integration_steps: 1,
      solver: :rk4
    ]
  end

  @doc """
  Get high-accuracy configuration using Dormand-Prince 4/5.

  Uses adaptive stepsize ODE solver for best accuracy.
  Slower but more precise continuous-time dynamics.
  """
  @spec high_accuracy_defaults() :: keyword()
  def high_accuracy_defaults do
    [
      hidden_size: 256,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1,
      # DOPRI5 uses adaptive stepping
      integration_steps: 1,
      solver: :dopri5
    ]
  end

  @doc """
  Initialize hidden state cache for O(1) incremental inference.
  """
  @spec init_cache(keyword()) :: map()
  def init_cache(opts \\ []) do
    batch_size = Keyword.get(opts, :batch_size, 1)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    # Per layer, cache the continuous hidden state
    layers =
      for layer_idx <- 1..num_layers, into: %{} do
        layer_cache = %{
          h: Nx.broadcast(0.0, {batch_size, hidden_size})
        }

        {"layer_#{layer_idx}", layer_cache}
      end

    %{
      layers: layers,
      step: 0,
      config: %{
        hidden_size: hidden_size,
        num_layers: num_layers
      }
    }
  end
end
