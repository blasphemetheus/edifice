defmodule Edifice.Energy.NeuralODE do
  @moduledoc """
  Neural ODE - Continuous-Depth Residual Networks.

  Neural ODEs parameterize the continuous dynamics of hidden states using
  a neural network. Instead of discrete residual layers h_{t+1} = h_t + f(h_t),
  Neural ODEs solve the continuous-time ODE:

      dh/dt = f(h(t), t; theta)

  This provides:
  - Constant memory cost (via adjoint sensitivity method)
  - Adaptive computation (solver controls accuracy)
  - Continuous-depth networks (no fixed number of layers)

  ## Architecture

  ```
  Input [batch, input_size]
        |
        v
  +--------------------------------------+
  | Input Projection                     |
  +--------------------------------------+
        |
        v
  +--------------------------------------+
  | ODE Solve: dh/dt = f(h, t)          |
  |   f = MLP(h, t)                     |
  |   Euler integration for num_steps    |
  +--------------------------------------+
        |
        v
  +--------------------------------------+
  | Output Projection                    |
  +--------------------------------------+
        |
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = NeuralODE.build(
        input_size: 256,
        hidden_size: 256,
        num_steps: 10,
        step_size: 0.1
      )

  ## References

  - Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
  - https://arxiv.org/abs/1806.07366
  """

  require Axon

  @default_hidden_size 256
  @default_num_steps 10
  @default_step_size 0.1

  @doc """
  Build a Neural ODE model.

  ## Options

  - `:input_size` - Input feature dimension (required)
  - `:hidden_size` - Hidden state dimension (default: 256)
  - `:num_steps` - Number of Euler integration steps (default: 10)
  - `:step_size` - Integration step size (default: 0.1)
  - `:output_size` - Output dimension, nil to use hidden_size (default: nil)

  ## Returns

  An Axon model: `[batch, input_size]` -> `[batch, output_size]`
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_steps = Keyword.get(opts, :num_steps, @default_num_steps)
    step_size = Keyword.get(opts, :step_size, @default_step_size)
    output_size = Keyword.get(opts, :output_size, nil)

    input = Axon.input("input", shape: {nil, input_size})

    # Project to hidden dimension
    h0 =
      if input_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_proj")
      else
        input
      end

    # Build the dynamics network f(h; theta) with SHARED weights across all steps.
    # This is the defining property of Neural ODEs: the same f is evaluated at every
    # integration step, unlike a regular ResNet which has different layers per step.
    # In Axon, using the same layer names shares parameters.
    h_final =
      Enum.reduce(0..(num_steps - 1), h0, fn step, h ->
        # Dynamics: dh/dt = f(h; theta) — shared weights via identical names
        dh =
          h
          |> Axon.dense(hidden_size, name: "dynamics_dense1")
          |> Axon.activation(:silu, name: "dynamics_act1")
          |> Axon.dense(hidden_size, name: "dynamics_dense2")
          |> Axon.activation(:tanh, name: "dynamics_act2")

        # Scale by step size
        scaled_dh =
          Axon.nx(dh, fn x -> Nx.multiply(x, step_size) end, name: "scale_step_#{step}")

        # Euler step: h_{t+dt} = h_t + dt * f(h_t; theta)
        Axon.add(h, scaled_dh, name: "euler_step_#{step}")
      end)

    # Final layer norm for stability
    h_final = Axon.layer_norm(h_final, name: "output_norm")

    # Optional output projection
    if output_size && output_size != hidden_size do
      Axon.dense(h_final, output_size, name: "output_proj")
    else
      h_final
    end
  end

  @doc """
  Build a Neural ODE with shared dynamics network.

  Delegates to `build/1` which now always uses shared dynamics weights
  (the same f(h; theta) is re-evaluated at every integration step).
  Kept for backward compatibility.

  ## Options

  Same as `build/1`.
  """
  @spec build_shared(keyword()) :: Axon.t()
  def build_shared(opts), do: build(opts)

  @doc """
  Get the output size of a Neural ODE model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    out = Keyword.get(opts, :output_size, nil)
    hidden = Keyword.get(opts, :hidden_size, @default_hidden_size)
    out || hidden
  end
end
