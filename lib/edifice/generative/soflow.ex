defmodule Edifice.Generative.SoFlow do
  @moduledoc """
  SoFlow: Solution Flow Models for One-Step Generative Modeling.

  Implements the SoFlow framework from "SoFlow: Solution Flow Models for
  One-Step Generative Modeling" (Luo et al., Princeton, Dec 2025). Instead
  of learning a velocity field and integrating an ODE at inference (like
  standard Flow Matching), SoFlow learns the ODE's **solution function**
  directly, enabling high-quality one-step generation.

  ## Key Innovation: Solution Function

  Standard Flow Matching learns `v(x_t, t)` and requires multi-step ODE
  integration at inference. SoFlow learns `f(x_t, t, s)` — the function
  that maps state at time `t` directly to state at time `s`.

  ```
  Flow Matching (multi-step):
    x_0 --v(x,0)--> x_{dt} --v(x,dt)--> ... --v(x,1-dt)--> x_1

  SoFlow (one-step):
    x_0 --f(x_0, 0, 1)--> x_1
  ```

  ## Euler Parameterization

  The solution function is parameterized as:
  `f_theta(x_t, t, s) = x_t + (s - t) * F_theta(x_t, t, s)`

  This automatically satisfies the identity `f(x_t, t, t) = x_t`.
  The network `F_theta` predicts a "normalized velocity" conditioned on
  both current time `t` and target time `s`.

  ## Training Loss

  Combined from two components:
  1. **Flow Matching Loss (L_FM)**: Ensures network derivatives match true
     velocity at the `t = s` boundary
  2. **Solution Consistency Loss (L_SCM)**: Enforces self-consistency of
     the solution function across different time intervals, using an EMA
     target network

  `L = lambda * L_FM + (1 - lambda) * L_SCM`

  ## Architecture

  Same as Flow Matching, but the velocity network takes **two** time
  inputs (current `t` and target `s`) instead of one:

  ```
  Inputs: (x_t, t, s, observations)
        |
        v
  [Time Embeddings] t_embed + s_embed + obs_embed + x_embed
        |
        v
  [Residual MLP Blocks x num_layers]
        |
        v
  F_theta(x_t, t, s) -- "normalized velocity"
  ```

  One-step inference: `x_generated = x_noise + F_theta(x_noise, 0, 1)`

  ## Comparison

  | Method | Steps | Distillation? | Quality |
  |--------|:-----:|:------------:|:-------:|
  | Flow Matching | 20-50 | No | High |
  | Consistency Model | 1 | Optional | Medium |
  | SoFlow | 1-2 | No | High |

  ## Usage

      model = SoFlow.build(
        obs_size: 287,
        action_dim: 64,
        action_horizon: 8
      )

      # One-step generation
      x_generated = SoFlow.one_step_sample(params, predict_fn, observations, noise)

      # Two-step refinement
      x_refined = SoFlow.multi_step_sample(params, predict_fn, observations, noise, steps: 2)

  ## References
  - Paper: https://arxiv.org/abs/2512.15657
  - Code: https://github.com/zlab-princeton/SoFlow
  """
  import Nx.Defn

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @default_action_horizon 8
  @default_hidden_size 256
  @default_num_layers 4

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a SoFlow model for one-step generative modeling.

  The key difference from Flow Matching: this network takes **two** time
  inputs — current time `t` and target time `s` — enabling direct solution
  function learning.

  ## Options
    - `:obs_size` - Size of observation/conditioning embedding (required)
    - `:action_dim` - Dimension of action/data space (required)
    - `:action_horizon` - Number of actions per sequence (default: 8)
    - `:hidden_size` - Hidden dimension (default: 256)
    - `:num_layers` - Number of MLP layers (default: 4)

  ## Returns
    An Axon model: (x_t, t, s, observations) -> F_theta(x_t, t, s)
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:action_dim, pos_integer()}
          | {:action_horizon, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:obs_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    obs_size = Keyword.fetch!(opts, :obs_size)
    action_dim = Keyword.fetch!(opts, :action_dim)
    action_horizon = Keyword.get(opts, :action_horizon, @default_action_horizon)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    # Inputs
    x_t = Axon.input("x_t", shape: {nil, action_horizon, action_dim})
    current_time = Axon.input("current_time", shape: {nil})
    target_time = Axon.input("target_time", shape: {nil})
    observations = Axon.input("observations", shape: {nil, obs_size})

    # Time embeddings for both t and s
    t_embed = build_time_embedding(current_time, hidden_size, "t_embed")
    s_embed = build_time_embedding(target_time, hidden_size, "s_embed")

    # Observation embedding
    obs_embed = Axon.dense(observations, hidden_size, name: "obs_embed")

    # State embedding
    x_flat = Axon.flatten(x_t, name: "flatten_x")
    x_embed = Axon.dense(x_flat, hidden_size, name: "x_embed")

    # Combine all embeddings
    combined = Axon.add([x_embed, t_embed, s_embed, obs_embed], name: "combine_embeds")

    # Normalized velocity prediction network
    f_theta =
      Enum.reduce(1..num_layers, combined, fn layer_idx, acc ->
        build_residual_block(acc, hidden_size, "layer_#{layer_idx}")
      end)

    # Project to action space
    f_theta = Axon.dense(f_theta, action_horizon * action_dim, name: "output_proj")

    # Reshape to [batch, action_horizon, action_dim]
    Axon.nx(
      f_theta,
      fn x ->
        batch = Nx.axis_size(x, 0)
        Nx.reshape(x, {batch, action_horizon, action_dim})
      end,
      name: "output_reshape"
    )
  end

  # ============================================================================
  # Time Embedding
  # ============================================================================

  defp build_time_embedding(timestep, hidden_size, name) do
    Axon.layer(
      &time_embedding_impl/2,
      [timestep],
      name: name,
      hidden_size: hidden_size,
      op_name: :time_embed
    )
  end

  defp time_embedding_impl(t, opts) do
    hidden_size = opts[:hidden_size]
    half_dim = div(hidden_size, 2)

    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half_dim}, type: :f32), half_dim)
        )
      )

    t_expanded = Nx.new_axis(t, -1)
    angles = Nx.multiply(t_expanded, freqs)
    Nx.concatenate([Nx.sin(angles), Nx.cos(angles)], axis: -1)
  end

  defp build_residual_block(input, hidden_size, name) do
    x = Axon.dense(input, hidden_size, name: "#{name}_dense1")
    x = Axon.activation(x, :silu, name: "#{name}_silu1")
    x = Axon.dense(x, hidden_size, name: "#{name}_dense2")
    x = Axon.activation(x, :silu, name: "#{name}_silu2")
    Axon.add(input, x, name: "#{name}_residual")
  end

  # ============================================================================
  # Solution Function (Euler Parameterization)
  # ============================================================================

  @doc """
  Apply the Euler parameterization to get the solution function value.

  `f(x_t, t, s) = x_t + (s - t) * F_theta(x_t, t, s)`

  ## Parameters
    - `x_t` - Current state [batch, horizon, dim]
    - `f_theta` - Network output (normalized velocity) [batch, horizon, dim]
    - `t` - Current time [batch]
    - `s` - Target time [batch]
  """
  @spec euler_parameterize(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          Nx.Tensor.t()
  defn euler_parameterize(x_t, f_theta, t, s) do
    dt = Nx.subtract(s, t) |> Nx.reshape({Nx.axis_size(t, 0), 1, 1})
    Nx.add(x_t, Nx.multiply(dt, f_theta))
  end

  # ============================================================================
  # Training Losses
  # ============================================================================

  @doc """
  Flow Matching loss component (L_FM).

  Ensures the network's behavior at the `t = s` boundary matches the true
  velocity. This grounds the solution function to the underlying ODE.

  ## Parameters
    - `f_theta_at_t_t` - F_theta(x_t, t, t) (network output when s = t)
    - `velocity_target` - True velocity: alpha_t' * x_0 + beta_t' * x_1

  For linear interpolation (alpha_t = 1-t, beta_t = t):
  velocity_target = x_1 - x_0
  """
  @spec flow_matching_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn flow_matching_loss(f_theta_at_t_t, velocity_target) do
    diff = Nx.subtract(f_theta_at_t_t, velocity_target)
    Nx.mean(Nx.multiply(diff, diff))
  end

  @doc """
  Solution Consistency loss component (L_SCM).

  Enforces that the solution function is self-consistent: starting from
  different points on the same trajectory should yield the same endpoint.

  Uses a Taylor step to advance from t to l, then checks consistency:
  `f(x_t, t, s) should equal f(x_t + v*(l-t), l, s)`

  The target uses an EMA (stop-gradient) network for stability.

  ## Parameters
    - `f_current` - f_theta(x_t, t, s) = x_t + (s-t) * F_theta(x_t, t, s)
    - `f_target` - stop_grad(f_ema(x_stepped, l, s)) from EMA network
  """
  @spec consistency_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn consistency_loss(f_current, f_target) do
    diff = Nx.subtract(f_current, Nx.Defn.Kernel.stop_grad(f_target))
    Nx.mean(Nx.multiply(diff, diff))
  end

  @doc """
  Combined SoFlow training loss.

  `L = lambda * L_FM + (1 - lambda) * L_SCM`

  ## Parameters
    - `fm_loss` - Flow matching loss
    - `scm_loss` - Solution consistency loss
    - `lambda` - Mixing ratio (default: 0.5)
  """
  @spec combined_loss(Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  defn combined_loss(fm_loss, scm_loss, lambda \\ 0.5) do
    Nx.add(Nx.multiply(lambda, fm_loss), Nx.multiply(1.0 - lambda, scm_loss))
  end

  # ============================================================================
  # Interpolation (same as Flow Matching)
  # ============================================================================

  @doc """
  Linear interpolation between noise and data.

  `x_t = (1 - t) * x_0 + t * x_1`
  """
  @spec interpolate(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn interpolate(x_0, x_1, t) do
    t_bc = Nx.reshape(t, {Nx.axis_size(t, 0), 1, 1})
    Nx.add(Nx.multiply(Nx.subtract(1.0, t_bc), x_0), Nx.multiply(t_bc, x_1))
  end

  # ============================================================================
  # Inference
  # ============================================================================

  @doc """
  One-step generation using the solution function.

  `x_generated = x_noise + F_theta(x_noise, 0, 1)`

  ## Parameters
    - `params` - Model parameters
    - `predict_fn` - The compiled prediction function
    - `observations` - Conditioning [batch, obs_size]
    - `noise` - Initial noise [batch, action_horizon, action_dim]
  """
  @spec one_step_sample(map(), (map(), map() -> Nx.Tensor.t()), Nx.Tensor.t(), Nx.Tensor.t()) ::
          Nx.Tensor.t()
  def one_step_sample(params, predict_fn, observations, noise) do
    batch_size = Nx.axis_size(noise, 0)
    t = Nx.broadcast(Nx.tensor(0.0, type: :f32), {batch_size})
    s = Nx.broadcast(Nx.tensor(1.0, type: :f32), {batch_size})

    f_theta =
      predict_fn.(params, %{
        "x_t" => noise,
        "current_time" => t,
        "target_time" => s,
        "observations" => observations
      })

    # Euler parameterization: x_0 + (1 - 0) * F_theta = noise + F_theta
    Nx.add(noise, f_theta)
  end

  @doc """
  Multi-step generation for improved quality.

  Divides [0, 1] into N equal segments and applies the solution function
  sequentially on each segment.

  ## Parameters
    - `params` - Model parameters
    - `predict_fn` - The compiled prediction function
    - `observations` - Conditioning [batch, obs_size]
    - `noise` - Initial noise [batch, action_horizon, action_dim]
    - `opts`:
      - `:steps` - Number of steps (default: 2)
  """
  @spec multi_step_sample(
          map(),
          (map(), map() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  def multi_step_sample(params, predict_fn, observations, noise, opts \\ []) do
    steps = Keyword.get(opts, :steps, 2)
    batch_size = Nx.axis_size(noise, 0)
    dt = 1.0 / steps

    {final_x, _} =
      Enum.reduce(0..(steps - 1), {noise, 0.0}, fn step_idx, {x, _} ->
        t_val = step_idx * dt
        s_val = (step_idx + 1) * dt

        t = Nx.broadcast(Nx.tensor(t_val, type: :f32), {batch_size})
        s = Nx.broadcast(Nx.tensor(s_val, type: :f32), {batch_size})

        f_theta =
          predict_fn.(params, %{
            "x_t" => x,
            "current_time" => t,
            "target_time" => s,
            "observations" => observations
          })

        # f(x_t, t, s) = x_t + (s - t) * F_theta
        x_next = Nx.add(x, Nx.multiply(dt, f_theta))
        {x_next, s_val}
      end)

    final_x
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a SoFlow model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    action_dim = Keyword.get(opts, :action_dim, 64)
    action_horizon = Keyword.get(opts, :action_horizon, @default_action_horizon)
    action_horizon * action_dim
  end

  @doc """
  Get recommended defaults.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      action_dim: 64,
      action_horizon: 8,
      hidden_size: 256,
      num_layers: 4
    ]
  end
end
