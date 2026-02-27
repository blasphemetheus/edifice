defmodule Edifice.Generative.RectifiedFlow do
  @moduledoc """
  Rectified Flow: straight-trajectory flow matching for few-step generation.

  Builds on Conditional Flow Matching (see `Edifice.Generative.FlowMatching`)
  with the key insight that iteratively "reflowing" — generating (noise, data)
  pairs from a trained model and retraining on straight-line paths between
  them — produces increasingly straight ODE trajectories. Straight paths can
  be accurately simulated with very few Euler steps (even 1).

  ## Relationship to Flow Matching

  The velocity network architecture is identical to standard flow matching
  (MLP with residual blocks, sinusoidal time embedding). The difference is
  purely in the training procedure:

  | Aspect | Flow Matching | Rectified Flow |
  |--------|--------------|----------------|
  | Training pairs | (noise, real data) | (noise, model-generated data) via reflow |
  | Trajectories | May curve | Progressively straightened |
  | Sampling steps | 10-50 typical | 1-5 after rectification |
  | Quality vs speed | Better with more steps | Near-optimal at few steps |

  ## Rectification (Reflow) Procedure

  ```
  1. Train standard flow matching model M_0 on (x_0 ~ noise, x_1 ~ data)
  2. Generate pairs: sample x_0 ~ noise, run M_0 to get x_1 = ODE(x_0)
  3. Train M_1 on these (x_0, x_1) pairs (straight-line target velocity)
  4. Repeat steps 2-3 to get M_2, M_3, ... (1-2 iterations usually enough)
  ```

  After rectification, straightness converges: S(Z^k) = O(1/k).

  ## One-Step Distillation

  For maximum speed, distill a multi-step teacher into a one-step student:

  ```
  teacher_output = sample(teacher_params, ..., num_steps: 10)
  student_output = x_0 + v_student(x_0, t=0)    (single Euler step)
  loss = ||student_output - teacher_output||^2
  ```

  ## Usage

      # Build model (same architecture as FlowMatching)
      model = RectifiedFlow.build(
        obs_size: 287,
        action_dim: 64,
        action_horizon: 8
      )

      # Train standard flow matching first, then reflow:
      {x_0, x_1} = RectifiedFlow.reflow_pairs(params, predict_fn, obs, noise)
      # Retrain on these pairs...

      # Sample with 1 step
      actions = RectifiedFlow.sample(params, predict_fn, obs, noise, num_steps: 1)

      # Measure straightness
      s = RectifiedFlow.straightness(params, predict_fn, obs, noise, actions)

  ## References

  - Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data
    with Rectified Flow" (ICLR 2023 Spotlight). arXiv:2209.03003
  - Used by: Stable Diffusion 3, Flux, InstaFlow
  """

  import Nx.Defn
  alias Edifice.Generative.FlowMatching

  @default_num_steps 1
  @default_num_reflow_steps 50

  # ============================================================================
  # Model Building (delegates to FlowMatching)
  # ============================================================================

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:action_dim, pos_integer()}
          | {:action_horizon, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:obs_size, pos_integer()}

  @doc """
  Build a Rectified Flow velocity network.

  Architecture is identical to `Edifice.Generative.FlowMatching.build/1`.
  The distinction is in training (reflow) and sampling (fewer steps).

  ## Options

    - `:obs_size` - Size of observation embedding (required)
    - `:action_dim` - Dimension of action space (required)
    - `:action_horizon` - Number of actions per sequence (default: 8)
    - `:hidden_size` - Hidden dimension (default: 256)
    - `:num_layers` - Number of MLP layers (default: 4)

  ## Returns

    An Axon model that predicts velocity given (x_t, t, obs).
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    FlowMatching.build(opts)
  end

  # ============================================================================
  # Rectification (Reflow)
  # ============================================================================

  @doc """
  Generate (noise, generated) pairs for rectification training.

  Runs the current model from noise to data, producing training pairs
  with straighter implicit paths. Retrain on these pairs to get a
  rectified model with fewer required sampling steps.

  ## Parameters

    - `params` - Current model parameters
    - `predict_fn` - Velocity prediction function
    - `observations` - Conditioning observations `{batch, obs_size}`
    - `noise` - Source noise (x_0) `{batch, action_horizon, action_dim}`
    - `opts` - Options:
      - `:num_steps` - Integration steps for pair generation (default: 50).
        Use more steps here for higher-quality pairs, even though the final
        rectified model will use fewer.
      - `:solver` - ODE solver (default: `:euler`)

  ## Returns

    `{x_0, x_1}` tuple where x_0 is the input noise and x_1 is the
    model-generated output. Retrain using these pairs with standard
    velocity loss: `||v_theta(x_t, t) - (x_1 - x_0)||^2`.
  """
  @spec reflow_pairs(
          map(),
          (map(), map() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def reflow_pairs(params, predict_fn, observations, noise, opts \\ []) do
    reflow_steps = Keyword.get(opts, :num_steps, @default_num_reflow_steps)
    solver = Keyword.get(opts, :solver, :euler)

    generated =
      FlowMatching.sample(params, predict_fn, observations, noise,
        num_steps: reflow_steps,
        solver: solver
      )

    {noise, generated}
  end

  # ============================================================================
  # Straightness Metric
  # ============================================================================

  @doc """
  Measure trajectory straightness.

  Computes how much the learned velocity field deviates from the
  constant straight-line velocity `(x_1 - x_0)` along the path.
  Lower values indicate straighter paths and better few-step sampling.

  ```
  S = E_t [||v_theta(x_t, t) - (x_1 - x_0)||^2]
  ```

  Estimated by evaluating at `num_eval_points` uniformly spaced times.

  ## Parameters

    - `params` - Model parameters
    - `predict_fn` - Velocity prediction function
    - `observations` - Conditioning `{batch, obs_size}`
    - `x_0` - Noise endpoints `{batch, action_horizon, action_dim}`
    - `x_1` - Data endpoints `{batch, action_horizon, action_dim}`
    - `opts` - Options:
      - `:num_eval_points` - Time points to evaluate (default: 10)

  ## Returns

    Scalar straightness measure (lower = straighter).
  """
  @spec straightness(
          map(),
          (map(), map() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  def straightness(params, predict_fn, observations, x_0, x_1, opts \\ []) do
    num_points = Keyword.get(opts, :num_eval_points, 10)
    batch_size = Nx.axis_size(x_0, 0)

    # Constant straight-line velocity
    straight_vel = FlowMatching.target_velocity(x_0, x_1)

    # Evaluate at uniformly spaced time points
    deviations =
      Enum.map(1..num_points, fn i ->
        t = i / (num_points + 1)
        t_tensor = Nx.broadcast(Nx.tensor(t, type: :f32), {batch_size})

        # Interpolated position
        x_t = FlowMatching.interpolate(x_0, x_1, t_tensor)

        # Predicted velocity at this point
        pred_vel =
          predict_fn.(params, %{
            "x_t" => x_t,
            "timestep" => t_tensor,
            "observations" => observations
          })

        # Squared deviation from straight-line velocity
        FlowMatching.velocity_loss(straight_vel, pred_vel)
      end)

    # Average over time points
    deviations
    |> Nx.stack()
    |> Nx.mean()
  end

  # ============================================================================
  # One-Step Distillation
  # ============================================================================

  @doc """
  Compute one-step distillation loss.

  Trains a student model to match a multi-step teacher in a single step.
  The student predicts `x_0 + v_student(x_0, 0)` and the teacher generates
  the target via multi-step ODE integration.

  ## Parameters

    - `teacher_params` - Teacher model parameters (multi-step)
    - `student_params` - Student model parameters (one-step)
    - `predict_fn` - Shared velocity prediction function
    - `observations` - Conditioning `{batch, obs_size}`
    - `noise` - Source noise (x_0) `{batch, action_horizon, action_dim}`
    - `opts` - Options:
      - `:teacher_steps` - Teacher integration steps (default: 50)
      - `:teacher_solver` - Teacher ODE solver (default: `:euler`)

  ## Returns

    Scalar distillation loss.
  """
  @spec distill_loss(
          map(),
          map(),
          (map(), map() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  def distill_loss(teacher_params, student_params, predict_fn, observations, noise, opts \\ []) do
    teacher_steps = Keyword.get(opts, :teacher_steps, @default_num_reflow_steps)
    teacher_solver = Keyword.get(opts, :teacher_solver, :euler)

    # Teacher generates target via multi-step integration
    teacher_output =
      FlowMatching.sample(teacher_params, predict_fn, observations, noise,
        num_steps: teacher_steps,
        solver: teacher_solver
      )

    # Student predicts in one step: x_1 = x_0 + v_student(x_0, t=0)
    batch_size = Nx.axis_size(noise, 0)
    t_zero = Nx.broadcast(Nx.tensor(0.0, type: :f32), {batch_size})

    student_vel =
      predict_fn.(student_params, %{
        "x_t" => noise,
        "timestep" => t_zero,
        "observations" => observations
      })

    student_output = Nx.add(noise, student_vel)

    # MSE between student one-step and teacher multi-step
    diff = Nx.subtract(student_output, teacher_output)
    Nx.mean(Nx.multiply(diff, diff))
  end

  # ============================================================================
  # Training Loss
  # ============================================================================

  @doc """
  Compute velocity matching loss (same as standard flow matching).

  For the first reflow iteration, train on (noise, real data) pairs.
  For subsequent iterations, train on pairs from `reflow_pairs/5`.

  ```
  L = ||v_theta(x_t, t) - (x_1 - x_0)||^2
  ```

  ## Parameters

    - `params` - Model parameters
    - `predict_fn` - Velocity prediction function
    - `observations` - Conditioning `{batch, obs_size}`
    - `x_1` - Target data `{batch, action_horizon, action_dim}`
    - `x_0` - Source noise `{batch, action_horizon, action_dim}`
    - `t` - Random timesteps `{batch}` in [0, 1]
  """
  @spec compute_loss(
          map(),
          (map(), map() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t()
        ) :: Nx.Tensor.t()
  def compute_loss(params, predict_fn, observations, x_1, x_0, t) do
    FlowMatching.compute_loss(params, predict_fn, observations, x_1, x_0, t)
  end

  # ============================================================================
  # Sampling
  # ============================================================================

  @doc """
  Sample actions via ODE integration.

  After rectification, uses very few steps (default: 1).

  ## Parameters

    - `params` - Model parameters
    - `predict_fn` - Velocity prediction function
    - `observations` - Conditioning `{batch, obs_size}`
    - `initial_noise` - Starting noise `{batch, action_horizon, action_dim}`
    - `opts` - Options:
      - `:num_steps` - Integration steps (default: 1 for rectified flow)
      - `:solver` - ODE solver (default: `:euler`)

  ## Returns

    Generated actions `{batch, action_horizon, action_dim}`.
  """
  @spec sample(
          map(),
          (map(), map() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  def sample(params, predict_fn, observations, initial_noise, opts \\ []) do
    num_steps = Keyword.get(opts, :num_steps, @default_num_steps)
    solver = Keyword.get(opts, :solver, :euler)

    FlowMatching.sample(params, predict_fn, observations, initial_noise,
      num_steps: num_steps,
      solver: solver
    )
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Interpolate between noise and data at time t.

  Delegates to `FlowMatching.interpolate/3`.
  """
  @spec interpolate(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn interpolate(x_0, x_1, t) do
    FlowMatching.interpolate(x_0, x_1, t)
  end

  @doc """
  Compute straight-line target velocity (x_1 - x_0).

  Delegates to `FlowMatching.target_velocity/2`.
  """
  @spec target_velocity(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn target_velocity(x_0, x_1) do
    FlowMatching.target_velocity(x_0, x_1)
  end

  @doc """
  Compute velocity matching MSE loss.

  Delegates to `FlowMatching.velocity_loss/2`.
  """
  @spec velocity_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn velocity_loss(target_vel, pred_vel) do
    FlowMatching.velocity_loss(target_vel, pred_vel)
  end

  @doc "Get the output size of a rectified flow model."
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    FlowMatching.output_size(opts)
  end
end
