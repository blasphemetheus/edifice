defmodule Edifice.Generative.FlowMatching do
  @moduledoc """
  Flow Matching: Action generation via continuous normalizing flows.

  Implements Conditional Flow Matching (CFM) from "Flow Matching for Generative
  Modeling" (Lipman et al., ICLR 2023). Learns a velocity field that transports
  samples from noise to data via an ODE.

  ## Key Innovation: Optimal Transport Paths

  Instead of diffusion's complex noise schedule, Flow Matching uses simple
  linear interpolation (optimal transport path):

  ```
  x_t = (1 - t) * x_0 + t * x_1    where x_0 ~ noise, x_1 ~ data
  v_target = x_1 - x_0             (constant velocity along path)
  ```

  Training minimizes: ||v_theta(x_t, t | obs) - v_target||^2

  ## Comparison with Diffusion

  | Feature | Diffusion | Flow Matching |
  |---------|-----------|---------------|
  | Path | Stochastic (SDE) | Deterministic (ODE) |
  | Schedule | Complex (beta schedule) | None needed |
  | Training | Noise prediction | Velocity prediction |
  | Inference | DDPM/DDIM sampling | ODE integration |
  | Steps | 20-100+ typical | 10-20 often sufficient |

  ## Architecture

  ```
  Observations [batch, obs_dim]
        |
        v
  +-------------------------------------+
  |  Observation Encoder                 |
  +-------------------------------------+
        |
        v obs_embed
  +-------------------------------------+
  |  Velocity Network                    |
  |  Input: (x_t, t, obs_embed)         |
  |  Output: v_theta (velocity field)    |
  +-------------------------------------+
        |
        v
  Actions [batch, action_horizon, action_dim]
  ```

  ## Training

  ```elixir
  # Forward process (create interpolated sample)
  x_t = FlowMatching.interpolate(noise, actions, t)
  target_velocity = actions - noise

  # Predict velocity
  pred_velocity = velocity_network(x_t, t, observations)

  # MSE loss
  loss = FlowMatching.velocity_loss(target_velocity, pred_velocity)
  ```

  ## Inference (ODE Integration)

  ```elixir
  # Start from noise
  x_0 = random_noise()

  # Euler integration (or higher order)
  for t <- 0..1 step dt:
    v = velocity_network(x_t, t, observations)
    x_{t+dt} = x_t + dt * v

  # Final x_1 is the generated action
  ```

  ## Usage

      # Build flow matching model
      model = FlowMatching.build(
        obs_size: 287,
        action_dim: 64,
        action_horizon: 8
      )

      # Training
      loss = FlowMatching.compute_loss(
        params, predict_fn, observations, actions, noise, t
      )

      # Inference
      actions = FlowMatching.sample(
        params, predict_fn, observations,
        num_steps: 20, solver: :euler
      )

  ## References
  - Flow Matching: https://arxiv.org/abs/2210.02747
  - Conditional Flow Matching: https://arxiv.org/abs/2302.00482
  - Rectified Flow: https://arxiv.org/abs/2209.03003
  """

  require Axon
  import Nx.Defn

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default action prediction horizon"
  @spec default_action_horizon() :: pos_integer()
  def default_action_horizon, do: 8

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default number of network layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 4

  @doc "Default number of ODE integration steps"
  @spec default_num_steps() :: pos_integer()
  def default_num_steps, do: 20

  @doc "Default ODE solver"
  @spec default_solver() :: atom()
  def default_solver, do: :euler

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Flow Matching model for action generation.

  ## Options
    - `:obs_size` - Size of observation embedding (required)
    - `:action_dim` - Dimension of action space (required)
    - `:action_horizon` - Number of actions per sequence (default: 8)
    - `:hidden_size` - Hidden dimension (default: 256)
    - `:num_layers` - Number of MLP layers (default: 4)

  ## Returns
    An Axon model that predicts velocity given (x_t, t, obs).
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
    action_horizon = Keyword.get(opts, :action_horizon, default_action_horizon())
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())

    # Inputs
    # x_t: current position in flow [batch, action_horizon, action_dim]
    x_t = Axon.input("x_t", shape: {nil, action_horizon, action_dim})
    # t: time in [0, 1] [batch]
    timestep = Axon.input("timestep", shape: {nil})
    # observations: conditioning [batch, obs_size]
    observations = Axon.input("observations", shape: {nil, obs_size})

    # Embed timestep with sinusoidal encoding
    time_embed = build_time_embedding(timestep, hidden_size)

    # Embed observations
    obs_embed = Axon.dense(observations, hidden_size, name: "obs_embed")

    # Flatten x_t for processing
    x_flat = Axon.flatten(x_t, name: "flatten_x")
    x_embed = Axon.dense(x_flat, hidden_size, name: "x_embed")

    # Combine all embeddings
    combined = Axon.add([x_embed, time_embed, obs_embed], name: "combine_embeds")

    # Velocity prediction network (MLP with residual connections)
    velocity_flat =
      Enum.reduce(1..num_layers, combined, fn layer_idx, acc ->
        build_residual_block(acc, hidden_size, "layer_#{layer_idx}")
      end)

    # Project to action space
    velocity_flat = Axon.dense(velocity_flat, action_horizon * action_dim, name: "velocity_proj")

    # Reshape to [batch, action_horizon, action_dim]
    # Use Axon.nx for dynamic batch size
    Axon.nx(
      velocity_flat,
      fn x ->
        batch = Nx.axis_size(x, 0)
        Nx.reshape(x, {batch, action_horizon, action_dim})
      end,
      name: "velocity_reshape"
    )
  end

  # Build sinusoidal time embedding
  defp build_time_embedding(timestep, hidden_size) do
    Axon.layer(
      &time_embedding_impl/2,
      [timestep],
      name: "time_embed",
      hidden_size: hidden_size,
      op_name: :time_embed
    )
  end

  defp time_embedding_impl(t, opts) do
    hidden_size = opts[:hidden_size]
    half_dim = div(hidden_size, 2)

    # Sinusoidal embedding
    # emb = exp(-log(10000) * i / half_dim) for i in 0..half_dim-1
    freqs =
      Nx.exp(
        Nx.multiply(
          Nx.negate(Nx.log(Nx.tensor(10_000.0))),
          Nx.divide(Nx.iota({half_dim}, type: :f32), half_dim)
        )
      )

    # t: [batch] -> [batch, 1]
    t_expanded = Nx.new_axis(t, -1)

    # [batch, 1] * [half_dim] -> [batch, half_dim]
    angles = Nx.multiply(t_expanded, freqs)

    # Concatenate sin and cos
    sin_embed = Nx.sin(angles)
    cos_embed = Nx.cos(angles)

    Nx.concatenate([sin_embed, cos_embed], axis: -1)
  end

  # Residual MLP block
  defp build_residual_block(input, hidden_size, name) do
    x = Axon.dense(input, hidden_size, name: "#{name}_dense1")
    x = Axon.activation(x, :silu, name: "#{name}_silu1")
    x = Axon.dense(x, hidden_size, name: "#{name}_dense2")
    x = Axon.activation(x, :silu, name: "#{name}_silu2")

    Axon.add(input, x, name: "#{name}_residual")
  end

  # ============================================================================
  # Flow Matching Operations
  # ============================================================================

  @doc """
  Interpolate between noise (x_0) and data (x_1) at time t.

  Uses optimal transport (linear) interpolation:
  x_t = (1 - t) * x_0 + t * x_1

  ## Parameters
    - `x_0` - Source (noise) [batch, action_horizon, action_dim]
    - `x_1` - Target (data/actions) [batch, action_horizon, action_dim]
    - `t` - Time in [0, 1] [batch]

  ## Returns
    Interpolated x_t with same shape as inputs.
  """
  @spec interpolate(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn interpolate(x_0, x_1, t) do
    # Broadcast t to match x shape: [batch] -> [batch, 1, 1]
    t_broadcast = Nx.reshape(t, {Nx.axis_size(t, 0), 1, 1})

    # x_t = (1 - t) * x_0 + t * x_1
    one_minus_t = Nx.subtract(1.0, t_broadcast)

    Nx.add(
      Nx.multiply(one_minus_t, x_0),
      Nx.multiply(t_broadcast, x_1)
    )
  end

  @doc """
  Compute the target velocity for training.

  For optimal transport path, velocity is constant:
  v_target = x_1 - x_0

  ## Parameters
    - `x_0` - Source (noise)
    - `x_1` - Target (data)

  ## Returns
    Target velocity (same shape as inputs).
  """
  @spec target_velocity(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn target_velocity(x_0, x_1) do
    Nx.subtract(x_1, x_0)
  end

  @doc """
  Compute velocity matching loss (MSE).

  L = ||v_theta(x_t, t) - v_target||^2

  ## Parameters
    - `pred_velocity` - Predicted velocity from network
    - `target_velocity` - Ground truth velocity (x_1 - x_0)

  ## Returns
    Scalar loss value.
  """
  @spec velocity_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn velocity_loss(target_vel, pred_vel) do
    diff = Nx.subtract(target_vel, pred_vel)
    Nx.mean(Nx.multiply(diff, diff))
  end

  @doc """
  Compute the complete training loss.

  ## Parameters
    - `params` - Model parameters
    - `predict_fn` - Velocity prediction function
    - `observations` - Conditioning observations [batch, obs_size]
    - `actions` - Target actions (x_1) [batch, action_horizon, action_dim]
    - `noise` - Source noise (x_0) [batch, action_horizon, action_dim]
    - `t` - Random timesteps [batch]

  ## Returns
    Scalar loss value.
  """
  @spec compute_loss(
          map(),
          (map(), map() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t()
        ) :: Nx.Tensor.t()
  def compute_loss(params, predict_fn, observations, actions, noise, t) do
    # Interpolate to get x_t
    x_t = interpolate(noise, actions, t)

    # Get target velocity
    target_vel = target_velocity(noise, actions)

    # Predict velocity
    pred_vel =
      predict_fn.(params, %{
        "x_t" => x_t,
        "timestep" => t,
        "observations" => observations
      })

    # MSE loss
    velocity_loss(target_vel, pred_vel)
  end

  # ============================================================================
  # ODE Solvers for Inference
  # ============================================================================

  @doc """
  Sample actions by integrating the learned ODE.

  Solves dx/dt = v_theta(x, t) from t=0 to t=1.

  ## Parameters
    - `params` - Model parameters
    - `predict_fn` - Velocity prediction function
    - `observations` - Conditioning [batch, obs_size]
    - `initial_noise` - Starting noise (x_0) [batch, action_horizon, action_dim]
    - `opts` - Options:
      - `:num_steps` - Integration steps (default: 20)
      - `:solver` - ODE solver: :euler, :midpoint, :rk4 (default: :euler)

  ## Returns
    Generated actions [batch, action_horizon, action_dim].
  """
  @spec sample(map(), (map(), map() -> Nx.Tensor.t()), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def sample(params, predict_fn, observations, initial_noise, opts \\ []) do
    num_steps = Keyword.get(opts, :num_steps, default_num_steps())
    solver = Keyword.get(opts, :solver, default_solver())

    dt = 1.0 / num_steps

    # Integrate from t=0 to t=1
    {final_x, _} =
      Enum.reduce(0..(num_steps - 1), {initial_noise, 0.0}, fn step, {x, _t} ->
        t = step / num_steps
        t_tensor = Nx.broadcast(Nx.tensor(t, type: :f32), {Nx.axis_size(x, 0)})

        x_next =
          case solver do
            :euler ->
              euler_step(params, predict_fn, x, t_tensor, observations, dt)

            :midpoint ->
              midpoint_step(params, predict_fn, x, t_tensor, observations, dt)

            :rk4 ->
              rk4_step(params, predict_fn, x, t_tensor, observations, dt)

            _ ->
              euler_step(params, predict_fn, x, t_tensor, observations, dt)
          end

        {x_next, t + dt}
      end)

    final_x
  end

  # Euler method: x_{t+dt} = x_t + dt * v(x_t, t)
  defp euler_step(params, predict_fn, x, t, observations, dt) do
    v =
      predict_fn.(params, %{
        "x_t" => x,
        "timestep" => t,
        "observations" => observations
      })

    Nx.add(x, Nx.multiply(dt, v))
  end

  # Midpoint method (2nd order): more accurate than Euler
  defp midpoint_step(params, predict_fn, x, t, observations, dt) do
    batch_size = Nx.axis_size(x, 0)

    # k1 = v(x, t)
    k1 =
      predict_fn.(params, %{
        "x_t" => x,
        "timestep" => t,
        "observations" => observations
      })

    # x_mid = x + dt/2 * k1
    x_mid = Nx.add(x, Nx.multiply(dt / 2, k1))
    t_mid = Nx.add(t, Nx.broadcast(dt / 2, {batch_size}))

    # k2 = v(x_mid, t + dt/2)
    k2 =
      predict_fn.(params, %{
        "x_t" => x_mid,
        "timestep" => t_mid,
        "observations" => observations
      })

    # x_{t+dt} = x + dt * k2
    Nx.add(x, Nx.multiply(dt, k2))
  end

  # RK4 method (4th order): highest accuracy
  defp rk4_step(params, predict_fn, x, t, observations, dt) do
    batch_size = Nx.axis_size(x, 0)
    half_dt = dt / 2

    # k1 = v(x, t)
    k1 =
      predict_fn.(params, %{
        "x_t" => x,
        "timestep" => t,
        "observations" => observations
      })

    # k2 = v(x + dt/2 * k1, t + dt/2)
    x2 = Nx.add(x, Nx.multiply(half_dt, k1))
    t2 = Nx.add(t, Nx.broadcast(half_dt, {batch_size}))

    k2 =
      predict_fn.(params, %{
        "x_t" => x2,
        "timestep" => t2,
        "observations" => observations
      })

    # k3 = v(x + dt/2 * k2, t + dt/2)
    x3 = Nx.add(x, Nx.multiply(half_dt, k2))

    k3 =
      predict_fn.(params, %{
        "x_t" => x3,
        "timestep" => t2,
        "observations" => observations
      })

    # k4 = v(x + dt * k3, t + dt)
    x4 = Nx.add(x, Nx.multiply(dt, k3))
    t4 = Nx.add(t, Nx.broadcast(dt, {batch_size}))

    k4 =
      predict_fn.(params, %{
        "x_t" => x4,
        "timestep" => t4,
        "observations" => observations
      })

    # x_{t+dt} = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    weighted_sum =
      Nx.add(
        Nx.add(k1, Nx.multiply(2.0, k2)),
        Nx.add(Nx.multiply(2.0, k3), k4)
      )

    Nx.add(x, Nx.multiply(dt / 6, weighted_sum))
  end

  # ============================================================================
  # Rectified Flow (Optional Enhancement)
  # ============================================================================

  @doc """
  Compute rectified flow loss with distillation.

  Rectified flow straightens the ODE paths by distilling from a trained
  flow model, allowing even fewer integration steps.

  This is a two-stage process:
  1. Train standard flow matching
  2. Generate (x_0, x_1) pairs by sampling, then retrain on straight paths
  """
  @spec rectified_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn rectified_loss(target_vel, pred_vel) do
    # Same as velocity_loss but on rectified pairs
    velocity_loss(target_vel, pred_vel)
  end

  @doc """
  Generate training pairs for rectified flow.

  Samples (noise, generated_action) pairs from a trained model
  to create straighter paths.
  """
  @spec generate_rectified_pairs(
          map(),
          (map(), map() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def generate_rectified_pairs(params, predict_fn, observations, noise, opts \\ []) do
    # Generate x_1 from x_0 using the current model
    generated = sample(params, predict_fn, observations, noise, opts)

    # Return (x_0, x_1) pair for straight-line training
    {noise, generated}
  end

  # ============================================================================
  # Guidance (Optional)
  # ============================================================================

  @doc """
  Sample with classifier-free guidance.

  Interpolates between conditional and unconditional predictions:
  v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)

  ## Parameters
    - `:guidance_scale` - Strength of guidance (default: 1.0, no guidance)
    - `:uncond_observations` - Unconditional observations (zeros or learned)
  """
  @spec sample_guided(
          map(),
          (map(), map() -> Nx.Tensor.t()),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          keyword()
        ) :: Nx.Tensor.t()
  def sample_guided(params, predict_fn, observations, initial_noise, opts \\ []) do
    num_steps = Keyword.get(opts, :num_steps, default_num_steps())
    guidance_scale = Keyword.get(opts, :guidance_scale, 1.0)

    # Create unconditional observations (zeros)
    uncond_obs = Nx.broadcast(0.0, Nx.shape(observations))

    dt = 1.0 / num_steps

    {final_x, _} =
      Enum.reduce(0..(num_steps - 1), {initial_noise, 0.0}, fn step, {x, _t} ->
        t = step / num_steps
        t_tensor = Nx.broadcast(Nx.tensor(t, type: :f32), {Nx.axis_size(x, 0)})

        # Conditional velocity
        v_cond =
          predict_fn.(params, %{
            "x_t" => x,
            "timestep" => t_tensor,
            "observations" => observations
          })

        # Unconditional velocity
        v_uncond =
          predict_fn.(params, %{
            "x_t" => x,
            "timestep" => t_tensor,
            "observations" => uncond_obs
          })

        # Guided velocity
        v_guided =
          Nx.add(
            v_uncond,
            Nx.multiply(guidance_scale, Nx.subtract(v_cond, v_uncond))
          )

        x_next = Nx.add(x, Nx.multiply(dt, v_guided))
        {x_next, t + dt}
      end)

    final_x
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a flow matching model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    action_dim = Keyword.get(opts, :action_dim, 64)
    action_horizon = Keyword.get(opts, :action_horizon, default_action_horizon())
    action_horizon * action_dim
  end

  @doc """
  Calculate approximate parameter count for a flow matching model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    obs_size = Keyword.get(opts, :obs_size, 287)
    action_dim = Keyword.get(opts, :action_dim, 64)
    action_horizon = Keyword.get(opts, :action_horizon, default_action_horizon())
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())

    action_flat = action_horizon * action_dim

    # Embeddings
    # Sinusoidal (no params, but we count dim)
    _time_embed = hidden_size
    obs_embed = obs_size * hidden_size
    x_embed = action_flat * hidden_size

    # Residual blocks: 2 dense layers each
    block_params = 2 * hidden_size * hidden_size

    # Output projection
    output_proj = hidden_size * action_flat

    obs_embed + x_embed + num_layers * block_params + output_proj
  end

  @doc """
  Get recommended defaults for action generation.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      action_dim: 64,
      action_horizon: 8,
      hidden_size: 256,
      num_layers: 4,
      num_steps: 20,
      solver: :euler
    ]
  end

  @doc """
  Get fast inference configuration.
  """
  @spec fast_inference_defaults() :: keyword()
  def fast_inference_defaults do
    [
      action_dim: 64,
      action_horizon: 4,
      hidden_size: 128,
      num_layers: 2,
      num_steps: 10,
      solver: :euler
    ]
  end

  @doc """
  Get high-quality configuration (more steps, better solver).
  """
  @spec quality_defaults() :: keyword()
  def quality_defaults do
    [
      action_dim: 64,
      action_horizon: 8,
      hidden_size: 256,
      num_layers: 4,
      num_steps: 50,
      solver: :rk4
    ]
  end
end
