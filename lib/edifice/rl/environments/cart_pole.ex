defmodule Edifice.RL.Environments.CartPole do
  @moduledoc """
  Classic CartPole balancing environment.

  A pole is attached by an un-actuated joint to a cart, which moves along
  a frictionless track. The agent applies a force of +10 or -10 N to the
  cart to keep the pole upright.

  ## Observation Space

  | Index | Variable     | Range            |
  |-------|-------------|------------------|
  | 0     | x (position) | [-2.4, 2.4]      |
  | 1     | x_dot (vel)  | (-inf, inf)      |
  | 2     | theta (angle)| [-0.21, 0.21] rad|
  | 3     | theta_dot    | (-inf, inf)      |

  ## Action Space

  Discrete(2): 0 = push left (-10N), 1 = push right (+10N)

  ## Reward

  +1.0 for every timestep the pole remains upright.

  ## Termination

  - Cart position |x| > 2.4
  - Pole angle |theta| > 12 degrees (~0.2095 rad)
  - Episode length > 500 steps

  ## Usage

      {obs, state} = CartPole.reset([])
      {next_obs, reward, done, info, next_state} = CartPole.step(1, state)

  ## References
  - Barto, Sutton, Anderson (1983) "Neuronlike adaptive elements..."
  - OpenAI Gymnasium CartPole-v1
  """

  @behaviour Edifice.RL.Environment

  # Physics constants
  @gravity 9.8
  @masspole 0.1
  @total_mass 1.1
  @length 0.5
  @polemass_length 0.05
  @force_mag 10.0
  @tau 0.02

  # Termination thresholds
  @x_threshold 2.4
  @theta_threshold 0.2095
  @max_steps 500

  @impl true
  def observation_space do
    %{shape: {4}, type: :f32, low: [-2.4, :neg_inf, -0.2095, :neg_inf], high: [2.4, :inf, 0.2095, :inf]}
  end

  @impl true
  def action_space do
    %{type: :discrete, n: 2}
  end

  @impl true
  def reset(_opts \\ []) do
    # Initialize state with small random perturbations
    x = :rand.uniform() * 0.1 - 0.05
    x_dot = :rand.uniform() * 0.1 - 0.05
    theta = :rand.uniform() * 0.1 - 0.05
    theta_dot = :rand.uniform() * 0.1 - 0.05

    state = %{x: x, x_dot: x_dot, theta: theta, theta_dot: theta_dot, step_count: 0}
    obs = state_to_obs(state)

    {obs, state}
  end

  @impl true
  def step(action, state) do
    %{x: x, x_dot: x_dot, theta: theta, theta_dot: theta_dot, step_count: step_count} = state

    # Determine force direction
    force = if action == 1, do: @force_mag, else: -@force_mag

    cos_theta = :math.cos(theta)
    sin_theta = :math.sin(theta)

    # Physics: nonlinear cart-pole dynamics
    temp = (force + @polemass_length * theta_dot * theta_dot * sin_theta) / @total_mass

    theta_acc =
      (@gravity * sin_theta - cos_theta * temp) /
        (@length * (4.0 / 3.0 - @masspole * cos_theta * cos_theta / @total_mass))

    x_acc = temp - @polemass_length * theta_acc * cos_theta / @total_mass

    # Euler integration
    new_x = x + @tau * x_dot
    new_x_dot = x_dot + @tau * x_acc
    new_theta = theta + @tau * theta_dot
    new_theta_dot = theta_dot + @tau * theta_acc

    new_step_count = step_count + 1

    # Check termination
    done =
      abs(new_x) > @x_threshold or
        abs(new_theta) > @theta_threshold or
        new_step_count >= @max_steps

    new_state = %{
      x: new_x,
      x_dot: new_x_dot,
      theta: new_theta,
      theta_dot: new_theta_dot,
      step_count: new_step_count
    }

    obs = state_to_obs(new_state)
    reward = 1.0
    info = %{step_count: new_step_count}

    {obs, reward, done, info, new_state}
  end

  defp state_to_obs(%{x: x, x_dot: x_dot, theta: theta, theta_dot: theta_dot}) do
    Nx.tensor([x, x_dot, theta, theta_dot], type: :f32)
  end
end
