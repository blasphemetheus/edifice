defmodule Edifice.RL.Environment do
  @moduledoc """
  Behaviour for reinforcement learning environments.

  Defines the standard interface for RL environments, compatible with
  the OpenAI Gym / Gymnasium convention adapted for Elixir.

  ## Callbacks

    - `observation_space/0` — Returns observation space descriptor
    - `action_space/0` — Returns action space descriptor
    - `reset/1` — Reset environment to initial state
    - `step/2` — Take an action, return next observation, reward, done flag

  ## Usage

      defmodule MyEnv do
        @behaviour Edifice.RL.Environment

        @impl true
        def observation_space, do: %{shape: {4}, type: :f32}

        @impl true
        def action_space, do: %{type: :discrete, n: 2}

        @impl true
        def reset(_opts), do: {Nx.broadcast(0.0, {4}), %{}}

        @impl true
        def step(action, state) do
          next_obs = Nx.broadcast(0.0, {4})
          reward = 1.0
          done = false
          info = %{}
          {next_obs, reward, done, info, state}
        end
      end

  ## Rollout Utility

  Use `rollout/4` to collect a trajectory from any environment implementing
  this behaviour:

      trajectory = Edifice.RL.Environment.rollout(MyEnv, policy_fn, 100)

  """

  @doc "Return observation space descriptor (e.g. `%{shape: {4}, type: :f32}`)."
  @callback observation_space() :: map()

  @doc "Return action space descriptor (e.g. `%{type: :discrete, n: 2}`)."
  @callback action_space() :: map()

  @doc """
  Reset the environment to an initial state.

  ## Parameters
    - `opts` - Environment-specific reset options

  ## Returns
    `{observation, state}` where `observation` is the initial observation tensor
    and `state` is opaque environment state.
  """
  @callback reset(opts :: keyword()) :: {Nx.Tensor.t(), any()}

  @doc """
  Take one step in the environment.

  ## Parameters
    - `action` - Action to take (tensor or integer)
    - `state` - Current environment state (from `reset/1` or previous `step/2`)

  ## Returns
    `{next_obs, reward, done, info, next_state}` tuple.
  """
  @callback step(action :: any(), state :: any()) ::
              {Nx.Tensor.t(), float(), boolean(), map(), any()}

  @doc """
  Collect a rollout trajectory from an environment.

  Runs a plain Elixir loop (not defn) calling `reset/1` then `step/2`
  repeatedly, using `policy_fn` to select actions.

  ## Parameters

    - `env_module` - Module implementing `Edifice.RL.Environment`
    - `policy_fn` - Function `observation -> action`
    - `num_steps` - Maximum number of steps to collect
    - `opts` - Options:
      - `:reset_opts` - Passed to `env_module.reset/1` (default: `[]`)

  ## Returns

    List of `%{obs: tensor, action: any, reward: float, done: boolean}` maps.
  """
  @spec rollout(module(), (Nx.Tensor.t() -> any()), pos_integer(), keyword()) :: [map()]
  def rollout(env_module, policy_fn, num_steps, opts \\ []) do
    reset_opts = Keyword.get(opts, :reset_opts, [])
    {obs, state} = env_module.reset(reset_opts)

    {trajectory, _obs, _state, _done} =
      Enum.reduce(1..num_steps, {[], obs, state, false}, fn _step, {traj, current_obs, current_state, done} ->
        if done do
          {traj, current_obs, current_state, true}
        else
          action = policy_fn.(current_obs)
          {next_obs, reward, next_done, _info, next_state} = env_module.step(action, current_state)

          experience = %{
            obs: current_obs,
            action: action,
            reward: reward,
            done: next_done
          }

          {[experience | traj], next_obs, next_state, next_done}
        end
      end)

    Enum.reverse(trajectory)
  end
end
