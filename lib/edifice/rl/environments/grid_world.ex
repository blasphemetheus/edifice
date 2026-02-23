defmodule Edifice.RL.Environments.GridWorld do
  @moduledoc """
  Simple grid world environment for testing RL algorithms.

  An agent starts at (0, 0) and must reach the goal at (size-1, size-1)
  on a square grid. Simpler than CartPole — useful for debugging training loops.

  ## Observation Space

  `[row/size, col/size]` — normalized position, shape {2}, type :f32

  ## Action Space

  Discrete(4): 0=up, 1=right, 2=down, 3=left

  ## Reward

  - -0.01 per step (time penalty encourages efficiency)
  - +1.0 for reaching the goal

  ## Termination

  - Reaching the goal cell
  - Exceeding max_steps (default: 100)

  ## Usage

      {obs, state} = GridWorld.reset(size: 5)
      {next_obs, reward, done, info, next_state} = GridWorld.step(1, state)
  """

  @behaviour Edifice.RL.Environment

  @default_size 5
  @default_max_steps 100

  @impl true
  def observation_space do
    %{shape: {2}, type: :f32, low: [0.0, 0.0], high: [1.0, 1.0]}
  end

  @impl true
  def action_space do
    %{type: :discrete, n: 4}
  end

  @impl true
  def reset(opts \\ []) do
    size = Keyword.get(opts, :size, @default_size)
    max_steps = Keyword.get(opts, :max_steps, @default_max_steps)

    state = %{row: 0, col: 0, size: size, max_steps: max_steps, step_count: 0}
    obs = state_to_obs(state)
    {obs, state}
  end

  @impl true
  def step(action, state) do
    %{row: row, col: col, size: size, max_steps: max_steps, step_count: step_count} = state

    # Apply action: 0=up, 1=right, 2=down, 3=left
    {new_row, new_col} =
      case action do
        0 -> {max(row - 1, 0), col}
        1 -> {row, min(col + 1, size - 1)}
        2 -> {min(row + 1, size - 1), col}
        3 -> {row, max(col - 1, 0)}
        _ -> {row, col}
      end

    new_step_count = step_count + 1
    at_goal = new_row == size - 1 and new_col == size - 1
    timed_out = new_step_count >= max_steps
    done = at_goal or timed_out

    reward = if at_goal, do: 1.0, else: -0.01

    new_state = %{state | row: new_row, col: new_col, step_count: new_step_count}
    obs = state_to_obs(new_state)
    info = %{step_count: new_step_count, at_goal: at_goal}

    {obs, reward, done, info, new_state}
  end

  defp state_to_obs(%{row: row, col: col, size: size}) do
    Nx.tensor([row / max(size - 1, 1), col / max(size - 1, 1)], type: :f32)
  end
end
