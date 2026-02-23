defmodule Edifice.RL.GAE do
  @moduledoc """
  Generalized Advantage Estimation (GAE).

  Computes advantage estimates that smoothly interpolate between high-bias/
  low-variance (TD(0)) and low-bias/high-variance (Monte Carlo) estimates.

  ## The GAE Formula

  ```
  delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
  A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
  ```

  Computed efficiently via reverse iteration:
  ```
  for t = T-1 down to 0:
      delta = rewards[t] + gamma * next_value * (1 - done[t]) - values[t]
      gae = delta + gamma * lambda * (1 - done[t]) * gae
      advantages[t] = gae
  ```

  ## Parameters

  - `lambda = 1.0` → Monte Carlo (low bias, high variance)
  - `lambda = 0.0` → TD(0) (high bias, low variance)
  - `lambda = 0.95` → good default

  ## Usage

      advantages = GAE.compute(rewards, values, dones, last_value,
        gamma: 0.99,
        lambda: 0.95
      )

  ## References
  - "High-Dimensional Continuous Control Using GAE" (Schulman et al., 2015)
  """

  @default_gamma 0.99
  @default_lambda 0.95

  @doc """
  Compute GAE advantages from a rollout.

  ## Parameters
    - `rewards` - List of float rewards, length T
    - `values` - List of float values V(s_t), length T
    - `dones` - List of boolean done flags, length T
    - `last_value` - V(s_T) — value of the state after the last step

  ## Options
    - `:gamma` - Discount factor (default: 0.99)
    - `:lambda` - GAE lambda for bias-variance tradeoff (default: 0.95)

  ## Returns
    `{advantages, returns}` where both are lists of floats, length T.
    `returns = advantages + values` (the value regression target).
  """
  @spec compute([float()], [float()], [boolean()], float(), keyword()) ::
          {[float()], [float()]}
  def compute(rewards, values, dones, last_value, opts \\ []) do
    gamma = Keyword.get(opts, :gamma, @default_gamma)
    lambda = Keyword.get(opts, :lambda, @default_lambda)

    t_max = length(rewards)

    # Convert to indexed lists for reverse iteration
    rewards_indexed = Enum.with_index(rewards) |> Map.new(fn {v, i} -> {i, v} end)
    values_indexed = Enum.with_index(values) |> Map.new(fn {v, i} -> {i, v} end)
    dones_indexed = Enum.with_index(dones) |> Map.new(fn {v, i} -> {i, v} end)

    # Reverse iteration to compute advantages
    {advantages_map, _gae} =
      Enum.reduce((t_max - 1)..0//-1, {%{}, 0.0}, fn t, {adv_map, gae} ->
        next_value =
          if t == t_max - 1 do
            last_value
          else
            values_indexed[t + 1]
          end

        non_terminal = if dones_indexed[t], do: 0.0, else: 1.0

        delta = rewards_indexed[t] + gamma * next_value * non_terminal - values_indexed[t]
        new_gae = delta + gamma * lambda * non_terminal * gae

        {Map.put(adv_map, t, new_gae), new_gae}
      end)

    # Convert back to ordered lists
    advantages = for t <- 0..(t_max - 1), do: advantages_map[t]
    returns = Enum.zip(advantages, values) |> Enum.map(fn {a, v} -> a + v end)

    {advantages, returns}
  end

  @doc """
  Normalize advantages to zero mean and unit variance.

  Standard practice in PPO to reduce variance of policy gradient estimates.
  """
  @spec normalize([float()]) :: [float()]
  def normalize(advantages) do
    n = length(advantages)
    mean = Enum.sum(advantages) / n
    variance = Enum.reduce(advantages, 0.0, fn a, acc -> acc + (a - mean) * (a - mean) end) / n
    std = :math.sqrt(variance + 1.0e-8)
    Enum.map(advantages, fn a -> (a - mean) / std end)
  end
end
