defmodule Edifice.RL.PPOTrainer do
  @moduledoc """
  Proximal Policy Optimization (PPO) trainer.

  Trains a PolicyValue network on an RL environment using the PPO algorithm.
  PPO is an on-policy actor-critic method that uses a clipped surrogate
  objective to ensure stable policy updates.

  ## Algorithm Overview

  1. Collect a rollout of T steps using the current policy
  2. Compute GAE advantages
  3. For K epochs: shuffle data into mini-batches, compute clipped PPO loss
  4. Repeat

  ## PPO Loss

  ```
  ratio = exp(new_log_prob - old_log_prob)
  surr1 = ratio * advantage
  surr2 = clip(ratio, 1-epsilon, 1+epsilon) * advantage
  policy_loss = -mean(min(surr1, surr2))
  value_loss = mean((new_value - return)^2)
  entropy_loss = -mean(entropy)
  total = policy_loss + c1 * value_loss + c2 * entropy_loss
  ```

  ## Usage

      alias Edifice.RL.{PPOTrainer, Environments.CartPole, PolicyValue}

      model = PolicyValue.build(input_size: 4, action_size: 2)
      results = PPOTrainer.train(model, CartPole,
        num_iterations: 100,
        rollout_steps: 200,
        num_epochs: 4,
        learning_rate: 3.0e-4
      )

  ## References
  - "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
  """

  alias Edifice.RL.GAE

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @default_num_iterations 100
  @default_rollout_steps 200
  @default_num_epochs 4
  @default_batch_size 64
  @default_gamma 0.99
  @default_lambda 0.95
  @default_epsilon 0.2
  @default_value_coeff 0.5
  @default_entropy_coeff 0.01
  @default_learning_rate 3.0e-4
  # @default_max_grad_norm 0.5 — TODO: implement gradient clipping

  # ============================================================================
  # Training
  # ============================================================================

  @doc """
  Train a PolicyValue model using PPO.

  ## Parameters
    - `model` - An Axon model built with `PolicyValue.build/1`
    - `env_module` - Module implementing `Edifice.RL.Environment`

  ## Options
    - `:num_iterations` - Number of collect-and-train cycles (default: 100)
    - `:rollout_steps` - Steps per rollout (default: 200)
    - `:num_epochs` - PPO update epochs per rollout (default: 4)
    - `:batch_size` - Mini-batch size (default: 64)
    - `:gamma` - Discount factor (default: 0.99)
    - `:lambda` - GAE lambda (default: 0.95)
    - `:epsilon` - PPO clip range (default: 0.2)
    - `:value_coeff` - Value loss coefficient (default: 0.5)
    - `:entropy_coeff` - Entropy bonus coefficient (default: 0.01)
    - `:learning_rate` - Learning rate (default: 3e-4)
    - `:reset_opts` - Environment reset options (default: [])
    - `:on_iteration` - Callback fn(iteration, avg_reward) for logging (optional)

  ## Returns
    `%{params: trained_params, reward_history: [avg_reward_per_iteration]}`
  """
  @spec train(Axon.t(), module(), keyword()) :: map()
  def train(model, env_module, opts \\ []) do
    num_iterations = Keyword.get(opts, :num_iterations, @default_num_iterations)
    rollout_steps = Keyword.get(opts, :rollout_steps, @default_rollout_steps)
    num_epochs = Keyword.get(opts, :num_epochs, @default_num_epochs)
    batch_size = Keyword.get(opts, :batch_size, @default_batch_size)
    gamma = Keyword.get(opts, :gamma, @default_gamma)
    lambda = Keyword.get(opts, :lambda, @default_lambda)
    epsilon = Keyword.get(opts, :epsilon, @default_epsilon)
    value_coeff = Keyword.get(opts, :value_coeff, @default_value_coeff)
    entropy_coeff = Keyword.get(opts, :entropy_coeff, @default_entropy_coeff)
    learning_rate = Keyword.get(opts, :learning_rate, @default_learning_rate)
    reset_opts = Keyword.get(opts, :reset_opts, [])
    on_iteration = Keyword.get(opts, :on_iteration, nil)

    # Get observation size from environment
    obs_space = env_module.observation_space()
    obs_size = elem(obs_space.shape, 0)

    # Initialize model parameters
    {init_fn, predict_fn} = Axon.build(model, mode: :train)
    params = init_fn.(Nx.template({1, obs_size}, :f32), Axon.ModelState.empty())

    # Initialize optimizer (Adam)
    {optimizer_init, optimizer_update} = Polaris.Optimizers.adam(learning_rate: learning_rate)
    optimizer_state = optimizer_init.(params)

    # Training loop
    {final_params, _opt_state, reward_history} =
      Enum.reduce(1..num_iterations, {params, optimizer_state, []}, fn iteration,
                                                                       {current_params, opt_state,
                                                                        rewards_acc} ->
        # 1. Collect rollout with policy
        {trajectory, avg_reward} =
          collect_rollout(env_module, predict_fn, current_params, rollout_steps, reset_opts)

        # 2. Compute GAE advantages
        traj_rewards = Enum.map(trajectory, & &1.reward)
        traj_values = Enum.map(trajectory, & &1.value)
        traj_dones = Enum.map(trajectory, & &1.done)

        # Last value bootstrap
        last_exp = List.last(trajectory)
        last_value = if last_exp.done, do: 0.0, else: last_exp.value

        {advantages, returns} =
          GAE.compute(traj_rewards, traj_values, traj_dones, last_value,
            gamma: gamma,
            lambda: lambda
          )

        # Normalize advantages
        advantages = GAE.normalize(advantages)

        # 3. Prepare training data as tensors
        obs_tensor = trajectory |> Enum.map(& &1.obs) |> Nx.stack()
        actions_tensor = trajectory |> Enum.map(& &1.action) |> Nx.tensor(type: :s32)
        old_log_probs_tensor = trajectory |> Enum.map(& &1.log_prob) |> Nx.tensor(type: :f32)
        advantages_tensor = Nx.tensor(advantages, type: :f32)
        returns_tensor = Nx.tensor(returns, type: :f32)

        num_samples = length(trajectory)

        # 4. PPO optimization epochs
        {updated_params, updated_opt_state} =
          Enum.reduce(1..num_epochs, {current_params, opt_state}, fn _epoch, {p, os} ->
            # Shuffle indices
            indices = Enum.shuffle(0..(num_samples - 1))

            # Mini-batch updates
            indices
            |> Enum.chunk_every(batch_size)
            |> Enum.reject(fn chunk -> length(chunk) < 2 end)
            |> Enum.reduce({p, os}, fn batch_indices, {bp, bos} ->
              idx = Nx.tensor(batch_indices, type: :s32)
              batch_obs = Nx.take(obs_tensor, idx)
              batch_actions = Nx.take(actions_tensor, idx)
              batch_old_log_probs = Nx.take(old_log_probs_tensor, idx)
              batch_advantages = Nx.take(advantages_tensor, idx)
              batch_returns = Nx.take(returns_tensor, idx)

              # Compute gradients and update
              {loss, grads} =
                ppo_loss_and_grad(
                  bp,
                  predict_fn,
                  batch_obs,
                  batch_actions,
                  batch_old_log_probs,
                  batch_advantages,
                  batch_returns,
                  epsilon,
                  value_coeff,
                  entropy_coeff
                )

              {updates, new_os} = optimizer_update.(grads, bos, bp)
              new_params = Polaris.Updates.apply_updates(bp, updates)

              _ = loss
              {new_params, new_os}
            end)
          end)

        # Log progress
        if on_iteration, do: on_iteration.(iteration, avg_reward)

        {updated_params, updated_opt_state, [avg_reward | rewards_acc]}
      end)

    %{
      params: final_params,
      reward_history: Enum.reverse(reward_history)
    }
  end

  # ============================================================================
  # Rollout Collection
  # ============================================================================

  defp collect_rollout(env_module, predict_fn, params, num_steps, reset_opts) do
    {obs, env_state} = env_module.reset(reset_opts)

    {trajectory, _obs, _state, _done, total_reward, episode_count} =
      Enum.reduce(1..num_steps, {[], obs, env_state, false, 0.0, 0}, fn _step,
                                                                        {traj, current_obs,
                                                                         current_state, done,
                                                                         total_r, ep_count} ->
        # Reset if previous episode ended
        {actual_obs, actual_state, new_ep_count} =
          if done do
            {new_obs, new_state} = env_module.reset(reset_opts)
            {new_obs, new_state, ep_count + 1}
          else
            {current_obs, current_state, ep_count}
          end

        # Get policy output
        input = Nx.new_axis(actual_obs, 0)
        %{policy: probs, value: value} = predict_fn.(params, %{"observation" => input})

        # Sample action from policy
        probs_flat = Nx.squeeze(probs)
        value_scalar = Nx.to_number(Nx.squeeze(value))

        action = sample_categorical(probs_flat)
        log_prob = Nx.to_number(Nx.log(Nx.add(probs_flat[action], 1.0e-8)))

        # Step environment
        {next_obs, reward, next_done, _info, next_state} = env_module.step(action, actual_state)

        experience = %{
          obs: actual_obs,
          action: action,
          reward: reward,
          done: next_done,
          log_prob: log_prob,
          value: value_scalar
        }

        {[experience | traj], next_obs, next_state, next_done, total_r + reward, new_ep_count}
      end)

    trajectory = Enum.reverse(trajectory)
    avg_reward = if episode_count > 0, do: total_reward / episode_count, else: total_reward

    {trajectory, avg_reward}
  end

  # Sample from a categorical distribution
  defp sample_categorical(probs) do
    probs_list = Nx.to_flat_list(probs)
    r = :rand.uniform()

    {action, _} =
      Enum.reduce_while(Enum.with_index(probs_list), {0, 0.0}, fn {p, i}, {_best, cumsum} ->
        new_cumsum = cumsum + p

        if new_cumsum >= r do
          {:halt, {i, new_cumsum}}
        else
          {:cont, {i, new_cumsum}}
        end
      end)

    action
  end

  # ============================================================================
  # PPO Loss Computation
  # ============================================================================

  defp ppo_loss_and_grad(
         params,
         predict_fn,
         obs,
         actions,
         old_log_probs,
         advantages,
         returns,
         epsilon,
         value_coeff,
         entropy_coeff
       ) do
    loss_fn = fn p ->
      %{policy: new_probs, value: new_values} = predict_fn.(p, %{"observation" => obs})

      # Get log probs of taken actions
      num_actions = Nx.axis_size(new_probs, 1)

      action_one_hot =
        Nx.equal(
          Nx.iota({Nx.axis_size(actions, 0), num_actions}, axis: 1),
          Nx.new_axis(actions, 1)
        )

      selected_probs = Nx.sum(Nx.multiply(new_probs, action_one_hot), axes: [1])
      new_log_probs = Nx.log(Nx.add(selected_probs, 1.0e-8))

      # Policy loss: clipped surrogate
      ratio = Nx.exp(Nx.subtract(new_log_probs, old_log_probs))
      surr1 = Nx.multiply(ratio, advantages)
      surr2 = Nx.multiply(Nx.clip(ratio, 1.0 - epsilon, 1.0 + epsilon), advantages)
      policy_loss = Nx.negate(Nx.mean(Nx.min(surr1, surr2)))

      # Value loss: MSE
      value_loss = Nx.mean(Nx.pow(Nx.subtract(new_values, returns), 2))

      # Entropy bonus: -sum(p * log(p))
      entropy =
        Nx.negate(Nx.sum(Nx.multiply(new_probs, Nx.log(Nx.add(new_probs, 1.0e-8))), axes: [1]))

      entropy_loss = Nx.negate(Nx.mean(entropy))

      # Total loss
      Nx.add(
        Nx.add(policy_loss, Nx.multiply(value_coeff, value_loss)),
        Nx.multiply(entropy_coeff, entropy_loss)
      )
    end

    Nx.Defn.value_and_grad(params, loss_fn)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Evaluate a trained policy on an environment.

  Runs `num_episodes` episodes with the given policy and returns average reward.
  """
  @spec evaluate(Axon.t(), map(), module(), keyword()) :: float()
  def evaluate(model, params, env_module, opts \\ []) do
    num_episodes = Keyword.get(opts, :num_episodes, 10)
    max_steps = Keyword.get(opts, :max_steps, 500)
    reset_opts = Keyword.get(opts, :reset_opts, [])

    {_init_fn, predict_fn} = Axon.build(model, mode: :inference)

    total_reward =
      Enum.reduce(1..num_episodes, 0.0, fn _ep, acc ->
        {obs, state} = env_module.reset(reset_opts)

        {ep_reward, _obs, _state} =
          Enum.reduce_while(1..max_steps, {0.0, obs, state}, fn _step,
                                                                {r, current_obs, current_state} ->
            input = Nx.new_axis(current_obs, 0)
            %{policy: probs} = predict_fn.(params, %{"observation" => input})

            # Greedy action selection
            action = Nx.to_number(Nx.argmax(Nx.squeeze(probs)))
            {next_obs, reward, done, _info, next_state} = env_module.step(action, current_state)

            if done do
              {:halt, {r + reward, next_obs, next_state}}
            else
              {:cont, {r + reward, next_obs, next_state}}
            end
          end)

        acc + ep_reward
      end)

    total_reward / num_episodes
  end
end
