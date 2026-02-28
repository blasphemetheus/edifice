defmodule Edifice.RL.EnvironmentTest do
  use ExUnit.Case, async: true
  @moduletag :rl

  alias Edifice.RL.Environment

  # Mock environment for testing
  defmodule MockEnv do
    @behaviour Edifice.RL.Environment

    @impl true
    def observation_space, do: %{shape: {4}, type: :f32}

    @impl true
    def action_space, do: %{type: :discrete, n: 2}

    @impl true
    def reset(_opts) do
      obs = Nx.tensor([1.0, 0.0, 0.0, 0.0])
      state = %{step: 0, max_steps: 5}
      {obs, state}
    end

    @impl true
    def step(action, state) do
      next_step = state.step + 1
      next_obs = Nx.tensor([0.0, 0.0, 0.0, Nx.to_number(action) / 1.0])
      reward = 1.0
      done = next_step >= state.max_steps
      info = %{step: next_step}
      next_state = %{state | step: next_step}
      {next_obs, reward, done, info, next_state}
    end
  end

  describe "behaviour callbacks" do
    test "observation_space returns valid descriptor" do
      space = MockEnv.observation_space()
      assert %{shape: {4}, type: :f32} = space
    end

    test "action_space returns valid descriptor" do
      space = MockEnv.action_space()
      assert %{type: :discrete, n: 2} = space
    end

    test "reset returns {obs, state}" do
      {obs, state} = MockEnv.reset([])
      assert Nx.shape(obs) == {4}
      assert state.step == 0
    end

    test "step returns {next_obs, reward, done, info, next_state}" do
      {_obs, state} = MockEnv.reset([])
      {next_obs, reward, done, info, next_state} = MockEnv.step(Nx.tensor(1), state)

      assert Nx.shape(next_obs) == {4}
      assert reward == 1.0
      assert done == false
      assert info.step == 1
      assert next_state.step == 1
    end

    test "episode terminates after max_steps" do
      {_obs, state} = MockEnv.reset([])

      {_obs, _reward, done, _info, final_state} =
        Enum.reduce(1..5, {nil, nil, false, nil, state}, fn _i, {_o, _r, _d, _info, s} ->
          MockEnv.step(Nx.tensor(0), s)
        end)

      assert done == true
      assert final_state.step == 5
    end
  end

  describe "rollout/4" do
    test "collects trajectory of correct length" do
      policy_fn = fn _obs -> Nx.tensor(0) end
      trajectory = Environment.rollout(MockEnv, policy_fn, 3)

      assert length(trajectory) == 3
    end

    test "each step contains required fields" do
      policy_fn = fn _obs -> Nx.tensor(1) end
      trajectory = Environment.rollout(MockEnv, policy_fn, 2)

      for step <- trajectory do
        assert Map.has_key?(step, :obs)
        assert Map.has_key?(step, :action)
        assert Map.has_key?(step, :reward)
        assert Map.has_key?(step, :done)
      end
    end

    test "stops collecting after done" do
      policy_fn = fn _obs -> Nx.tensor(0) end
      # Max steps is 5, request 10 — should stop at 5
      trajectory = Environment.rollout(MockEnv, policy_fn, 10)

      assert length(trajectory) == 5
      assert List.last(trajectory).done == true
    end

    test "first observation matches reset" do
      policy_fn = fn _obs -> Nx.tensor(0) end
      trajectory = Environment.rollout(MockEnv, policy_fn, 3)

      first_obs = hd(trajectory).obs
      assert Enum.at(Nx.to_flat_list(first_obs), 0) == 1.0
    end
  end
end
