defmodule Edifice.RL.PPOTrainerTest do
  use ExUnit.Case, async: true
  alias Edifice.RL.{PPOTrainer, PolicyValue}

  defmodule Counter do
    def observation_space, do: %{shape: {1}, type: :f32}
    def action_space, do: %{type: :discrete, n: 2}

    def reset(opts) do
      if opts[:notify], do: send(opts[:notify], :environment_reset)
      {Nx.tensor([0.0]), {0, Keyword.get(opts, :terminal, 100)}}
    end

    def step(_action, {n, terminal}) do
      {Nx.tensor([n + 1.0]), 1.0, n + 1 == terminal, %{}, {n + 1, terminal}}
    end
  end

  defp value_prediction(_params, %{"observation" => obs}) do
    %{
      policy: Nx.broadcast(Nx.tensor([[0.5, 0.5]]), {Nx.axis_size(obs, 0), 2}),
      value: Nx.squeeze(obs, axes: [1])
    }
  end

  test "rollout bootstraps the next observation, and zeroes only true terminal boundaries" do
    {trajectory, reward, value} =
      PPOTrainer.collect_rollout(Counter, &value_prediction/2, %{}, 3, [])

    assert Enum.map(trajectory, & &1.value) == [0.0, 1.0, 2.0]
    assert value == 3.0
    assert reward == 3.0

    {_, reward, value} =
      PPOTrainer.collect_rollout(Counter, &value_prediction/2, %{}, 4, terminal: 2)

    assert value == 0.0
    assert reward == 2.0
  end

  test "training updates finite parameters, including a singleton minibatch" do
    model = PolicyValue.build(input_size: 1, action_size: 2, hidden_size: 4)
    {init, _} = Axon.build(model, mode: :inference, seed: 42)
    initial = init.(%{"observation" => Nx.template({1, 1}, :f32)}, Axon.ModelState.empty())

    result =
      PPOTrainer.train(model, Counter,
        num_iterations: 2,
        num_epochs: 1,
        rollout_steps: 1,
        batch_size: 1,
        seed: 42,
        reset_opts: [notify: self()]
      )

    assert_received :environment_reset
    refute_received :environment_reset

    refute Nx.to_flat_list(result.params.data["pv_value_head"]["bias"]) ==
             Nx.to_flat_list(initial.data["pv_value_head"]["bias"])

    for {_name, params} <- result.params.data, {_key, tensor} <- params do
      assert Nx.to_number(Nx.any(Nx.logical_or(Nx.is_nan(tensor), Nx.is_infinity(tensor)))) == 0
    end

    assert result.reward_history == [1.0, 1.0]

    assert PPOTrainer.evaluate(model, result.params, Counter, num_episodes: 2, max_steps: 3) ==
             3.0
  end

  @tag :exla_only
  test "compiled host training executes clipping and Adam with finite updates" do
    model = PolicyValue.build(input_size: 1, action_size: 2, hidden_size: 4)

    result =
      PPOTrainer.train(model, Counter,
        num_iterations: 2,
        num_epochs: 2,
        rollout_steps: 3,
        batch_size: 2,
        compiler: EXLA,
        client: :host
      )

    assert result.reward_history == [3.0, 3.0]
    assert Nx.to_number(Nx.any(Nx.is_nan(result.params.data["pv_value_head"]["kernel"]))) == 0
  end

  test "invalid loop and clipping options fail before training" do
    model = PolicyValue.build(input_size: 1, action_size: 2, hidden_size: 4)

    for opts <- [[num_iterations: 0], [rollout_steps: 0], [batch_size: 0], [max_grad_norm: 0]] do
      assert_raise ArgumentError, fn -> PPOTrainer.train(model, Counter, opts) end
    end
  end
end
