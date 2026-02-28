defmodule Edifice.Robotics.DiffusionPolicyTest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  @batch 2
  @action_dim 4
  @obs_dim 8
  @tp 8
  @to 2

  @opts [
    action_dim: @action_dim,
    obs_dim: @obs_dim,
    prediction_horizon: @tp,
    observation_horizon: @to,
    down_dims: [16, 32],
    diffusion_step_embed_dim: 16,
    n_groups: 4,
    num_train_timesteps: 10
  ]

  defp build_and_predict(opts \\ @opts) do
    model = Edifice.Robotics.DiffusionPolicy.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    tp = Keyword.get(opts, :prediction_horizon, @tp)
    to = Keyword.get(opts, :observation_horizon, @to)
    action_dim = Keyword.get(opts, :action_dim, @action_dim)
    obs_dim = Keyword.get(opts, :obs_dim, @obs_dim)
    batch = Keyword.get(opts, :batch, @batch)

    input = %{
      "noisy_actions" => random_tensor({batch, tp, action_dim}),
      "timestep" => Nx.tensor(Enum.map(1..batch, fn _ -> :rand.uniform(9) end)),
      "observations" => random_tensor({batch, to, obs_dim})
    }

    params = init_fn.(input, %{})
    output = predict_fn.(params, input)
    {output, {batch, tp, action_dim}}
  end

  describe "build/1" do
    test "produces correct output shape" do
      {output, {batch, tp, action_dim}} = build_and_predict()
      assert Nx.shape(output) == {batch, tp, action_dim}
    end

    test "output values are finite" do
      {output, _shape} = build_and_predict()
      assert_finite!(output)
    end

    test "handles batch_size=1" do
      {output, _} = build_and_predict(Keyword.put(@opts, :batch, 1))
      assert {1, @tp, @action_dim} == Nx.shape(output)
      assert_finite!(output)
    end

    test "works with different action dimensions" do
      opts = Keyword.merge(@opts, action_dim: 7, obs_dim: 20)
      {output, _} = build_and_predict(opts)
      assert {2, @tp, 7} == Nx.shape(output)
    end

    test "works with different prediction horizon" do
      opts = Keyword.put(@opts, :prediction_horizon, 4)
      {output, _} = build_and_predict(opts)
      assert {2, 4, @action_dim} == Nx.shape(output)
    end

    test "works with single down_dim level" do
      opts = Keyword.put(@opts, :down_dims, [16])
      {output, _} = build_and_predict(opts)
      assert {2, @tp, @action_dim} == Nx.shape(output)
      assert_finite!(output)
    end
  end

  describe "make_cosine_schedule/1" do
    test "returns correct schedule fields" do
      schedule = Edifice.Robotics.DiffusionPolicy.make_cosine_schedule(num_steps: 10)

      assert schedule.num_steps == 10
      assert Nx.shape(schedule.betas) == {10}
      assert Nx.shape(schedule.alphas_cumprod) == {10}
      assert Nx.shape(schedule.sqrt_alphas_cumprod) == {10}
      assert Nx.shape(schedule.sqrt_one_minus_alphas_cumprod) == {10}
    end

    test "alphas_cumprod is monotonically decreasing" do
      schedule = Edifice.Robotics.DiffusionPolicy.make_cosine_schedule(num_steps: 50)
      ac = Nx.to_flat_list(schedule.alphas_cumprod)

      for [a, b] <- Enum.chunk_every(ac, 2, 1, :discard) do
        assert a >= b, "alphas_cumprod should be monotonically decreasing"
      end
    end

    test "betas are in valid range" do
      schedule = Edifice.Robotics.DiffusionPolicy.make_cosine_schedule(num_steps: 100)
      betas = Nx.to_flat_list(schedule.betas)

      for b <- betas do
        assert b >= 0.0 and b <= 1.0, "betas should be in [0, 1]"
      end
    end
  end

  describe "add_noise/4" do
    test "noisy actions have same shape as input" do
      schedule = Edifice.Robotics.DiffusionPolicy.make_cosine_schedule(num_steps: 10)
      actions = random_tensor({2, 8, 4})
      noise = random_tensor({2, 8, 4})
      timesteps = Nx.tensor([3, 7])

      noisy = Edifice.Robotics.DiffusionPolicy.add_noise(actions, noise, timesteps, schedule)
      assert Nx.shape(noisy) == {2, 8, 4}
      assert_finite!(noisy)
    end
  end

  describe "compute_loss/2" do
    test "returns scalar loss" do
      pred = random_tensor({2, 8, 4})
      actual = random_tensor({2, 8, 4})
      loss = Edifice.Robotics.DiffusionPolicy.compute_loss(pred, actual)
      assert Nx.shape(loss) == {}
      assert_finite!(loss)
    end
  end

  describe "output_size/1" do
    test "returns action_dim * prediction_horizon" do
      assert Edifice.Robotics.DiffusionPolicy.output_size(
               action_dim: 7,
               prediction_horizon: 16
             ) == 112
    end
  end
end
