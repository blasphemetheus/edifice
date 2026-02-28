defmodule Edifice.Generative.RectifiedFlowTest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.RectifiedFlow

  @batch 2
  @obs_size 8
  @action_dim 4
  @action_horizon 2
  @hidden_size 16
  @num_layers 1

  @build_opts [
    obs_size: @obs_size,
    action_dim: @action_dim,
    action_horizon: @action_horizon,
    hidden_size: @hidden_size,
    num_layers: @num_layers
  ]

  defp build_model do
    model = RectifiedFlow.build(@build_opts)
    {init_fn, predict_fn} = Axon.build(model, mode: :inference)

    template = %{
      "x_t" => Nx.template({@batch, @action_horizon, @action_dim}, :f32),
      "timestep" => Nx.template({@batch}, :f32),
      "observations" => Nx.template({@batch, @obs_size}, :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())
    {params, predict_fn}
  end

  defp random_inputs do
    key = Nx.Random.key(42)
    {obs, key} = Nx.Random.uniform(key, shape: {@batch, @obs_size})
    {noise, _key} = Nx.Random.uniform(key, shape: {@batch, @action_horizon, @action_dim})
    {obs, noise}
  end

  describe "build/1" do
    test "returns an Axon model" do
      model = RectifiedFlow.build(@build_opts)
      assert %Axon{} = model
    end

    test "builds via registry" do
      model = Edifice.build(:rectified_flow, @build_opts)
      assert %Axon{} = model
    end
  end

  describe "reflow_pairs/5" do
    test "returns {noise, generated} with correct shapes" do
      {params, predict_fn} = build_model()
      {obs, noise} = random_inputs()

      {x_0, x_1} = RectifiedFlow.reflow_pairs(params, predict_fn, obs, noise, num_steps: 5)

      assert Nx.shape(x_0) == {@batch, @action_horizon, @action_dim}
      assert Nx.shape(x_1) == {@batch, @action_horizon, @action_dim}
      # x_0 should be the original noise
      assert Nx.to_number(Nx.all(Nx.equal(x_0, noise))) == 1
    end
  end

  describe "straightness/6" do
    test "returns a scalar" do
      {params, predict_fn} = build_model()
      {obs, noise} = random_inputs()

      # Generate x_1 endpoints
      x_1 = RectifiedFlow.sample(params, predict_fn, obs, noise, num_steps: 5)

      s = RectifiedFlow.straightness(params, predict_fn, obs, noise, x_1, num_eval_points: 3)

      assert Nx.shape(s) == {}
      assert Nx.to_number(s) >= 0.0
    end
  end

  describe "distill_loss/6" do
    test "returns a scalar loss" do
      {params, predict_fn} = build_model()
      {obs, noise} = random_inputs()

      loss =
        RectifiedFlow.distill_loss(params, params, predict_fn, obs, noise, teacher_steps: 5)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) >= 0.0
    end
  end

  describe "sample/5" do
    test "defaults to 1 step" do
      {params, predict_fn} = build_model()
      {obs, noise} = random_inputs()

      result = RectifiedFlow.sample(params, predict_fn, obs, noise)

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      refute Nx.to_number(Nx.any(Nx.is_nan(result))) == 1
    end

    test "works with multiple steps" do
      {params, predict_fn} = build_model()
      {obs, noise} = random_inputs()

      result = RectifiedFlow.sample(params, predict_fn, obs, noise, num_steps: 5)

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
    end
  end

  describe "delegated utilities" do
    test "interpolate/3" do
      x_0 = Nx.broadcast(0.0, {@batch, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      t = Nx.tensor([0.5, 0.5], type: :f32)

      result = RectifiedFlow.interpolate(x_0, x_1, t)
      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}

      val = Nx.to_number(result[0][0][0])
      assert abs(val - 0.5) < 1.0e-5
    end

    test "target_velocity/2" do
      x_0 = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(3.0, {@batch, @action_horizon, @action_dim})

      result = RectifiedFlow.target_velocity(x_0, x_1)
      val = Nx.to_number(result[0][0][0])
      assert abs(val - 2.0) < 1.0e-5
    end

    test "velocity_loss/2" do
      target = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      pred = Nx.broadcast(0.0, {@batch, @action_horizon, @action_dim})

      loss = RectifiedFlow.velocity_loss(target, pred)
      val = Nx.to_number(loss)
      assert abs(val - 1.0) < 1.0e-5
    end
  end

  describe "output_size/1" do
    test "returns action_horizon * action_dim" do
      assert RectifiedFlow.output_size(action_dim: 10, action_horizon: 4) == 40
    end
  end
end
