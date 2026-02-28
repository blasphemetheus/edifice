defmodule Edifice.Generative.FlowMatchingTest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.FlowMatching

  @batch 2
  @obs_size 8
  @action_dim 4
  @action_horizon 2

  describe "build/1" do
    test "builds an Axon model" do
      model =
        FlowMatching.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 16,
          num_layers: 1
        )

      assert %Axon{} = model
    end
  end

  describe "interpolate/3" do
    test "interpolates between noise and data" do
      x_0 = Nx.broadcast(0.0, {@batch, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      t = Nx.tensor([0.5, 0.5], type: :f32)

      result = FlowMatching.interpolate(x_0, x_1, t)
      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}

      # At t=0.5, result should be 0.5
      val = Nx.to_number(result[0][0][0])
      assert abs(val - 0.5) < 1.0e-5
    end

    test "t=0 returns noise" do
      x_0 = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(2.0, {@batch, @action_horizon, @action_dim})
      t = Nx.tensor([0.0, 0.0], type: :f32)

      result = FlowMatching.interpolate(x_0, x_1, t)
      val = Nx.to_number(result[0][0][0])
      assert abs(val - 1.0) < 1.0e-5
    end

    test "t=1 returns data" do
      x_0 = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(2.0, {@batch, @action_horizon, @action_dim})
      t = Nx.tensor([1.0, 1.0], type: :f32)

      result = FlowMatching.interpolate(x_0, x_1, t)
      val = Nx.to_number(result[0][0][0])
      assert abs(val - 2.0) < 1.0e-5
    end
  end

  describe "target_velocity/2" do
    test "returns x_1 - x_0" do
      x_0 = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(3.0, {@batch, @action_horizon, @action_dim})

      result = FlowMatching.target_velocity(x_0, x_1)
      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}

      val = Nx.to_number(result[0][0][0])
      assert abs(val - 2.0) < 1.0e-5
    end
  end

  describe "velocity_loss/2" do
    test "computes MSE loss" do
      target = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      pred = Nx.broadcast(0.0, {@batch, @action_horizon, @action_dim})

      loss = FlowMatching.velocity_loss(target, pred)
      assert Nx.shape(loss) == {}

      val = Nx.to_number(loss)
      assert abs(val - 1.0) < 1.0e-5
    end

    test "zero loss when pred equals target" do
      target = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      pred = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})

      loss = FlowMatching.velocity_loss(target, pred)
      val = Nx.to_number(loss)
      assert abs(val) < 1.0e-5
    end
  end

  describe "rectified_loss/2" do
    test "same as velocity_loss" do
      target = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      pred = Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim})

      r_loss = FlowMatching.rectified_loss(target, pred)
      v_loss = FlowMatching.velocity_loss(target, pred)

      assert Nx.to_number(r_loss) == Nx.to_number(v_loss)
    end
  end

  describe "output_size/1" do
    test "returns action_horizon * action_dim" do
      assert FlowMatching.output_size(action_dim: 10, action_horizon: 4) == 40
    end

    test "uses defaults" do
      size = FlowMatching.output_size()
      assert is_integer(size)
      assert size > 0
    end
  end

  describe "param_count/1" do
    test "returns positive count" do
      count =
        FlowMatching.param_count(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 32,
          num_layers: 2
        )

      assert is_integer(count)
      assert count > 0
    end
  end

  describe "defaults" do
    test "recommended_defaults returns keyword list" do
      defaults = FlowMatching.recommended_defaults()
      assert Keyword.has_key?(defaults, :action_dim)
      assert Keyword.has_key?(defaults, :solver)
    end

    test "fast_inference_defaults" do
      defaults = FlowMatching.fast_inference_defaults()
      assert Keyword.has_key?(defaults, :num_steps)
    end

    test "quality_defaults uses rk4" do
      defaults = FlowMatching.quality_defaults()
      assert defaults[:solver] == :rk4
    end

    test "default_action_horizon" do
      assert FlowMatching.default_action_horizon() == 8
    end

    test "default_hidden_size" do
      assert FlowMatching.default_hidden_size() == 256
    end

    test "default_num_layers" do
      assert FlowMatching.default_num_layers() == 4
    end

    test "default_num_steps" do
      assert FlowMatching.default_num_steps() == 20
    end

    test "default_solver" do
      assert FlowMatching.default_solver() == :euler
    end
  end
end
