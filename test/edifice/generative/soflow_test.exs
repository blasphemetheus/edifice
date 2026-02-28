defmodule Edifice.Generative.SoFlowTest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.SoFlow

  @batch 2
  @obs_size 16
  @action_dim 8
  @action_horizon 4
  @hidden_size 32
  @num_layers 2

  @opts [
    obs_size: @obs_size,
    action_dim: @action_dim,
    action_horizon: @action_horizon,
    hidden_size: @hidden_size,
    num_layers: @num_layers
  ]

  defp random_inputs do
    key = Nx.Random.key(42)
    {x_t, key} = Nx.Random.uniform(key, shape: {@batch, @action_horizon, @action_dim})
    {current_time, key} = Nx.Random.uniform(key, shape: {@batch})
    {target_time, key} = Nx.Random.uniform(key, shape: {@batch})
    {observations, _key} = Nx.Random.uniform(key, shape: {@batch, @obs_size})

    %{
      "x_t" => x_t,
      "current_time" => current_time,
      "target_time" => target_time,
      "observations" => observations
    }
  end

  describe "build/1" do
    test "builds an Axon model" do
      model = SoFlow.build(@opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = SoFlow.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch, @action_horizon, @action_dim}, :f32),
            "current_time" => Nx.template({@batch}, :f32),
            "target_time" => Nx.template({@batch}, :f32),
            "observations" => Nx.template({@batch, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_inputs())

      assert Nx.shape(output) == {@batch, @action_horizon, @action_dim}
    end

    test "output contains finite values" do
      model = SoFlow.build(@opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "x_t" => Nx.template({@batch, @action_horizon, @action_dim}, :f32),
            "current_time" => Nx.template({@batch}, :f32),
            "target_time" => Nx.template({@batch}, :f32),
            "observations" => Nx.template({@batch, @obs_size}, :f32)
          },
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_inputs())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "euler_parameterize/4" do
    test "computes f(x_t, t, s) = x_t + (s - t) * F_theta" do
      x_t = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      f_theta = Nx.broadcast(2.0, {@batch, @action_horizon, @action_dim})
      t = Nx.tensor([0.0, 0.0], type: :f32)
      s = Nx.tensor([1.0, 1.0], type: :f32)

      result = SoFlow.euler_parameterize(x_t, f_theta, t, s)

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      # x_t + (1 - 0) * f_theta = 1 + 2 = 3
      val = Nx.to_number(result[0][0][0])
      assert abs(val - 3.0) < 1.0e-5
    end

    test "returns x_t when t equals s" do
      x_t = Nx.broadcast(5.0, {@batch, @action_horizon, @action_dim})
      f_theta = Nx.broadcast(10.0, {@batch, @action_horizon, @action_dim})
      t = Nx.tensor([0.5, 0.5], type: :f32)
      s = Nx.tensor([0.5, 0.5], type: :f32)

      result = SoFlow.euler_parameterize(x_t, f_theta, t, s)

      val = Nx.to_number(result[0][0][0])
      assert abs(val - 5.0) < 1.0e-5
    end
  end

  describe "interpolate/3" do
    test "linear interpolation between x_0 and x_1" do
      x_0 = Nx.broadcast(0.0, {@batch, @action_horizon, @action_dim})
      x_1 = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      t = Nx.tensor([0.5, 0.5], type: :f32)

      result = SoFlow.interpolate(x_0, x_1, t)

      val = Nx.to_number(result[0][0][0])
      assert abs(val - 0.5) < 1.0e-5
    end
  end

  describe "losses" do
    test "flow_matching_loss computes MSE" do
      pred = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      target = Nx.broadcast(0.0, {@batch, @action_horizon, @action_dim})

      loss = SoFlow.flow_matching_loss(pred, target)
      assert Nx.shape(loss) == {}
      val = Nx.to_number(loss)
      assert abs(val - 1.0) < 1.0e-5
    end

    test "consistency_loss computes MSE with stop_grad" do
      current = Nx.broadcast(1.0, {@batch, @action_horizon, @action_dim})
      target = Nx.broadcast(0.0, {@batch, @action_horizon, @action_dim})

      loss = SoFlow.consistency_loss(current, target)
      assert Nx.shape(loss) == {}
    end

    test "combined_loss mixes fm and scm losses" do
      fm_loss = Nx.tensor(1.0, type: :f32)
      scm_loss = Nx.tensor(2.0, type: :f32)

      combined = SoFlow.combined_loss(fm_loss, scm_loss, 0.5)
      # 0.5 * 1.0 + 0.5 * 2.0 = 1.5
      val = Nx.to_number(combined)
      assert abs(val - 1.5) < 1.0e-5
    end
  end

  describe "output_size/1" do
    test "returns action_horizon * action_dim" do
      assert SoFlow.output_size(@opts) == @action_horizon * @action_dim
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = SoFlow.recommended_defaults()
      assert Keyword.has_key?(defaults, :action_dim)
      assert Keyword.has_key?(defaults, :action_horizon)
      assert Keyword.has_key?(defaults, :hidden_size)
    end
  end
end
