defmodule Edifice.Generative.DDIMCoverageTest do
  @moduledoc """
  Coverage tests for DDIM module.
  Targets uncovered branches: ddim_sample full loop, ddim_step with non-zero eta,
  schedule variations, recommended_defaults, and edge cases.
  """
  use ExUnit.Case, async: true

  alias Edifice.Generative.DDIM

  @batch 2
  @action_dim 4
  @action_horizon 2
  @obs_size 16
  @hidden_size 16

  @base_opts [
    obs_size: @obs_size,
    action_dim: @action_dim,
    action_horizon: @action_horizon,
    hidden_size: @hidden_size,
    num_layers: 1,
    num_steps: 20
  ]

  # ============================================================================
  # ddim_sample - full sampling loop
  # ============================================================================

  describe "ddim_sample/5" do
    test "deterministic sampling (eta=0.0) produces correct shape" do
      model = DDIM.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "noisy_actions" => Nx.template({@batch, @action_horizon, @action_dim}, :f32),
        "timestep" => Nx.template({@batch}, :f32),
        "observations" => Nx.template({@batch, @obs_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      schedule = DDIM.make_schedule(num_steps: 20)
      observations = Nx.broadcast(0.5, {@batch, @obs_size})
      initial_noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result =
        DDIM.ddim_sample(params, predict_fn, observations, initial_noise,
          schedule: schedule,
          ddim_steps: 5,
          eta: 0.0
        )

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "deterministic sampling is reproducible" do
      model = DDIM.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "noisy_actions" => Nx.template({@batch, @action_horizon, @action_dim}, :f32),
        "timestep" => Nx.template({@batch}, :f32),
        "observations" => Nx.template({@batch, @obs_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      schedule = DDIM.make_schedule(num_steps: 20)
      observations = Nx.broadcast(0.5, {@batch, @obs_size})
      initial_noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result1 =
        DDIM.ddim_sample(params, predict_fn, observations, initial_noise,
          schedule: schedule,
          ddim_steps: 5,
          eta: 0.0
        )

      result2 =
        DDIM.ddim_sample(params, predict_fn, observations, initial_noise,
          schedule: schedule,
          ddim_steps: 5,
          eta: 0.0
        )

      diff = Nx.subtract(result1, result2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-5, "Deterministic DDIM sampling should be reproducible"
    end

    test "sampling with fewer steps still produces valid output" do
      model = DDIM.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "noisy_actions" => Nx.template({@batch, @action_horizon, @action_dim}, :f32),
        "timestep" => Nx.template({@batch}, :f32),
        "observations" => Nx.template({@batch, @obs_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      schedule = DDIM.make_schedule(num_steps: 20)
      observations = Nx.broadcast(0.5, {@batch, @obs_size})
      initial_noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      # Use very few steps (stride will be large)
      result =
        DDIM.ddim_sample(params, predict_fn, observations, initial_noise,
          schedule: schedule,
          ddim_steps: 2,
          eta: 0.0
        )

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "sampling with default ddim_steps and eta" do
      # Build with more steps to test default ddim_steps=50
      opts = Keyword.put(@base_opts, :num_steps, 100)
      model = DDIM.build(opts)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "noisy_actions" => Nx.template({@batch, @action_horizon, @action_dim}, :f32),
        "timestep" => Nx.template({@batch}, :f32),
        "observations" => Nx.template({@batch, @obs_size}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      schedule = DDIM.make_schedule(num_steps: 100)
      observations = Nx.broadcast(0.5, {@batch, @obs_size})
      initial_noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      # Use explicit ddim_steps but still exercise the default eta path
      result =
        DDIM.ddim_sample(params, predict_fn, observations, initial_noise,
          schedule: schedule,
          ddim_steps: 10
        )

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # ddim_step with non-zero eta (stochastic mode)
  # ============================================================================

  describe "ddim_step with non-zero eta" do
    test "eta=1.0 (DDPM-like) produces valid output" do
      schedule = DDIM.make_schedule(num_steps: 20)
      noisy = Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim})
      pred_noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result = DDIM.ddim_step(noisy, pred_noise, 10, 5, schedule, 1.0)

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "eta=0.5 (partially stochastic) produces valid output" do
      schedule = DDIM.make_schedule(num_steps: 20)
      noisy = Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim})
      pred_noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result = DDIM.ddim_step(noisy, pred_noise, 15, 10, schedule, 0.5)

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "eta=0.0 deterministic step" do
      schedule = DDIM.make_schedule(num_steps: 20)
      noisy = Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim})
      pred_noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result1 = DDIM.ddim_step(noisy, pred_noise, 10, 5, schedule, 0.0)
      result2 = DDIM.ddim_step(noisy, pred_noise, 10, 5, schedule, 0.0)

      diff = Nx.subtract(result1, result2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6, "Deterministic step should be reproducible"
    end

    test "step at t=0 boundary" do
      schedule = DDIM.make_schedule(num_steps: 20)
      noisy = Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim})
      pred_noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result = DDIM.ddim_step(noisy, pred_noise, 1, 0, schedule, 0.0)

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "step near end of schedule (high t)" do
      schedule = DDIM.make_schedule(num_steps: 20)
      noisy = Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim})
      pred_noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result = DDIM.ddim_step(noisy, pred_noise, 19, 14, schedule, 0.0)

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Schedule variations
  # ============================================================================

  describe "make_schedule variations" do
    test "default schedule with default parameters" do
      schedule = DDIM.make_schedule()

      assert schedule.num_steps == 1000
      assert Nx.shape(schedule.betas) == {1000}
      assert Nx.shape(schedule.alphas) == {1000}
      assert Nx.shape(schedule.alphas_cumprod) == {1000}
      assert Nx.shape(schedule.sqrt_alphas_cumprod) == {1000}
      assert Nx.shape(schedule.sqrt_one_minus_alphas_cumprod) == {1000}
    end

    test "custom beta_start and beta_end" do
      schedule = DDIM.make_schedule(num_steps: 50, beta_start: 1.0e-5, beta_end: 0.01)

      assert schedule.num_steps == 50
      assert Nx.shape(schedule.betas) == {50}

      # Betas should be monotonically increasing (linspace)
      first_beta = Nx.to_number(schedule.betas[0])
      last_beta = Nx.to_number(schedule.betas[49])

      assert_in_delta first_beta, 1.0e-5, 1.0e-4
      assert_in_delta last_beta, 0.01, 1.0e-3
      assert last_beta > first_beta
    end

    test "alphas_cumprod is monotonically decreasing" do
      schedule = DDIM.make_schedule(num_steps: 100)

      ac = schedule.alphas_cumprod
      first = Nx.to_number(ac[0])
      last = Nx.to_number(ac[99])

      # alphas_cumprod should decrease from near 1 to near 0
      assert first > last
      assert first > 0.9
      assert last < 0.5
    end

    test "sqrt values are consistent with alphas_cumprod" do
      schedule = DDIM.make_schedule(num_steps: 50)

      ac = schedule.alphas_cumprod
      sqrt_ac = schedule.sqrt_alphas_cumprod
      sqrt_omc = schedule.sqrt_one_minus_alphas_cumprod

      # sqrt(ac)^2 should approximately equal ac
      recon_ac = Nx.multiply(sqrt_ac, sqrt_ac)
      diff_ac = Nx.subtract(recon_ac, ac) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff_ac < 1.0e-5

      # sqrt(1-ac)^2 should approximately equal 1-ac
      one_minus_ac = Nx.subtract(1.0, ac)
      recon_omc = Nx.multiply(sqrt_omc, sqrt_omc)

      diff_omc =
        Nx.subtract(recon_omc, one_minus_ac) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert diff_omc < 1.0e-4
    end

    test "all schedule values are finite" do
      schedule = DDIM.make_schedule(num_steps: 200)

      assert Nx.all(Nx.is_nan(schedule.betas) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(schedule.alphas) |> Nx.logical_not()) |> Nx.to_number() == 1

      assert Nx.all(Nx.is_nan(schedule.alphas_cumprod) |> Nx.logical_not()) |> Nx.to_number() ==
               1

      assert Nx.all(Nx.is_nan(schedule.sqrt_alphas_cumprod) |> Nx.logical_not())
             |> Nx.to_number() == 1

      assert Nx.all(Nx.is_nan(schedule.sqrt_one_minus_alphas_cumprod) |> Nx.logical_not())
             |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Build model variations
  # ============================================================================

  describe "build variations" do
    test "different action_horizon" do
      model =
        DDIM.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: 6,
          hidden_size: @hidden_size,
          num_layers: 1,
          num_steps: 20
        )

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_actions" => Nx.broadcast(0.5, {@batch, 6, @action_dim}),
        "timestep" => Nx.tensor([5, 10], type: :f32),
        "observations" => Nx.broadcast(0.5, {@batch, @obs_size})
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, 6, @action_dim}
    end

    test "different num_layers" do
      model =
        DDIM.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: @hidden_size,
          num_layers: 3,
          num_steps: 20
        )

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_actions" => Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim}),
        "timestep" => Nx.tensor([5, 10], type: :f32),
        "observations" => Nx.broadcast(0.5, {@batch, @obs_size})
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @action_horizon, @action_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "different hidden_size" do
      model =
        DDIM.build(
          obs_size: @obs_size,
          action_dim: @action_dim,
          action_horizon: @action_horizon,
          hidden_size: 32,
          num_layers: 1,
          num_steps: 20
        )

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_actions" => Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim}),
        "timestep" => Nx.tensor([5, 10], type: :f32),
        "observations" => Nx.broadcast(0.5, {@batch, @obs_size})
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @action_horizon, @action_dim}
    end

    test "output is deterministic" do
      model = DDIM.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_actions" => Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim}),
        "timestep" => Nx.tensor([5, 10], type: :f32),
        "observations" => Nx.broadcast(0.5, {@batch, @obs_size})
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end

  # ============================================================================
  # Utility functions
  # ============================================================================

  describe "output_size/1" do
    test "with defaults" do
      # Default: action_dim=64, action_horizon=8 => 512
      assert DDIM.output_size() == 512
    end

    test "with custom values" do
      assert DDIM.output_size(action_dim: 8, action_horizon: 4) == 32
    end
  end

  describe "param_count/1" do
    test "returns positive integer" do
      count = DDIM.param_count(@base_opts)
      assert is_integer(count)
      assert count > 0
    end

    test "more layers means more params" do
      count_few = DDIM.param_count(Keyword.put(@base_opts, :num_layers, 1))
      count_many = DDIM.param_count(Keyword.put(@base_opts, :num_layers, 4))
      assert count_many > count_few
    end

    test "larger hidden_size means more params" do
      count_small = DDIM.param_count(Keyword.put(@base_opts, :hidden_size, 16))
      count_large = DDIM.param_count(Keyword.put(@base_opts, :hidden_size, 64))
      assert count_large > count_small
    end

    test "uses defaults when options not provided" do
      count = DDIM.param_count([])
      assert is_integer(count)
      assert count > 0
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = DDIM.recommended_defaults()

      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :action_dim)
      assert Keyword.has_key?(defaults, :action_horizon)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :num_steps)
      assert Keyword.has_key?(defaults, :ddim_steps)
      assert Keyword.has_key?(defaults, :eta)
    end

    test "eta default is 0.0 (deterministic)" do
      defaults = DDIM.recommended_defaults()
      assert Keyword.get(defaults, :eta) == 0.0
    end
  end

  # ============================================================================
  # Edge cases
  # ============================================================================

  describe "edge cases" do
    test "batch size 1" do
      model = DDIM.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_actions" => Nx.broadcast(0.5, {1, @action_horizon, @action_dim}),
        "timestep" => Nx.tensor([5], type: :f32),
        "observations" => Nx.broadcast(0.5, {1, @obs_size})
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, @action_horizon, @action_dim}
    end

    test "ddim_step with t == t_prev (same timestep)" do
      schedule = DDIM.make_schedule(num_steps: 20)
      noisy = Nx.broadcast(0.5, {@batch, @action_horizon, @action_dim})
      pred_noise = Nx.broadcast(0.1, {@batch, @action_horizon, @action_dim})

      result = DDIM.ddim_step(noisy, pred_noise, 5, 5, schedule, 0.0)

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "ddim_step with larger noise magnitude" do
      schedule = DDIM.make_schedule(num_steps: 20)
      noisy = Nx.broadcast(5.0, {@batch, @action_horizon, @action_dim})
      pred_noise = Nx.broadcast(2.0, {@batch, @action_horizon, @action_dim})

      result = DDIM.ddim_step(noisy, pred_noise, 10, 5, schedule, 0.0)

      assert Nx.shape(result) == {@batch, @action_horizon, @action_dim}
      assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "schedule with small num_steps" do
      schedule = DDIM.make_schedule(num_steps: 5)

      assert schedule.num_steps == 5
      assert Nx.shape(schedule.betas) == {5}
      assert Nx.all(Nx.is_nan(schedule.alphas_cumprod) |> Nx.logical_not()) |> Nx.to_number() ==
               1
    end
  end
end
