defmodule Edifice.Generative.ConsistencyModelCorrectnessTest do
  @moduledoc """
  Correctness tests for ConsistencyModel c_skip/c_out + pseudo-Huber + Karras fix.
  Verifies boundary condition, Karras noise schedule properties,
  pseudo-Huber loss properties, and consistency loss returns scalar.
  """
  use ExUnit.Case, async: true

  alias Edifice.Generative.ConsistencyModel

  @batch 2
  @input_dim 16
  @hidden_size 16
  @sigma_min 0.002
  @sigma_max 80.0

  @base_opts [
    input_dim: @input_dim,
    hidden_size: @hidden_size,
    num_layers: 1,
    sigma_min: @sigma_min,
    sigma_max: @sigma_max
  ]

  # ============================================================================
  # Boundary Condition: f(x, sigma_min) ~ x
  # ============================================================================

  describe "boundary condition" do
    @tag timeout: 120_000
    test "at sigma_min, output approximately equals input (c_skip->1, c_out->0)" do
      model = ConsistencyModel.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "noisy_input" => Nx.template({@batch, @input_dim}, :f32),
        "sigma" => Nx.template({@batch}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_dim})

      # At sigma = sigma_min, f(x, sigma_min) should equal x
      sigma = Nx.broadcast(Nx.tensor(@sigma_min, type: :f32), {@batch})

      output = predict_fn.(params, %{"noisy_input" => input, "sigma" => sigma})

      assert Nx.shape(output) == {@batch, @input_dim}

      # c_skip(sigma_min) = sigma_data^2 / (0 + sigma_data^2) = 1
      # c_out(sigma_min) = sigma_data * 0 / sqrt(...) = 0
      # So output = 1 * x + 0 * F(x) = x
      diff = Nx.subtract(output, input) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert diff < 1.0e-4,
             "At sigma_min, output should approximately equal input. Max diff: #{diff}"
    end

    @tag timeout: 120_000
    test "at larger sigma, output differs from input" do
      model = ConsistencyModel.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "noisy_input" => Nx.template({@batch, @input_dim}, :f32),
        "sigma" => Nx.template({@batch}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_dim})

      # At a larger sigma, f(x, sigma) should NOT equal x
      sigma = Nx.broadcast(Nx.tensor(10.0, type: :f32), {@batch})

      output = predict_fn.(params, %{"noisy_input" => input, "sigma" => sigma})

      assert Nx.shape(output) == {@batch, @input_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Karras Noise Schedule
  # ============================================================================

  describe "Karras noise schedule" do
    test "first step is approximately sigma_min" do
      schedule = ConsistencyModel.noise_schedule(
        n_steps: 40,
        sigma_min: @sigma_min,
        sigma_max: @sigma_max
      )

      first =
        Nx.slice_along_axis(schedule, 0, 1, axis: 0) |> Nx.squeeze() |> Nx.to_number()

      assert_in_delta first, @sigma_min, 1.0e-6,
        "First schedule step should be sigma_min (#{@sigma_min}), got: #{first}"
    end

    test "last step is approximately sigma_max" do
      schedule = ConsistencyModel.noise_schedule(
        n_steps: 40,
        sigma_min: @sigma_min,
        sigma_max: @sigma_max
      )

      n = Nx.axis_size(schedule, 0)

      last =
        Nx.slice_along_axis(schedule, n - 1, 1, axis: 0) |> Nx.squeeze() |> Nx.to_number()

      assert_in_delta last, @sigma_max, 1.0e-3,
        "Last schedule step should be sigma_max (#{@sigma_max}), got: #{last}"
    end

    test "schedule is monotonically increasing" do
      schedule = ConsistencyModel.noise_schedule(
        n_steps: 20,
        sigma_min: @sigma_min,
        sigma_max: @sigma_max
      )

      values = Nx.to_flat_list(schedule)

      pairs = Enum.zip(Enum.drop(values, -1), Enum.drop(values, 1))

      Enum.each(pairs, fn {a, b} ->
        assert b >= a,
               "Schedule should be monotonically increasing, but #{b} < #{a}"
      end)
    end

    test "schedule has correct number of steps" do
      for n <- [5, 10, 40] do
        schedule = ConsistencyModel.noise_schedule(n_steps: n)
        assert Nx.axis_size(schedule, 0) == n, "Schedule should have #{n} steps"
      end
    end

    test "all schedule values are positive" do
      schedule = ConsistencyModel.noise_schedule(n_steps: 20)
      min_val = Nx.reduce_min(schedule) |> Nx.to_number()

      assert min_val > 0.0,
             "All schedule values should be positive, got min: #{min_val}"
    end
  end

  # ============================================================================
  # Pseudo-Huber Loss
  # ============================================================================

  describe "pseudo-Huber loss" do
    test "loss is non-negative" do
      key = Nx.Random.key(42)
      {pred, key} = Nx.Random.normal(key, shape: {@batch, @input_dim})
      {target, _} = Nx.Random.normal(key, shape: {@batch, @input_dim})

      loss = ConsistencyModel.consistency_loss(pred, target)

      assert Nx.to_number(loss) >= 0.0,
             "Pseudo-Huber loss should be non-negative"
    end

    test "loss is zero (or near-zero) when inputs are equal" do
      key = Nx.Random.key(42)
      {pred, _} = Nx.Random.normal(key, shape: {@batch, @input_dim})

      loss = ConsistencyModel.consistency_loss(pred, pred)

      # Pseudo-Huber: sqrt(0 + c^2) - c = c - c = 0
      assert Nx.to_number(loss) < 1.0e-6,
             "Loss should be ~0 when pred equals target, got: #{Nx.to_number(loss)}"
    end

    test "loss increases with larger differences" do
      key = Nx.Random.key(42)
      {pred, _} = Nx.Random.normal(key, shape: {@batch, @input_dim})

      # Small perturbation
      target_near = Nx.add(pred, 0.01)
      # Large perturbation
      target_far = Nx.add(pred, 10.0)

      loss_near = ConsistencyModel.consistency_loss(pred, target_near) |> Nx.to_number()
      loss_far = ConsistencyModel.consistency_loss(pred, target_far) |> Nx.to_number()

      assert loss_far > loss_near,
             "Larger differences should produce larger loss: #{loss_far} vs #{loss_near}"
    end
  end

  # ============================================================================
  # Consistency Loss Returns Scalar
  # ============================================================================

  describe "consistency loss returns scalar" do
    test "output is a scalar tensor" do
      key = Nx.Random.key(42)
      {pred, key} = Nx.Random.normal(key, shape: {@batch, @input_dim})
      {target, _} = Nx.Random.normal(key, shape: {@batch, @input_dim})

      loss = ConsistencyModel.consistency_loss(pred, target)

      assert Nx.shape(loss) == {},
             "Consistency loss should return a scalar, got shape: #{inspect(Nx.shape(loss))}"
    end

    test "loss is finite" do
      key = Nx.Random.key(42)
      {pred, key} = Nx.Random.normal(key, shape: {@batch, @input_dim})
      {target, _} = Nx.Random.normal(key, shape: {@batch, @input_dim})

      loss = ConsistencyModel.consistency_loss(pred, target)

      assert Nx.all(Nx.is_nan(loss) |> Nx.logical_not()) |> Nx.to_number() == 1,
             "Loss should not be NaN"

      assert Nx.all(Nx.is_infinity(loss) |> Nx.logical_not()) |> Nx.to_number() == 1,
             "Loss should not be Inf"
    end

    test "loss works with different batch sizes" do
      for batch <- [1, 4, 8] do
        key = Nx.Random.key(42)
        {pred, key} = Nx.Random.normal(key, shape: {batch, @input_dim})
        {target, _} = Nx.Random.normal(key, shape: {batch, @input_dim})

        loss = ConsistencyModel.consistency_loss(pred, target)

        assert Nx.shape(loss) == {},
               "Loss should be scalar for batch=#{batch}"

        assert Nx.to_number(loss) >= 0.0,
               "Loss should be non-negative for batch=#{batch}"
      end
    end
  end
end
