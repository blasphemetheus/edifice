defmodule Edifice.Energy.NeuralODECorrectnessTest do
  use ExUnit.Case, async: true
  @moduletag :energy

  alias Edifice.Energy.NeuralODE

  @batch 2
  @input_size 32
  @hidden_size 32

  @base_opts [
    input_size: @input_size,
    hidden_size: @hidden_size,
    num_steps: 5,
    step_size: 0.1
  ]

  # ============================================================================
  # Shared Dynamics Network
  # ============================================================================

  describe "shared dynamics weights" do
    test "params contain shared dynamics_dense1 (not per-step names)" do
      model = NeuralODE.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should have shared dynamics layers
      dynamics_keys = Enum.filter(param_keys, &String.contains?(&1, "dynamics_dense1"))

      assert dynamics_keys != [],
             "Should have 'dynamics_dense1' param, got keys: #{inspect(param_keys)}"

      # Should NOT have per-step dynamics layers
      per_step_keys = Enum.filter(param_keys, &String.contains?(&1, "dynamics_dense1_step_0"))

      assert per_step_keys == [],
             "Should not have per-step dynamics names, but found: #{inspect(per_step_keys)}"
    end

    test "different num_steps produce correct output shape" do
      for num_steps <- [3, 10] do
        opts = Keyword.put(@base_opts, :num_steps, num_steps)
        model = NeuralODE.build(opts)
        {init_fn, predict_fn} = Axon.build(model, mode: :inference)
        params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

        key = Nx.Random.key(42)
        {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})
        output = predict_fn.(params, input)

        assert Nx.shape(output) == {@batch, @hidden_size},
               "num_steps=#{num_steps} should output {#{@batch}, #{@hidden_size}}, got #{inspect(Nx.shape(output))}"
      end
    end

    test "build_shared/1 produces identical param keys to build/1" do
      model_build = NeuralODE.build(@base_opts)
      model_shared = NeuralODE.build_shared(@base_opts)

      {init_build, _} = Axon.build(model_build, mode: :inference)
      {init_shared, _} = Axon.build(model_shared, mode: :inference)

      params_build =
        init_build.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      params_shared =
        init_shared.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      keys_build = Map.keys(params_build.data) |> Enum.sort()
      keys_shared = Map.keys(params_shared.data) |> Enum.sort()

      assert keys_build == keys_shared,
             "build/1 and build_shared/1 should produce identical param keys"
    end
  end

  # ============================================================================
  # Determinism and Stability
  # ============================================================================

  describe "determinism and stability" do
    test "output is deterministic (same params + same input = same output)" do
      model = NeuralODE.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "output is finite (no NaN/Inf)" do
      model = NeuralODE.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "different inputs produce different outputs" do
      model = NeuralODE.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      {input2, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})

      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different inputs should produce different outputs"
    end
  end
end
