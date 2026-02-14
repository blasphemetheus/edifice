defmodule Edifice.Neuromorphic.SNNCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.Neuromorphic.SNN

  @batch 2
  @input_size 16
  @output_size 4
  @hidden_sizes [8]
  @num_timesteps 10

  @base_opts [
    input_size: @input_size,
    hidden_sizes: @hidden_sizes,
    output_size: @output_size,
    num_timesteps: @num_timesteps,
    tau: 2.0,
    threshold: 1.0
  ]

  # ============================================================================
  # Multi-Layer LIF Architecture
  # ============================================================================

  describe "multi-layer LIF architecture" do
    test "params contain per-layer dense + LIF layers" do
      model = SNN.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should have dense layer for each hidden size
      layer_0_keys = Enum.filter(param_keys, &String.contains?(&1, "snn_layer_0_dense"))

      assert layer_0_keys != [],
             "Should have dense layer for hidden layer 0, got: #{inspect(param_keys)}"

      # Should have output readout layer
      output_keys = Enum.filter(param_keys, &String.contains?(&1, "snn_output"))

      assert output_keys != [],
             "Should have output readout layer, got: #{inspect(param_keys)}"
    end

    test "multi-layer SNN has more params than single-layer" do
      single = SNN.build(Keyword.put(@base_opts, :hidden_sizes, [8]))
      multi = SNN.build(Keyword.put(@base_opts, :hidden_sizes, [8, 8]))

      {init_s, _} = Axon.build(single, mode: :inference)
      {init_m, _} = Axon.build(multi, mode: :inference)

      params_s = init_s.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())
      params_m = init_m.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      keys_s = Map.keys(params_s.data) |> length()
      keys_m = Map.keys(params_m.data) |> length()

      assert keys_m > keys_s,
             "Multi-layer should have more param groups: #{keys_m} vs #{keys_s}"
    end
  end

  # ============================================================================
  # Output Properties
  # ============================================================================

  describe "output properties" do
    test "output shape is [batch, output_size]" do
      model = SNN.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @output_size}
    end

    test "output is finite" do
      model = SNN.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output is deterministic" do
      model = SNN.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})

      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "different inputs produce different outputs" do
      model = SNN.build(@base_opts)
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
