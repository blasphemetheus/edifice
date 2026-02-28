defmodule Edifice.Recurrent.ReservoirCorrectnessTest do
  use ExUnit.Case, async: true
  @moduletag :recurrent

  alias Edifice.Recurrent.Reservoir

  @batch 2
  @input_size 8
  @reservoir_size 32
  @output_size 4
  @seq_len 4

  @base_opts [
    input_size: @input_size,
    reservoir_size: @reservoir_size,
    output_size: @output_size,
    spectral_radius: 0.9,
    sparsity: 0.5,
    seq_len: @seq_len
  ]

  # ============================================================================
  # Fixed Reservoir Weights
  # ============================================================================

  describe "fixed reservoir weights" do
    test "reservoir weights are NOT in trainable params" do
      # ESN: only readout layer is trainable. Reservoir weights are constants.
      model = Reservoir.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @input_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should only have readout params, no reservoir weight params
      reservoir_param_keys =
        Enum.filter(param_keys, fn key ->
          String.contains?(key, "reservoir") and not String.contains?(key, "readout")
        end)

      # The reservoir layer should NOT have trainable params (weights are Axon.constant)
      assert reservoir_param_keys == [],
             "Reservoir weights should be constants, not trainable params: #{inspect(reservoir_param_keys)}"
    end

    test "readout layer IS trainable" do
      model = Reservoir.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @input_size}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)
      readout_keys = Enum.filter(param_keys, &String.contains?(&1, "readout"))

      assert readout_keys != [],
             "Should have trainable readout params, got: #{inspect(param_keys)}"
    end

    test "same model produces identical outputs with shared readout params (deterministic reservoir)" do
      # Two separately built models with same opts should have identical reservoir
      # constants. We share the readout params to isolate the reservoir behavior.
      model1 = Reservoir.build(@base_opts)
      model2 = Reservoir.build(@base_opts)

      {init_fn1, predict_fn1} = Axon.build(model1, mode: :inference)
      {_init_fn2, predict_fn2} = Axon.build(model2, mode: :inference)

      # Initialize params from model1 and reuse for model2
      # (readout params are random, but reservoir constants are deterministic)
      params =
        init_fn1.(Nx.template({@batch, @seq_len, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @input_size})

      output1 = predict_fn1.(params, input)
      output2 = predict_fn2.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert diff < 1.0e-5,
             "Same opts + same readout params should produce identical output, diff = #{diff}"
    end
  end

  # ============================================================================
  # Output Properties
  # ============================================================================

  describe "output properties" do
    test "output shape is [batch, output_size]" do
      model = Reservoir.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @output_size}
    end

    test "output is finite" do
      model = Reservoir.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @input_size})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "different inputs produce different outputs" do
      model = Reservoir.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      params =
        init_fn.(Nx.template({@batch, @seq_len, @input_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @input_size})
      {input2, _} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @input_size})

      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different inputs should produce different outputs"
    end
  end
end
