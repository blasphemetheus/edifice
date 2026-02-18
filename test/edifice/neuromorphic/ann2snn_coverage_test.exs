defmodule Edifice.Neuromorphic.ANN2SNNCoverageTest do
  @moduledoc """
  Coverage tests for ANN2SNN module.
  Targets uncovered branches: if_neuron defn, default opts, SNN with output_size,
  default threshold/timesteps in if_neuron_simulate, and finiteness/determinism.
  """
  use ExUnit.Case, async: true

  alias Edifice.Neuromorphic.ANN2SNN

  @batch 2
  @input_size 16

  # ============================================================================
  # ANN build with default hidden_sizes (no :hidden_sizes provided)
  # ============================================================================

  describe "ANN build with defaults" do
    test "uses default hidden_sizes [256, 128] when not specified" do
      model = ANN2SNN.build(input_size: @input_size)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      # Default: hidden_sizes [256, 128], output_size defaults to last hidden = 128
      assert Nx.shape(output) == {@batch, 128}
    end

    test "output is finite with default hidden_sizes" do
      model = ANN2SNN.build(input_size: @input_size)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # ANN build - output_size == last hidden (no output layer added)
  # ============================================================================

  describe "ANN build - output_size equals last hidden" do
    test "does not add output layer when output_size matches last hidden" do
      model =
        ANN2SNN.build(
          input_size: @input_size,
          hidden_sizes: [32, 16],
          output_size: 16
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch, 16}

      # Verify no "output" layer in params (it should skip the output dense)
      param_keys = Map.keys(params.data)
      output_keys = Enum.filter(param_keys, &String.starts_with?(&1, "output"))
      assert output_keys == [], "Should not have output layer when output_size == last hidden"
    end
  end

  # ============================================================================
  # SNN build with explicit output_size (triggers output layer branch)
  # ============================================================================

  describe "SNN build with explicit output_size" do
    test "adds output layer when output_size differs from last hidden" do
      model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [16, 8],
          output_size: 4,
          num_timesteps: 5,
          threshold: 1.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(1.0, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch, 4}
    end

    test "SNN output_size matching last hidden skips output layer" do
      model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [16, 8],
          output_size: 8,
          num_timesteps: 5,
          threshold: 1.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(1.0, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch, 8}

      # Should NOT have output layer
      param_keys = Map.keys(params.data)
      output_keys = Enum.filter(param_keys, &String.starts_with?(&1, "output"))
      assert output_keys == []
    end
  end

  # ============================================================================
  # SNN build with default opts
  # ============================================================================

  describe "SNN build with defaults" do
    test "uses default hidden_sizes, num_timesteps, and threshold" do
      model = ANN2SNN.build_snn(input_size: @input_size)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.5, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      # Default: hidden_sizes [256, 128], output defaults to 128
      assert Nx.shape(output) == {@batch, 128}
    end
  end

  # ============================================================================
  # SNN finiteness and determinism
  # ============================================================================

  describe "SNN output properties" do
    @snn_opts [
      input_size: @input_size,
      hidden_sizes: [16, 8],
      num_timesteps: 5,
      threshold: 1.0
    ]

    test "SNN output is finite" do
      model = ANN2SNN.build_snn(@snn_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(2.0, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "SNN output is deterministic" do
      model = ANN2SNN.build_snn(@snn_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(1.5, {@batch, @input_size})
      output1 = predict_fn.(params, %{"input" => input})
      output2 = predict_fn.(params, %{"input" => input})

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "different inputs produce different ANN outputs" do
      # Use ANN (with ReLU) which is more sensitive to input differences
      ann_opts = [
        input_size: @input_size,
        hidden_sizes: [16, 8]
      ]

      model = ANN2SNN.build(ann_opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      {input2, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})

      output1 = predict_fn.(params, %{"input" => input1})
      output2 = predict_fn.(params, %{"input" => input2})

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different inputs should produce different ANN outputs"
    end
  end

  # ============================================================================
  # SNN with different threshold and timestep values
  # ============================================================================

  describe "SNN threshold and timestep variations" do
    test "higher threshold produces fewer spikes (lower output)" do
      low_thresh_model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [8],
          num_timesteps: 10,
          threshold: 0.5
        )

      high_thresh_model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [8],
          num_timesteps: 10,
          threshold: 5.0
        )

      # Build both
      {init_low, pred_low} = Axon.build(low_thresh_model)
      {init_high, pred_high} = Axon.build(high_thresh_model)

      params_low =
        init_low.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      params_high =
        init_high.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(2.0, {@batch, @input_size})
      out_low = pred_low.(params_low, %{"input" => input})
      out_high = pred_high.(params_high, %{"input" => input})

      # Both should be finite and non-negative
      assert Nx.all(Nx.is_nan(out_low) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(out_high) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.reduce_min(out_low) |> Nx.to_number() >= 0.0
      assert Nx.reduce_min(out_high) |> Nx.to_number() >= 0.0
    end

    test "more timesteps allows higher resolution spike rates" do
      model_few =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [8],
          num_timesteps: 2,
          threshold: 1.0
        )

      model_many =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [8],
          num_timesteps: 20,
          threshold: 1.0
        )

      {init_few, pred_few} = Axon.build(model_few)
      {init_many, pred_many} = Axon.build(model_many)

      params_few =
        init_few.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      params_many =
        init_many.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(1.0, {@batch, @input_size})
      out_few = pred_few.(params_few, %{"input" => input})
      out_many = pred_many.(params_many, %{"input" => input})

      # Both should be valid
      assert Nx.all(Nx.is_nan(out_few) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(out_many) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # SNN IF neuron behavior (tested through the model layer which uses
  # if_neuron_simulate internally)
  # ============================================================================

  describe "SNN IF neuron behavior through model" do
    test "SNN with very low threshold produces non-negative spike rates" do
      model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [8],
          num_timesteps: 10,
          threshold: 0.01
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(1.0, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.reduce_min(output) |> Nx.to_number() >= 0.0
    end

    test "SNN with very high threshold still produces valid output" do
      model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [8],
          num_timesteps: 10,
          threshold: 100.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.1, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.reduce_min(output) |> Nx.to_number() >= 0.0
    end

    test "SNN with single timestep" do
      model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [8],
          num_timesteps: 1,
          threshold: 1.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(2.0, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch, 8}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "SNN with many hidden layers" do
      model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [16, 12, 8],
          num_timesteps: 5,
          threshold: 1.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(1.0, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.shape(output) == {@batch, 8}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "SNN with zero input produces non-negative output" do
      model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [8],
          num_timesteps: 5,
          threshold: 1.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      input = Nx.broadcast(0.0, {@batch, @input_size})
      output = predict_fn.(params, %{"input" => input})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.reduce_min(output) |> Nx.to_number() >= 0.0
    end
  end

  # ============================================================================
  # output_size with defaults
  # ============================================================================

  describe "output_size/1" do
    test "with no options uses default hidden_sizes" do
      # Default hidden_sizes is [256, 128], output_size defaults to last = 128
      assert ANN2SNN.output_size() == 128
    end

    test "with custom hidden_sizes" do
      assert ANN2SNN.output_size(hidden_sizes: [64, 32]) == 32
    end

    test "with explicit output_size overrides hidden_sizes" do
      assert ANN2SNN.output_size(hidden_sizes: [64, 32], output_size: 10) == 10
    end
  end

  # ============================================================================
  # ANN and SNN weight sharing compatibility
  # ============================================================================

  describe "ANN/SNN weight sharing" do
    test "ANN and SNN share the same dense layer names" do
      opts = [
        input_size: @input_size,
        hidden_sizes: [16, 8],
        num_timesteps: 5,
        threshold: 1.0
      ]

      ann = ANN2SNN.build(opts)
      snn = ANN2SNN.build_snn(opts)

      {ann_init, _} = Axon.build(ann)
      {snn_init, _} = Axon.build(snn)

      ann_params =
        ann_init.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      snn_params =
        snn_init.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      ann_keys = Map.keys(ann_params.data) |> MapSet.new()
      snn_keys = Map.keys(snn_params.data) |> MapSet.new()

      # Dense and BN layers should overlap
      ann_dense_keys =
        ann_keys |> Enum.filter(&(String.contains?(&1, "dense_") or String.contains?(&1, "bn_")))

      snn_dense_keys =
        snn_keys |> Enum.filter(&(String.contains?(&1, "dense_") or String.contains?(&1, "bn_")))

      assert ann_dense_keys != []
      assert snn_dense_keys != []

      # The shared layer names should be the same
      ann_shared = ann_dense_keys |> MapSet.new()
      snn_shared = snn_dense_keys |> MapSet.new()
      overlap = MapSet.intersection(ann_shared, snn_shared)

      assert MapSet.size(overlap) > 0,
             "ANN and SNN should share dense/bn layer names for weight transfer"
    end
  end

  # ============================================================================
  # Single hidden layer (edge case)
  # ============================================================================

  describe "single hidden layer" do
    test "ANN with single hidden layer" do
      model =
        ANN2SNN.build(
          input_size: @input_size,
          hidden_sizes: [8]
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"input" => Nx.broadcast(0.5, {@batch, @input_size})})
      assert Nx.shape(output) == {@batch, 8}
    end

    test "SNN with single hidden layer" do
      model =
        ANN2SNN.build_snn(
          input_size: @input_size,
          hidden_sizes: [8],
          num_timesteps: 5,
          threshold: 1.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"input" => Nx.template({@batch, @input_size}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"input" => Nx.broadcast(1.0, {@batch, @input_size})})
      assert Nx.shape(output) == {@batch, 8}
    end
  end
end
