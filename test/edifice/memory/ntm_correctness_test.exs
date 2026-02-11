defmodule Edifice.Memory.NTMCorrectnessTest do
  @moduledoc """
  Correctness tests for NTM content+location addressing fix.
  Verifies model builds and runs, output shape is correct,
  and read/write heads function properly.
  """
  use ExUnit.Case, async: true

  alias Edifice.Memory.NTM

  @batch 2
  @input_size 16
  @memory_size 8
  @memory_dim 8
  @controller_size 16
  @output_size 8

  @base_opts [
    input_size: @input_size,
    memory_size: @memory_size,
    memory_dim: @memory_dim,
    controller_size: @controller_size,
    num_heads: 1,
    output_size: @output_size
  ]

  # ============================================================================
  # Model Builds and Runs
  # ============================================================================

  describe "model builds and runs" do
    test "builds an Axon model without error" do
      model = NTM.build(@base_opts)
      assert %Axon{} = model
    end

    @tag timeout: 120_000
    test "forward pass completes without error" do
      model = NTM.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "input" => Nx.template({@batch, @input_size}, :f32),
        "memory" => Nx.template({@batch, @memory_size, @memory_dim}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      {memory, _} = Nx.Random.uniform(key, shape: {@batch, @memory_size, @memory_dim})

      output = predict_fn.(params, %{"input" => input, "memory" => memory})

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Output Shape
  # ============================================================================

  describe "output shape" do
    @tag timeout: 120_000
    test "output shape is [batch, output_size]" do
      model = NTM.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "input" => Nx.template({@batch, @input_size}, :f32),
        "memory" => Nx.template({@batch, @memory_size, @memory_dim}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      {memory, _} = Nx.Random.uniform(key, shape: {@batch, @memory_size, @memory_dim})

      output = predict_fn.(params, %{"input" => input, "memory" => memory})

      assert Nx.shape(output) == {@batch, @output_size}
    end

    @tag timeout: 120_000
    test "output is deterministic" do
      model = NTM.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "input" => Nx.template({@batch, @input_size}, :f32),
        "memory" => Nx.template({@batch, @memory_size, @memory_dim}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      {memory, _} = Nx.Random.uniform(key, shape: {@batch, @memory_size, @memory_dim})

      input_map = %{"input" => input, "memory" => memory}

      output1 = predict_fn.(params, input_map)
      output2 = predict_fn.(params, input_map)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end

  # ============================================================================
  # Read and Write Heads Function
  # ============================================================================

  describe "read and write heads" do
    @tag timeout: 120_000
    test "different inputs produce different outputs" do
      model = NTM.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "input" => Nx.template({@batch, @input_size}, :f32),
        "memory" => Nx.template({@batch, @memory_size, @memory_dim}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input1, key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      {input2, key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      {memory, _} = Nx.Random.uniform(key, shape: {@batch, @memory_size, @memory_dim})

      output1 = predict_fn.(params, %{"input" => input1, "memory" => memory})
      output2 = predict_fn.(params, %{"input" => input2, "memory" => memory})

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different inputs should produce different outputs"
    end

    @tag timeout: 120_000
    test "different memory contents produce different outputs" do
      model = NTM.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "input" => Nx.template({@batch, @input_size}, :f32),
        "memory" => Nx.template({@batch, @memory_size, @memory_dim}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      {memory1, key} = Nx.Random.uniform(key, shape: {@batch, @memory_size, @memory_dim})
      {memory2, _} = Nx.Random.uniform(key, shape: {@batch, @memory_size, @memory_dim})

      output1 = predict_fn.(params, %{"input" => input, "memory" => memory1})
      output2 = predict_fn.(params, %{"input" => input, "memory" => memory2})

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different memory contents should produce different outputs"
    end

    @tag timeout: 120_000
    test "params contain addressing pipeline components" do
      model = NTM.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "input" => Nx.template({@batch, @input_size}, :f32),
        "memory" => Nx.template({@batch, @memory_size, @memory_dim}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())
      param_keys = Map.keys(params.data)

      # Verify addressing pipeline components exist for both read and write heads
      for head_type <- ["read_head", "write_head"] do
        # Key projection for content addressing
        key_params = Enum.filter(param_keys, &String.contains?(&1, "#{head_type}_key"))

        assert length(key_params) > 0,
               "Should have key projection for #{head_type}, got: #{inspect(param_keys)}"

        # Beta for content sharpness
        beta_params = Enum.filter(param_keys, &String.contains?(&1, "#{head_type}_beta"))

        assert length(beta_params) > 0,
               "Should have beta parameter for #{head_type}"

        # Shift kernel for location addressing
        shift_params = Enum.filter(param_keys, &String.contains?(&1, "#{head_type}_shift"))

        assert length(shift_params) > 0,
               "Should have shift kernel for #{head_type}"

        # Gamma for sharpening
        gamma_params = Enum.filter(param_keys, &String.contains?(&1, "#{head_type}_gamma"))

        assert length(gamma_params) > 0,
               "Should have gamma parameter for #{head_type}"
      end
    end

    @tag timeout: 120_000
    test "write head has erase and add vectors" do
      model = NTM.build(@base_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)

      template = %{
        "input" => Nx.template({@batch, @input_size}, :f32),
        "memory" => Nx.template({@batch, @memory_size, @memory_dim}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())
      param_keys = Map.keys(params.data)

      erase_keys = Enum.filter(param_keys, &String.contains?(&1, "write_head_erase"))

      assert length(erase_keys) > 0,
             "Write head should have erase vector params"

      add_keys = Enum.filter(param_keys, &String.contains?(&1, "write_head_add"))

      assert length(add_keys) > 0,
             "Write head should have add vector params"
    end
  end

  # ============================================================================
  # Content Addressing (Unit Test)
  # ============================================================================

  describe "content addressing" do
    test "produces valid probability distribution" do
      key = Nx.Random.key(42)
      {query_key, key} = Nx.Random.normal(key, shape: {@batch, @memory_dim})
      {memory, _} = Nx.Random.normal(key, shape: {@batch, @memory_size, @memory_dim})

      beta = Nx.broadcast(Nx.tensor(5.0), {@batch, 1})

      weights = NTM.content_addressing(query_key, memory, beta)

      # Shape should be [batch, memory_size]
      assert Nx.shape(weights) == {@batch, @memory_size}

      # Weights should be non-negative
      min_val = Nx.reduce_min(weights) |> Nx.to_number()
      assert min_val >= 0.0, "Addressing weights should be non-negative, got min: #{min_val}"

      # Weights should sum to approximately 1 (probability distribution)
      sums = Nx.sum(weights, axes: [1])

      sums_list = Nx.to_flat_list(sums)

      for s <- sums_list do
        assert_in_delta s, 1.0, 0.01, "Addressing weights should sum to ~1.0, got: #{s}"
      end
    end
  end
end
