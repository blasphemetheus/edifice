defmodule Edifice.Generative.DiTCoverageTest do
  @moduledoc """
  Coverage tests for DiT module.
  Targets uncovered branches: class conditioning (num_classes), build_dit_block
  with various options, param_count, recommended_defaults, and default options.
  """
  use ExUnit.Case, async: true

  alias Edifice.Generative.DiT

  @batch 2
  @input_dim 16
  @hidden_size 16

  # ============================================================================
  # DiT with class conditioning (num_classes != nil)
  # ============================================================================

  describe "DiT with class conditioning" do
    @class_opts [
      input_dim: @input_dim,
      hidden_size: @hidden_size,
      depth: 2,
      num_heads: 2,
      num_classes: 5,
      num_steps: 50
    ]

    test "builds model that accepts class_label input" do
      model = DiT.build(@class_opts)
      assert %Axon{} = model
    end

    test "forward pass with class labels produces correct shape" do
      model = DiT.build(@class_opts)

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([10, 20]),
        "class_label" => Nx.tensor([0, 3])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @input_dim}
    end

    test "output is finite with class conditioning" do
      model = DiT.build(@class_opts)

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([10, 20]),
        "class_label" => Nx.tensor([1, 4])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "different class labels produce different outputs" do
      model = DiT.build(@class_opts)

      {init_fn, predict_fn} = Axon.build(model)

      input1 = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([10, 20]),
        "class_label" => Nx.tensor([0, 0])
      }

      input2 = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([10, 20]),
        "class_label" => Nx.tensor([4, 4])
      }

      params = init_fn.(input1, Axon.ModelState.empty())
      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different class labels should produce different outputs"
    end

    test "class conditioning with boundary class values" do
      model = DiT.build(@class_opts)

      {init_fn, predict_fn} = Axon.build(model)

      # Test with first and last class
      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([0, 49]),
        "class_label" => Nx.tensor([0, 4])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @input_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # DiT unconditional (no num_classes) - different configurations
  # ============================================================================

  describe "DiT unconditional variations" do
    test "single depth (depth=1)" do
      model =
        DiT.build(
          input_dim: @input_dim,
          hidden_size: @hidden_size,
          depth: 1,
          num_heads: 2,
          num_steps: 50
        )

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([10, 20])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @input_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "different mlp_ratio" do
      model =
        DiT.build(
          input_dim: @input_dim,
          hidden_size: @hidden_size,
          depth: 1,
          num_heads: 2,
          mlp_ratio: 2.0,
          num_steps: 50
        )

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([5, 15])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @input_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "different num_heads" do
      model =
        DiT.build(
          input_dim: @input_dim,
          hidden_size: @hidden_size,
          depth: 1,
          num_heads: 4,
          num_steps: 50
        )

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([25, 30])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @input_dim}
    end

    test "output is deterministic" do
      model =
        DiT.build(
          input_dim: @input_dim,
          hidden_size: @hidden_size,
          depth: 2,
          num_heads: 2,
          num_steps: 50
        )

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([10, 20])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output1 = predict_fn.(params, input)
      output2 = predict_fn.(params, input)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "different timesteps produce different outputs" do
      model =
        DiT.build(
          input_dim: @input_dim,
          hidden_size: @hidden_size,
          depth: 2,
          num_heads: 2,
          num_steps: 100
        )

      {init_fn, predict_fn} = Axon.build(model)

      input1 = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([5, 5])
      }

      input2 = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([90, 90])
      }

      params = init_fn.(input1, Axon.ModelState.empty())
      output1 = predict_fn.(params, input1)
      output2 = predict_fn.(params, input2)

      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6, "Different timesteps should produce different outputs"
    end
  end

  # ============================================================================
  # build_dit_block directly
  # ============================================================================

  describe "build_dit_block/3" do
    test "builds a single block with default name" do
      input = Axon.input("input", shape: {nil, @hidden_size})
      condition = Axon.input("condition", shape: {nil, @hidden_size})

      block =
        DiT.build_dit_block(input, condition,
          hidden_size: @hidden_size,
          num_heads: 2,
          mlp_ratio: 4.0
        )

      assert %Axon{} = block

      {init_fn, predict_fn} = Axon.build(block)

      inp = %{
        "input" => Nx.broadcast(0.5, {@batch, @hidden_size}),
        "condition" => Nx.broadcast(0.1, {@batch, @hidden_size})
      }

      params = init_fn.(inp, Axon.ModelState.empty())
      output = predict_fn.(params, inp)

      assert Nx.shape(output) == {@batch, @hidden_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "builds a block with custom name" do
      input = Axon.input("input", shape: {nil, @hidden_size})
      condition = Axon.input("condition", shape: {nil, @hidden_size})

      block =
        DiT.build_dit_block(input, condition,
          hidden_size: @hidden_size,
          num_heads: 2,
          mlp_ratio: 2.0,
          name: "custom_block"
        )

      assert %Axon{} = block

      {init_fn, predict_fn} = Axon.build(block)

      inp = %{
        "input" => Nx.broadcast(0.5, {@batch, @hidden_size}),
        "condition" => Nx.broadcast(0.1, {@batch, @hidden_size})
      }

      params = init_fn.(inp, Axon.ModelState.empty())
      output = predict_fn.(params, inp)

      assert Nx.shape(output) == {@batch, @hidden_size}

      # Verify custom name is used in params
      param_keys = Map.keys(params.data)
      custom_keys = Enum.filter(param_keys, &String.contains?(&1, "custom_block"))
      assert length(custom_keys) > 0, "Should have params with custom block name"
    end

    test "block uses default options when not all specified" do
      input = Axon.input("input", shape: {nil, @hidden_size})
      condition = Axon.input("condition", shape: {nil, @hidden_size})

      # Only specify hidden_size, rely on defaults for num_heads, mlp_ratio, name
      block = DiT.build_dit_block(input, condition, hidden_size: @hidden_size)

      assert %Axon{} = block
    end
  end

  # ============================================================================
  # Utility functions
  # ============================================================================

  describe "param_count/1" do
    test "returns positive integer" do
      count =
        DiT.param_count(
          input_dim: @input_dim,
          hidden_size: @hidden_size,
          depth: 2,
          mlp_ratio: 4.0
        )

      assert is_integer(count)
      assert count > 0
    end

    test "deeper model has more params" do
      count_shallow =
        DiT.param_count(
          input_dim: @input_dim,
          hidden_size: @hidden_size,
          depth: 2,
          mlp_ratio: 4.0
        )

      count_deep =
        DiT.param_count(
          input_dim: @input_dim,
          hidden_size: @hidden_size,
          depth: 6,
          mlp_ratio: 4.0
        )

      assert count_deep > count_shallow
    end

    test "wider model has more params" do
      count_narrow =
        DiT.param_count(
          input_dim: @input_dim,
          hidden_size: 16,
          depth: 2,
          mlp_ratio: 4.0
        )

      count_wide =
        DiT.param_count(
          input_dim: @input_dim,
          hidden_size: 64,
          depth: 2,
          mlp_ratio: 4.0
        )

      assert count_wide > count_narrow
    end

    test "uses defaults when options not provided" do
      count = DiT.param_count([])
      assert is_integer(count)
      assert count > 0
    end
  end

  describe "output_size/1" do
    test "returns default 64 when no options given" do
      assert DiT.output_size() == 64
    end

    test "returns specified input_dim" do
      assert DiT.output_size(input_dim: 128) == 128
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = DiT.recommended_defaults()

      assert is_list(defaults)
      assert Keyword.has_key?(defaults, :input_dim)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :depth)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :mlp_ratio)
      assert Keyword.has_key?(defaults, :num_steps)
    end

    test "defaults can be used to build a model" do
      defaults = DiT.recommended_defaults()
      model = DiT.build(defaults)
      assert %Axon{} = model
    end
  end

  # ============================================================================
  # Edge cases
  # ============================================================================

  describe "edge cases" do
    test "timestep 0 works" do
      model =
        DiT.build(
          input_dim: @input_dim,
          hidden_size: @hidden_size,
          depth: 1,
          num_heads: 2,
          num_steps: 50
        )

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([0, 0])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @input_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "large timestep value" do
      model =
        DiT.build(
          input_dim: @input_dim,
          hidden_size: @hidden_size,
          depth: 1,
          num_heads: 2,
          num_steps: 1000
        )

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {@batch, @input_dim}),
        "timestep" => Nx.tensor([999, 500])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @input_dim}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "batch size 1" do
      model =
        DiT.build(
          input_dim: @input_dim,
          hidden_size: @hidden_size,
          depth: 1,
          num_heads: 2,
          num_steps: 50
        )

      {init_fn, predict_fn} = Axon.build(model)

      input = %{
        "noisy_input" => Nx.broadcast(0.5, {1, @input_dim}),
        "timestep" => Nx.tensor([10])
      }

      params = init_fn.(input, Axon.ModelState.empty())
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, @input_dim}
    end
  end
end
