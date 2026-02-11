defmodule Edifice.Vision.UNetCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.Vision.UNet

  @batch 2

  # Small UNet config for fast testing
  @small_opts [
    in_channels: 1,
    out_channels: 1,
    image_size: 8,
    base_features: 4,
    depth: 2
  ]

  # ============================================================================
  # Spatial Dimension Preservation
  # ============================================================================

  describe "spatial dimension preservation" do
    test "output has same spatial dimensions as input" do
      model = UNet.build(@small_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 1, 8, 8}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 1, 8, 8})
      output = predict_fn.(params, input)

      # UNet preserves spatial dimensions exactly
      assert Nx.shape(output) == {@batch, 1, 8, 8}
    end

    test "works with multi-channel input and different output channels" do
      opts = Keyword.merge(@small_opts, in_channels: 3, out_channels: 2)
      model = UNet.build(opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 3, 8, 8}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 3, 8, 8})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, 2, 8, 8}
    end
  end

  # ============================================================================
  # Skip Connections
  # ============================================================================

  describe "skip connections" do
    test "different depths produce different outputs" do
      # Depth 1 vs depth 2 should give structurally different models
      model_d1 = UNet.build(Keyword.put(@small_opts, :depth, 1))
      model_d2 = UNet.build(Keyword.put(@small_opts, :depth, 2))

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 1, 8, 8})

      {init_fn1, predict_fn1} = Axon.build(model_d1)
      params1 = init_fn1.(Nx.template({@batch, 1, 8, 8}, :f32), Axon.ModelState.empty())
      output1 = predict_fn1.(params1, input)

      {init_fn2, predict_fn2} = Axon.build(model_d2)
      params2 = init_fn2.(Nx.template({@batch, 1, 8, 8}, :f32), Axon.ModelState.empty())
      output2 = predict_fn2.(params2, input)

      # Both should have correct shape
      assert Nx.shape(output1) == {@batch, 1, 8, 8}
      assert Nx.shape(output2) == {@batch, 1, 8, 8}

      # But different content (different architectures)
      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6
    end
  end

  # ============================================================================
  # Convolutional Structure
  # ============================================================================

  describe "convolutional structure" do
    test "model uses convolutional layers (params have kernel shapes)" do
      model = UNet.build(@small_opts)
      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 1, 8, 8}, :f32), Axon.ModelState.empty())

      # Extract parameter names from the ModelState struct
      param_keys = Map.keys(params.data)

      # Should have encoder conv layers
      assert Enum.any?(param_keys, &String.contains?(&1, "enc_0_conv1"))
      assert Enum.any?(param_keys, &String.contains?(&1, "enc_0_conv2"))

      # Should have decoder conv layers
      assert Enum.any?(param_keys, &String.contains?(&1, "dec_0"))

      # Should have bottleneck conv layers
      assert Enum.any?(param_keys, &String.contains?(&1, "bottleneck_conv"))

      # Should have output 1x1 conv
      assert Enum.any?(param_keys, &String.contains?(&1, "output_conv"))
    end

    test "conv kernels have spatial dimensions (not just dense weights)" do
      model = UNet.build(@small_opts)
      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 1, 8, 8}, :f32), Axon.ModelState.empty())

      # Find a 3x3 conv kernel — should be 4D: [height, width, in_channels, out_channels]
      kernel = params.data["enc_0_conv1"]["kernel"]

      # Conv kernel should be 4D (3x3 spatial kernel)
      assert tuple_size(Nx.shape(kernel)) == 4

      # First two dims should be 3x3 (our kernel_size)
      {kh, kw, _in_c, _out_c} = Nx.shape(kernel)
      assert kh == 3
      assert kw == 3
    end
  end

  # ============================================================================
  # Output Properties
  # ============================================================================

  describe "output properties" do
    test "output is finite (no NaN or Inf)" do
      model = UNet.build(@small_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 1, 8, 8}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 1, 8, 8})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output varies spatially (not constant across pixels)" do
      model = UNet.build(@small_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 1, 8, 8}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 1, 8, 8})
      output = predict_fn.(params, input)

      # Check that the output varies across spatial positions
      # (a constant-valued output would suggest the spatial structure is lost)
      first_batch = Nx.slice(output, [0, 0, 0, 0], [1, 1, 8, 8]) |> Nx.reshape({64})
      variance = Nx.variance(first_batch) |> Nx.to_number()
      assert variance > 1.0e-10
    end
  end

  # ============================================================================
  # Attention Gate (optional)
  # ============================================================================

  describe "attention gate" do
    test "model with attention produces valid output" do
      opts = Keyword.put(@small_opts, :use_attention, true)
      model = UNet.build(opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 1, 8, 8}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 1, 8, 8})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, 1, 8, 8}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
