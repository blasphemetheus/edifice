defmodule Edifice.Convolutional.ResNetCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.Convolutional.ResNet

  @batch 2
  @input_shape {nil, 32, 32, 3}
  @initial_channels 16
  @num_classes 10

  @resnet18_opts [
    input_shape: @input_shape,
    num_classes: @num_classes,
    block_sizes: [2, 2, 2, 2],
    block_type: :residual,
    initial_channels: @initial_channels
  ]

  # ============================================================================
  # Identity Skip Connections (Residual Blocks)
  # ============================================================================

  describe "identity skip connections (residual)" do
    test "second block in stage 0 uses identity skip (no skip_proj)" do
      # In stage 0, all blocks have the same number of channels (initial_channels).
      # The second block (block1) should reuse channels from block0's output,
      # so it should NOT need a 1x1 projection on the skip connection.
      model = ResNet.build(@resnet18_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, 32, 32, 3}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # stage0_block1 should use identity shortcut (channels match, stride 1)
      stage0_block1_skip =
        Enum.filter(param_keys, &String.contains?(&1, "stage0_block1_skip_proj"))

      assert stage0_block1_skip == [],
        "stage0_block1 should use identity skip (no projection), but found: #{inspect(stage0_block1_skip)}"
    end

    test "first block in stage 1 uses projection skip (stride 2 + channel change)" do
      # Stage 1 doubles channels and uses stride 2 for downsampling.
      # The first block (block0) must project the skip connection.
      model = ResNet.build(@resnet18_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, 32, 32, 3}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      stage1_block0_skip =
        Enum.filter(param_keys, &String.contains?(&1, "stage1_block0_skip_proj"))

      assert length(stage1_block0_skip) > 0,
        "stage1_block0 should have skip_proj for stride 2 + channel change"
    end

    test "output shape is [batch, num_classes]" do
      model = ResNet.build(@resnet18_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, 32, 32, 3}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 32, 32, 3})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @num_classes}
    end

    test "identity shortcuts reduce total parameter count" do
      # A model with identity shortcuts should have fewer parameters than one
      # that projects every skip connection. We verify this indirectly:
      # - stage0: both blocks use identity (stem outputs initial_channels, stage 0 uses initial_channels)
      # - stage1-3: block0 needs projection (channel doubling + stride 2), block1 uses identity
      model = ResNet.build(@resnet18_opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, 32, 32, 3}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Count skip projection layers across all stages
      skip_proj_keys = Enum.filter(param_keys, &String.contains?(&1, "skip_proj"))

      # With 4 stages of [2,2,2,2] blocks and initial_channels matching stem output:
      # - stage0: block0 identity (stem -> initial_channels matches), block1 identity
      # - stage1: block0 needs projection (channel doubling + stride 2), block1 identity
      # - stage2: block0 needs projection, block1 identity
      # - stage3: block0 needs projection, block1 identity
      # So we expect exactly 3 skip_proj layers (one per stage 1-3)
      assert length(skip_proj_keys) == 3,
        "Expected 3 skip_proj param groups (stages 1-3 only), got #{length(skip_proj_keys)}: #{inspect(skip_proj_keys)}"

      # Verify no second-block projections exist in any stage
      second_block_skip_keys = Enum.filter(param_keys, fn key ->
        String.contains?(key, "block1_skip_proj")
      end)

      assert second_block_skip_keys == [],
        "Second blocks should all use identity skip, but found: #{inspect(second_block_skip_keys)}"
    end

    test "output is finite (no NaN/Inf)" do
      model = ResNet.build(@resnet18_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, 32, 32, 3}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 32, 32, 3})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Identity Skip Connections (Bottleneck Blocks)
  # ============================================================================

  describe "identity skip connections (bottleneck)" do
    test "second bottleneck in stage 0 uses identity skip (no skip_proj)" do
      # For bottleneck blocks, out_channels = channels * 4 (expansion factor).
      # In stage 0: channels = initial_channels, out_channels = initial_channels * 4.
      # Block 0 projects from stem output to initial_channels*4.
      # Block 1 has in_channels = initial_channels*4, out_channels = initial_channels*4,
      # stride = 1, so it should use identity shortcut.
      model =
        ResNet.build(
          input_shape: @input_shape,
          num_classes: @num_classes,
          block_sizes: [2, 2, 2, 2],
          block_type: :bottleneck,
          initial_channels: @initial_channels
        )

      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, 32, 32, 3}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      stage0_block1_skip =
        Enum.filter(param_keys, &String.contains?(&1, "stage0_block1_skip_proj"))

      assert stage0_block1_skip == [],
        "Bottleneck stage0_block1 should use identity skip, but found: #{inspect(stage0_block1_skip)}"
    end

    test "bottleneck output shape is [batch, num_classes]" do
      model =
        ResNet.build(
          input_shape: @input_shape,
          num_classes: @num_classes,
          block_sizes: [2, 2, 2, 2],
          block_type: :bottleneck,
          initial_channels: @initial_channels
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(Nx.template({@batch, 32, 32, 3}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, 32, 32, 3})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @num_classes}
    end
  end
end
