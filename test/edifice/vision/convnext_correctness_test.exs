defmodule Edifice.Vision.ConvNeXtCorrectnessTest do
  use ExUnit.Case, async: true

  alias Edifice.Vision.ConvNeXt

  @batch 2
  @channels 3

  @convnext_opts [
    image_size: 32,
    patch_size: 4,
    in_channels: @channels,
    depths: [2, 2],
    dims: [16, 32]
  ]

  # ============================================================================
  # Depthwise Convolution Structure
  # ============================================================================

  describe "depthwise convolution structure" do
    test "model has depthwise conv parameters with correct kernel shape" do
      model = ConvNeXt.build(@convnext_opts)
      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should have depthwise conv layers
      dw_conv_keys = Enum.filter(param_keys, &String.contains?(&1, "dw_conv"))
      assert length(dw_conv_keys) > 0

      # Depthwise conv kernels should be 7x7
      # In Axon, conv kernels are named "kernel" within the layer
      for key <- dw_conv_keys do
        layer_params = params.data[key]

        if Map.has_key?(layer_params, "kernel") do
          kernel = layer_params["kernel"]
          shape = Nx.shape(kernel)
          # Axon conv kernel shape: {height, width, in_channels/groups, out_channels}
          assert elem(shape, 0) == 7, "kernel height should be 7, got #{elem(shape, 0)}"
          assert elem(shape, 1) == 7, "kernel width should be 7, got #{elem(shape, 1)}"
        end
      end
    end

    test "model has pointwise (1x1) conv parameters" do
      model = ConvNeXt.build(@convnext_opts)
      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should have pointwise expand and project convolutions
      pw_expand_keys = Enum.filter(param_keys, &String.contains?(&1, "pw_expand"))
      pw_project_keys = Enum.filter(param_keys, &String.contains?(&1, "pw_project"))

      assert length(pw_expand_keys) > 0
      assert length(pw_project_keys) > 0

      # Pointwise kernels should be 1x1
      for key <- pw_expand_keys do
        layer_params = params.data[key]

        if Map.has_key?(layer_params, "kernel") do
          kernel = layer_params["kernel"]
          shape = Nx.shape(kernel)
          assert elem(shape, 0) == 1, "pointwise kernel height should be 1"
          assert elem(shape, 1) == 1, "pointwise kernel width should be 1"
        end
      end
    end
  end

  # ============================================================================
  # Stem (Strided Conv)
  # ============================================================================

  describe "stem" do
    test "stem uses strided convolution (not patch embedding)" do
      model = ConvNeXt.build(@convnext_opts)
      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should have stem_conv (strided conv), not patch_embed
      assert Enum.any?(param_keys, &String.contains?(&1, "stem_conv"))
      refute Enum.any?(param_keys, &String.contains?(&1, "patch_embed"))

      # Stem conv kernel should be patch_size x patch_size (4x4)
      stem_params = params.data["stem_conv"]
      kernel = stem_params["kernel"]
      shape = Nx.shape(kernel)
      assert elem(shape, 0) == 4, "stem kernel height should equal patch_size"
      assert elem(shape, 1) == 4, "stem kernel width should equal patch_size"
    end
  end

  # ============================================================================
  # Downsampling
  # ============================================================================

  describe "downsampling" do
    test "downsample uses 2x2 strided conv" do
      model = ConvNeXt.build(@convnext_opts)
      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())

      param_keys = Map.keys(params.data)

      # Should have downsample convolution
      ds_keys = Enum.filter(param_keys, &String.contains?(&1, "downsample") and String.contains?(&1, "conv"))
      assert length(ds_keys) > 0

      # Downsample conv kernel should be 2x2
      for key <- ds_keys do
        layer_params = params.data[key]

        if Map.has_key?(layer_params, "kernel") do
          kernel = layer_params["kernel"]
          shape = Nx.shape(kernel)
          assert elem(shape, 0) == 2, "downsample kernel height should be 2"
          assert elem(shape, 1) == 2, "downsample kernel width should be 2"
        end
      end
    end
  end

  # ============================================================================
  # Spatial Behavior
  # ============================================================================

  describe "spatial behavior" do
    test "different spatial regions produce different outputs" do
      model = ConvNeXt.build(@convnext_opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())

      # Image with activity in top-left only
      image1 = Nx.broadcast(0.0, {@batch, @channels, 32, 32})

      image1 =
        Nx.indexed_put(
          image1,
          Nx.tensor([[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 3, 3],
                     [1, 0, 0, 0], [1, 0, 1, 1], [1, 0, 2, 2], [1, 0, 3, 3]]),
          Nx.broadcast(1.0, {8})
        )

      # Image with activity in bottom-right only
      image2 = Nx.broadcast(0.0, {@batch, @channels, 32, 32})

      image2 =
        Nx.indexed_put(
          image2,
          Nx.tensor([[0, 0, 28, 28], [0, 0, 29, 29], [0, 0, 30, 30], [0, 0, 31, 31],
                     [1, 0, 28, 28], [1, 0, 29, 29], [1, 0, 30, 30], [1, 0, 31, 31]]),
          Nx.broadcast(1.0, {8})
        )

      output1 = predict_fn.(params, image1)
      output2 = predict_fn.(params, image2)

      # Different spatial inputs should produce different outputs
      diff = Nx.subtract(output1, output2) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff > 1.0e-6
    end

    test "inverted bottleneck expands then contracts channels" do
      # With dims: [16, 32], stage 0 blocks should have:
      # - Depthwise conv: 16 -> 16 (same channels)
      # - Pointwise expand: 16 -> 64 (4x expansion)
      # - Pointwise project: 64 -> 16 (back to original)
      model = ConvNeXt.build(@convnext_opts)
      {init_fn, _predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())

      # Stage 0 block 0 pointwise expand should output 64 channels (16 * 4)
      expand_params = params.data["stage0_block0_pw_expand"]
      expand_kernel = expand_params["kernel"]
      # kernel shape: {1, 1, in_channels, out_channels}
      out_channels = elem(Nx.shape(expand_kernel), 3)
      assert out_channels == 64, "expansion should be 4x: 16 -> 64, got #{out_channels}"

      # Stage 0 block 0 pointwise project should output 16 channels
      project_params = params.data["stage0_block0_pw_project"]
      project_kernel = project_params["kernel"]
      out_channels = elem(Nx.shape(project_kernel), 3)
      assert out_channels == 16, "projection should go back to dim: 64 -> 16, got #{out_channels}"
    end
  end
end
