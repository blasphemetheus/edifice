defmodule Edifice.Vision.VisionTest do
  use ExUnit.Case, async: true

  alias Edifice.Vision.{ConvNeXt, DeiT, MLPMixer, SwinTransformer, UNet, ViT}

  # Small dimensions for fast testing
  @batch 2
  @channels 3
  @image_size 32
  @patch_size 8
  @embed_dim 32

  defp random_image do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @channels, @image_size, @image_size})
    input
  end

  # ============================================================================
  # ViT Tests
  # ============================================================================

  describe "ViT" do
    @vit_opts [
      image_size: @image_size,
      patch_size: @patch_size,
      in_channels: @channels,
      embed_dim: @embed_dim,
      depth: 2,
      num_heads: 4
    ]

    test "build/1 returns an Axon model" do
      model = ViT.build(@vit_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = ViT.build(@vit_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_image())

      assert Nx.shape(output) == {@batch, @embed_dim}
    end

    test "forward pass with num_classes produces classification output" do
      opts = Keyword.put(@vit_opts, :num_classes, 10)
      model = ViT.build(opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_image())

      assert Nx.shape(output) == {@batch, 10}
    end

    test "output contains finite values" do
      model = ViT.build(@vit_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_image())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns embed_dim by default" do
      assert ViT.output_size(@vit_opts) == @embed_dim
    end

    test "output_size/1 returns num_classes when set" do
      assert ViT.output_size(Keyword.put(@vit_opts, :num_classes, 10)) == 10
    end
  end

  # ============================================================================
  # DeiT Tests
  # ============================================================================

  describe "DeiT" do
    @deit_opts [
      image_size: @image_size,
      patch_size: @patch_size,
      in_channels: @channels,
      embed_dim: @embed_dim,
      depth: 2,
      num_heads: 4
    ]

    test "build/1 returns an Axon model" do
      model = DeiT.build(@deit_opts)
      assert %Axon{} = model
    end

    test "forward pass without classification returns CLS token" do
      model = DeiT.build(@deit_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_image())

      assert Nx.shape(output) == {@batch, @embed_dim}
    end

    test "forward pass with num_classes produces classification output" do
      opts = Keyword.put(@deit_opts, :num_classes, 10)
      model = DeiT.build(opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_image())

      assert Nx.shape(output) == {@batch, 10}
    end

    test "forward pass with distillation returns container" do
      opts = @deit_opts |> Keyword.put(:num_classes, 10) |> Keyword.put(:teacher_num_classes, 10)
      model = DeiT.build(opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_image())

      assert %{cls: cls_out, dist: dist_out} = output
      assert Nx.shape(cls_out) == {@batch, 10}
      assert Nx.shape(dist_out) == {@batch, 10}
    end

    test "output contains finite values" do
      model = DeiT.build(@deit_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_image())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns embed_dim by default" do
      assert DeiT.output_size(@deit_opts) == @embed_dim
    end
  end

  # ============================================================================
  # SwinTransformer Tests
  # ============================================================================

  describe "SwinTransformer" do
    # Swin needs image_size divisible by patch_size, and enough patches for merging.
    # With image=32, patch=4 -> 64 patches (8x8 grid), 2 stages with merging -> 4x4 -> 2x2
    @swin_opts [
      image_size: 32,
      patch_size: 4,
      in_channels: @channels,
      embed_dim: 16,
      depths: [2, 2],
      num_heads: [2, 4],
      window_size: 4
    ]

    test "build/1 returns an Axon model" do
      model = SwinTransformer.build(@swin_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = SwinTransformer.build(@swin_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_image())

      # Output dim = embed_dim * 2^(num_stages - 1) = 16 * 2^1 = 32
      assert Nx.shape(output) == {@batch, 32}
    end

    test "forward pass with num_classes produces classification output" do
      opts = Keyword.put(@swin_opts, :num_classes, 10)
      model = SwinTransformer.build(opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_image())

      assert Nx.shape(output) == {@batch, 10}
    end

    test "output contains finite values" do
      model = SwinTransformer.build(@swin_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_image())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns correct dimension" do
      # embed_dim * 2^(num_stages - 1) = 16 * 2 = 32
      assert SwinTransformer.output_size(@swin_opts) == 32
    end
  end

  # ============================================================================
  # UNet Tests
  # ============================================================================

  describe "UNet" do
    # Use very small dimensions to keep test fast
    @unet_opts [
      in_channels: 1,
      out_channels: 1,
      image_size: 8,
      base_features: 8,
      depth: 2
    ]

    test "build/1 returns an Axon model" do
      model = UNet.build(@unet_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct spatial output shape" do
      model = UNet.build(@unet_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 1, 8, 8}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(99)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch, 1, 8, 8})
      output = predict_fn.(params, input)

      # UNet outputs [batch, out_channels, image_size, image_size]
      assert Nx.shape(output) == {@batch, 1, 8, 8}
    end

    test "output contains finite values" do
      model = UNet.build(@unet_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 1, 8, 8}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(99)
      {input, _key} = Nx.Random.uniform(key, shape: {@batch, 1, 8, 8})
      output = predict_fn.(params, input)

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns flattened spatial size" do
      # out_channels * image_size * image_size = 1 * 8 * 8 = 64
      assert UNet.output_size(@unet_opts) == 64
    end
  end

  # ============================================================================
  # ConvNeXt Tests
  # ============================================================================

  describe "ConvNeXt" do
    # Need image_size divisible by patch_size, and enough patches for downsampling.
    # image=32, patch=4 -> 64 patches (8x8 grid), 2 stages with downsampling -> 4x4
    @convnext_opts [
      image_size: 32,
      patch_size: 4,
      in_channels: @channels,
      depths: [2, 2],
      dims: [16, 32]
    ]

    test "build/1 returns an Axon model" do
      model = ConvNeXt.build(@convnext_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = ConvNeXt.build(@convnext_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_image())

      # Output is last dim = 32
      assert Nx.shape(output) == {@batch, 32}
    end

    test "forward pass with num_classes produces classification output" do
      opts = Keyword.put(@convnext_opts, :num_classes, 10)
      model = ConvNeXt.build(opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_image())

      assert Nx.shape(output) == {@batch, 10}
    end

    test "output contains finite values" do
      model = ConvNeXt.build(@convnext_opts)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, random_image())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns last dim" do
      assert ConvNeXt.output_size(@convnext_opts) == 32
    end

    test "output_size/1 returns num_classes when set" do
      assert ConvNeXt.output_size(Keyword.put(@convnext_opts, :num_classes, 10)) == 10
    end
  end

  # ============================================================================
  # MLPMixer Tests
  # ============================================================================

  describe "MLPMixer" do
    @mixer_opts [
      image_size: @image_size,
      patch_size: @patch_size,
      in_channels: @channels,
      hidden_dim: @embed_dim,
      num_layers: 2,
      token_mlp_dim: 16,
      channel_mlp_dim: 64
    ]

    test "build/1 returns an Axon model" do
      model = MLPMixer.build(@mixer_opts)
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      model = MLPMixer.build(@mixer_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_image())

      assert Nx.shape(output) == {@batch, @embed_dim}
    end

    test "forward pass with num_classes produces classification output" do
      opts = Keyword.put(@mixer_opts, :num_classes, 10)
      model = MLPMixer.build(opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_image())

      assert Nx.shape(output) == {@batch, 10}
    end

    test "output contains finite values" do
      model = MLPMixer.build(@mixer_opts)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, random_image())

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "output_size/1 returns hidden_dim by default" do
      assert MLPMixer.output_size(@mixer_opts) == @embed_dim
    end

    test "output_size/1 returns num_classes when set" do
      assert MLPMixer.output_size(Keyword.put(@mixer_opts, :num_classes, 10)) == 10
    end
  end
end
