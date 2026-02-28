defmodule Edifice.Vision.EfficientViTTest do
  use ExUnit.Case, async: true
  @moduletag :vision

  alias Edifice.Vision.EfficientViT

  @batch 2
  @channels 3
  @image_size 32
  @patch_size 8
  @embed_dim 32

  @base_opts [
    image_size: @image_size,
    patch_size: @patch_size,
    in_channels: @channels,
    embed_dim: @embed_dim,
    depths: [1, 2],
    num_heads: [4, 4]
  ]

  defp random_image do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @channels, @image_size, @image_size})
    input
  end

  defp build_and_predict(opts) do
    model = EfficientViT.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    params =
      init_fn.(
        Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
        Axon.ModelState.empty()
      )

    predict_fn.(params, random_image())
  end

  describe "build/1" do
    test "returns an Axon model" do
      model = EfficientViT.build(@base_opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      output = build_and_predict(@base_opts)
      # 2 stages: dim doubles each stage: 32 -> 64
      assert Nx.shape(output) == {@batch, 64}
    end

    test "works with num_classes" do
      opts = Keyword.put(@base_opts, :num_classes, 10)
      output = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, 10}
    end

    test "output contains finite values" do
      output = build_and_predict(@base_opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with single stage" do
      opts = [
        image_size: @image_size,
        patch_size: @patch_size,
        in_channels: @channels,
        embed_dim: @embed_dim,
        depths: [2],
        num_heads: [4]
      ]

      output = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, @embed_dim}
    end

    test "works with three stages" do
      opts = [
        image_size: @image_size,
        patch_size: @patch_size,
        in_channels: @channels,
        embed_dim: 16,
        depths: [1, 1, 1],
        num_heads: [2, 2, 2]
      ]

      output = build_and_predict(opts)
      # 3 stages: 16 -> 32 -> 64
      assert Nx.shape(output) == {@batch, 64}
    end
  end

  describe "output_size/1" do
    test "returns last stage dim by default" do
      # 2 stages starting at 32: 32, 64 -> last = 64
      assert EfficientViT.output_size(@base_opts) == 64
    end

    test "returns num_classes when set" do
      assert EfficientViT.output_size(Keyword.put(@base_opts, :num_classes, 10)) == 10
    end
  end
end
