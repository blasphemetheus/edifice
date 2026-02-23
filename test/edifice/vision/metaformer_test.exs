defmodule Edifice.Vision.MetaFormerTest do
  use ExUnit.Case, async: true

  alias Edifice.Vision.MetaFormer

  @batch 2
  @channels 3
  @image_size 32
  @patch_size 8

  @base_opts [
    image_size: @image_size,
    patch_size: @patch_size,
    in_channels: @channels,
    depths: [2, 2],
    dims: [32, 64]
  ]

  defp random_image do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @channels, @image_size, @image_size})
    input
  end

  defp build_and_predict_metaformer(opts) do
    model = MetaFormer.build_metaformer(opts)
    {init_fn, predict_fn} = Axon.build(model)

    params =
      init_fn.(
        Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
        Axon.ModelState.empty()
      )

    predict_fn.(params, random_image())
  end

  defp build_and_predict_caformer(opts) do
    model = MetaFormer.build_caformer(opts)
    {init_fn, predict_fn} = Axon.build(model)

    params =
      init_fn.(
        Nx.template({@batch, @channels, @image_size, @image_size}, :f32),
        Axon.ModelState.empty()
      )

    predict_fn.(params, random_image())
  end

  describe "build_metaformer/1" do
    test "returns an Axon model" do
      model = MetaFormer.build_metaformer(@base_opts)
      assert %Axon{} = model
    end

    test "produces correct output shape with pooling mixer" do
      opts = Keyword.put(@base_opts, :token_mixer, :pooling)
      output = build_and_predict_metaformer(opts)
      assert Nx.shape(output) == {@batch, 64}
    end

    test "produces correct output shape with conv mixer" do
      opts = Keyword.put(@base_opts, :token_mixer, :conv)
      output = build_and_predict_metaformer(opts)
      assert Nx.shape(output) == {@batch, 64}
    end

    test "produces correct output shape with attention mixer" do
      opts = Keyword.put(@base_opts, :token_mixer, :attention)
      output = build_and_predict_metaformer(opts)
      assert Nx.shape(output) == {@batch, 64}
    end

    test "works with num_classes" do
      opts = @base_opts |> Keyword.put(:token_mixer, :pooling) |> Keyword.put(:num_classes, 10)
      output = build_and_predict_metaformer(opts)
      assert Nx.shape(output) == {@batch, 10}
    end

    test "output contains finite values" do
      opts = Keyword.put(@base_opts, :token_mixer, :pooling)
      output = build_and_predict_metaformer(opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build_caformer/1" do
    test "returns an Axon model" do
      model = MetaFormer.build_caformer(@base_opts)
      assert %Axon{} = model
    end

    test "produces correct output shape" do
      output = build_and_predict_caformer(@base_opts)
      assert Nx.shape(output) == {@batch, 64}
    end

    test "works with num_classes" do
      opts = Keyword.put(@base_opts, :num_classes, 10)
      output = build_and_predict_caformer(opts)
      assert Nx.shape(output) == {@batch, 10}
    end

    test "output contains finite values" do
      output = build_and_predict_caformer(@base_opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "build/1" do
    test "aliases build_metaformer" do
      model = MetaFormer.build(@base_opts)
      assert %Axon{} = model
    end
  end

  describe "output_size/1" do
    test "returns last dim by default" do
      assert MetaFormer.output_size(@base_opts) == 64
    end

    test "returns num_classes when set" do
      assert MetaFormer.output_size(Keyword.put(@base_opts, :num_classes, 10)) == 10
    end
  end
end
