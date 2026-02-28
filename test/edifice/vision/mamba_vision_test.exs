defmodule Edifice.Vision.MambaVisionTest do
  use ExUnit.Case, async: true
  @moduletag :vision

  alias Edifice.Vision.MambaVision

  @batch 2
  @channels 3
  @image_size 32
  @dim 16
  @depths [1, 1, 2, 1]
  @num_heads [1, 2, 2, 4]

  @base_opts [
    image_size: @image_size,
    in_channels: @channels,
    dim: @dim,
    depths: @depths,
    num_heads: @num_heads,
    mlp_ratio: 2,
    dropout: 0.0,
    d_state: 4,
    d_conv: 3
  ]

  defp random_image do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @channels, @image_size, @image_size})
    input
  end

  defp build_and_predict(opts) do
    model = MambaVision.build(opts)
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
      model = MambaVision.build(@base_opts)
      assert %Axon{} = model
    end

    test "produces correct output shape without num_classes" do
      output = build_and_predict(@base_opts)
      # Output is 8 * dim
      assert Nx.shape(output) == {@batch, @dim * 8}
    end

    test "produces correct output shape with num_classes" do
      output = build_and_predict(Keyword.put(@base_opts, :num_classes, 10))
      assert Nx.shape(output) == {@batch, 10}
    end

    test "output contains finite values" do
      output = build_and_predict(@base_opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns 8*dim by default" do
      assert MambaVision.output_size(@base_opts) == @dim * 8
    end

    test "returns num_classes when set" do
      assert MambaVision.output_size(Keyword.put(@base_opts, :num_classes, 10)) == 10
    end
  end

  describe "config variants" do
    test "tiny_config returns expected structure" do
      config = MambaVision.tiny_config()
      assert Keyword.has_key?(config, :dim)
      assert Keyword.has_key?(config, :depths)
      assert Keyword.has_key?(config, :num_heads)
      assert config[:dim] == 80
    end

    test "small_config returns expected structure" do
      config = MambaVision.small_config()
      assert config[:dim] == 96
    end

    test "base_config returns expected structure" do
      config = MambaVision.base_config()
      assert config[:dim] == 128
    end
  end
end
