defmodule Edifice.Vision.PoolFormerTest do
  use ExUnit.Case, async: true
  @moduletag :vision

  alias Edifice.Vision.PoolFormer

  @batch 2
  @channels 3
  @image_size 32
  @patch_size 8
  @hidden_size 32

  @base_opts [
    image_size: @image_size,
    patch_size: @patch_size,
    in_channels: @channels,
    hidden_size: @hidden_size,
    num_layers: 2,
    pool_size: 3
  ]

  defp random_image do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @channels, @image_size, @image_size})
    input
  end

  defp build_and_predict(opts) do
    model = PoolFormer.build(opts)
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
      model = PoolFormer.build(@base_opts)
      assert %Axon{} = model
    end

    test "produces correct output shape without num_classes" do
      output = build_and_predict(@base_opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "produces correct output shape with num_classes" do
      output = build_and_predict(Keyword.put(@base_opts, :num_classes, 10))
      assert Nx.shape(output) == {@batch, 10}
    end

    test "output contains finite values" do
      output = build_and_predict(@base_opts)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with different pool sizes" do
      output = build_and_predict(Keyword.put(@base_opts, :pool_size, 5))
      assert Nx.shape(output) == {@batch, @hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size by default" do
      assert PoolFormer.output_size(@base_opts) == @hidden_size
    end

    test "returns num_classes when set" do
      assert PoolFormer.output_size(Keyword.put(@base_opts, :num_classes, 10)) == 10
    end
  end
end
