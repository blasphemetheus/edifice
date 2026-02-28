defmodule Edifice.Vision.VisionKANTest do
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  alias Edifice.Vision.VisionKAN

  @image_size 16
  @in_channels 1
  @channels [8, 16]
  @depths [1, 1]
  @patch_size 4
  @batch 2

  defp build_opts(overrides \\ []) do
    Keyword.merge(
      [
        image_size: @image_size,
        in_channels: @in_channels,
        channels: @channels,
        depths: @depths,
        patch_size: @patch_size,
        num_rbf_centers: 3,
        kan_patch_size: 4,
        dw_kernel_size: 3,
        global_reduction: 2,
        dropout: 0.0,
        ffn_expansion: 2
      ],
      overrides
    )
  end

  defp build_and_run(opts \\ [], batch \\ @batch) do
    model = VisionKAN.build(build_opts(opts))
    {init_fn, predict_fn} = Axon.build(model)
    input = random_tensor({batch, @in_channels, @image_size, @image_size})
    params = init_fn.(input, %{})
    output = predict_fn.(params, input)
    {output, params}
  end

  describe "build/1" do
    test "builds a valid model" do
      model = VisionKAN.build(build_opts())
      assert %Axon{} = model
    end

    test "forward pass produces correct output shape" do
      {output, _params} = build_and_run()
      last_dim = List.last(@channels)
      assert Nx.shape(output) == {@batch, last_dim}
      assert_finite!(output)
    end

    test "batch=1 works" do
      {output, _params} = build_and_run([], 1)
      last_dim = List.last(@channels)
      assert Nx.shape(output) == {1, last_dim}
      assert_finite!(output)
    end

    test "with classification head" do
      num_classes = 10
      {output, _params} = build_and_run(num_classes: num_classes)
      assert Nx.shape(output) == {@batch, num_classes}
      assert_finite!(output)
    end

    test "3-channel input" do
      opts = [in_channels: 3, input_shape: {nil, 3, @image_size, @image_size}]
      model = VisionKAN.build(build_opts(opts))
      {init_fn, predict_fn} = Axon.build(model)
      input = random_tensor({@batch, 3, @image_size, @image_size})
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)
      last_dim = List.last(@channels)
      assert Nx.shape(output) == {@batch, last_dim}
      assert_finite!(output)
    end

    test "single stage" do
      {output, _params} = build_and_run(channels: [16], depths: [2])
      assert Nx.shape(output) == {@batch, 16}
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns last channel dim" do
      assert VisionKAN.output_size(channels: [32, 64, 128]) == 128
    end

    test "returns default" do
      assert VisionKAN.output_size() == 256
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = VisionKAN.recommended_defaults()
      assert Keyword.has_key?(defaults, :channels)
      assert Keyword.has_key?(defaults, :depths)
      assert Keyword.has_key?(defaults, :num_rbf_centers)
      assert Keyword.has_key?(defaults, :kan_patch_size)
    end
  end

  describe "backbone behaviour" do
    test "build_backbone excludes num_classes" do
      model = VisionKAN.build_backbone(build_opts(num_classes: 10))
      assert %Axon{} = model
    end

    test "feature_size returns last channel" do
      assert VisionKAN.feature_size(channels: [32, 64]) == 64
    end

    test "input_shape returns NCHW tuple" do
      shape = VisionKAN.input_shape(image_size: 32, in_channels: 3)
      assert shape == {nil, 3, 32, 32}
    end
  end

  describe "registry" do
    test "builds via Edifice.build/2" do
      model = Edifice.build(:vision_kan, build_opts())
      assert %Axon{} = model
    end
  end
end
