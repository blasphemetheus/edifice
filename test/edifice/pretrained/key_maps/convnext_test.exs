defmodule Edifice.Pretrained.KeyMaps.ConvNeXtTest do
  use ExUnit.Case, async: true

  alias Edifice.Pretrained.KeyMaps.ConvNeXt

  setup do
    assert Code.ensure_loaded?(Safetensors),
           "safetensors package must be available for these tests"

    :ok
  end

  defp write_fixture(tensors) do
    path =
      Path.join(
        System.tmp_dir!(),
        "edifice_convnext_test_#{System.unique_integer([:positive])}.safetensors"
      )

    Safetensors.write!(path, tensors)
    ExUnit.Callbacks.on_exit(fn -> File.rm(path) end)
    path
  end

  describe "map_key/1 - stem" do
    test "maps patch embedding conv and norm" do
      assert ConvNeXt.map_key("convnext.embeddings.patch_embeddings.weight") ==
               "stem_conv.kernel"

      assert ConvNeXt.map_key("convnext.embeddings.patch_embeddings.bias") ==
               "stem_conv.bias"

      assert ConvNeXt.map_key("convnext.embeddings.layernorm.weight") ==
               "stem_norm.gamma"

      assert ConvNeXt.map_key("convnext.embeddings.layernorm.bias") ==
               "stem_norm.beta"
    end
  end

  describe "map_key/1 - blocks" do
    test "maps depthwise conv" do
      assert ConvNeXt.map_key("convnext.encoder.stages.0.layers.0.dwconv.weight") ==
               "stage0_block0_dw_conv.kernel"

      assert ConvNeXt.map_key("convnext.encoder.stages.0.layers.0.dwconv.bias") ==
               "stage0_block0_dw_conv.bias"
    end

    test "maps block layernorm" do
      assert ConvNeXt.map_key("convnext.encoder.stages.0.layers.0.layernorm.weight") ==
               "stage0_block0_norm.gamma"

      assert ConvNeXt.map_key("convnext.encoder.stages.0.layers.0.layernorm.bias") ==
               "stage0_block0_norm.beta"
    end

    test "maps pointwise convolutions" do
      assert ConvNeXt.map_key("convnext.encoder.stages.0.layers.0.pwconv1.weight") ==
               "stage0_block0_pw_expand.kernel"

      assert ConvNeXt.map_key("convnext.encoder.stages.0.layers.0.pwconv1.bias") ==
               "stage0_block0_pw_expand.bias"

      assert ConvNeXt.map_key("convnext.encoder.stages.0.layers.0.pwconv2.weight") ==
               "stage0_block0_pw_project.kernel"

      assert ConvNeXt.map_key("convnext.encoder.stages.0.layers.0.pwconv2.bias") ==
               "stage0_block0_pw_project.bias"
    end

    test "maps layer scale parameter" do
      assert ConvNeXt.map_key(
               "convnext.encoder.stages.0.layers.0.layer_scale_parameter"
             ) == "stage0_block0_layer_scale.stage0_block0_gamma"

      assert ConvNeXt.map_key(
               "convnext.encoder.stages.2.layers.5.layer_scale_parameter"
             ) == "stage2_block5_layer_scale.stage2_block5_gamma"
    end

    test "maps higher stage/block indices" do
      assert ConvNeXt.map_key("convnext.encoder.stages.3.layers.2.dwconv.weight") ==
               "stage3_block2_dw_conv.kernel"

      assert ConvNeXt.map_key("convnext.encoder.stages.2.layers.8.pwconv1.weight") ==
               "stage2_block8_pw_expand.kernel"
    end
  end

  describe "map_key/1 - downsampling" do
    test "maps downsampling with index shift (HF stage N → Edifice downsample N-1)" do
      # HF stages.1.downsampling_layer → Edifice downsample_0
      assert ConvNeXt.map_key(
               "convnext.encoder.stages.1.downsampling_layer.0.weight"
             ) == "downsample_0_norm.gamma"

      assert ConvNeXt.map_key(
               "convnext.encoder.stages.1.downsampling_layer.0.bias"
             ) == "downsample_0_norm.beta"

      assert ConvNeXt.map_key(
               "convnext.encoder.stages.1.downsampling_layer.1.weight"
             ) == "downsample_0_conv.kernel"

      assert ConvNeXt.map_key(
               "convnext.encoder.stages.1.downsampling_layer.1.bias"
             ) == "downsample_0_conv.bias"

      # HF stages.3.downsampling_layer → Edifice downsample_2
      assert ConvNeXt.map_key(
               "convnext.encoder.stages.3.downsampling_layer.0.weight"
             ) == "downsample_2_norm.gamma"

      assert ConvNeXt.map_key(
               "convnext.encoder.stages.3.downsampling_layer.1.weight"
             ) == "downsample_2_conv.kernel"
    end
  end

  describe "map_key/1 - head" do
    test "maps final norm and classifier" do
      assert ConvNeXt.map_key("convnext.layernorm.weight") == "final_norm.gamma"
      assert ConvNeXt.map_key("convnext.layernorm.bias") == "final_norm.beta"
      assert ConvNeXt.map_key("classifier.weight") == "classifier.kernel"
      assert ConvNeXt.map_key("classifier.bias") == "classifier.bias"
    end
  end

  describe "map_key/1 - unmapped" do
    test "returns :unmapped for unknown keys" do
      assert ConvNeXt.map_key("some.random.key") == :unmapped
    end
  end

  describe "tensor_transforms/0" do
    test "permutes standard conv2d kernels from OIHW to HWIO" do
      transforms = ConvNeXt.tensor_transforms()
      # PyTorch: {out=16, in=3, kH=4, kW=4}
      tensor = Nx.iota({16, 3, 4, 4}, type: :f32)

      result =
        Edifice.Pretrained.Transform.apply_transform(
          "stem_conv.kernel",
          transforms,
          tensor
        )

      # Axon: {kH=4, kW=4, in=3, out=16}
      assert Nx.shape(result) == {4, 4, 3, 16}
    end

    test "converts linear weight to 1x1 conv kernel for pointwise layers" do
      transforms = ConvNeXt.tensor_transforms()
      # HF Linear: {out=384, in=96}
      tensor = Nx.iota({384, 96}, type: :f32)

      result =
        Edifice.Pretrained.Transform.apply_transform(
          "stage0_block0_pw_expand.kernel",
          transforms,
          tensor
        )

      # Axon conv 1x1: {1, 1, in=96, out=384}
      assert Nx.shape(result) == {1, 1, 96, 384}
    end

    test "reshapes layer scale from {dim} to {1,1,1,dim}" do
      transforms = ConvNeXt.tensor_transforms()
      tensor = Nx.iota({96}, type: :f32)

      result =
        Edifice.Pretrained.Transform.apply_transform(
          "stage0_block0_layer_scale.stage0_block0_gamma",
          transforms,
          tensor
        )

      assert Nx.shape(result) == {1, 1, 1, 96}
    end

    test "leaves bias tensors unchanged" do
      transforms = ConvNeXt.tensor_transforms()
      tensor = Nx.iota({96}, type: :f32)

      result =
        Edifice.Pretrained.Transform.apply_transform(
          "stage0_block0_dw_conv.bias",
          transforms,
          tensor
        )

      assert Nx.shape(result) == {96}
    end

    test "transposes rank-2 classifier kernel" do
      transforms = ConvNeXt.tensor_transforms()
      tensor = Nx.iota({1000, 768}, type: :f32)

      result =
        Edifice.Pretrained.Transform.apply_transform(
          "classifier.kernel",
          transforms,
          tensor
        )

      assert Nx.shape(result) == {768, 1000}
    end

    test "leaves norm scale unchanged" do
      transforms = ConvNeXt.tensor_transforms()
      tensor = Nx.iota({96}, type: :f32)

      result =
        Edifice.Pretrained.Transform.apply_transform(
          "stage0_block0_norm.gamma",
          transforms,
          tensor
        )

      assert Nx.shape(result) == {96}
    end
  end

  describe "end-to-end loading" do
    test "loads minimal ConvNeXt checkpoint into ModelState" do
      dim = 4

      tensors = %{
        # Stem
        "convnext.embeddings.patch_embeddings.weight" =>
          Nx.iota({dim, 3, 4, 4}, type: :f32),
        "convnext.embeddings.patch_embeddings.bias" =>
          Nx.iota({dim}, type: :f32),
        "convnext.embeddings.layernorm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {dim}),
        "convnext.embeddings.layernorm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {dim}),
        # One block in stage 0
        "convnext.encoder.stages.0.layers.0.dwconv.weight" =>
          Nx.iota({dim, 1, 7, 7}, type: :f32),
        "convnext.encoder.stages.0.layers.0.dwconv.bias" =>
          Nx.iota({dim}, type: :f32),
        "convnext.encoder.stages.0.layers.0.layernorm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {dim}),
        "convnext.encoder.stages.0.layers.0.layernorm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {dim}),
        "convnext.encoder.stages.0.layers.0.pwconv1.weight" =>
          Nx.iota({dim * 4, dim}, type: :f32),
        "convnext.encoder.stages.0.layers.0.pwconv1.bias" =>
          Nx.iota({dim * 4}, type: :f32),
        "convnext.encoder.stages.0.layers.0.pwconv2.weight" =>
          Nx.iota({dim, dim * 4}, type: :f32),
        "convnext.encoder.stages.0.layers.0.pwconv2.bias" =>
          Nx.iota({dim}, type: :f32),
        "convnext.encoder.stages.0.layers.0.layer_scale_parameter" =>
          Nx.broadcast(Nx.tensor(1.0e-6, type: :f32), {dim}),
        # Downsample between stage 0 and 1 (in HF: stages.1.downsampling_layer)
        "convnext.encoder.stages.1.downsampling_layer.0.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {dim}),
        "convnext.encoder.stages.1.downsampling_layer.0.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {dim}),
        "convnext.encoder.stages.1.downsampling_layer.1.weight" =>
          Nx.iota({dim * 2, dim, 2, 2}, type: :f32),
        "convnext.encoder.stages.1.downsampling_layer.1.bias" =>
          Nx.iota({dim * 2}, type: :f32),
        # Final norm + classifier
        "convnext.layernorm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {dim}),
        "convnext.layernorm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {dim}),
        "classifier.weight" =>
          Nx.iota({10, dim}, type: :f32),
        "classifier.bias" =>
          Nx.iota({10}, type: :f32)
      }

      path = write_fixture(tensors)
      model_state = Edifice.Pretrained.load(ConvNeXt, path)

      assert %Axon.ModelState{} = model_state
      data = model_state.data

      # Stem conv: OIHW {4,3,4,4} → HWIO {4,4,3,4}
      assert Nx.shape(data["stem_conv"]["kernel"]) == {4, 4, 3, dim}
      assert %Nx.Tensor{} = data["stem_norm"]["gamma"]

      # Block depthwise conv: {4,1,7,7} → {7,7,1,4}
      assert Nx.shape(data["stage0_block0_dw_conv"]["kernel"]) == {7, 7, 1, dim}

      # Pointwise expand: Linear {4*dim, dim} → Conv {1, 1, dim, 4*dim}
      assert Nx.shape(data["stage0_block0_pw_expand"]["kernel"]) == {1, 1, dim, dim * 4}

      # Pointwise project: Linear {dim, 4*dim} → Conv {1, 1, 4*dim, dim}
      assert Nx.shape(data["stage0_block0_pw_project"]["kernel"]) == {1, 1, dim * 4, dim}

      # Layer scale: {dim} → {1, 1, 1, dim}
      gamma = data["stage0_block0_layer_scale"]["stage0_block0_gamma"]
      assert Nx.shape(gamma) == {1, 1, 1, dim}

      # Downsample conv: OIHW {8,4,2,2} → HWIO {2,2,4,8}
      assert Nx.shape(data["downsample_0_conv"]["kernel"]) == {2, 2, dim, dim * 2}
      assert %Nx.Tensor{} = data["downsample_0_norm"]["gamma"]

      # Classifier: Linear {10, dim} → transposed to {dim, 10}
      assert Nx.shape(data["classifier"]["kernel"]) == {dim, 10}
      assert %Nx.Tensor{} = data["final_norm"]["gamma"]
    end
  end
end
