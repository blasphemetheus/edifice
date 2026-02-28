defmodule Edifice.Vision.BackboneTest do
  use ExUnit.Case, async: true
  @moduletag :vision

  alias Edifice.Vision.Backbone

  # Small dimensions for fast BinaryBackend tests
  @batch 2

  # ============================================================================
  # Backbone modules with minimal opts for fast testing
  # ============================================================================

  @backbone_modules [
    {Edifice.Vision.ViT, [embed_dim: 32, depth: 1, num_heads: 2, image_size: 32, patch_size: 8]},
    {Edifice.Vision.DeiT, [embed_dim: 32, depth: 1, num_heads: 2, image_size: 32, patch_size: 8]},
    {Edifice.Vision.SwinTransformer,
     [
       embed_dim: 32,
       depths: [1],
       num_heads: [2],
       window_size: 4,
       image_size: 32,
       patch_size: 4
     ]},
    {Edifice.Vision.ConvNeXt, [depths: [1], dims: [32], image_size: 32, patch_size: 4]},
    {Edifice.Vision.MLPMixer,
     [
       hidden_size: 32,
       num_layers: 1,
       token_mlp_dim: 16,
       channel_mlp_dim: 64,
       image_size: 32,
       patch_size: 8
     ]},
    {Edifice.Vision.PoolFormer, [hidden_size: 32, num_layers: 1, image_size: 32, patch_size: 8]},
    {Edifice.Vision.FocalNet,
     [hidden_size: 32, num_layers: 1, focal_levels: 2, image_size: 32, patch_size: 8]},
    {Edifice.Vision.MetaFormer, [depths: [1], dims: [32], image_size: 32, patch_size: 4]},
    {Edifice.Vision.EfficientViT,
     [embed_dim: 32, depths: [1], num_heads: [2], image_size: 32, patch_size: 8]},
    {Edifice.Vision.MambaVision,
     [dim: 8, depths: [1, 1, 1, 1], num_heads: [1, 1, 1, 1], image_size: 32]},
    {Edifice.Vision.DINOv2,
     [
       embed_dim: 32,
       num_heads: 2,
       num_layers: 1,
       image_size: 32,
       patch_size: 8,
       num_register_tokens: 0,
       include_head: false
     ]},
    {Edifice.Vision.DINOv3,
     [
       embed_dim: 32,
       num_heads: 2,
       num_layers: 1,
       image_size: 32,
       patch_size: 8,
       num_register_tokens: 0,
       include_head: false
     ]}
  ]

  # ============================================================================
  # Contract Tests — verify every adopter satisfies the Backbone interface
  # ============================================================================

  for {mod, opts} <- @backbone_modules do
    mod_name = mod |> Module.split() |> List.last()

    describe "#{mod_name} Backbone" do
      test "build_backbone/1 returns an Axon model" do
        model = unquote(mod).build_backbone(unquote(Macro.escape(opts)))
        assert %Axon{} = model
      end

      test "feature_size/1 returns a positive integer" do
        size = unquote(mod).feature_size(unquote(Macro.escape(opts)))
        assert is_integer(size) and size > 0
      end

      test "input_shape/1 returns {nil, C, H, W}" do
        {nil, c, h, w} = unquote(mod).input_shape(unquote(Macro.escape(opts)))
        assert is_integer(c) and c > 0
        assert is_integer(h) and h > 0
        assert is_integer(w) and w > 0
      end

      test "forward pass produces [batch, feature_size]" do
        opts = unquote(Macro.escape(opts))
        model = unquote(mod).build_backbone(opts)
        size = unquote(mod).feature_size(opts)
        {nil, c, h, w} = unquote(mod).input_shape(opts)

        {init_fn, predict_fn} = Axon.build(model)
        template = Nx.template({@batch, c, h, w}, :f32)
        params = init_fn.(template, Axon.ModelState.empty())
        input = Nx.broadcast(0.5, {@batch, c, h, w})
        output = predict_fn.(params, %{"image" => input})

        assert Nx.shape(output) == {@batch, size}
      end
    end
  end

  # ============================================================================
  # Dispatch Helper Tests
  # ============================================================================

  describe "Backbone.build_backbone/2 dispatch" do
    test "dispatches to ViT" do
      opts = [embed_dim: 32, depth: 1, num_heads: 2, image_size: 32, patch_size: 8]
      model = Backbone.build_backbone(Edifice.Vision.ViT, opts)
      assert %Axon{} = model
    end

    test "dispatches to ConvNeXt" do
      opts = [depths: [1], dims: [32], image_size: 32, patch_size: 4]
      model = Backbone.build_backbone(Edifice.Vision.ConvNeXt, opts)
      assert %Axon{} = model
    end
  end

  describe "Backbone.feature_size/2 dispatch" do
    test "dispatches to ViT" do
      assert Backbone.feature_size(Edifice.Vision.ViT, embed_dim: 64) == 64
    end

    test "dispatches to Swin" do
      assert Backbone.feature_size(Edifice.Vision.SwinTransformer, embed_dim: 32, depths: [1, 1]) ==
               64
    end
  end

  describe "Backbone.input_shape/2 dispatch" do
    test "returns NCHW shape" do
      assert Backbone.input_shape(Edifice.Vision.ViT, image_size: 64, in_channels: 1) ==
               {nil, 1, 64, 64}
    end
  end
end
