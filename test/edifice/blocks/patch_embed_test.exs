defmodule Edifice.Blocks.PatchEmbedTest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.PatchEmbed

  @batch 2

  describe "num_patches/2" do
    test "calculates correctly for standard sizes" do
      assert PatchEmbed.num_patches(224, 16) == 196
      assert PatchEmbed.num_patches(32, 8) == 16
      assert PatchEmbed.num_patches(64, 16) == 16
    end

    test "handles square-exact divisions" do
      assert PatchEmbed.num_patches(256, 32) == 64
      assert PatchEmbed.num_patches(16, 4) == 16
    end
  end

  describe "layer/2" do
    test "produces correct number of patch embeddings" do
      image = Axon.input("image", shape: {nil, 3, 32, 32})
      model = PatchEmbed.layer(image, embed_dim: 64, patch_size: 8, in_channels: 3)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 3, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 3, 32, 32}))

      # 32/8 = 4, 4*4 = 16 patches
      assert Nx.shape(output) == {@batch, 16, 64}
    end

    test "patches are correctly shaped with different patch sizes" do
      image = Axon.input("image", shape: {nil, 3, 64, 64})
      model = PatchEmbed.layer(image, embed_dim: 128, patch_size: 16, in_channels: 3)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 3, 64, 64}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 3, 64, 64}))

      # 64/16 = 4, 4*4 = 16 patches
      assert Nx.shape(output) == {@batch, 16, 128}
    end

    test "handles batch_size=1" do
      image = Axon.input("image", shape: {nil, 3, 32, 32})
      model = PatchEmbed.layer(image, embed_dim: 64, patch_size: 8, in_channels: 3)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 3, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {1, 3, 32, 32}))

      assert Nx.shape(output) == {1, 16, 64}
    end

    test "single-channel input" do
      image = Axon.input("image", shape: {nil, 1, 16, 16})
      model = PatchEmbed.layer(image, embed_dim: 32, patch_size: 4, in_channels: 1)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 1, 16, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 1, 16, 16}))

      # 16/4 = 4, 4*4 = 16 patches
      assert Nx.shape(output) == {@batch, 16, 32}
    end

    test "output is finite" do
      image = Axon.input("image", shape: {nil, 3, 32, 32})
      model = PatchEmbed.layer(image, embed_dim: 64, patch_size: 8, in_channels: 3)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 3, 32, 32}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, 3, 32, 32})
      output = predict_fn.(params, test_input)

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end
end
