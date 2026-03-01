defmodule Edifice.Pretrained.KeyMaps.ViTTest do
  use ExUnit.Case, async: true

  alias Edifice.Pretrained.KeyMaps.ViT

  setup do
    assert Code.ensure_loaded?(Safetensors),
           "safetensors package must be available for these tests"

    :ok
  end

  defp write_fixture(tensors) do
    path =
      Path.join(
        System.tmp_dir!(),
        "edifice_vit_test_#{System.unique_integer([:positive])}.safetensors"
      )

    Safetensors.write!(path, tensors)
    ExUnit.Callbacks.on_exit(fn -> File.rm(path) end)
    path
  end

  describe "map_key/1 - embeddings" do
    test "maps patch embedding projection" do
      assert ViT.map_key("vit.embeddings.patch_embeddings.projection.weight") ==
               "patch_embed_proj.kernel"

      assert ViT.map_key("vit.embeddings.patch_embeddings.projection.bias") ==
               "patch_embed_proj.bias"
    end

    test "maps CLS token" do
      assert ViT.map_key("vit.embeddings.cls_token") == "cls_token_proj.kernel"
    end

    test "maps position embeddings" do
      assert ViT.map_key("vit.embeddings.position_embeddings") == "pos_embed_proj.kernel"
    end
  end

  describe "map_key/1 - encoder blocks" do
    test "maps layer norms" do
      assert ViT.map_key("vit.encoder.layer.0.layernorm_before.weight") ==
               "block_0_norm1.gamma"

      assert ViT.map_key("vit.encoder.layer.0.layernorm_before.bias") ==
               "block_0_norm1.beta"

      assert ViT.map_key("vit.encoder.layer.3.layernorm_after.weight") ==
               "block_3_norm2.gamma"

      assert ViT.map_key("vit.encoder.layer.3.layernorm_after.bias") ==
               "block_3_norm2.beta"
    end

    test "maps Q/K/V to intermediate keys for concat" do
      assert ViT.map_key("vit.encoder.layer.0.attention.attention.query.weight") ==
               "block_0_attn_q.kernel"

      assert ViT.map_key("vit.encoder.layer.0.attention.attention.key.weight") ==
               "block_0_attn_k.kernel"

      assert ViT.map_key("vit.encoder.layer.0.attention.attention.value.weight") ==
               "block_0_attn_v.kernel"

      assert ViT.map_key("vit.encoder.layer.5.attention.attention.query.bias") ==
               "block_5_attn_q.bias"

      assert ViT.map_key("vit.encoder.layer.5.attention.attention.key.bias") ==
               "block_5_attn_k.bias"

      assert ViT.map_key("vit.encoder.layer.5.attention.attention.value.bias") ==
               "block_5_attn_v.bias"
    end

    test "maps attention output projection" do
      assert ViT.map_key("vit.encoder.layer.0.attention.output.dense.weight") ==
               "block_0_attn_proj.kernel"

      assert ViT.map_key("vit.encoder.layer.0.attention.output.dense.bias") ==
               "block_0_attn_proj.bias"
    end

    test "maps MLP layers" do
      assert ViT.map_key("vit.encoder.layer.2.intermediate.dense.weight") ==
               "block_2_mlp_fc1.kernel"

      assert ViT.map_key("vit.encoder.layer.2.intermediate.dense.bias") ==
               "block_2_mlp_fc1.bias"

      assert ViT.map_key("vit.encoder.layer.2.output.dense.weight") ==
               "block_2_mlp_fc2.kernel"

      assert ViT.map_key("vit.encoder.layer.2.output.dense.bias") ==
               "block_2_mlp_fc2.bias"
    end
  end

  describe "map_key/1 - final norm and classifier" do
    test "maps final layer norm" do
      assert ViT.map_key("vit.layernorm.weight") == "final_norm.gamma"
      assert ViT.map_key("vit.layernorm.bias") == "final_norm.beta"
    end

    test "maps classifier head" do
      assert ViT.map_key("classifier.weight") == "classifier.kernel"
      assert ViT.map_key("classifier.bias") == "classifier.bias"
    end
  end

  describe "map_key/1 - skip and unmapped" do
    test "skips pooler keys" do
      assert ViT.map_key("vit.pooler.dense.weight") == :skip
      assert ViT.map_key("vit.pooler.dense.bias") == :skip
    end

    test "returns :unmapped for unknown keys" do
      assert ViT.map_key("some.random.key") == :unmapped
    end
  end

  describe "concat_keys/0" do
    test "returns QKV concat rules for 12 layers" do
      concat = ViT.concat_keys()

      # Should have entries for both kernel and bias, for 12 layers = 24 entries
      assert map_size(concat) == 24

      # Check one kernel group
      assert {sources, 1} = concat["block_0_attn_qkv.kernel"]

      assert sources == [
               "block_0_attn_q.kernel",
               "block_0_attn_k.kernel",
               "block_0_attn_v.kernel"
             ]

      # Check one bias group (biases concat along axis 0)
      assert {sources, 0} = concat["block_11_attn_qkv.bias"]

      assert sources == [
               "block_11_attn_q.bias",
               "block_11_attn_k.bias",
               "block_11_attn_v.bias"
             ]
    end
  end

  describe "build_concat_keys/1" do
    test "builds concat rules for custom layer count" do
      concat = ViT.build_concat_keys(2)
      assert map_size(concat) == 4

      assert Map.has_key?(concat, "block_0_attn_qkv.kernel")
      assert Map.has_key?(concat, "block_0_attn_qkv.bias")
      assert Map.has_key?(concat, "block_1_attn_qkv.kernel")
      assert Map.has_key?(concat, "block_1_attn_qkv.bias")
    end
  end

  describe "tensor_transforms/0" do
    test "transposes rank-2 kernel tensors" do
      transforms = ViT.tensor_transforms()
      tensor = Nx.iota({4, 3}, type: :f32)

      result =
        Edifice.Pretrained.Transform.apply_transform("dense.kernel", transforms, tensor)

      assert Nx.shape(result) == {3, 4}
    end

    test "squeezes rank-3 kernel tensors (CLS/pos embed)" do
      transforms = ViT.tensor_transforms()
      tensor = Nx.iota({1, 1, 768}, type: :f32)

      result =
        Edifice.Pretrained.Transform.apply_transform(
          "cls_token_proj.kernel",
          transforms,
          tensor
        )

      assert Nx.shape(result) == {1, 768}
    end

    test "leaves bias tensors unchanged" do
      transforms = ViT.tensor_transforms()
      tensor = Nx.iota({4}, type: :f32)

      result =
        Edifice.Pretrained.Transform.apply_transform("dense.bias", transforms, tensor)

      assert Nx.shape(result) == {4}
    end
  end

  describe "QKV concatenation" do
    test "concatenates separate Q/K/V weights into combined QKV" do
      embed_dim = 8

      tensors = %{
        # Q/K/V weights for block 0 — each [embed_dim, embed_dim] in HF (transposed by transform)
        "vit.encoder.layer.0.attention.attention.query.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {embed_dim, embed_dim}),
        "vit.encoder.layer.0.attention.attention.key.weight" =>
          Nx.broadcast(Nx.tensor(2.0, type: :f32), {embed_dim, embed_dim}),
        "vit.encoder.layer.0.attention.attention.value.weight" =>
          Nx.broadcast(Nx.tensor(3.0, type: :f32), {embed_dim, embed_dim}),
        # Q/K/V biases
        "vit.encoder.layer.0.attention.attention.query.bias" =>
          Nx.broadcast(Nx.tensor(0.1, type: :f32), {embed_dim}),
        "vit.encoder.layer.0.attention.attention.key.bias" =>
          Nx.broadcast(Nx.tensor(0.2, type: :f32), {embed_dim}),
        "vit.encoder.layer.0.attention.attention.value.bias" =>
          Nx.broadcast(Nx.tensor(0.3, type: :f32), {embed_dim}),
        # Other required keys
        "vit.encoder.layer.0.attention.output.dense.weight" =>
          Nx.iota({embed_dim, embed_dim}, type: :f32),
        "vit.encoder.layer.0.attention.output.dense.bias" =>
          Nx.iota({embed_dim}, type: :f32),
        "vit.encoder.layer.0.layernorm_before.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {embed_dim}),
        "vit.encoder.layer.0.layernorm_before.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {embed_dim}),
        "vit.encoder.layer.0.layernorm_after.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {embed_dim}),
        "vit.encoder.layer.0.layernorm_after.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {embed_dim}),
        "vit.encoder.layer.0.intermediate.dense.weight" =>
          Nx.iota({embed_dim * 4, embed_dim}, type: :f32),
        "vit.encoder.layer.0.intermediate.dense.bias" =>
          Nx.iota({embed_dim * 4}, type: :f32),
        "vit.encoder.layer.0.output.dense.weight" =>
          Nx.iota({embed_dim, embed_dim * 4}, type: :f32),
        "vit.encoder.layer.0.output.dense.bias" =>
          Nx.iota({embed_dim}, type: :f32),
        "vit.layernorm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {embed_dim}),
        "vit.layernorm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {embed_dim}),
        "vit.embeddings.patch_embeddings.projection.weight" =>
          Nx.iota({embed_dim, 4}, type: :f32),
        "vit.embeddings.patch_embeddings.projection.bias" =>
          Nx.iota({embed_dim}, type: :f32),
        "vit.embeddings.cls_token" =>
          Nx.iota({1, 1, embed_dim}, type: :f32),
        "vit.embeddings.position_embeddings" =>
          Nx.iota({1, 10, embed_dim}, type: :f32)
      }

      # Use build_concat_keys(1) since we only have 1 layer
      path = write_fixture(tensors)

      # Load with a custom key map that uses 1-layer concat
      model_state =
        Edifice.Pretrained.load(
          Edifice.Pretrained.KeyMaps.ViT,
          path,
          strict: false
        )

      flat = Edifice.Pretrained.Transform.flatten_params(model_state)

      # QKV kernel: after transpose, each is [embed_dim, embed_dim]
      # Concatenated along axis 1 → [embed_dim, 3*embed_dim]
      qkv_kernel = flat["block_0_attn_qkv.kernel"]
      assert Nx.shape(qkv_kernel) == {embed_dim, 3 * embed_dim}

      # First third should be all 1.0 (transposed Q), second 2.0 (K), third 3.0 (V)
      q_part = Nx.slice(qkv_kernel, [0, 0], [embed_dim, embed_dim])
      k_part = Nx.slice(qkv_kernel, [0, embed_dim], [embed_dim, embed_dim])
      v_part = Nx.slice(qkv_kernel, [0, 2 * embed_dim], [embed_dim, embed_dim])

      assert Nx.to_number(Nx.mean(q_part)) == 1.0
      assert Nx.to_number(Nx.mean(k_part)) == 2.0
      assert Nx.to_number(Nx.mean(v_part)) == 3.0

      # QKV bias: concatenated along axis 0 → [3*embed_dim]
      qkv_bias = flat["block_0_attn_qkv.bias"]
      assert Nx.shape(qkv_bias) == {3 * embed_dim}
    end
  end

  describe "CLS token and position embedding reshaping" do
    test "CLS token is squeezed from [1,1,D] to [1,D]" do
      embed_dim = 4

      tensors = %{
        "vit.embeddings.cls_token" =>
          Nx.iota({1, 1, embed_dim}, type: :f32),
        "vit.embeddings.position_embeddings" =>
          Nx.iota({1, 5, embed_dim}, type: :f32),
        "vit.embeddings.patch_embeddings.projection.weight" =>
          Nx.iota({embed_dim, 3}, type: :f32),
        "vit.embeddings.patch_embeddings.projection.bias" =>
          Nx.iota({embed_dim}, type: :f32),
        "vit.layernorm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {embed_dim}),
        "vit.layernorm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {embed_dim})
      }

      path = write_fixture(tensors)
      model_state = Edifice.Pretrained.load(ViT, path, strict: false)
      flat = Edifice.Pretrained.Transform.flatten_params(model_state)

      # CLS token: [1,1,D] → squeeze → [1,D]
      cls = flat["cls_token_proj.kernel"]
      assert Nx.shape(cls) == {1, embed_dim}

      # Position embedding: [1,S,D] → squeeze → [S,D]
      pos = flat["pos_embed_proj.kernel"]
      assert Nx.shape(pos) == {5, embed_dim}
    end
  end

  describe "end-to-end loading" do
    test "loads minimal 1-layer ViT checkpoint into ModelState" do
      d = 8

      tensors = %{
        "vit.embeddings.patch_embeddings.projection.weight" =>
          Nx.iota({d, 4}, type: :f32),
        "vit.embeddings.patch_embeddings.projection.bias" =>
          Nx.iota({d}, type: :f32),
        "vit.embeddings.cls_token" => Nx.iota({1, 1, d}, type: :f32),
        "vit.embeddings.position_embeddings" => Nx.iota({1, 10, d}, type: :f32),
        "vit.encoder.layer.0.layernorm_before.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
        "vit.encoder.layer.0.layernorm_before.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {d}),
        "vit.encoder.layer.0.attention.attention.query.weight" =>
          Nx.iota({d, d}, type: :f32),
        "vit.encoder.layer.0.attention.attention.query.bias" =>
          Nx.iota({d}, type: :f32),
        "vit.encoder.layer.0.attention.attention.key.weight" =>
          Nx.iota({d, d}, type: :f32),
        "vit.encoder.layer.0.attention.attention.key.bias" =>
          Nx.iota({d}, type: :f32),
        "vit.encoder.layer.0.attention.attention.value.weight" =>
          Nx.iota({d, d}, type: :f32),
        "vit.encoder.layer.0.attention.attention.value.bias" =>
          Nx.iota({d}, type: :f32),
        "vit.encoder.layer.0.attention.output.dense.weight" =>
          Nx.iota({d, d}, type: :f32),
        "vit.encoder.layer.0.attention.output.dense.bias" =>
          Nx.iota({d}, type: :f32),
        "vit.encoder.layer.0.layernorm_after.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
        "vit.encoder.layer.0.layernorm_after.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {d}),
        "vit.encoder.layer.0.intermediate.dense.weight" =>
          Nx.iota({d * 4, d}, type: :f32),
        "vit.encoder.layer.0.intermediate.dense.bias" =>
          Nx.iota({d * 4}, type: :f32),
        "vit.encoder.layer.0.output.dense.weight" =>
          Nx.iota({d, d * 4}, type: :f32),
        "vit.encoder.layer.0.output.dense.bias" =>
          Nx.iota({d}, type: :f32),
        "vit.layernorm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
        "vit.layernorm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {d}),
        "classifier.weight" =>
          Nx.iota({10, d}, type: :f32),
        "classifier.bias" =>
          Nx.iota({10}, type: :f32)
      }

      path = write_fixture(tensors)
      model_state = Edifice.Pretrained.load(ViT, path, strict: false)

      assert %Axon.ModelState{} = model_state
      data = model_state.data

      # Check key structure
      assert is_map(data["patch_embed_proj"])
      assert %Nx.Tensor{} = data["patch_embed_proj"]["kernel"]
      assert %Nx.Tensor{} = data["cls_token_proj"]["kernel"]
      assert %Nx.Tensor{} = data["pos_embed_proj"]["kernel"]
      assert %Nx.Tensor{} = data["final_norm"]["gamma"]
      assert %Nx.Tensor{} = data["classifier"]["kernel"]

      # QKV should be combined
      assert %Nx.Tensor{} = data["block_0_attn_qkv"]["kernel"]
      assert Nx.shape(data["block_0_attn_qkv"]["kernel"]) == {d, 3 * d}
      assert %Nx.Tensor{} = data["block_0_attn_qkv"]["bias"]
      assert Nx.shape(data["block_0_attn_qkv"]["bias"]) == {3 * d}

      # Attention proj should be transposed
      assert Nx.shape(data["block_0_attn_proj"]["kernel"]) == {d, d}

      # MLP layers should be transposed
      assert Nx.shape(data["block_0_mlp_fc1"]["kernel"]) == {d, d * 4}
      assert Nx.shape(data["block_0_mlp_fc2"]["kernel"]) == {d * 4, d}
    end
  end
end
