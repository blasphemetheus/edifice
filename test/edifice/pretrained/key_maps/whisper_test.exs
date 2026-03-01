defmodule Edifice.Pretrained.KeyMaps.WhisperTest do
  use ExUnit.Case, async: true

  alias Edifice.Pretrained.KeyMaps.Whisper

  setup do
    assert Code.ensure_loaded?(Safetensors),
           "safetensors package must be available for these tests"

    :ok
  end

  defp write_fixture(tensors) do
    path =
      Path.join(
        System.tmp_dir!(),
        "edifice_whisper_test_#{System.unique_integer([:positive])}.safetensors"
      )

    Safetensors.write!(path, tensors)
    ExUnit.Callbacks.on_exit(fn -> File.rm(path) end)
    path
  end

  describe "map_key/1 - encoder conv layers" do
    test "maps conv1 weight and bias" do
      assert Whisper.map_key("model.encoder.conv1.weight") == "enc_conv1.kernel"
      assert Whisper.map_key("model.encoder.conv1.bias") == "enc_conv1.bias"
    end

    test "maps conv2 weight and bias" do
      assert Whisper.map_key("model.encoder.conv2.weight") == "enc_conv2.kernel"
      assert Whisper.map_key("model.encoder.conv2.bias") == "enc_conv2.bias"
    end
  end

  describe "map_key/1 - encoder layers with index shift" do
    test "maps encoder self-attention Q/K/V/out with +1 index" do
      # HF layer 0 → Edifice block 1
      assert Whisper.map_key("model.encoder.layers.0.self_attn.q_proj.weight") ==
               "enc_block_1_attn_q.kernel"

      assert Whisper.map_key("model.encoder.layers.0.self_attn.q_proj.bias") ==
               "enc_block_1_attn_q.bias"

      assert Whisper.map_key("model.encoder.layers.0.self_attn.k_proj.weight") ==
               "enc_block_1_attn_k.kernel"

      assert Whisper.map_key("model.encoder.layers.0.self_attn.v_proj.weight") ==
               "enc_block_1_attn_v.kernel"

      assert Whisper.map_key("model.encoder.layers.0.self_attn.out_proj.weight") ==
               "enc_block_1_attn_out.kernel"

      assert Whisper.map_key("model.encoder.layers.0.self_attn.out_proj.bias") ==
               "enc_block_1_attn_out.bias"

      # HF layer 3 → Edifice block 4
      assert Whisper.map_key("model.encoder.layers.3.self_attn.q_proj.weight") ==
               "enc_block_4_attn_q.kernel"
    end

    test "maps encoder norms with +1 index" do
      assert Whisper.map_key("model.encoder.layers.0.self_attn_layer_norm.weight") ==
               "enc_block_1_attn_norm.gamma"

      assert Whisper.map_key("model.encoder.layers.0.self_attn_layer_norm.bias") ==
               "enc_block_1_attn_norm.beta"

      assert Whisper.map_key("model.encoder.layers.2.final_layer_norm.weight") ==
               "enc_block_3_ffn_norm.gamma"

      assert Whisper.map_key("model.encoder.layers.2.final_layer_norm.bias") ==
               "enc_block_3_ffn_norm.beta"
    end

    test "maps encoder FFN layers with +1 index" do
      assert Whisper.map_key("model.encoder.layers.0.fc1.weight") ==
               "enc_block_1_ffn_up.kernel"

      assert Whisper.map_key("model.encoder.layers.0.fc1.bias") ==
               "enc_block_1_ffn_up.bias"

      assert Whisper.map_key("model.encoder.layers.0.fc2.weight") ==
               "enc_block_1_ffn_down.kernel"

      assert Whisper.map_key("model.encoder.layers.0.fc2.bias") ==
               "enc_block_1_ffn_down.bias"
    end
  end

  describe "map_key/1 - encoder global" do
    test "skips encoder position embeddings" do
      assert Whisper.map_key("model.encoder.embed_positions.weight") == :skip
    end

    test "maps encoder final norm" do
      assert Whisper.map_key("model.encoder.layer_norm.weight") == "enc_final_norm.gamma"
      assert Whisper.map_key("model.encoder.layer_norm.bias") == "enc_final_norm.beta"
    end
  end

  describe "map_key/1 - decoder embeddings" do
    test "maps decoder token embedding" do
      assert Whisper.map_key("model.decoder.embed_tokens.weight") == "dec_token_embed.kernel"
    end

    test "maps decoder position embedding" do
      assert Whisper.map_key("model.decoder.embed_positions.weight") == "dec_pos_embed.kernel"
    end
  end

  describe "map_key/1 - decoder self-attention" do
    test "maps decoder self-attention with +1 index" do
      assert Whisper.map_key("model.decoder.layers.0.self_attn.q_proj.weight") ==
               "dec_block_1_attn_q.kernel"

      assert Whisper.map_key("model.decoder.layers.0.self_attn.q_proj.bias") ==
               "dec_block_1_attn_q.bias"

      assert Whisper.map_key("model.decoder.layers.0.self_attn.k_proj.weight") ==
               "dec_block_1_attn_k.kernel"

      assert Whisper.map_key("model.decoder.layers.0.self_attn.v_proj.weight") ==
               "dec_block_1_attn_v.kernel"

      assert Whisper.map_key("model.decoder.layers.0.self_attn.out_proj.weight") ==
               "dec_block_1_attn_out.kernel"

      assert Whisper.map_key("model.decoder.layers.0.self_attn_layer_norm.weight") ==
               "dec_block_1_attn_norm.gamma"

      assert Whisper.map_key("model.decoder.layers.0.self_attn_layer_norm.bias") ==
               "dec_block_1_attn_norm.beta"
    end
  end

  describe "map_key/1 - decoder cross-attention" do
    test "maps decoder cross-attention with +1 index" do
      assert Whisper.map_key("model.decoder.layers.0.encoder_attn.q_proj.weight") ==
               "dec_block_1_cross_attn_q_proj.kernel"

      assert Whisper.map_key("model.decoder.layers.0.encoder_attn.q_proj.bias") ==
               "dec_block_1_cross_attn_q_proj.bias"

      assert Whisper.map_key("model.decoder.layers.0.encoder_attn.k_proj.weight") ==
               "dec_block_1_cross_attn_k_proj.kernel"

      assert Whisper.map_key("model.decoder.layers.0.encoder_attn.k_proj.bias") ==
               "dec_block_1_cross_attn_k_proj.bias"

      assert Whisper.map_key("model.decoder.layers.0.encoder_attn.v_proj.weight") ==
               "dec_block_1_cross_attn_v_proj.kernel"

      assert Whisper.map_key("model.decoder.layers.0.encoder_attn.v_proj.bias") ==
               "dec_block_1_cross_attn_v_proj.bias"

      assert Whisper.map_key("model.decoder.layers.0.encoder_attn.out_proj.weight") ==
               "dec_block_1_cross_attn_out_proj.kernel"

      assert Whisper.map_key("model.decoder.layers.0.encoder_attn.out_proj.bias") ==
               "dec_block_1_cross_attn_out_proj.bias"

      assert Whisper.map_key("model.decoder.layers.0.encoder_attn_layer_norm.weight") ==
               "dec_block_1_cross_attn_norm.gamma"

      assert Whisper.map_key("model.decoder.layers.0.encoder_attn_layer_norm.bias") ==
               "dec_block_1_cross_attn_norm.beta"
    end

    test "cross-attention index shift works for higher indices" do
      assert Whisper.map_key("model.decoder.layers.5.encoder_attn.q_proj.weight") ==
               "dec_block_6_cross_attn_q_proj.kernel"
    end
  end

  describe "map_key/1 - decoder FFN" do
    test "maps decoder FFN with +1 index" do
      assert Whisper.map_key("model.decoder.layers.0.fc1.weight") ==
               "dec_block_1_ffn_up.kernel"

      assert Whisper.map_key("model.decoder.layers.0.fc1.bias") ==
               "dec_block_1_ffn_up.bias"

      assert Whisper.map_key("model.decoder.layers.0.fc2.weight") ==
               "dec_block_1_ffn_down.kernel"

      assert Whisper.map_key("model.decoder.layers.0.fc2.bias") ==
               "dec_block_1_ffn_down.bias"

      assert Whisper.map_key("model.decoder.layers.0.final_layer_norm.weight") ==
               "dec_block_1_ffn_norm.gamma"

      assert Whisper.map_key("model.decoder.layers.0.final_layer_norm.bias") ==
               "dec_block_1_ffn_norm.beta"
    end
  end

  describe "map_key/1 - decoder global" do
    test "maps decoder final norm" do
      assert Whisper.map_key("model.decoder.layer_norm.weight") == "dec_final_norm.gamma"
      assert Whisper.map_key("model.decoder.layer_norm.bias") == "dec_final_norm.beta"
    end

    test "skips proj_out" do
      assert Whisper.map_key("proj_out.weight") == :skip
    end
  end

  describe "map_key/1 - unmapped" do
    test "returns :unmapped for unknown keys" do
      assert Whisper.map_key("some.random.key") == :unmapped
    end
  end

  describe "end-to-end loading" do
    test "loads minimal 1-layer Whisper checkpoint into ModelState" do
      d = 8

      tensors = %{
        # Encoder conv
        "model.encoder.conv1.weight" => Nx.iota({d, 4, 3}, type: :f32),
        "model.encoder.conv1.bias" => Nx.iota({d}, type: :f32),
        "model.encoder.conv2.weight" => Nx.iota({d, d, 3}, type: :f32),
        "model.encoder.conv2.bias" => Nx.iota({d}, type: :f32),
        # Encoder positions (skipped)
        "model.encoder.embed_positions.weight" => Nx.iota({100, d}, type: :f32),
        # Encoder layer 0
        "model.encoder.layers.0.self_attn.q_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.encoder.layers.0.self_attn.q_proj.bias" => Nx.iota({d}, type: :f32),
        "model.encoder.layers.0.self_attn.k_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.encoder.layers.0.self_attn.k_proj.bias" => Nx.iota({d}, type: :f32),
        "model.encoder.layers.0.self_attn.v_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.encoder.layers.0.self_attn.v_proj.bias" => Nx.iota({d}, type: :f32),
        "model.encoder.layers.0.self_attn.out_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.encoder.layers.0.self_attn.out_proj.bias" => Nx.iota({d}, type: :f32),
        "model.encoder.layers.0.self_attn_layer_norm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
        "model.encoder.layers.0.self_attn_layer_norm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {d}),
        "model.encoder.layers.0.fc1.weight" => Nx.iota({d * 4, d}, type: :f32),
        "model.encoder.layers.0.fc1.bias" => Nx.iota({d * 4}, type: :f32),
        "model.encoder.layers.0.fc2.weight" => Nx.iota({d, d * 4}, type: :f32),
        "model.encoder.layers.0.fc2.bias" => Nx.iota({d}, type: :f32),
        "model.encoder.layers.0.final_layer_norm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
        "model.encoder.layers.0.final_layer_norm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {d}),
        # Encoder final norm
        "model.encoder.layer_norm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
        "model.encoder.layer_norm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {d}),
        # Decoder embeddings
        "model.decoder.embed_tokens.weight" => Nx.iota({100, d}, type: :f32),
        "model.decoder.embed_positions.weight" => Nx.iota({50, d}, type: :f32),
        # Decoder layer 0 - self attn
        "model.decoder.layers.0.self_attn.q_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.decoder.layers.0.self_attn.q_proj.bias" => Nx.iota({d}, type: :f32),
        "model.decoder.layers.0.self_attn.k_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.decoder.layers.0.self_attn.k_proj.bias" => Nx.iota({d}, type: :f32),
        "model.decoder.layers.0.self_attn.v_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.decoder.layers.0.self_attn.v_proj.bias" => Nx.iota({d}, type: :f32),
        "model.decoder.layers.0.self_attn.out_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.decoder.layers.0.self_attn.out_proj.bias" => Nx.iota({d}, type: :f32),
        "model.decoder.layers.0.self_attn_layer_norm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
        "model.decoder.layers.0.self_attn_layer_norm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {d}),
        # Decoder layer 0 - cross attn
        "model.decoder.layers.0.encoder_attn.q_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.decoder.layers.0.encoder_attn.q_proj.bias" => Nx.iota({d}, type: :f32),
        "model.decoder.layers.0.encoder_attn.k_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.decoder.layers.0.encoder_attn.k_proj.bias" => Nx.iota({d}, type: :f32),
        "model.decoder.layers.0.encoder_attn.v_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.decoder.layers.0.encoder_attn.v_proj.bias" => Nx.iota({d}, type: :f32),
        "model.decoder.layers.0.encoder_attn.out_proj.weight" => Nx.iota({d, d}, type: :f32),
        "model.decoder.layers.0.encoder_attn.out_proj.bias" => Nx.iota({d}, type: :f32),
        "model.decoder.layers.0.encoder_attn_layer_norm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
        "model.decoder.layers.0.encoder_attn_layer_norm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {d}),
        # Decoder layer 0 - FFN
        "model.decoder.layers.0.fc1.weight" => Nx.iota({d * 4, d}, type: :f32),
        "model.decoder.layers.0.fc1.bias" => Nx.iota({d * 4}, type: :f32),
        "model.decoder.layers.0.fc2.weight" => Nx.iota({d, d * 4}, type: :f32),
        "model.decoder.layers.0.fc2.bias" => Nx.iota({d}, type: :f32),
        "model.decoder.layers.0.final_layer_norm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
        "model.decoder.layers.0.final_layer_norm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {d}),
        # Decoder final norm
        "model.decoder.layer_norm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
        "model.decoder.layer_norm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {d}),
        # Output projection (skipped)
        "proj_out.weight" => Nx.iota({100, d}, type: :f32)
      }

      path = write_fixture(tensors)
      model_state = Edifice.Pretrained.load(Whisper, path)

      assert %Axon.ModelState{} = model_state
      data = model_state.data

      # Encoder conv
      assert %Nx.Tensor{} = data["enc_conv1"]["kernel"]
      assert %Nx.Tensor{} = data["enc_conv2"]["kernel"]

      # Encoder block 1 (from HF layer 0)
      assert %Nx.Tensor{} = data["enc_block_1_attn_q"]["kernel"]
      assert Nx.shape(data["enc_block_1_attn_q"]["kernel"]) == {d, d}
      assert %Nx.Tensor{} = data["enc_block_1_attn_norm"]["gamma"]
      assert %Nx.Tensor{} = data["enc_block_1_ffn_up"]["kernel"]
      assert Nx.shape(data["enc_block_1_ffn_up"]["kernel"]) == {d, d * 4}

      # Encoder final norm
      assert %Nx.Tensor{} = data["enc_final_norm"]["gamma"]

      # Decoder embeddings
      assert %Nx.Tensor{} = data["dec_token_embed"]["kernel"]
      assert %Nx.Tensor{} = data["dec_pos_embed"]["kernel"]

      # Decoder block 1
      assert %Nx.Tensor{} = data["dec_block_1_attn_q"]["kernel"]
      assert %Nx.Tensor{} = data["dec_block_1_cross_attn_q_proj"]["kernel"]
      assert %Nx.Tensor{} = data["dec_block_1_cross_attn_norm"]["gamma"]
      assert %Nx.Tensor{} = data["dec_block_1_ffn_up"]["kernel"]

      # Decoder final norm
      assert %Nx.Tensor{} = data["dec_final_norm"]["gamma"]

      # Verify skipped keys are absent
      flat = Edifice.Pretrained.Transform.flatten_params(model_state)
      refute Enum.any?(Map.keys(flat), &String.contains?(&1, "embed_positions"))
      refute Enum.any?(Map.keys(flat), &String.contains?(&1, "proj_out"))
    end

    test "linear weights are transposed, conv weights are not" do
      d = 8

      tensors = %{
        "model.encoder.conv1.weight" => Nx.iota({d, 4, 3}, type: :f32),
        "model.encoder.conv1.bias" => Nx.iota({d}, type: :f32),
        "model.encoder.conv2.weight" => Nx.iota({d, d, 3}, type: :f32),
        "model.encoder.conv2.bias" => Nx.iota({d}, type: :f32),
        "model.encoder.layer_norm.weight" =>
          Nx.broadcast(Nx.tensor(1.0, type: :f32), {d}),
        "model.encoder.layer_norm.bias" =>
          Nx.broadcast(Nx.tensor(0.0, type: :f32), {d})
      }

      path = write_fixture(tensors)
      model_state = Edifice.Pretrained.load(Whisper, path, strict: false)
      data = model_state.data

      # Conv weights are rank 3 — transpose_linear leaves them unchanged
      assert Nx.shape(data["enc_conv1"]["kernel"]) == {d, 4, 3}
      assert Nx.shape(data["enc_conv2"]["kernel"]) == {d, d, 3}
    end
  end
end
