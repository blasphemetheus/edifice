defmodule Edifice.Pretrained.RoundTripTest do
  @moduledoc """
  Round-trip tests for pretrained weight loading.

  For each reference architecture (ViT, Whisper, ConvNeXt):
  1. Build a real (small) Edifice model and init random weights
  2. Flatten the ModelState to get all Axon parameter paths
  3. Reverse-map those params to HuggingFace checkpoint format
  4. Write the HF checkpoint to a SafeTensors file
  5. Load it back using the forward key map
  6. Assert every parameter matches the original
  """
  use ExUnit.Case, async: true

  alias Edifice.Pretrained.Transform

  setup do
    assert Code.ensure_loaded?(Safetensors),
           "safetensors package must be available for these tests"

    :ok
  end

  defp write_fixture(tensors) do
    path =
      Path.join(
        System.tmp_dir!(),
        "edifice_roundtrip_#{System.unique_integer([:positive])}.safetensors"
      )

    Safetensors.write!(path, tensors)
    ExUnit.Callbacks.on_exit(fn -> File.rm(path) end)
    path
  end

  defp assert_all_close(a, b, opts \\ []) do
    atol = Keyword.get(opts, :atol, 1.0e-5)

    assert Nx.shape(a) == Nx.shape(b),
           "Shape mismatch: #{inspect(Nx.shape(a))} vs #{inspect(Nx.shape(b))}"

    diff = a |> Nx.subtract(b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

    assert diff < atol,
           "Max absolute difference #{diff} exceeds tolerance #{atol}"
  end

  # ============================================================================
  # ViT Round-Trip
  # ============================================================================

  describe "ViT round-trip" do
    @vit_opts [
      image_size: 8,
      patch_size: 4,
      embed_dim: 16,
      depth: 2,
      num_heads: 2,
      num_classes: 5,
      in_channels: 3
    ]

    test "all model params survive HF→load→compare cycle" do
      # 1. Build model, init params
      model = Edifice.Vision.ViT.build(@vit_opts)
      {init_fn, _} = Axon.build(model, mode: :inference)
      template = %{"image" => Nx.template({1, 3, 8, 8}, :f32)}
      original_state = init_fn.(template, Axon.ModelState.empty())
      original_flat = Transform.flatten_params(original_state)

      # 2. Reverse-map to HF checkpoint format
      hf_checkpoint = vit_axon_to_hf(original_flat, @vit_opts)

      # 3. Write → Load
      path = write_fixture(hf_checkpoint)

      loaded_state =
        Edifice.Pretrained.load(
          Edifice.Pretrained.KeyMaps.ViT,
          path,
          strict: false
        )

      loaded_flat = Transform.flatten_params(loaded_state)

      # 4. Assert all original params are present and match
      # Skip biases for CLS token and position embed Dense layers — HF has no bias for these
      skip_keys = MapSet.new(["cls_token_proj.bias", "pos_embed_proj.bias"])

      for {axon_key, original_tensor} <- original_flat,
          axon_key not in skip_keys do
        assert Map.has_key?(loaded_flat, axon_key),
               "Missing param after round-trip: #{axon_key}"

        assert_all_close(original_tensor, loaded_flat[axon_key])
      end
    end
  end

  # Reverse-maps Axon ViT flat params to HF checkpoint format
  defp vit_axon_to_hf(flat_params, opts) do
    depth = Keyword.fetch!(opts, :depth)
    embed_dim = Keyword.fetch!(opts, :embed_dim)

    Enum.flat_map(flat_params, fn {key, tensor} ->
      vit_reverse_key(key, tensor, depth, embed_dim)
    end)
    |> Map.new()
  end

  defp vit_reverse_key("patch_embed_proj.kernel", t, _depth, _dim),
    do: [{"vit.embeddings.patch_embeddings.projection.weight", Nx.transpose(t)}]

  defp vit_reverse_key("patch_embed_proj.bias", t, _depth, _dim),
    do: [{"vit.embeddings.patch_embeddings.projection.bias", t}]

  defp vit_reverse_key("cls_token_proj.kernel", t, _depth, _dim) do
    # Axon: {1, D} → HF: {1, 1, D}
    {rows, cols} = Nx.shape(t)
    [{"vit.embeddings.cls_token", Nx.reshape(t, {1, rows, cols})}]
  end

  defp vit_reverse_key("pos_embed_proj.kernel", t, _depth, _dim) do
    # Axon: {S, D} → HF: {1, S, D}
    {seq, cols} = Nx.shape(t)
    [{"vit.embeddings.position_embeddings", Nx.reshape(t, {1, seq, cols})}]
  end

  defp vit_reverse_key("final_norm.gamma", t, _depth, _dim),
    do: [{"vit.layernorm.weight", t}]

  defp vit_reverse_key("final_norm.beta", t, _depth, _dim),
    do: [{"vit.layernorm.bias", t}]

  defp vit_reverse_key("classifier.kernel", t, _depth, _dim),
    do: [{"classifier.weight", Nx.transpose(t)}]

  defp vit_reverse_key("classifier.bias", t, _depth, _dim),
    do: [{"classifier.bias", t}]

  defp vit_reverse_key(key, tensor, _depth, embed_dim) do
    cond do
      match = Regex.run(~r/^block_(\d+)_attn_qkv\.kernel$/, key) ->
        [_, idx] = match
        # Axon kernel: {D, 3*D} (in, out) → split along axis 1 into 3 x {D, D}, transpose each → {D, D}
        q = Nx.slice(tensor, [0, 0], [embed_dim, embed_dim]) |> Nx.transpose()
        k = Nx.slice(tensor, [0, embed_dim], [embed_dim, embed_dim]) |> Nx.transpose()
        v = Nx.slice(tensor, [0, 2 * embed_dim], [embed_dim, embed_dim]) |> Nx.transpose()

        [
          {"vit.encoder.layer.#{idx}.attention.attention.query.weight", q},
          {"vit.encoder.layer.#{idx}.attention.attention.key.weight", k},
          {"vit.encoder.layer.#{idx}.attention.attention.value.weight", v}
        ]

      match = Regex.run(~r/^block_(\d+)_attn_qkv\.bias$/, key) ->
        [_, idx] = match
        q = Nx.slice(tensor, [0], [embed_dim])
        k = Nx.slice(tensor, [embed_dim], [embed_dim])
        v = Nx.slice(tensor, [2 * embed_dim], [embed_dim])

        [
          {"vit.encoder.layer.#{idx}.attention.attention.query.bias", q},
          {"vit.encoder.layer.#{idx}.attention.attention.key.bias", k},
          {"vit.encoder.layer.#{idx}.attention.attention.value.bias", v}
        ]

      match = Regex.run(~r/^block_(\d+)_attn_proj\.kernel$/, key) ->
        [_, idx] = match
        [{"vit.encoder.layer.#{idx}.attention.output.dense.weight", Nx.transpose(tensor)}]

      match = Regex.run(~r/^block_(\d+)_attn_proj\.bias$/, key) ->
        [_, idx] = match
        [{"vit.encoder.layer.#{idx}.attention.output.dense.bias", tensor}]

      match = Regex.run(~r/^block_(\d+)_norm1\.gamma$/, key) ->
        [_, idx] = match
        [{"vit.encoder.layer.#{idx}.layernorm_before.weight", tensor}]

      match = Regex.run(~r/^block_(\d+)_norm1\.beta$/, key) ->
        [_, idx] = match
        [{"vit.encoder.layer.#{idx}.layernorm_before.bias", tensor}]

      match = Regex.run(~r/^block_(\d+)_norm2\.gamma$/, key) ->
        [_, idx] = match
        [{"vit.encoder.layer.#{idx}.layernorm_after.weight", tensor}]

      match = Regex.run(~r/^block_(\d+)_norm2\.beta$/, key) ->
        [_, idx] = match
        [{"vit.encoder.layer.#{idx}.layernorm_after.bias", tensor}]

      match = Regex.run(~r/^block_(\d+)_mlp_fc1\.kernel$/, key) ->
        [_, idx] = match
        [{"vit.encoder.layer.#{idx}.intermediate.dense.weight", Nx.transpose(tensor)}]

      match = Regex.run(~r/^block_(\d+)_mlp_fc1\.bias$/, key) ->
        [_, idx] = match
        [{"vit.encoder.layer.#{idx}.intermediate.dense.bias", tensor}]

      match = Regex.run(~r/^block_(\d+)_mlp_fc2\.kernel$/, key) ->
        [_, idx] = match
        [{"vit.encoder.layer.#{idx}.output.dense.weight", Nx.transpose(tensor)}]

      match = Regex.run(~r/^block_(\d+)_mlp_fc2\.bias$/, key) ->
        [_, idx] = match
        [{"vit.encoder.layer.#{idx}.output.dense.bias", tensor}]

      true ->
        # Skip non-trainable layers (dropout, activations, etc.)
        []
    end
  end

  # ============================================================================
  # Whisper Round-Trip
  # ============================================================================

  describe "Whisper round-trip" do
    @whisper_opts [
      n_mels: 8,
      hidden_dim: 16,
      encoder_layers: 1,
      decoder_layers: 1,
      num_heads: 2,
      ffn_dim: 32,
      vocab_size: 50,
      max_audio_len: 20,
      max_dec_len: 10
    ]

    test "encoder params survive HF→load→compare cycle" do
      encoder = Edifice.Audio.Whisper.build_encoder(@whisper_opts)
      {init_fn, _} = Axon.build(encoder, mode: :inference)

      template = %{"mel_spectrogram" => Nx.template({1, 8, 20}, :f32)}
      original_state = init_fn.(template, Axon.ModelState.empty())
      original_flat = Transform.flatten_params(original_state)

      hf_checkpoint = whisper_encoder_to_hf(original_flat, @whisper_opts)
      path = write_fixture(hf_checkpoint)

      loaded_state =
        Edifice.Pretrained.load(
          Edifice.Pretrained.KeyMaps.Whisper,
          path,
          strict: false
        )

      loaded_flat = Transform.flatten_params(loaded_state)

      # Check all encoder params
      enc_keys =
        Enum.filter(Map.keys(original_flat), &String.starts_with?(&1, "enc_"))

      for axon_key <- enc_keys do
        assert Map.has_key?(loaded_flat, axon_key),
               "Missing encoder param after round-trip: #{axon_key}"

        assert_all_close(original_flat[axon_key], loaded_flat[axon_key])
      end
    end

    test "decoder params survive HF→load→compare cycle" do
      decoder = Edifice.Audio.Whisper.build_decoder(@whisper_opts)
      {init_fn, _} = Axon.build(decoder, mode: :inference)
      enc_len = div(20, 2)

      template = %{
        "token_ids" => Nx.template({1, 10}, :s64),
        "encoder_output" => Nx.template({1, enc_len, 16}, :f32)
      }

      original_state = init_fn.(template, Axon.ModelState.empty())
      original_flat = Transform.flatten_params(original_state)

      hf_checkpoint = whisper_decoder_to_hf(original_flat, @whisper_opts)
      path = write_fixture(hf_checkpoint)

      loaded_state =
        Edifice.Pretrained.load(
          Edifice.Pretrained.KeyMaps.Whisper,
          path,
          strict: false
        )

      loaded_flat = Transform.flatten_params(loaded_state)

      # Check all decoder params except dec_output_proj (Edifice-only, no HF source)
      dec_keys =
        original_flat
        |> Map.keys()
        |> Enum.filter(&String.starts_with?(&1, "dec_"))
        |> Enum.reject(&String.starts_with?(&1, "dec_output_proj"))

      for axon_key <- dec_keys do
        assert Map.has_key?(loaded_flat, axon_key),
               "Missing decoder param after round-trip: #{axon_key}"

        assert_all_close(original_flat[axon_key], loaded_flat[axon_key])
      end
    end
  end

  defp whisper_encoder_to_hf(flat_params, _opts) do
    Enum.flat_map(flat_params, fn {key, tensor} ->
      whisper_enc_reverse(key, tensor)
    end)
    |> Map.new()
  end

  # Encoder conv stem
  defp whisper_enc_reverse("enc_conv1.kernel", t), do: [{"model.encoder.conv1.weight", t}]
  defp whisper_enc_reverse("enc_conv1.bias", t), do: [{"model.encoder.conv1.bias", t}]
  defp whisper_enc_reverse("enc_conv2.kernel", t), do: [{"model.encoder.conv2.weight", t}]
  defp whisper_enc_reverse("enc_conv2.bias", t), do: [{"model.encoder.conv2.bias", t}]

  # Encoder final norm
  defp whisper_enc_reverse("enc_final_norm.gamma", t),
    do: [{"model.encoder.layer_norm.weight", t}]

  defp whisper_enc_reverse("enc_final_norm.beta", t),
    do: [{"model.encoder.layer_norm.bias", t}]

  defp whisper_enc_reverse(key, tensor) do
    # enc_block_{i}_... → model.encoder.layers.{i-1}....
    case Regex.run(~r/^enc_block_(\d+)_(.+)$/, key) do
      [_, idx_str, rest] ->
        hf_idx = String.to_integer(idx_str) - 1
        whisper_enc_block_reverse(hf_idx, rest, tensor)

      nil ->
        []
    end
  end

  defp whisper_enc_block_reverse(i, "attn_q.kernel", t),
    do: [{"model.encoder.layers.#{i}.self_attn.q_proj.weight", Nx.transpose(t)}]

  defp whisper_enc_block_reverse(i, "attn_q.bias", t),
    do: [{"model.encoder.layers.#{i}.self_attn.q_proj.bias", t}]

  defp whisper_enc_block_reverse(i, "attn_k.kernel", t),
    do: [{"model.encoder.layers.#{i}.self_attn.k_proj.weight", Nx.transpose(t)}]

  defp whisper_enc_block_reverse(i, "attn_k.bias", t),
    do: [{"model.encoder.layers.#{i}.self_attn.k_proj.bias", t}]

  defp whisper_enc_block_reverse(i, "attn_v.kernel", t),
    do: [{"model.encoder.layers.#{i}.self_attn.v_proj.weight", Nx.transpose(t)}]

  defp whisper_enc_block_reverse(i, "attn_v.bias", t),
    do: [{"model.encoder.layers.#{i}.self_attn.v_proj.bias", t}]

  defp whisper_enc_block_reverse(i, "attn_out.kernel", t),
    do: [{"model.encoder.layers.#{i}.self_attn.out_proj.weight", Nx.transpose(t)}]

  defp whisper_enc_block_reverse(i, "attn_out.bias", t),
    do: [{"model.encoder.layers.#{i}.self_attn.out_proj.bias", t}]

  defp whisper_enc_block_reverse(i, "attn_norm.gamma", t),
    do: [{"model.encoder.layers.#{i}.self_attn_layer_norm.weight", t}]

  defp whisper_enc_block_reverse(i, "attn_norm.beta", t),
    do: [{"model.encoder.layers.#{i}.self_attn_layer_norm.bias", t}]

  defp whisper_enc_block_reverse(i, "ffn_up.kernel", t),
    do: [{"model.encoder.layers.#{i}.fc1.weight", Nx.transpose(t)}]

  defp whisper_enc_block_reverse(i, "ffn_up.bias", t),
    do: [{"model.encoder.layers.#{i}.fc1.bias", t}]

  defp whisper_enc_block_reverse(i, "ffn_down.kernel", t),
    do: [{"model.encoder.layers.#{i}.fc2.weight", Nx.transpose(t)}]

  defp whisper_enc_block_reverse(i, "ffn_down.bias", t),
    do: [{"model.encoder.layers.#{i}.fc2.bias", t}]

  defp whisper_enc_block_reverse(i, "ffn_norm.gamma", t),
    do: [{"model.encoder.layers.#{i}.final_layer_norm.weight", t}]

  defp whisper_enc_block_reverse(i, "ffn_norm.beta", t),
    do: [{"model.encoder.layers.#{i}.final_layer_norm.bias", t}]

  defp whisper_enc_block_reverse(_i, _rest, _t), do: []

  defp whisper_decoder_to_hf(flat_params, _opts) do
    Enum.flat_map(flat_params, fn {key, tensor} ->
      whisper_dec_reverse(key, tensor)
    end)
    |> Map.new()
  end

  defp whisper_dec_reverse("dec_token_embed.kernel", t),
    do: [{"model.decoder.embed_tokens.weight", t}]

  defp whisper_dec_reverse("dec_pos_embed.kernel", t),
    do: [{"model.decoder.embed_positions.weight", t}]

  defp whisper_dec_reverse("dec_final_norm.gamma", t),
    do: [{"model.decoder.layer_norm.weight", t}]

  defp whisper_dec_reverse("dec_final_norm.beta", t),
    do: [{"model.decoder.layer_norm.bias", t}]

  defp whisper_dec_reverse(key, tensor) do
    case Regex.run(~r/^dec_block_(\d+)_(.+)$/, key) do
      [_, idx_str, rest] ->
        hf_idx = String.to_integer(idx_str) - 1
        whisper_dec_block_reverse(hf_idx, rest, tensor)

      nil ->
        []
    end
  end

  # Decoder self-attention
  defp whisper_dec_block_reverse(i, "attn_q.kernel", t),
    do: [{"model.decoder.layers.#{i}.self_attn.q_proj.weight", Nx.transpose(t)}]

  defp whisper_dec_block_reverse(i, "attn_q.bias", t),
    do: [{"model.decoder.layers.#{i}.self_attn.q_proj.bias", t}]

  defp whisper_dec_block_reverse(i, "attn_k.kernel", t),
    do: [{"model.decoder.layers.#{i}.self_attn.k_proj.weight", Nx.transpose(t)}]

  defp whisper_dec_block_reverse(i, "attn_k.bias", t),
    do: [{"model.decoder.layers.#{i}.self_attn.k_proj.bias", t}]

  defp whisper_dec_block_reverse(i, "attn_v.kernel", t),
    do: [{"model.decoder.layers.#{i}.self_attn.v_proj.weight", Nx.transpose(t)}]

  defp whisper_dec_block_reverse(i, "attn_v.bias", t),
    do: [{"model.decoder.layers.#{i}.self_attn.v_proj.bias", t}]

  defp whisper_dec_block_reverse(i, "attn_out.kernel", t),
    do: [{"model.decoder.layers.#{i}.self_attn.out_proj.weight", Nx.transpose(t)}]

  defp whisper_dec_block_reverse(i, "attn_out.bias", t),
    do: [{"model.decoder.layers.#{i}.self_attn.out_proj.bias", t}]

  defp whisper_dec_block_reverse(i, "attn_norm.gamma", t),
    do: [{"model.decoder.layers.#{i}.self_attn_layer_norm.weight", t}]

  defp whisper_dec_block_reverse(i, "attn_norm.beta", t),
    do: [{"model.decoder.layers.#{i}.self_attn_layer_norm.bias", t}]

  # Decoder cross-attention
  defp whisper_dec_block_reverse(i, "cross_attn_q_proj.kernel", t),
    do: [{"model.decoder.layers.#{i}.encoder_attn.q_proj.weight", Nx.transpose(t)}]

  defp whisper_dec_block_reverse(i, "cross_attn_q_proj.bias", t),
    do: [{"model.decoder.layers.#{i}.encoder_attn.q_proj.bias", t}]

  defp whisper_dec_block_reverse(i, "cross_attn_k_proj.kernel", t),
    do: [{"model.decoder.layers.#{i}.encoder_attn.k_proj.weight", Nx.transpose(t)}]

  defp whisper_dec_block_reverse(i, "cross_attn_k_proj.bias", t),
    do: [{"model.decoder.layers.#{i}.encoder_attn.k_proj.bias", t}]

  defp whisper_dec_block_reverse(i, "cross_attn_v_proj.kernel", t),
    do: [{"model.decoder.layers.#{i}.encoder_attn.v_proj.weight", Nx.transpose(t)}]

  defp whisper_dec_block_reverse(i, "cross_attn_v_proj.bias", t),
    do: [{"model.decoder.layers.#{i}.encoder_attn.v_proj.bias", t}]

  defp whisper_dec_block_reverse(i, "cross_attn_out_proj.kernel", t),
    do: [{"model.decoder.layers.#{i}.encoder_attn.out_proj.weight", Nx.transpose(t)}]

  defp whisper_dec_block_reverse(i, "cross_attn_out_proj.bias", t),
    do: [{"model.decoder.layers.#{i}.encoder_attn.out_proj.bias", t}]

  defp whisper_dec_block_reverse(i, "cross_attn_norm.gamma", t),
    do: [{"model.decoder.layers.#{i}.encoder_attn_layer_norm.weight", t}]

  defp whisper_dec_block_reverse(i, "cross_attn_norm.beta", t),
    do: [{"model.decoder.layers.#{i}.encoder_attn_layer_norm.bias", t}]

  # Decoder FFN
  defp whisper_dec_block_reverse(i, "ffn_up.kernel", t),
    do: [{"model.decoder.layers.#{i}.fc1.weight", Nx.transpose(t)}]

  defp whisper_dec_block_reverse(i, "ffn_up.bias", t),
    do: [{"model.decoder.layers.#{i}.fc1.bias", t}]

  defp whisper_dec_block_reverse(i, "ffn_down.kernel", t),
    do: [{"model.decoder.layers.#{i}.fc2.weight", Nx.transpose(t)}]

  defp whisper_dec_block_reverse(i, "ffn_down.bias", t),
    do: [{"model.decoder.layers.#{i}.fc2.bias", t}]

  defp whisper_dec_block_reverse(i, "ffn_norm.gamma", t),
    do: [{"model.decoder.layers.#{i}.final_layer_norm.weight", t}]

  defp whisper_dec_block_reverse(i, "ffn_norm.beta", t),
    do: [{"model.decoder.layers.#{i}.final_layer_norm.bias", t}]

  defp whisper_dec_block_reverse(_i, _rest, _t), do: []

  # ============================================================================
  # ConvNeXt Round-Trip
  # ============================================================================

  describe "ConvNeXt round-trip" do
    @cnx_opts [
      image_size: 8,
      patch_size: 2,
      dims: [8, 16],
      depths: [1, 1],
      in_channels: 3,
      num_classes: 5
    ]

    test "all model params survive HF→load→compare cycle" do
      model = Edifice.Vision.ConvNeXt.build(@cnx_opts)
      {init_fn, _} = Axon.build(model, mode: :inference)
      template = Nx.template({1, 3, 8, 8}, :f32)
      original_state = init_fn.(template, Axon.ModelState.empty())
      original_flat = Transform.flatten_params(original_state)

      hf_checkpoint = convnext_axon_to_hf(original_flat)
      path = write_fixture(hf_checkpoint)

      loaded_state =
        Edifice.Pretrained.load(
          Edifice.Pretrained.KeyMaps.ConvNeXt,
          path,
          strict: false
        )

      loaded_flat = Transform.flatten_params(loaded_state)

      for {axon_key, original_tensor} <- original_flat do
        assert Map.has_key?(loaded_flat, axon_key),
               "Missing param after round-trip: #{axon_key}"

        assert_all_close(original_tensor, loaded_flat[axon_key])
      end
    end
  end

  defp convnext_axon_to_hf(flat_params) do
    Enum.flat_map(flat_params, fn {key, tensor} ->
      cnx_reverse_key(key, tensor)
    end)
    |> Map.new()
  end

  # Stem
  defp cnx_reverse_key("stem_conv.kernel", t),
    do: [{"convnext.embeddings.patch_embeddings.weight", Nx.transpose(t, axes: [3, 2, 0, 1])}]

  defp cnx_reverse_key("stem_conv.bias", t),
    do: [{"convnext.embeddings.patch_embeddings.bias", t}]

  defp cnx_reverse_key("stem_norm.gamma", t),
    do: [{"convnext.embeddings.layernorm.weight", t}]

  defp cnx_reverse_key("stem_norm.beta", t),
    do: [{"convnext.embeddings.layernorm.bias", t}]

  # Final norm + classifier
  defp cnx_reverse_key("final_norm.gamma", t),
    do: [{"convnext.layernorm.weight", t}]

  defp cnx_reverse_key("final_norm.beta", t),
    do: [{"convnext.layernorm.bias", t}]

  defp cnx_reverse_key("classifier.kernel", t),
    do: [{"classifier.weight", Nx.transpose(t)}]

  defp cnx_reverse_key("classifier.bias", t),
    do: [{"classifier.bias", t}]

  defp cnx_reverse_key(key, tensor) do
    cond do
      # Downsample norm
      match = Regex.run(~r/^downsample_(\d+)_norm\.gamma$/, key) ->
        [_, idx] = match
        hf_stage = String.to_integer(idx) + 1

        [{"convnext.encoder.stages.#{hf_stage}.downsampling_layer.0.weight", tensor}]

      match = Regex.run(~r/^downsample_(\d+)_norm\.beta$/, key) ->
        [_, idx] = match
        hf_stage = String.to_integer(idx) + 1

        [{"convnext.encoder.stages.#{hf_stage}.downsampling_layer.0.bias", tensor}]

      # Downsample conv
      match = Regex.run(~r/^downsample_(\d+)_conv\.kernel$/, key) ->
        [_, idx] = match
        hf_stage = String.to_integer(idx) + 1
        hf_t = Nx.transpose(tensor, axes: [3, 2, 0, 1])

        [{"convnext.encoder.stages.#{hf_stage}.downsampling_layer.1.weight", hf_t}]

      match = Regex.run(~r/^downsample_(\d+)_conv\.bias$/, key) ->
        [_, idx] = match
        hf_stage = String.to_integer(idx) + 1

        [{"convnext.encoder.stages.#{hf_stage}.downsampling_layer.1.bias", tensor}]

      # Block depthwise conv
      match = Regex.run(~r/^stage(\d+)_block(\d+)_dw_conv\.kernel$/, key) ->
        [_, s, b] = match
        hf_t = Nx.transpose(tensor, axes: [3, 2, 0, 1])

        [{"convnext.encoder.stages.#{s}.layers.#{b}.dwconv.weight", hf_t}]

      match = Regex.run(~r/^stage(\d+)_block(\d+)_dw_conv\.bias$/, key) ->
        [_, s, b] = match
        [{"convnext.encoder.stages.#{s}.layers.#{b}.dwconv.bias", tensor}]

      # Block layernorm
      match = Regex.run(~r/^stage(\d+)_block(\d+)_norm\.gamma$/, key) ->
        [_, s, b] = match
        [{"convnext.encoder.stages.#{s}.layers.#{b}.layernorm.weight", tensor}]

      match = Regex.run(~r/^stage(\d+)_block(\d+)_norm\.beta$/, key) ->
        [_, s, b] = match
        [{"convnext.encoder.stages.#{s}.layers.#{b}.layernorm.bias", tensor}]

      # Pointwise expand (Axon 1x1 conv → HF Linear)
      match = Regex.run(~r/^stage(\d+)_block(\d+)_pw_expand\.kernel$/, key) ->
        [_, s, b] = match
        {1, 1, inp, out} = Nx.shape(tensor)
        hf_t = tensor |> Nx.reshape({inp, out}) |> Nx.transpose()
        [{"convnext.encoder.stages.#{s}.layers.#{b}.pwconv1.weight", hf_t}]

      match = Regex.run(~r/^stage(\d+)_block(\d+)_pw_expand\.bias$/, key) ->
        [_, s, b] = match
        [{"convnext.encoder.stages.#{s}.layers.#{b}.pwconv1.bias", tensor}]

      # Pointwise project (Axon 1x1 conv → HF Linear)
      match = Regex.run(~r/^stage(\d+)_block(\d+)_pw_project\.kernel$/, key) ->
        [_, s, b] = match
        {1, 1, inp, out} = Nx.shape(tensor)
        hf_t = tensor |> Nx.reshape({inp, out}) |> Nx.transpose()
        [{"convnext.encoder.stages.#{s}.layers.#{b}.pwconv2.weight", hf_t}]

      match = Regex.run(~r/^stage(\d+)_block(\d+)_pw_project\.bias$/, key) ->
        [_, s, b] = match
        [{"convnext.encoder.stages.#{s}.layers.#{b}.pwconv2.bias", tensor}]

      # Layer scale gamma
      match =
          Regex.run(~r/^stage(\d+)_block(\d+)_layer_scale\.stage\d+_block\d+_gamma$/, key) ->
        [_, s, b] = match
        {1, 1, 1, dim} = Nx.shape(tensor)
        hf_t = Nx.reshape(tensor, {dim})

        [{"convnext.encoder.stages.#{s}.layers.#{b}.layer_scale_parameter", hf_t}]

      true ->
        []
    end
  end
end
