defmodule Edifice.Audio.WhisperTest do
  use ExUnit.Case, async: true
  @moduletag :audio

  alias Edifice.Audio.Whisper

  @batch_size 2
  @n_mels 80
  @audio_len 64
  @dec_len 8
  @hidden_dim 64
  @num_heads 4
  @num_layers 2
  @ffn_dim 128
  @vocab_size 32

  @small_opts [
    n_mels: @n_mels,
    max_audio_len: @audio_len,
    hidden_dim: @hidden_dim,
    encoder_layers: @num_layers,
    decoder_layers: @num_layers,
    num_heads: @num_heads,
    ffn_dim: @ffn_dim,
    vocab_size: @vocab_size,
    max_dec_len: @dec_len
  ]

  describe "Whisper.build/1" do
    test "produces encoder and decoder tuple" do
      {encoder, decoder} = Whisper.build(@small_opts)
      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end
  end

  describe "Whisper.build_encoder/1" do
    test "produces correct output shape" do
      encoder = Whisper.build_encoder(@small_opts)
      {init_fn, predict_fn} = Axon.build(encoder, mode: :inference)

      params =
        init_fn.(
          %{"mel_spectrogram" => Nx.template({@batch_size, @n_mels, @audio_len}, :f32)},
          Axon.ModelState.empty()
        )

      mel = Nx.broadcast(0.0, {@batch_size, @n_mels, @audio_len})
      output = predict_fn.(params, %{"mel_spectrogram" => mel})

      # Conv stem stride=2 halves the time dimension
      expected_enc_len = div(@audio_len, 2)
      assert Nx.shape(output) == {@batch_size, expected_enc_len, @hidden_dim}
    end

    test "output values are finite" do
      encoder = Whisper.build_encoder(@small_opts)
      {init_fn, predict_fn} = Axon.build(encoder, mode: :inference)

      params =
        init_fn.(
          %{"mel_spectrogram" => Nx.template({@batch_size, @n_mels, @audio_len}, :f32)},
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)
      {mel, _} = Nx.Random.normal(key, shape: {@batch_size, @n_mels, @audio_len})
      output = predict_fn.(params, %{"mel_spectrogram" => mel})

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end

  describe "Whisper.build_decoder/1" do
    test "produces correct output shape" do
      decoder = Whisper.build_decoder(@small_opts)
      {init_fn, predict_fn} = Axon.build(decoder, mode: :inference)

      enc_len = div(@audio_len, 2)

      params =
        init_fn.(
          %{
            "token_ids" => Nx.template({@batch_size, @dec_len}, :s64),
            "encoder_output" => Nx.template({@batch_size, enc_len, @hidden_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      tokens = Nx.broadcast(0, {@batch_size, @dec_len}) |> Nx.as_type(:s64)
      enc_out = Nx.broadcast(0.0, {@batch_size, enc_len, @hidden_dim})

      output =
        predict_fn.(params, %{
          "token_ids" => tokens,
          "encoder_output" => enc_out
        })

      assert Nx.shape(output) == {@batch_size, @dec_len, @vocab_size}
    end

    test "output values are finite" do
      decoder = Whisper.build_decoder(@small_opts)
      {init_fn, predict_fn} = Axon.build(decoder, mode: :inference)

      enc_len = div(@audio_len, 2)

      params =
        init_fn.(
          %{
            "token_ids" => Nx.template({@batch_size, @dec_len}, :s64),
            "encoder_output" => Nx.template({@batch_size, enc_len, @hidden_dim}, :f32)
          },
          Axon.ModelState.empty()
        )

      tokens = Nx.broadcast(1, {@batch_size, @dec_len}) |> Nx.as_type(:s64)

      key = Nx.Random.key(99)
      {enc_out, _} = Nx.Random.normal(key, shape: {@batch_size, enc_len, @hidden_dim})

      output =
        predict_fn.(params, %{
          "token_ids" => tokens,
          "encoder_output" => enc_out
        })

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end

  describe "Whisper.build/1 with minimal configuration" do
    test "works with 1 layer" do
      {encoder, decoder} =
        Whisper.build(
          n_mels: 40,
          max_audio_len: 32,
          hidden_dim: 32,
          encoder_layers: 1,
          decoder_layers: 1,
          num_heads: 2,
          ffn_dim: 64,
          vocab_size: 16,
          max_dec_len: 4
        )

      # Encoder forward
      {enc_init, enc_predict} = Axon.build(encoder, mode: :inference)

      enc_params =
        enc_init.(
          %{"mel_spectrogram" => Nx.template({1, 40, 32}, :f32)},
          Axon.ModelState.empty()
        )

      mel = Nx.broadcast(0.0, {1, 40, 32})
      enc_out = enc_predict.(enc_params, %{"mel_spectrogram" => mel})
      assert Nx.shape(enc_out) == {1, 16, 32}

      # Decoder forward
      {dec_init, dec_predict} = Axon.build(decoder, mode: :inference)

      dec_params =
        dec_init.(
          %{
            "token_ids" => Nx.template({1, 4}, :s64),
            "encoder_output" => Nx.template({1, 16, 32}, :f32)
          },
          Axon.ModelState.empty()
        )

      tokens = Nx.broadcast(0, {1, 4}) |> Nx.as_type(:s64)

      logits =
        dec_predict.(dec_params, %{
          "token_ids" => tokens,
          "encoder_output" => enc_out
        })

      assert Nx.shape(logits) == {1, 4, 16}
    end
  end

  describe "Whisper.output_size/1" do
    test "returns vocab size" do
      assert Whisper.output_size(vocab_size: 1024) == 1024
      assert Whisper.output_size(vocab_size: 51_865) == 51_865
      # default
      assert Whisper.output_size() == 51_865
    end
  end

  describe "Edifice.build/2 integration" do
    test "can build Whisper via registry" do
      {encoder, decoder} = Edifice.build(:whisper, @small_opts)
      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end
  end
end
