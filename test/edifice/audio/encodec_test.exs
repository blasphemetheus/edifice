defmodule Edifice.Audio.EnCodecTest do
  use ExUnit.Case, async: true
  @moduletag :audio
  # Conv-heavy model is slow on BinaryBackend (no EXLA in tests)
  @moduletag timeout: 300_000

  alias Edifice.Audio.EnCodec

  @batch_size 2
  @hidden_dim 16
  @num_codebooks 4
  @codebook_size 32
  @embedding_dim @hidden_dim * 16

  describe "EnCodec.build/1" do
    test "produces encoder and decoder tuple" do
      {encoder, decoder} = EnCodec.build(hidden_dim: @hidden_dim)

      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end
  end

  describe "EnCodec.build_encoder/1" do
    test "produces correct output shape" do
      encoder = EnCodec.build_encoder(hidden_dim: @hidden_dim)

      {init_fn, predict_fn} = Axon.build(encoder, mode: :inference)

      # Input: waveform [batch, 1, samples]
      # samples must be divisible by 320 (total downsampling factor)
      samples = 320 * 4

      params =
        init_fn.(
          %{"waveform" => Nx.template({@batch_size, 1, samples}, :f32)},
          Axon.ModelState.empty()
        )

      waveform = Nx.broadcast(0.0, {@batch_size, 1, samples})
      output = predict_fn.(params, %{"waveform" => waveform})

      # Output: [batch, T, dim] where T = samples / 320, dim = hidden_dim * 16
      expected_t = div(samples, 320)
      assert Nx.shape(output) == {@batch_size, expected_t, @embedding_dim}
    end
  end

  describe "EnCodec.build_decoder/1" do
    @tag :slow
    test "produces correct output shape" do
      decoder = EnCodec.build_decoder(hidden_dim: @hidden_dim)

      {init_fn, predict_fn} = Axon.build(decoder, mode: :inference)

      # Input: embeddings [batch, T, dim]
      t = 4

      params =
        init_fn.(
          %{"embeddings" => Nx.template({@batch_size, t, @embedding_dim}, :f32)},
          Axon.ModelState.empty()
        )

      embeddings = Nx.broadcast(0.0, {@batch_size, t, @embedding_dim})
      output = predict_fn.(params, %{"embeddings" => embeddings})

      # Output: [batch, 1, samples] where samples = T * 320
      expected_samples = t * 320
      assert Nx.shape(output) == {@batch_size, 1, expected_samples}
    end
  end

  describe "EnCodec.build_rvq/1" do
    test "returns RVQ configuration" do
      config =
        EnCodec.build_rvq(
          num_codebooks: @num_codebooks,
          codebook_size: @codebook_size,
          hidden_dim: @hidden_dim
        )

      assert config.num_codebooks == @num_codebooks
      assert config.codebook_size == @codebook_size
      assert config.embedding_dim == @embedding_dim
    end
  end

  describe "EnCodec.init_codebooks/4" do
    test "initializes codebooks with correct shape" do
      codebooks = EnCodec.init_codebooks(@num_codebooks, @codebook_size, @embedding_dim)

      assert Nx.shape(codebooks) == {@num_codebooks, @codebook_size, @embedding_dim}
      assert Nx.type(codebooks) == {:f, 32}
    end
  end

  describe "EnCodec.rvq_quantize/2" do
    test "produces correct shapes" do
      seq_len = 4
      z_e = Nx.broadcast(0.1, {@batch_size, seq_len, @embedding_dim})
      codebooks = EnCodec.init_codebooks(@num_codebooks, @codebook_size, @embedding_dim)

      {z_q, indices} = EnCodec.rvq_quantize(z_e, codebooks)

      # z_q should have same shape as z_e
      assert Nx.shape(z_q) == {@batch_size, seq_len, @embedding_dim}

      # indices: [batch, num_codebooks, seq_len]
      assert Nx.shape(indices) == {@batch_size, @num_codebooks, seq_len}
    end

    test "indices are in valid range" do
      seq_len = 4

      z_e =
        Nx.Random.uniform(Nx.Random.key(42), shape: {@batch_size, seq_len, @embedding_dim})
        |> elem(0)

      codebooks = EnCodec.init_codebooks(@num_codebooks, @codebook_size, @embedding_dim)

      {_z_q, indices} = EnCodec.rvq_quantize(z_e, codebooks)

      min_idx = Nx.reduce_min(indices) |> Nx.to_number()
      max_idx = Nx.reduce_max(indices) |> Nx.to_number()

      assert min_idx >= 0
      assert max_idx < @codebook_size
    end
  end

  describe "EnCodec.rvq_dequantize/2" do
    test "produces correct shape" do
      seq_len = 4
      indices = Nx.broadcast(0, {@batch_size, @num_codebooks, seq_len}) |> Nx.as_type(:s32)
      codebooks = EnCodec.init_codebooks(@num_codebooks, @codebook_size, @embedding_dim)

      z_q = EnCodec.rvq_dequantize(indices, codebooks)

      assert Nx.shape(z_q) == {@batch_size, seq_len, @embedding_dim}
    end

    test "roundtrip quantize-dequantize produces valid embeddings" do
      seq_len = 4

      z_e =
        Nx.Random.uniform(Nx.Random.key(123), shape: {@batch_size, seq_len, @embedding_dim})
        |> elem(0)

      codebooks = EnCodec.init_codebooks(@num_codebooks, @codebook_size, @embedding_dim)

      {_z_q_st, indices} = EnCodec.rvq_quantize(z_e, codebooks)
      z_q_reconstructed = EnCodec.rvq_dequantize(indices, codebooks)

      # Dequantized should have valid (non-NaN, non-Inf) values
      assert Nx.any(Nx.is_nan(z_q_reconstructed)) |> Nx.to_number() == 0
      assert Nx.any(Nx.is_infinity(z_q_reconstructed)) |> Nx.to_number() == 0
    end
  end

  describe "EnCodec.spectral_loss/2" do
    test "computes loss" do
      shape = {@batch_size, 1, 640}
      reconstruction = Nx.broadcast(0.5, shape)
      target = Nx.broadcast(0.0, shape)

      loss = EnCodec.spectral_loss(reconstruction, target)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end
  end

  describe "EnCodec.commitment_loss/2" do
    test "computes loss" do
      shape = {@batch_size, 4, @embedding_dim}
      z_e = Nx.broadcast(1.0, shape)
      z_q = Nx.broadcast(0.5, shape)

      loss = EnCodec.commitment_loss(z_e, z_q)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end
  end

  describe "EnCodec.loss/5" do
    test "computes combined loss" do
      waveform_shape = {@batch_size, 1, 640}
      embedding_shape = {@batch_size, 4, @embedding_dim}

      reconstruction = Nx.broadcast(0.5, waveform_shape)
      target = Nx.broadcast(0.0, waveform_shape)
      z_e = Nx.broadcast(1.0, embedding_shape)
      z_q = Nx.broadcast(0.5, embedding_shape)

      loss = EnCodec.loss(reconstruction, target, z_e, z_q)

      assert Nx.shape(loss) == {}
      assert Nx.to_number(loss) > 0
    end
  end

  describe "EnCodec.output_size/1" do
    test "returns embedding dimension" do
      assert EnCodec.output_size(hidden_dim: 128) == 128 * 16
      assert EnCodec.output_size(hidden_dim: 64) == 64 * 16
      # default
      assert EnCodec.output_size() == 128 * 16
    end
  end

  describe "Edifice.build/2 integration" do
    test "can build EnCodec via registry" do
      {encoder, decoder} = Edifice.build(:encodec, hidden_dim: @hidden_dim)

      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end
  end
end
