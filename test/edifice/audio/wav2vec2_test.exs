defmodule Edifice.Audio.Wav2Vec2Test do
  use ExUnit.Case, async: true
  @moduletag :audio

  import Edifice.TestHelpers

  alias Edifice.Audio.Wav2Vec2

  @moduletag timeout: 120_000

  @batch 2
  # 16 kHz * 0.1 seconds = 1600 samples (small for testing)
  @samples 1600

  @small_opts [
    hidden_dim: 16,
    encoder_layers: 1,
    num_heads: 4,
    ffn_dim: 32,
    dropout: 0.0,
    conv_pos_kernel: 8,
    conv_pos_groups: 4,
    cnn_channels: 16,
    num_codebook_groups: 2,
    codebook_entries: 16,
    codevector_dim: 16
  ]

  defp random_waveform(batch \\ @batch, samples \\ @samples) do
    key = Nx.Random.key(42)
    {waveform, _key} = Nx.Random.normal(key, shape: {batch, samples})
    waveform
  end

  describe "build/1 encoder" do
    test "produces temporal features from waveform" do
      {encoder, _quantizer} = Wav2Vec2.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(encoder)

      waveform = random_waveform()
      template = %{"waveform" => Nx.template({@batch, @samples}, :f32)}
      params = init_fn.(template, %{})
      output = predict_fn.(params, %{"waveform" => waveform})

      # After CNN with total stride 320: T = floor((1600 - receptive_field) / 320) + 1
      # The exact T depends on valid padding; just check it's [batch, T, hidden_dim]
      {b, _t, d} = Nx.shape(output)
      assert b == @batch
      assert d == 16
      assert_finite!(output)
    end

    test "works with batch_size=1" do
      {encoder, _quantizer} = Wav2Vec2.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(encoder)

      waveform = random_waveform(1)
      template = %{"waveform" => Nx.template({1, @samples}, :f32)}
      params = init_fn.(template, %{})
      output = predict_fn.(params, %{"waveform" => waveform})

      {b, _t, d} = Nx.shape(output)
      assert b == 1
      assert d == 16
      assert_finite!(output)
    end

    test "different hidden dimensions" do
      opts = Keyword.merge(@small_opts, hidden_dim: 32, conv_pos_groups: 4)
      {encoder, _quantizer} = Wav2Vec2.build(opts)
      {init_fn, predict_fn} = Axon.build(encoder)

      waveform = random_waveform()
      template = %{"waveform" => Nx.template({@batch, @samples}, :f32)}
      params = init_fn.(template, %{})
      output = predict_fn.(params, %{"waveform" => waveform})

      {_b, _t, d} = Nx.shape(output)
      assert d == 32
      assert_finite!(output)
    end

    test "single transformer layer" do
      opts = Keyword.put(@small_opts, :encoder_layers, 1)
      {encoder, _quantizer} = Wav2Vec2.build(opts)
      {init_fn, predict_fn} = Axon.build(encoder)

      waveform = random_waveform()
      template = %{"waveform" => Nx.template({@batch, @samples}, :f32)}
      params = init_fn.(template, %{})
      output = predict_fn.(params, %{"waveform" => waveform})

      assert_finite!(output)
    end
  end

  describe "build/1 quantizer" do
    test "produces quantized vectors" do
      {_encoder, quantizer} = Wav2Vec2.build(@small_opts)
      {init_fn, predict_fn} = Axon.build(quantizer)

      key = Nx.Random.key(99)
      {cnn_features, _} = Nx.Random.normal(key, shape: {@batch, 4, 16})
      template = %{"cnn_features" => Nx.template({@batch, 4, 16}, :f32)}
      params = init_fn.(template, %{})
      output = predict_fn.(params, %{"cnn_features" => cnn_features})

      {b, t, d} = Nx.shape(output)
      assert b == @batch
      assert t == 4
      assert d == 16
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns hidden_dim" do
      assert Wav2Vec2.output_size(hidden_dim: 768) == 768
    end

    test "returns default hidden_dim" do
      assert Wav2Vec2.output_size([]) == 768
    end
  end
end
