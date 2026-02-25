defmodule Edifice.Generative.CogVideoXTest do
  use ExUnit.Case, async: true
  # 3D convolutions are slow on BinaryBackend (no EXLA in tests)
  @moduletag timeout: 300_000

  alias Edifice.Generative.CogVideoX

  # Use smaller dimensions for faster testing
  @batch 1
  @num_frames 9
  @latent_frames 3
  @in_channels 3
  @latent_channels 4
  @height 16
  @width 16
  @text_len 4
  @hidden_size 32
  @num_heads 4
  @num_layers 2
  @text_hidden_size 16

  @vae_opts [
    in_channels: @in_channels,
    latent_channels: @latent_channels,
    num_frames: @num_frames,
    spatial_downsample: 4,
    temporal_downsample: 2,
    base_channels: 8
  ]

  @transformer_opts [
    patch_size: [1, 2, 2],
    hidden_size: @hidden_size,
    num_heads: @num_heads,
    num_layers: @num_layers,
    text_hidden_size: @text_hidden_size,
    mlp_ratio: 2.0
  ]

  # ============================================================================
  # build_vae/1
  # ============================================================================

  describe "build_vae/1" do
    test "returns encoder and decoder tuple" do
      {encoder, decoder} = CogVideoX.build_vae(@vae_opts)
      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end

    test "encoder produces latent representation" do
      {encoder, _decoder} = CogVideoX.build_vae(@vae_opts)
      {init_fn, predict_fn} = Axon.build(encoder)

      template = %{"video" => Nx.template({@batch, @num_frames, @in_channels, @height, @width}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {video, _key} = Nx.Random.uniform(key, shape: {@batch, @num_frames, @in_channels, @height, @width})

      latent = predict_fn.(params, %{"video" => video})

      # Should have reduced spatial and temporal dimensions
      {_batch, frames, channels, h, w} = Nx.shape(latent)
      assert frames < @num_frames
      assert channels == @latent_channels
      assert h < @height
      assert w < @width
    end

    test "encoder output contains finite values" do
      {encoder, _decoder} = CogVideoX.build_vae(@vae_opts)
      {init_fn, predict_fn} = Axon.build(encoder)

      template = %{"video" => Nx.template({@batch, @num_frames, @in_channels, @height, @width}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {video, _key} = Nx.Random.uniform(key, shape: {@batch, @num_frames, @in_channels, @height, @width})

      latent = predict_fn.(params, %{"video" => video})

      assert Nx.all(Nx.is_nan(latent) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(latent) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "decoder produces output with correct channels" do
      {_encoder, decoder} = CogVideoX.build_vae(@vae_opts)
      {init_fn, predict_fn} = Axon.build(decoder)

      # Decoder input shape (post-encoding dimensions)
      template = %{"latent" => Nx.template({@batch, @latent_frames, @latent_channels, 4, 4}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {latent, _key} = Nx.Random.uniform(key, shape: {@batch, @latent_frames, @latent_channels, 4, 4})

      output = predict_fn.(params, %{"latent" => latent})

      # Output should have 3 channels (RGB)
      assert Nx.axis_size(output, 2) == @in_channels
    end
  end

  # ============================================================================
  # build_transformer/1
  # ============================================================================

  describe "build_transformer/1" do
    test "returns an Axon model" do
      model = CogVideoX.build_transformer(@transformer_opts)
      assert %Axon{} = model
    end

    test "forward pass produces output" do
      model = CogVideoX.build_transformer(@transformer_opts)
      {init_fn, predict_fn} = Axon.build(model)

      # Video latent: [batch, frames, channels, h, w]
      video_latent_shape = {@batch, @latent_frames, @latent_channels, 4, 4}
      text_embed_shape = {@batch, @text_len, @text_hidden_size}
      timestep_shape = {@batch}

      template = %{
        "video_latent" => Nx.template(video_latent_shape, :f32),
        "text_embed" => Nx.template(text_embed_shape, :f32),
        "timestep" => Nx.template(timestep_shape, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {video_latent, key} = Nx.Random.uniform(key, shape: video_latent_shape)
      {text_embed, key} = Nx.Random.uniform(key, shape: text_embed_shape)
      {timestep, _key} = Nx.Random.uniform(key, shape: timestep_shape)

      output = predict_fn.(params, %{
        "video_latent" => video_latent,
        "text_embed" => text_embed,
        "timestep" => timestep
      })

      # Output should be video tokens (excluding text)
      assert Nx.rank(output) == 3
      assert Nx.axis_size(output, 0) == @batch
    end

    test "output contains finite values" do
      model = CogVideoX.build_transformer(@transformer_opts)
      {init_fn, predict_fn} = Axon.build(model)

      video_latent_shape = {@batch, @latent_frames, @latent_channels, 4, 4}
      text_embed_shape = {@batch, @text_len, @text_hidden_size}
      timestep_shape = {@batch}

      template = %{
        "video_latent" => Nx.template(video_latent_shape, :f32),
        "text_embed" => Nx.template(text_embed_shape, :f32),
        "timestep" => Nx.template(timestep_shape, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {video_latent, key} = Nx.Random.uniform(key, shape: video_latent_shape)
      {text_embed, key} = Nx.Random.uniform(key, shape: text_embed_shape)
      {timestep, _key} = Nx.Random.uniform(key, shape: timestep_shape)

      output = predict_fn.(params, %{
        "video_latent" => video_latent,
        "text_embed" => text_embed,
        "timestep" => timestep
      })

      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "works with single layer" do
      opts = Keyword.put(@transformer_opts, :num_layers, 1)
      model = CogVideoX.build_transformer(opts)
      assert %Axon{} = model
    end
  end

  # ============================================================================
  # build/1 (full pipeline)
  # ============================================================================

  describe "build/1" do
    test "returns an Axon model" do
      model = CogVideoX.build(@transformer_opts)
      assert %Axon{} = model
    end
  end

  # ============================================================================
  # rope3d_freqs/4
  # ============================================================================

  describe "rope3d_freqs/4" do
    test "returns sin and cos frequency tensors" do
      {sin_freqs, cos_freqs} = CogVideoX.rope3d_freqs(4, 4, 4, hidden_size: 48, num_heads: 4)

      assert Nx.rank(sin_freqs) == 2
      assert Nx.rank(cos_freqs) == 2

      # Total positions = 4 * 4 * 4 = 64
      assert Nx.axis_size(sin_freqs, 0) == 64
      assert Nx.axis_size(cos_freqs, 0) == 64
    end

    test "frequencies are finite" do
      {sin_freqs, cos_freqs} = CogVideoX.rope3d_freqs(2, 2, 2, hidden_size: 24, num_heads: 4)

      assert Nx.all(Nx.is_nan(sin_freqs) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_nan(cos_freqs) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "sin values are in [-1, 1]" do
      {sin_freqs, _cos_freqs} = CogVideoX.rope3d_freqs(2, 2, 2, hidden_size: 24, num_heads: 4)

      assert Nx.all(Nx.greater_equal(sin_freqs, -1.0)) |> Nx.to_number() == 1
      assert Nx.all(Nx.less_equal(sin_freqs, 1.0)) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = CogVideoX.recommended_defaults()

      assert Keyword.has_key?(defaults, :in_channels)
      assert Keyword.has_key?(defaults, :latent_channels)
      assert Keyword.has_key?(defaults, :num_frames)
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :num_heads)
      assert Keyword.has_key?(defaults, :num_layers)
    end
  end

  describe "param_count/1" do
    test "returns positive integer" do
      count = CogVideoX.param_count(@transformer_opts)
      assert is_integer(count)
      assert count > 0
    end

    test "increases with more layers" do
      count_2 = CogVideoX.param_count(num_layers: 2, hidden_size: 32)
      count_4 = CogVideoX.param_count(num_layers: 4, hidden_size: 32)
      assert count_4 > count_2
    end
  end

  # ============================================================================
  # Registry Integration
  # ============================================================================

  describe "Edifice registry" do
    test "cogvideox is registered" do
      assert :cogvideox in Edifice.list_architectures()
    end

    test "can build via Edifice.build/2" do
      model = Edifice.build(:cogvideox, @transformer_opts)
      assert %Axon{} = model
    end
  end
end
