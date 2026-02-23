defmodule Edifice.Transformer.ByteLatentTransformerTest do
  use ExUnit.Case, async: true

  alias Edifice.Transformer.ByteLatentTransformer

  @batch 2
  @max_byte_len 32
  @patch_size 4
  @vocab_size 256
  @latent_dim 32
  @byte_dim 16
  @num_patches div(@max_byte_len, @patch_size)

  @opts [
    vocab_size: @vocab_size,
    patch_size: @patch_size,
    latent_dim: @latent_dim,
    byte_dim: @byte_dim,
    num_encoder_layers: 1,
    num_latent_layers: 2,
    num_decoder_layers: 1,
    num_heads: 4,
    num_kv_heads: 2,
    max_byte_len: @max_byte_len,
    dropout: 0.0
  ]

  describe "build/1" do
    test "returns a 3-tuple" do
      result = ByteLatentTransformer.build(@opts)
      assert tuple_size(result) == 3
      {encoder, latent, decoder} = result
      assert %Axon{} = encoder
      assert %Axon{} = latent
      assert %Axon{} = decoder
    end

    test "encoder produces [batch, num_patches, latent_dim]" do
      {encoder, _latent, _decoder} = ByteLatentTransformer.build(@opts)

      {init_fn, predict_fn} = Axon.build(encoder)

      # Byte IDs input: integers in 0..255
      key = Nx.Random.key(42)
      {byte_input, _} = Nx.Random.uniform(key, shape: {@batch, @max_byte_len})
      byte_ids = Nx.as_type(Nx.multiply(byte_input, 255), :s64)

      params =
        init_fn.(
          %{"byte_ids" => Nx.template({@batch, @max_byte_len}, :s64)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"byte_ids" => byte_ids})

      assert Nx.shape(output) == {@batch, @num_patches, @latent_dim}
    end

    test "latent transformer produces [batch, num_patches, latent_dim]" do
      {_encoder, latent, _decoder} = ByteLatentTransformer.build(@opts)

      {init_fn, predict_fn} = Axon.build(latent)

      key = Nx.Random.key(42)
      {latent_input, _} = Nx.Random.uniform(key, shape: {@batch, @num_patches, @latent_dim})

      params =
        init_fn.(
          %{"latent_patches" => Nx.template({@batch, @num_patches, @latent_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"latent_patches" => latent_input})

      assert Nx.shape(output) == {@batch, @num_patches, @latent_dim}
    end

    test "decoder produces [batch, byte_len, vocab_size]" do
      {_encoder, _latent, decoder} = ByteLatentTransformer.build(@opts)

      {init_fn, predict_fn} = Axon.build(decoder)

      key = Nx.Random.key(42)
      {latent_input, _} = Nx.Random.uniform(key, shape: {@batch, @num_patches, @latent_dim})

      params =
        init_fn.(
          %{"latent_output" => Nx.template({@batch, @num_patches, @latent_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"latent_output" => latent_input})

      assert Nx.shape(output) == {@batch, @max_byte_len, @vocab_size}
    end

    test "all components produce finite values" do
      {encoder, latent, decoder} = ByteLatentTransformer.build(@opts)
      key = Nx.Random.key(42)

      # Encoder
      {init_fn, predict_fn} = Axon.build(encoder)
      {byte_input, _key} = Nx.Random.uniform(key, shape: {@batch, @max_byte_len})
      byte_ids = Nx.as_type(Nx.multiply(byte_input, 255), :s64)

      params =
        init_fn.(
          %{"byte_ids" => Nx.template({@batch, @max_byte_len}, :s64)},
          Axon.ModelState.empty()
        )

      enc_out = predict_fn.(params, %{"byte_ids" => byte_ids})
      assert Nx.all(Nx.is_nan(enc_out) |> Nx.logical_not()) |> Nx.to_number() == 1

      # Latent
      {init_fn2, predict_fn2} = Axon.build(latent)

      params2 =
        init_fn2.(
          %{"latent_patches" => Nx.template({@batch, @num_patches, @latent_dim}, :f32)},
          Axon.ModelState.empty()
        )

      lat_out = predict_fn2.(params2, %{"latent_patches" => enc_out})
      assert Nx.all(Nx.is_nan(lat_out) |> Nx.logical_not()) |> Nx.to_number() == 1

      # Decoder
      {init_fn3, predict_fn3} = Axon.build(decoder)

      params3 =
        init_fn3.(
          %{"latent_output" => Nx.template({@batch, @num_patches, @latent_dim}, :f32)},
          Axon.ModelState.empty()
        )

      dec_out = predict_fn3.(params3, %{"latent_output" => lat_out})
      assert Nx.all(Nx.is_nan(dec_out) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns latent_dim" do
      assert ByteLatentTransformer.output_size(@opts) == @latent_dim
    end
  end
end
