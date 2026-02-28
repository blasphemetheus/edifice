defmodule Edifice.Generative.MAGVIT2Test do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.MAGVIT2

  @opts [input_size: 32, latent_dim: 8]

  describe "build/1" do
    test "encoder produces correct output shape" do
      {encoder, _decoder} = MAGVIT2.build(@opts)
      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({2, 32}, :f32), %{})
      out = predict_fn.(params, Nx.iota({2, 32}, type: :f32) |> Nx.divide(100))

      assert Nx.shape(out) == {2, 8}
    end

    test "decoder produces correct output shape" do
      {_encoder, decoder} = MAGVIT2.build(@opts)
      {init_fn, predict_fn} = Axon.build(decoder)
      params = init_fn.(Nx.template({2, 8}, :f32), %{})
      out = predict_fn.(params, Nx.iota({2, 8}, type: :f32) |> Nx.divide(10))

      assert Nx.shape(out) == {2, 32}
    end

    test "outputs are finite" do
      {encoder, decoder} = MAGVIT2.build(@opts)

      {enc_init, enc_pred} = Axon.build(encoder)
      enc_params = enc_init.(Nx.template({2, 32}, :f32), %{})
      z_e = enc_pred.(enc_params, Nx.iota({2, 32}, type: :f32) |> Nx.divide(100))

      assert z_e |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert z_e |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0

      {dec_init, dec_pred} = Axon.build(decoder)
      dec_params = dec_init.(Nx.template({2, 8}, :f32), %{})
      recon = dec_pred.(dec_params, z_e)

      assert recon |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert recon |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      {encoder, decoder} = MAGVIT2.build(@opts)

      {enc_init, enc_pred} = Axon.build(encoder)
      enc_params = enc_init.(Nx.template({1, 32}, :f32), %{})
      z_e = enc_pred.(enc_params, Nx.iota({1, 32}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(z_e) == {1, 8}

      {dec_init, dec_pred} = Axon.build(decoder)
      dec_params = dec_init.(Nx.template({1, 8}, :f32), %{})
      recon = dec_pred.(dec_params, z_e)
      assert Nx.shape(recon) == {1, 32}
    end

    test "different latent dim" do
      opts = Keyword.put(@opts, :latent_dim, 14)
      {encoder, decoder} = MAGVIT2.build(opts)

      {enc_init, enc_pred} = Axon.build(encoder)
      enc_params = enc_init.(Nx.template({2, 32}, :f32), %{})
      z_e = enc_pred.(enc_params, Nx.iota({2, 32}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(z_e) == {2, 14}

      {dec_init, dec_pred} = Axon.build(decoder)
      dec_params = dec_init.(Nx.template({2, 14}, :f32), %{})
      recon = dec_pred.(dec_params, z_e)
      assert Nx.shape(recon) == {2, 32}
    end

    test "custom encoder/decoder sizes" do
      opts = @opts ++ [encoder_sizes: [64, 32], decoder_sizes: [32, 64]]
      {encoder, decoder} = MAGVIT2.build(opts)

      {enc_init, enc_pred} = Axon.build(encoder)
      enc_params = enc_init.(Nx.template({2, 32}, :f32), %{})
      z_e = enc_pred.(enc_params, Nx.iota({2, 32}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(z_e) == {2, 8}

      {dec_init, dec_pred} = Axon.build(decoder)
      dec_params = dec_init.(Nx.template({2, 8}, :f32), %{})
      recon = dec_pred.(dec_params, z_e)
      assert Nx.shape(recon) == {2, 32}
    end
  end

  describe "quantize/1" do
    test "produces binary codes in {-1, +1}" do
      z_e = Nx.tensor([[0.5, -0.3, 0.1, -0.8], [0.0, 1.2, -0.5, 0.3]])
      {z_q, _indices} = MAGVIT2.quantize(z_e)

      # All values should be -1 or +1
      abs_vals = Nx.abs(z_q)
      assert Nx.all(Nx.equal(abs_vals, 1.0)) |> Nx.to_number() == 1
    end

    test "produces correct integer indices" do
      # [+, -, +, -] -> bits [1,0,1,0] -> 1*8 + 0*4 + 1*2 + 0*1 = 10
      z_e = Nx.tensor([[1.0, -1.0, 1.0, -1.0]])
      {_z_q, indices} = MAGVIT2.quantize(z_e)

      assert Nx.to_number(indices[0]) == 10
    end

    test "indices range is valid" do
      z_e = Nx.Random.normal(Nx.Random.key(42), shape: {16, 8}) |> elem(0)
      {_z_q, indices} = MAGVIT2.quantize(z_e)

      # All indices should be in [0, 2^8 - 1] = [0, 255]
      assert Nx.reduce_min(indices) |> Nx.to_number() >= 0
      assert Nx.reduce_max(indices) |> Nx.to_number() <= 255
    end
  end

  describe "codebook_size/1" do
    test "returns 2^latent_dim" do
      assert MAGVIT2.codebook_size(18) == 262_144
      assert MAGVIT2.codebook_size(14) == 16_384
      assert MAGVIT2.codebook_size(8) == 256
    end
  end

  describe "loss functions" do
    test "entropy_loss returns scalar" do
      z_e = Nx.Random.normal(Nx.Random.key(42), shape: {8, 4}) |> elem(0)
      loss = MAGVIT2.entropy_loss(z_e)
      assert Nx.shape(loss) == {}
      assert loss |> Nx.is_nan() |> Nx.to_number() == 0
    end

    test "commitment_loss returns scalar" do
      z_e = Nx.Random.normal(Nx.Random.key(42), shape: {8, 4}) |> elem(0)
      loss = MAGVIT2.commitment_loss(z_e)
      assert Nx.shape(loss) == {}
      assert loss |> Nx.is_nan() |> Nx.to_number() == 0
      # Commitment loss should be non-negative
      assert Nx.to_number(loss) >= 0.0
    end

    test "combined loss returns scalar" do
      recon = Nx.Random.normal(Nx.Random.key(42), shape: {8, 32}) |> elem(0)
      target = Nx.Random.normal(Nx.Random.key(42), shape: {8, 32}) |> elem(0)
      z_e = Nx.Random.normal(Nx.Random.key(42), shape: {8, 4}) |> elem(0)
      loss = MAGVIT2.loss(recon, target, z_e)
      assert Nx.shape(loss) == {}
      assert loss |> Nx.is_nan() |> Nx.to_number() == 0
    end
  end

  describe "output_size/1" do
    test "returns input_size" do
      assert MAGVIT2.output_size(input_size: 32) == 32
      assert MAGVIT2.output_size(input_size: 64) == 64
    end
  end

  describe "Edifice.build/2" do
    test "builds magvit2 via registry" do
      {encoder, decoder} = Edifice.build(:magvit2, @opts)
      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end
  end
end
