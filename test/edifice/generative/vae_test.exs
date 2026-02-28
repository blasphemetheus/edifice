defmodule Edifice.Generative.VAETest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.VAE

  describe "build/1" do
    test "returns {encoder, decoder} tuple" do
      {encoder, decoder} = VAE.build(input_size: 64)
      assert %Axon{} = encoder
      assert %Axon{} = decoder
    end
  end

  describe "encoder" do
    test "outputs mu and log_var with correct shapes" do
      {encoder, _decoder} = VAE.build(input_size: 64, latent_size: 16)
      {init_fn, predict_fn} = Axon.build(encoder)

      params = init_fn.(Nx.template({4, 64}, :f32), Axon.ModelState.empty())
      input = Nx.broadcast(0.5, {4, 64})
      output = predict_fn.(params, input)

      assert %{mu: mu, log_var: log_var} = output
      assert Nx.shape(mu) == {4, 16}
      assert Nx.shape(log_var) == {4, 16}
    end

    test "respects custom encoder_sizes" do
      {encoder, _decoder} =
        VAE.build(input_size: 32, latent_size: 8, encoder_sizes: [128, 64])

      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({2, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {2, 32}))

      assert %{mu: mu, log_var: log_var} = output
      assert Nx.shape(mu) == {2, 8}
      assert Nx.shape(log_var) == {2, 8}
    end

    test "uses default latent_size of 32" do
      {encoder, _decoder} = VAE.build(input_size: 64)
      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({1, 64}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {1, 64}))

      assert Nx.shape(output.mu) == {1, 32}
    end
  end

  describe "decoder" do
    test "reconstructs to input_size" do
      {_encoder, decoder} = VAE.build(input_size: 64, latent_size: 16)
      {init_fn, predict_fn} = Axon.build(decoder)

      params = init_fn.(Nx.template({4, 16}, :f32), Axon.ModelState.empty())
      latent = Nx.broadcast(0.5, {4, 16})
      output = predict_fn.(params, latent)

      assert Nx.shape(output) == {4, 64}
    end

    test "respects custom decoder_sizes" do
      {_encoder, decoder} =
        VAE.build(input_size: 48, latent_size: 8, decoder_sizes: [64, 128])

      {init_fn, predict_fn} = Axon.build(decoder)
      params = init_fn.(Nx.template({2, 8}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {2, 8}))

      assert Nx.shape(output) == {2, 48}
    end
  end

  describe "build_encoder/1" do
    test "builds encoder independently" do
      encoder = VAE.build_encoder(input_size: 32, latent_size: 8)
      {init_fn, predict_fn} = Axon.build(encoder)
      params = init_fn.(Nx.template({2, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {2, 32}))

      assert %{mu: mu, log_var: log_var} = output
      assert Nx.shape(mu) == {2, 8}
      assert Nx.shape(log_var) == {2, 8}
    end
  end

  describe "build_decoder/1" do
    test "builds decoder independently" do
      decoder = VAE.build_decoder(input_size: 64, latent_size: 16)
      {init_fn, predict_fn} = Axon.build(decoder)
      params = init_fn.(Nx.template({3, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {3, 16}))

      assert Nx.shape(output) == {3, 64}
    end
  end

  describe "reparameterize/3" do
    test "produces output with same shape as mu" do
      mu = Nx.broadcast(0.5, {4, 16})
      log_var = Nx.broadcast(0.5, {4, 16})
      key = Nx.Random.key(0)
      {z, _key} = VAE.reparameterize(mu, log_var, key)

      assert Nx.shape(z) == {4, 16}
    end

    test "output type matches input type" do
      mu = Nx.broadcast(0.5, {2, 8})
      log_var = Nx.broadcast(0.5, {2, 8})
      key = Nx.Random.key(0)
      {z, _key} = VAE.reparameterize(mu, log_var, key)

      assert Nx.type(z) == {:f, 32}
    end

    test "different keys produce different samples" do
      mu = Nx.broadcast(0.0, {2, 8})
      log_var = Nx.broadcast(0.0, {2, 8})
      {z1, _} = VAE.reparameterize(mu, log_var, Nx.Random.key(1))
      {z2, _} = VAE.reparameterize(mu, log_var, Nx.Random.key(2))

      refute Nx.to_flat_list(z1) == Nx.to_flat_list(z2)
    end
  end

  describe "kl_divergence/2" do
    test "returns a scalar" do
      mu = Nx.broadcast(0.5, {4, 16})
      log_var = Nx.broadcast(0.5, {4, 16})
      kl = VAE.kl_divergence(mu, log_var)

      assert Nx.shape(kl) == {}
    end

    test "is zero when mu=0 and log_var=0 (standard normal)" do
      mu = Nx.broadcast(0.0, {4, 16})
      log_var = Nx.broadcast(0.0, {4, 16})
      kl = VAE.kl_divergence(mu, log_var)

      assert_in_delta Nx.to_number(kl), 0.0, 1.0e-5
    end

    test "is positive for non-standard distributions" do
      mu = Nx.broadcast(1.0, {4, 16})
      log_var = Nx.broadcast(0.5, {4, 16})
      kl = VAE.kl_divergence(mu, log_var)

      assert Nx.to_number(kl) > 0.0
    end
  end

  describe "loss/5" do
    test "returns a scalar loss" do
      reconstruction = Nx.broadcast(0.5, {4, 64})
      target = Nx.broadcast(0.5, {4, 64})
      mu = Nx.broadcast(0.5, {4, 16})
      log_var = Nx.broadcast(0.5, {4, 16})

      loss = VAE.loss(reconstruction, target, mu, log_var)
      assert Nx.shape(loss) == {}
    end

    test "loss is non-negative" do
      reconstruction = Nx.broadcast(0.5, {4, 64})
      target = Nx.broadcast(0.5, {4, 64})
      mu = Nx.broadcast(0.5, {4, 16})
      log_var = Nx.broadcast(0.5, {4, 16})

      loss = VAE.loss(reconstruction, target, mu, log_var)
      assert Nx.to_number(loss) >= 0.0
    end

    test "respects beta parameter" do
      reconstruction = Nx.broadcast(0.5, {4, 64})
      target = Nx.broadcast(0.5, {4, 64})
      mu = Nx.broadcast(1.0, {4, 16})
      log_var = Nx.broadcast(0.5, {4, 16})

      loss_beta_1 = VAE.loss(reconstruction, target, mu, log_var, beta: 1.0)
      loss_beta_0 = VAE.loss(reconstruction, target, mu, log_var, beta: 0.0)

      # With beta=0, only reconstruction loss, so it should be less
      assert Nx.to_number(loss_beta_1) > Nx.to_number(loss_beta_0)
    end
  end
end
