defmodule Edifice.Generative.GANTest do
  use ExUnit.Case, async: true
  @moduletag :generative

  alias Edifice.Generative.GAN

  describe "build/1" do
    test "returns {generator, discriminator} tuple" do
      {generator, discriminator} = GAN.build(output_size: 64)
      assert %Axon{} = generator
      assert %Axon{} = discriminator
    end
  end

  describe "generator" do
    test "maps noise to output_size" do
      {generator, _discriminator} =
        GAN.build(latent_size: 32, output_size: 64, generator_sizes: [128])

      {init_fn, predict_fn} = Axon.build(generator)
      params = init_fn.(Nx.template({4, 32}, :f32), Axon.ModelState.empty())
      noise = Nx.broadcast(0.5, {4, 32})
      output = predict_fn.(params, noise)

      assert Nx.shape(output) == {4, 64}
    end

    test "uses default latent_size of 128" do
      {generator, _discriminator} =
        GAN.build(output_size: 64, generator_sizes: [128])

      {init_fn, predict_fn} = Axon.build(generator)
      params = init_fn.(Nx.template({2, 128}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {2, 128}))

      assert Nx.shape(output) == {2, 64}
    end

    test "output values are bounded by sigmoid activation" do
      {generator, _discriminator} =
        GAN.build(latent_size: 16, output_size: 32, generator_sizes: [64])

      {init_fn, predict_fn} = Axon.build(generator)
      params = init_fn.(Nx.template({4, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {4, 16}))

      min_val = Nx.to_number(Nx.reduce_min(output))
      max_val = Nx.to_number(Nx.reduce_max(output))
      assert min_val >= 0.0
      assert max_val <= 1.0
    end
  end

  describe "discriminator" do
    test "maps data to single score per sample" do
      {_generator, discriminator} =
        GAN.build(output_size: 64, discriminator_sizes: [128])

      {init_fn, predict_fn} = Axon.build(discriminator, mode: :inference)
      params = init_fn.(Nx.template({4, 64}, :f32), Axon.ModelState.empty())
      data = Nx.broadcast(0.5, {4, 64})
      output = predict_fn.(params, data)

      assert Nx.shape(output) == {4, 1}
    end

    test "respects custom discriminator_sizes" do
      {_generator, discriminator} =
        GAN.build(output_size: 32, discriminator_sizes: [256, 128])

      {init_fn, predict_fn} = Axon.build(discriminator, mode: :inference)
      params = init_fn.(Nx.template({2, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {2, 32}))

      assert Nx.shape(output) == {2, 1}
    end
  end

  describe "build_generator/1" do
    test "builds generator independently" do
      generator = GAN.build_generator(latent_size: 16, output_size: 48)
      {init_fn, predict_fn} = Axon.build(generator)
      params = init_fn.(Nx.template({3, 16}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {3, 16}))

      assert Nx.shape(output) == {3, 48}
    end
  end

  describe "build_discriminator/1" do
    test "builds discriminator independently" do
      discriminator = GAN.build_discriminator(output_size: 48)
      {init_fn, predict_fn} = Axon.build(discriminator, mode: :inference)
      params = init_fn.(Nx.template({3, 48}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {3, 48}))

      assert Nx.shape(output) == {3, 1}
    end
  end

  describe "build_conditional_generator/1" do
    test "takes noise and condition inputs" do
      cgen =
        GAN.build_conditional_generator(
          latent_size: 16,
          condition_size: 10,
          output_size: 64,
          generator_sizes: [128]
        )

      {init_fn, predict_fn} = Axon.build(cgen)

      params =
        init_fn.(
          %{
            "noise" => Nx.template({4, 16}, :f32),
            "condition" => Nx.template({4, 10}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "noise" => Nx.broadcast(0.5, {4, 16}),
          "condition" => Nx.broadcast(0.5, {4, 10})
        })

      assert Nx.shape(output) == {4, 64}
    end
  end

  describe "loss functions" do
    test "discriminator_loss returns scalar" do
      real_scores = Nx.broadcast(0.5, {8, 1})
      fake_scores = Nx.broadcast(0.5, {8, 1})
      loss = GAN.discriminator_loss(real_scores, fake_scores)

      assert Nx.shape(loss) == {}
    end

    test "generator_loss returns scalar" do
      fake_scores = Nx.broadcast(0.5, {8, 1})
      loss = GAN.generator_loss(fake_scores)

      assert Nx.shape(loss) == {}
    end

    test "wasserstein_critic_loss returns scalar" do
      real_scores = Nx.broadcast(0.5, {8, 1})
      fake_scores = Nx.broadcast(0.5, {8, 1})
      loss = GAN.wasserstein_critic_loss(real_scores, fake_scores)

      assert Nx.shape(loss) == {}
    end

    test "wasserstein_generator_loss returns scalar" do
      fake_scores = Nx.broadcast(0.5, {8, 1})
      loss = GAN.wasserstein_generator_loss(fake_scores)

      assert Nx.shape(loss) == {}
    end
  end
end
