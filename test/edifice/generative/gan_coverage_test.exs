defmodule Edifice.Generative.GANCoverageTest do
  @moduledoc """
  Coverage tests for Edifice.Generative.GAN.
  Covers WGAN-GP gradient penalty, different activations, loss value correctness,
  conditional generator variations, and code branches not exercised by existing tests.
  """
  use ExUnit.Case, async: true

  alias Edifice.Generative.GAN

  @batch 4
  @latent_size 16
  @output_size 32

  # ============================================================================
  # Generator with different activations
  # ============================================================================

  describe "generator activation variations" do
    test "generator with :tanh output activation" do
      generator =
        GAN.build_generator(
          latent_size: @latent_size,
          output_size: @output_size,
          generator_sizes: [32],
          output_activation: :tanh
        )

      {init_fn, predict_fn} = Axon.build(generator)
      params = init_fn.(Nx.template({@batch, @latent_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @latent_size}))

      assert Nx.shape(output) == {@batch, @output_size}

      # Tanh bounds output to [-1, 1]
      min_val = Nx.to_number(Nx.reduce_min(output))
      max_val = Nx.to_number(Nx.reduce_max(output))
      assert min_val >= -1.0
      assert max_val <= 1.0
    end

    test "generator with :silu activation in hidden layers" do
      generator =
        GAN.build_generator(
          latent_size: @latent_size,
          output_size: @output_size,
          generator_sizes: [32, 64],
          activation: :silu
        )

      {init_fn, predict_fn} = Axon.build(generator)
      params = init_fn.(Nx.template({@batch, @latent_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @latent_size}))

      assert Nx.shape(output) == {@batch, @output_size}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "generator with multiple hidden layers" do
      generator =
        GAN.build_generator(
          latent_size: @latent_size,
          output_size: @output_size,
          generator_sizes: [16, 32, 64]
        )

      {init_fn, predict_fn} = Axon.build(generator)
      params = init_fn.(Nx.template({@batch, @latent_size}, :f32), Axon.ModelState.empty())

      # Verify has all layers
      param_keys = Map.keys(params.data)
      assert Enum.any?(param_keys, &String.contains?(&1, "gen_dense_0"))
      assert Enum.any?(param_keys, &String.contains?(&1, "gen_dense_1"))
      assert Enum.any?(param_keys, &String.contains?(&1, "gen_dense_2"))

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @latent_size}))
      assert Nx.shape(output) == {@batch, @output_size}
    end
  end

  # ============================================================================
  # Discriminator variations
  # ============================================================================

  describe "discriminator activation variations" do
    test "discriminator with :leaky_relu activation" do
      disc =
        GAN.build_discriminator(
          output_size: @output_size,
          discriminator_sizes: [64, 32],
          activation: :leaky_relu
        )

      {init_fn, predict_fn} = Axon.build(disc, mode: :inference)
      params = init_fn.(Nx.template({@batch, @output_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @output_size}))

      assert Nx.shape(output) == {@batch, 1}
    end

    test "discriminator with single hidden layer" do
      disc =
        GAN.build_discriminator(
          output_size: @output_size,
          discriminator_sizes: [64]
        )

      {init_fn, predict_fn} = Axon.build(disc, mode: :inference)
      params = init_fn.(Nx.template({@batch, @output_size}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @output_size}))

      assert Nx.shape(output) == {@batch, 1}
    end
  end

  # ============================================================================
  # Conditional generator variations
  # ============================================================================

  describe "conditional generator variations" do
    test "conditional generator with different activations" do
      cgen =
        GAN.build_conditional_generator(
          latent_size: @latent_size,
          condition_size: 8,
          output_size: @output_size,
          generator_sizes: [32],
          activation: :silu,
          output_activation: :tanh
        )

      {init_fn, predict_fn} = Axon.build(cgen)

      params =
        init_fn.(
          %{
            "noise" => Nx.template({@batch, @latent_size}, :f32),
            "condition" => Nx.template({@batch, 8}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "noise" => Nx.broadcast(0.5, {@batch, @latent_size}),
          "condition" => Nx.broadcast(0.5, {@batch, 8})
        })

      assert Nx.shape(output) == {@batch, @output_size}
      # tanh bounds to [-1, 1]
      assert Nx.to_number(Nx.reduce_min(output)) >= -1.0
      assert Nx.to_number(Nx.reduce_max(output)) <= 1.0
    end

    test "conditional generator with default latent_size" do
      cgen =
        GAN.build_conditional_generator(
          condition_size: 4,
          output_size: @output_size,
          generator_sizes: [32]
        )

      {init_fn, predict_fn} = Axon.build(cgen)

      # Default latent_size is 128
      params =
        init_fn.(
          %{
            "noise" => Nx.template({@batch, 128}, :f32),
            "condition" => Nx.template({@batch, 4}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "noise" => Nx.broadcast(0.5, {@batch, 128}),
          "condition" => Nx.broadcast(0.5, {@batch, 4})
        })

      assert Nx.shape(output) == {@batch, @output_size}
    end

    test "conditional generator with multiple hidden layers" do
      cgen =
        GAN.build_conditional_generator(
          latent_size: @latent_size,
          condition_size: 8,
          output_size: @output_size,
          generator_sizes: [16, 32, 64]
        )

      {init_fn, predict_fn} = Axon.build(cgen)

      params =
        init_fn.(
          %{
            "noise" => Nx.template({@batch, @latent_size}, :f32),
            "condition" => Nx.template({@batch, 8}, :f32)
          },
          Axon.ModelState.empty()
        )

      param_keys = Map.keys(params.data)
      assert Enum.any?(param_keys, &String.contains?(&1, "cgen_dense_0"))
      assert Enum.any?(param_keys, &String.contains?(&1, "cgen_dense_1"))
      assert Enum.any?(param_keys, &String.contains?(&1, "cgen_dense_2"))

      output =
        predict_fn.(params, %{
          "noise" => Nx.broadcast(0.5, {@batch, @latent_size}),
          "condition" => Nx.broadcast(0.5, {@batch, 8})
        })

      assert Nx.shape(output) == {@batch, @output_size}
    end
  end

  # ============================================================================
  # Loss function correctness
  # ============================================================================

  describe "discriminator_loss/2 correctness" do
    test "returns positive loss" do
      real_scores = Nx.broadcast(0.5, {@batch, 1})
      fake_scores = Nx.broadcast(0.5, {@batch, 1})
      loss = GAN.discriminator_loss(real_scores, fake_scores)

      assert Nx.to_number(loss) > 0.0
    end

    test "loss is lower when disc correctly classifies" do
      # Real scores high (close to 1), fake scores low (close to 0)
      good_real = Nx.broadcast(2.0, {@batch, 1})
      good_fake = Nx.broadcast(-2.0, {@batch, 1})
      good_loss = GAN.discriminator_loss(good_real, good_fake)

      # Real scores low, fake scores high (wrong)
      bad_real = Nx.broadcast(-2.0, {@batch, 1})
      bad_fake = Nx.broadcast(2.0, {@batch, 1})
      bad_loss = GAN.discriminator_loss(bad_real, bad_fake)

      assert Nx.to_number(good_loss) < Nx.to_number(bad_loss)
    end

    test "discriminator loss is finite with extreme inputs" do
      real_scores = Nx.broadcast(10.0, {@batch, 1})
      fake_scores = Nx.broadcast(-10.0, {@batch, 1})
      loss = GAN.discriminator_loss(real_scores, fake_scores)

      assert Nx.all(Nx.is_nan(loss) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(loss) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end

  describe "generator_loss/1 correctness" do
    test "loss is lower when fake scores are high" do
      good_fake = Nx.broadcast(2.0, {@batch, 1})
      bad_fake = Nx.broadcast(-2.0, {@batch, 1})

      good_loss = GAN.generator_loss(good_fake)
      bad_loss = GAN.generator_loss(bad_fake)

      assert Nx.to_number(good_loss) < Nx.to_number(bad_loss)
    end

    test "generator loss is positive" do
      scores = Nx.broadcast(0.0, {@batch, 1})
      loss = GAN.generator_loss(scores)
      assert Nx.to_number(loss) > 0.0
    end
  end

  describe "wasserstein_critic_loss/2 correctness" do
    test "critic loss is negative when real > fake" do
      real_scores = Nx.broadcast(1.0, {@batch, 1})
      fake_scores = Nx.broadcast(-1.0, {@batch, 1})
      loss = GAN.wasserstein_critic_loss(real_scores, fake_scores)

      # mean(fake) - mean(real) = -1 - 1 = -2
      assert Nx.to_number(loss) < 0.0
    end

    test "critic loss is zero when real == fake" do
      scores = Nx.broadcast(0.5, {@batch, 1})
      loss = GAN.wasserstein_critic_loss(scores, scores)

      assert_in_delta Nx.to_number(loss), 0.0, 1.0e-6
    end

    test "critic loss is positive when fake > real" do
      real_scores = Nx.broadcast(-1.0, {@batch, 1})
      fake_scores = Nx.broadcast(1.0, {@batch, 1})
      loss = GAN.wasserstein_critic_loss(real_scores, fake_scores)

      assert Nx.to_number(loss) > 0.0
    end
  end

  describe "wasserstein_generator_loss/1 correctness" do
    test "lower loss for higher fake scores" do
      high_scores = Nx.broadcast(2.0, {@batch, 1})
      low_scores = Nx.broadcast(-2.0, {@batch, 1})

      high_loss = GAN.wasserstein_generator_loss(high_scores)
      low_loss = GAN.wasserstein_generator_loss(low_scores)

      assert Nx.to_number(high_loss) < Nx.to_number(low_loss)
    end

    test "zero scores give zero loss" do
      scores = Nx.broadcast(0.0, {@batch, 1})
      loss = GAN.wasserstein_generator_loss(scores)

      assert_in_delta Nx.to_number(loss), 0.0, 1.0e-6
    end
  end

  # ============================================================================
  # Gradient Penalty (WGAN-GP)
  # ============================================================================

  describe "gradient_penalty/5" do
    test "returns finite scalar" do
      key = Nx.Random.key(300)
      {real_data, key} = Nx.Random.uniform(key, shape: {@batch, @output_size})
      {fake_data, key} = Nx.Random.uniform(key, shape: {@batch, @output_size})

      # Simple critic function: linear transform
      critic_fn = fn _params, x -> Nx.sum(x, axes: [1], keep_axes: true) end

      gp = GAN.gradient_penalty(real_data, fake_data, critic_fn, %{}, key)

      assert Nx.shape(gp) == {}
      assert Nx.all(Nx.is_nan(gp) |> Nx.logical_not()) |> Nx.to_number() == 1
      assert Nx.all(Nx.is_infinity(gp) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "gradient penalty is non-negative" do
      key = Nx.Random.key(301)
      {real_data, key} = Nx.Random.uniform(key, shape: {@batch, @output_size})
      {fake_data, key} = Nx.Random.uniform(key, shape: {@batch, @output_size})

      critic_fn = fn _params, x -> Nx.sum(x, axes: [1], keep_axes: true) end

      gp = GAN.gradient_penalty(real_data, fake_data, critic_fn, %{}, key)

      assert Nx.to_number(gp) >= 0.0
    end

    test "gradient penalty is small when critic gradient has near-unit norm" do
      # If critic is f(x) = sum_i x_i / sqrt(d), gradients are 1/sqrt(d) per dim,
      # and ||grad||^2 = d * (1/sqrt(d))^2 = 1, so penalty should be small
      d = @output_size

      key = Nx.Random.key(302)
      {real_data, key} = Nx.Random.uniform(key, shape: {@batch, d})
      {fake_data, key} = Nx.Random.uniform(key, shape: {@batch, d})

      scale = :math.sqrt(d)

      critic_fn = fn _params, x ->
        Nx.sum(x, axes: [1], keep_axes: true) |> Nx.divide(scale)
      end

      gp = GAN.gradient_penalty(real_data, fake_data, critic_fn, %{}, key)

      # Penalty should be relatively small compared to when gradients are far from 1
      assert Nx.to_number(gp) < 2.0
    end
  end

  # ============================================================================
  # Full build/1 with different configs
  # ============================================================================

  describe "build/1 with various configs" do
    test "build with all custom options" do
      {gen, disc} =
        GAN.build(
          latent_size: 8,
          output_size: 16,
          generator_sizes: [32],
          discriminator_sizes: [32],
          activation: :silu,
          output_activation: :tanh
        )

      assert %Axon{} = gen
      assert %Axon{} = disc

      {init_fn, predict_fn} = Axon.build(gen)
      params = init_fn.(Nx.template({@batch, 8}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 8}))

      assert Nx.shape(output) == {@batch, 16}
    end

    @tag :slow
    test "build with default generator and discriminator sizes" do
      {gen, disc} = GAN.build(output_size: @output_size)

      # Default generator_sizes: [256, 512]
      {init_fn, _} = Axon.build(gen)
      params = init_fn.(Nx.template({@batch, 128}, :f32), Axon.ModelState.empty())
      param_keys = Map.keys(params.data)
      assert Enum.any?(param_keys, &String.contains?(&1, "gen_dense_0"))
      assert Enum.any?(param_keys, &String.contains?(&1, "gen_dense_1"))

      # Default discriminator_sizes: [512, 256]
      {init_fn_d, _} = Axon.build(disc, mode: :inference)
      params_d = init_fn_d.(Nx.template({@batch, @output_size}, :f32), Axon.ModelState.empty())
      param_keys_d = Map.keys(params_d.data)
      assert Enum.any?(param_keys_d, &String.contains?(&1, "disc_dense_0"))
      assert Enum.any?(param_keys_d, &String.contains?(&1, "disc_dense_1"))
    end
  end

  # ============================================================================
  # Generator -> Discriminator pipeline
  # ============================================================================

  describe "end-to-end pipeline" do
    test "generator output feeds into discriminator" do
      {gen, disc} =
        GAN.build(
          latent_size: @latent_size,
          output_size: @output_size,
          generator_sizes: [32],
          discriminator_sizes: [32]
        )

      # Generate fake data
      {gen_init, gen_pred} = Axon.build(gen)
      gen_params = gen_init.(Nx.template({@batch, @latent_size}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(310)
      {noise, _} = Nx.Random.normal(key, shape: {@batch, @latent_size})
      fake_data = gen_pred.(gen_params, noise)

      assert Nx.shape(fake_data) == {@batch, @output_size}

      # Feed to discriminator
      {disc_init, disc_pred} = Axon.build(disc, mode: :inference)
      disc_params = disc_init.(Nx.template({@batch, @output_size}, :f32), Axon.ModelState.empty())
      disc_out = disc_pred.(disc_params, fake_data)

      assert Nx.shape(disc_out) == {@batch, 1}
      assert Nx.all(Nx.is_nan(disc_out) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
