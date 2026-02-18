defmodule Edifice.Generative.GAN do
  @moduledoc """
  Generative Adversarial Network framework.

  Provides building blocks for GAN architectures including standard GAN,
  WGAN (Wasserstein), and conditional GAN variants.

  ## Architecture

  ```
  Noise z ~ N(0, I)          Real data x
       |                          |
       v                          v
  +----------+              +----------+
  | Generator|              |Discrimin.|
  | G(z) -> x'|             | D(x) -> p|
  +----------+              +----------+
       |                          |
       v                          v
  Fake samples              Real/Fake score
  ```

  ## Training

  GANs use adversarial training:
  - **Discriminator**: maximize log(D(x)) + log(1 - D(G(z)))
  - **Generator**: maximize log(D(G(z))) (or minimize -log(D(G(z))))

  ## Usage

      {generator, discriminator} = GAN.build(
        latent_size: 128,
        output_size: 784,
        generator_sizes: [256, 512],
        discriminator_sizes: [512, 256]
      )
  """

  require Axon
  import Nx.Defn

  @doc """
  Build generator and discriminator networks.

  ## Options
    - `:latent_size` - Size of noise vector z (default: 128)
    - `:output_size` - Size of generated output (required)
    - `:generator_sizes` - Hidden layer sizes for G (default: [256, 512])
    - `:discriminator_sizes` - Hidden layer sizes for D (default: [512, 256])
    - `:activation` - Activation function (default: :relu)
    - `:output_activation` - Generator output activation (default: :sigmoid)

  ## Returns
    Tuple of `{generator, discriminator}` Axon models.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:output_size, pos_integer()}
          | {:latent_size, pos_integer()}
          | {:generator_sizes, [pos_integer()]}
          | {:discriminator_sizes, [pos_integer()]}
          | {:activation, atom()}
          | {:output_activation, atom()}

  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    {build_generator(opts), build_discriminator(opts)}
  end

  @doc """
  Build the generator network.

  Maps latent noise z to data space.
  """
  @spec build_generator(keyword()) :: Axon.t()
  def build_generator(opts \\ []) do
    latent_size = Keyword.get(opts, :latent_size, 128)
    output_size = Keyword.fetch!(opts, :output_size)
    hidden_sizes = Keyword.get(opts, :generator_sizes, [256, 512])
    activation = Keyword.get(opts, :activation, :relu)
    output_activation = Keyword.get(opts, :output_activation, :sigmoid)

    input = Axon.input("noise", shape: {nil, latent_size})

    # Hidden layers
    hidden =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "gen_dense_#{idx}")
        |> Axon.batch_norm(name: "gen_bn_#{idx}")
        |> Axon.activation(activation)
      end)

    # Output layer
    hidden
    |> Axon.dense(output_size, name: "gen_output")
    |> Axon.activation(output_activation)
  end

  @doc """
  Build the discriminator network.

  Maps data to a real/fake probability.
  """
  @spec build_discriminator(keyword()) :: Axon.t()
  def build_discriminator(opts \\ []) do
    output_size = Keyword.fetch!(opts, :output_size)
    hidden_sizes = Keyword.get(opts, :discriminator_sizes, [512, 256])
    activation = Keyword.get(opts, :activation, :relu)

    input = Axon.input("data", shape: {nil, output_size})

    # Hidden layers with dropout (common for discriminator stability)
    hidden =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "disc_dense_#{idx}")
        |> Axon.activation(activation)
        |> Axon.dropout(rate: 0.3, name: "disc_dropout_#{idx}")
      end)

    # Output: single probability
    Axon.dense(hidden, 1, name: "disc_output")
  end

  @doc """
  Build a conditional generator.

  Takes both noise z and conditioning label y.
  """
  @spec build_conditional_generator(keyword()) :: Axon.t()
  def build_conditional_generator(opts \\ []) do
    latent_size = Keyword.get(opts, :latent_size, 128)
    condition_size = Keyword.fetch!(opts, :condition_size)
    output_size = Keyword.fetch!(opts, :output_size)
    hidden_sizes = Keyword.get(opts, :generator_sizes, [256, 512])
    activation = Keyword.get(opts, :activation, :relu)
    output_activation = Keyword.get(opts, :output_activation, :sigmoid)

    noise = Axon.input("noise", shape: {nil, latent_size})
    condition = Axon.input("condition", shape: {nil, condition_size})

    # Concatenate noise and condition
    combined = Axon.concatenate(noise, condition, name: "gen_concat")

    # Hidden layers
    hidden =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(combined, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "cgen_dense_#{idx}")
        |> Axon.batch_norm(name: "cgen_bn_#{idx}")
        |> Axon.activation(activation)
      end)

    hidden
    |> Axon.dense(output_size, name: "cgen_output")
    |> Axon.activation(output_activation)
  end

  @doc """
  Standard GAN discriminator loss.

  L_D = -mean(log(D(real))) - mean(log(1 - D(fake)))
  """
  defn discriminator_loss(real_scores, fake_scores) do
    real_loss = Nx.mean(Nx.log(Nx.add(Nx.sigmoid(real_scores), 1.0e-8)))
    fake_loss = Nx.mean(Nx.log(Nx.add(Nx.subtract(1.0, Nx.sigmoid(fake_scores)), 1.0e-8)))
    Nx.negate(Nx.add(real_loss, fake_loss))
  end

  @doc """
  Standard GAN generator loss (non-saturating).

  L_G = -mean(log(D(G(z))))
  """
  defn generator_loss(fake_scores) do
    Nx.negate(Nx.mean(Nx.log(Nx.add(Nx.sigmoid(fake_scores), 1.0e-8))))
  end

  @doc """
  Wasserstein GAN discriminator (critic) loss.

  L_D = mean(D(fake)) - mean(D(real))
  """
  defn wasserstein_critic_loss(real_scores, fake_scores) do
    Nx.subtract(Nx.mean(fake_scores), Nx.mean(real_scores))
  end

  @doc """
  Wasserstein GAN generator loss.

  L_G = -mean(D(G(z)))
  """
  defn wasserstein_generator_loss(fake_scores) do
    Nx.negate(Nx.mean(fake_scores))
  end

  @doc """
  Gradient penalty for WGAN-GP.

  Penalizes gradients that deviate from unit norm along interpolations
  between real and fake data.

  Requires a PRNG `key` for sampling random interpolation coefficients.
  """
  defn gradient_penalty(real_data, fake_data, critic_fn, params, key) do
    batch_size = Nx.axis_size(real_data, 0)
    {alpha, _key} = Nx.Random.uniform(key, shape: {batch_size, 1})

    interpolated =
      Nx.add(
        Nx.multiply(alpha, real_data),
        Nx.multiply(Nx.subtract(1.0, alpha), fake_data)
      )

    grad_fn = Nx.Defn.grad(fn x -> Nx.mean(critic_fn.(params, x)) end)
    gradients = grad_fn.(interpolated)

    gradient_norm = Nx.sqrt(Nx.sum(Nx.pow(gradients, 2), axes: [1]))
    Nx.mean(Nx.pow(Nx.subtract(gradient_norm, 1.0), 2))
  end
end
