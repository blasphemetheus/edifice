defmodule Edifice.Generative.VAE do
  @moduledoc """
  Variational Autoencoder (VAE).

  Learns a smooth, continuous latent space by encoding inputs into
  distributions (parameterized by mu and log_var) rather than point
  estimates. The reparameterization trick enables gradient flow through
  the stochastic sampling step, making end-to-end training possible.

  ## Architecture

  ```
  Input [batch, input_size]
        |
        v
  +-------------------+
  | Encoder           |
  | Dense layers      |
  +---+----------+----+
      |          |
      v          v
    mu [L]   log_var [L]     L = latent_size
      |          |
      +----+-----+
           |
     reparameterize
     z = mu + eps * exp(0.5 * log_var)
           |
           v
  +-------------------+
  | Decoder           |
  | Dense layers      |
  +-------------------+
        |
        v
  Reconstruction [batch, input_size]
  ```

  ## Loss

  The VAE loss combines reconstruction error with a KL divergence
  regularizer that pushes the learned posterior toward a standard
  normal prior:

      L = reconstruction_loss + beta * KL(q(z|x) || p(z))

  The beta parameter (default 1.0) controls the trade-off. Values
  less than 1.0 yield a beta-VAE with better reconstructions at the
  cost of a less regular latent space.

  ## Usage

      # Build full VAE
      {encoder, decoder} = VAE.build(input_size: 784, latent_size: 32)

      # Build encoder only (for inference / embedding)
      encoder = VAE.build_encoder(input_size: 784, latent_size: 32)

      # Reparameterization and loss (in training loop)
      key = Nx.Random.key(System.system_time())
      {z, _key} = VAE.reparameterize(mu, log_var, key)
      kl = VAE.kl_divergence(mu, log_var)
      total = VAE.loss(reconstruction, target, mu, log_var, beta: 1.0)
  """

  require Axon
  import Nx.Defn

  @default_latent_size 32
  @default_encoder_sizes [256, 128]
  @default_decoder_sizes [128, 256]
  @default_activation :relu

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_size, pos_integer()}
          | {:latent_size, pos_integer()}
          | {:encoder_sizes, [pos_integer()]}
          | {:decoder_sizes, [pos_integer()]}
          | {:activation, atom()}

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a complete VAE (encoder + decoder).

  Returns a tuple `{encoder, decoder}` where the encoder outputs
  `{mu, log_var}` via `Axon.container/1` and the decoder reconstructs
  from a latent vector.

  ## Options
    - `:input_size` - Input feature dimension (required)
    - `:latent_size` - Latent space dimension (default: 32)
    - `:encoder_sizes` - Hidden layer sizes for encoder (default: [256, 128])
    - `:decoder_sizes` - Hidden layer sizes for decoder (default: [128, 256])
    - `:activation` - Activation function (default: :relu)

  ## Returns
    `{encoder, decoder}` - Tuple of Axon models.
    - Encoder: input -> `%{mu: [batch, latent], log_var: [batch, latent]}`
    - Decoder: latent -> `[batch, input_size]`
  """
  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)

    encoder = build_encoder(opts)
    decoder = build_decoder(Keyword.put(opts, :output_size, input_size))

    {encoder, decoder}
  end

  @doc """
  Build the encoder network.

  Maps input to a distribution in latent space, parameterized by
  mu (mean) and log_var (log variance). Uses `Axon.container/1` to
  return both outputs as a map.

  ## Options
    - `:input_size` - Input feature dimension (required)
    - `:latent_size` - Latent dimension (default: 32)
    - `:encoder_sizes` - Hidden layer sizes (default: [256, 128])
    - `:activation` - Activation function (default: :relu)

  ## Returns
    An Axon model outputting `%{mu: [batch, latent_size], log_var: [batch, latent_size]}`.
  """
  @spec build_encoder(keyword()) :: Axon.t()
  def build_encoder(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    latent_size = Keyword.get(opts, :latent_size, @default_latent_size)
    encoder_sizes = Keyword.get(opts, :encoder_sizes, @default_encoder_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)

    input = Axon.input("input", shape: {nil, input_size})

    # Shared encoder trunk
    trunk =
      Enum.with_index(encoder_sizes)
      |> Enum.reduce(input, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "encoder_dense_#{idx}")
        |> Axon.activation(activation, name: "encoder_act_#{idx}")
      end)

    # Two separate heads: mu and log_var
    mu = Axon.dense(trunk, latent_size, name: "mu")
    log_var = Axon.dense(trunk, latent_size, name: "log_var")

    # Return both as a container (Axon graph with multiple outputs)
    Axon.container(%{mu: mu, log_var: log_var})
  end

  @doc """
  Build the decoder network.

  Maps a latent vector back to the input space.

  ## Options
    - `:input_size` or `:output_size` - Reconstruction output dimension (required)
    - `:latent_size` - Latent dimension (default: 32)
    - `:decoder_sizes` - Hidden layer sizes (default: [128, 256])
    - `:activation` - Activation function (default: :relu)

  ## Returns
    An Axon model: `[batch, latent_size]` -> `[batch, output_size]`.
  """
  @spec build_decoder(keyword()) :: Axon.t()
  def build_decoder(opts \\ []) do
    output_size = Keyword.get(opts, :output_size) || Keyword.fetch!(opts, :input_size)
    latent_size = Keyword.get(opts, :latent_size, @default_latent_size)
    decoder_sizes = Keyword.get(opts, :decoder_sizes, @default_decoder_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)

    latent_input = Axon.input("latent", shape: {nil, latent_size})

    # Decoder hidden layers
    trunk =
      Enum.with_index(decoder_sizes)
      |> Enum.reduce(latent_input, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "decoder_dense_#{idx}")
        |> Axon.activation(activation, name: "decoder_act_#{idx}")
      end)

    # Output reconstruction (no activation - let the loss handle it)
    Axon.dense(trunk, output_size, name: "decoder_output")
  end

  # ============================================================================
  # Reparameterization Trick
  # ============================================================================

  @doc """
  Reparameterization trick: sample z from q(z|x) = N(mu, sigma^2).

  Computes `z = mu + eps * exp(0.5 * log_var)` where `eps ~ N(0, I)`.
  This moves the stochasticity outside the computational graph,
  allowing gradients to flow through mu and log_var.

  ## Parameters
    - `mu` - Mean of the approximate posterior `[batch, latent_size]`
    - `log_var` - Log variance of the approximate posterior `[batch, latent_size]`
    - `key` - PRNG key from `Nx.Random.key/1` (required for proper stochastic sampling)

  ## Returns
    `{z, new_key}` — Sampled latent vector and updated PRNG key.
  """
  @spec reparameterize(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  defn reparameterize(mu, log_var, key) do
    # Standard deviation: exp(0.5 * log_var) = sqrt(var)
    std = Nx.exp(0.5 * log_var)

    # Sample epsilon from standard normal
    {eps, key} = Nx.Random.normal(key, shape: Nx.shape(mu), type: Nx.type(mu))

    # z = mu + eps * std
    {mu + eps * std, key}
  end

  # ============================================================================
  # Loss Functions
  # ============================================================================

  @doc """
  KL divergence between the learned posterior q(z|x) and the prior p(z) = N(0, I).

  Computed in closed form:
      KL = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))

  ## Parameters
    - `mu` - Mean `[batch, latent_size]`
    - `log_var` - Log variance `[batch, latent_size]`

  ## Returns
    KL divergence scalar (mean over batch).
  """
  @spec kl_divergence(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn kl_divergence(mu, log_var) do
    # Per-sample KL: sum over latent dimensions
    kl_per_sample =
      -0.5 * Nx.sum(1.0 + log_var - Nx.pow(mu, 2) - Nx.exp(log_var), axes: [-1])

    # Mean over batch
    Nx.mean(kl_per_sample)
  end

  @doc """
  Combined VAE loss: reconstruction + beta * KL divergence.

  Uses mean squared error for reconstruction by default. The beta
  parameter controls the regularization strength:
  - beta = 1.0: standard VAE (ELBO)
  - beta < 1.0: weaker regularization, better reconstructions
  - beta > 1.0: beta-VAE, more disentangled but blurrier

  ## Parameters
    - `reconstruction` - Decoder output `[batch, input_size]`
    - `target` - Original input `[batch, input_size]`
    - `mu` - Encoder mean `[batch, latent_size]`
    - `log_var` - Encoder log variance `[batch, latent_size]`
    - `beta` - KL weight (default: 1.0)

  ## Returns
    Combined loss scalar.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def loss(reconstruction, target, mu, log_var, opts \\ []) do
    beta = Keyword.get(opts, :beta, 1.0)
    do_loss(reconstruction, target, mu, log_var, beta)
  end

  defnp do_loss(reconstruction, target, mu, log_var, beta) do
    # MSE reconstruction loss (mean over features and batch)
    recon_loss = Nx.mean(Nx.pow(reconstruction - target, 2))

    # KL divergence
    kl = kl_divergence(mu, log_var)

    recon_loss + beta * kl
  end
end
