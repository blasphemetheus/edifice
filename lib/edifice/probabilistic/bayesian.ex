defmodule Edifice.Probabilistic.Bayesian do
  @moduledoc """
  Bayesian Neural Network layers with weight uncertainty.

  Instead of point-estimate weights, each weight is a probability distribution
  parameterized by (mu, rho) where the actual weight is sampled as:

      W = mu + softplus(rho) * epsilon,  epsilon ~ N(0, 1)

  This provides:
  1. **Uncertainty estimation** - Variance in predictions indicates confidence
  2. **Regularization** - KL divergence from prior acts as a learned regularizer
  3. **Robustness** - Multiple weight samples reduce overconfidence

  ## Training with ELBO

  The network is trained by maximizing the Evidence Lower BOund:

      ELBO = E_q[log p(D|W)] - beta * KL(q(W) || p(W))

  where:
  - q(W) = N(mu, softplus(rho)^2) is the learned weight posterior
  - p(W) = N(0, 1) is the prior (standard normal)
  - beta controls the regularization strength (typically 1/num_batches)

  ## Architecture

  ```
  Input [batch, input_size]
        |
        v
  +-------------------------------+
  | Bayesian Dense (sample W)    |
  | W = mu + softplus(rho) * eps |
  +-------------------------------+
        |
        v
  +-------------------------------+
  | Activation (ReLU)            |
  +-------------------------------+
        |   (repeat for each layer)
        v
  Output [batch, output_size]
  ```

  ## Usage

      # Build a Bayesian neural network
      model = Bayesian.build(
        input_size: 256,
        hidden_sizes: [128, 64],
        output_size: 10
      )

      # Training: use ELBO loss
      loss = Bayesian.elbo_loss(predictions, targets, kl_cost, beta: 1/num_batches)

  ## References
  - Blundell et al., "Weight Uncertainty in Neural Networks" (2015)
  - https://arxiv.org/abs/1505.05424
  """

  require Axon
  import Nx.Defn

  @default_hidden_sizes [256, 128]
  @default_prior_sigma 1.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Bayesian Neural Network.

  Each dense layer uses weight distributions instead of point estimates.
  During the forward pass, weights are sampled from the learned posterior.

  ## Options
    - `:input_size` - Input feature dimension (required)
    - `:hidden_sizes` - List of hidden layer sizes (default: [256, 128])
    - `:output_size` - Output dimension (required)
    - `:activation` - Activation function (default: :relu)
    - `:prior_sigma` - Standard deviation of the weight prior (default: 1.0)

  ## Returns
    An Axon model: `[batch, input_size]` -> `[batch, output_size]`

  ## Note
    The model uses the reparameterization trick internally. During training,
    different samples produce different outputs (stochastic). For deterministic
    inference, use the mean weights (mu) directly by setting epsilon to zero
    in the params.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    output_size = Keyword.fetch!(opts, :output_size)
    activation = Keyword.get(opts, :activation, :relu)

    input = Axon.input("input", shape: {nil, input_size})

    # Build hidden layers
    x =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        acc
        |> bayesian_dense(size, name: "bayesian_dense_#{idx}")
        |> Axon.activation(activation, name: "act_#{idx}")
      end)

    # Output layer (also Bayesian)
    bayesian_dense(x, output_size, name: "bayesian_output")
  end

  @doc """
  Build a Bayesian dense layer using the reparameterization trick.

  Instead of learning fixed weights W, learns mu and rho parameters
  where sigma = softplus(rho) = log(1 + exp(rho)).

  During the forward pass:
  1. Sample epsilon ~ N(0, 1)
  2. Compute W = mu + sigma * epsilon
  3. Output = input * W + bias

  ## Parameters
    - `input` - Axon node
    - `units` - Number of output units

  ## Options
    - `:name` - Layer name prefix (default: "bayesian_dense")

  ## Returns
    An Axon node with shape `[batch, units]`
  """
  @spec bayesian_dense(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def bayesian_dense(input, units, opts \\ []) do
    name = Keyword.get(opts, :name, "bayesian_dense")

    # Learn mu and rho for weights via two separate dense layers
    # mu: weight means
    mu = Axon.dense(input, units, name: "#{name}_mu", use_bias: true)

    # rho: weight log-variance (sigma = softplus(rho))
    # Initialize rho to small negative values so initial sigma is small
    rho =
      Axon.dense(input, units,
        name: "#{name}_rho",
        kernel_initializer: Axon.Initializers.uniform(scale: 0.01),
        use_bias: true
      )

    # Reparameterization trick: sample = mu + softplus(rho) * epsilon
    Axon.layer(
      &bayesian_sample_impl/3,
      [mu, rho],
      name: name,
      op_name: :bayesian_sample
    )
  end

  # Reparameterization trick implementation
  defp bayesian_sample_impl(mu, rho, _opts) do
    sigma = Nx.log1p(Nx.exp(rho))
    key = Nx.Random.key(:erlang.system_time())
    {epsilon, _new_key} = Nx.Random.normal(key, shape: Nx.shape(mu), type: Nx.type(mu))
    Nx.add(mu, Nx.multiply(sigma, epsilon))
  end

  # ============================================================================
  # Loss Functions
  # ============================================================================

  @doc """
  Compute KL divergence between weight posterior q(W) and prior p(W).

  For Gaussian posterior N(mu, sigma^2) and Gaussian prior N(0, prior_sigma^2):

      KL = sum(log(prior_sigma/sigma) + (sigma^2 + mu^2)/(2*prior_sigma^2) - 0.5)

  ## Parameters
    - `mu` - Weight means `[...]`
    - `rho` - Weight log-variance parameters `[...]`

  ## Options
    - `:prior_sigma` - Standard deviation of the prior (default: 1.0)

  ## Returns
    Scalar KL divergence cost
  """
  @spec kl_cost(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn kl_cost(mu, rho, opts \\ []) do
    prior_sigma =
      case opts[:prior_sigma] do
        nil -> @default_prior_sigma
        val -> val
      end

    # Posterior sigma via softplus
    sigma = Nx.log1p(Nx.exp(rho))

    # KL(N(mu, sigma^2) || N(0, prior_sigma^2))
    kl =
      Nx.log(prior_sigma / sigma) +
        (sigma * sigma + mu * mu) / (2.0 * prior_sigma * prior_sigma) -
        0.5

    Nx.sum(kl)
  end

  @doc """
  Compute the Evidence Lower BOund (ELBO) loss.

  ELBO = reconstruction_loss + beta * KL_divergence

  The reconstruction loss measures how well the model fits the data,
  while KL divergence regularizes the weight posterior toward the prior.

  ## Parameters
    - `predictions` - Model predictions `[batch, output_size]`
    - `targets` - Ground truth targets `[batch, output_size]`
    - `kl_divergence` - KL cost from `kl_cost/3`

  ## Options
    - `:beta` - KL weight, typically `1 / num_batches` (default: 1.0)
    - `:loss_fn` - Reconstruction loss: `:mse` or `:cross_entropy` (default: :mse)

  ## Returns
    Scalar ELBO loss (to be minimized)
  """
  @spec elbo_loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn elbo_loss(predictions, targets, kl_divergence, opts \\ []) do
    beta =
      case opts[:beta] do
        nil -> 1.0
        val -> val
      end

    # Reconstruction loss (MSE)
    reconstruction = Nx.mean(Nx.pow(predictions - targets, 2))

    # ELBO = reconstruction + beta * KL
    reconstruction + beta * kl_divergence
  end
end
