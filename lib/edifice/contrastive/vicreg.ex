defmodule Edifice.Contrastive.VICReg do
  @moduledoc """
  VICReg - Variance-Invariance-Covariance Regularization.

  Implements VICReg from "VICReg: Variance-Invariance-Covariance Regularization
  for Self-Supervised Learning" (Bardes et al., ICLR 2022). VICReg prevents
  representation collapse through three explicit regularization terms applied
  directly to the embedding vectors, without requiring negative pairs,
  asymmetric networks, or momentum encoders.

  ## Key Innovations

  - **Explicit collapse prevention**: Three distinct terms each prevent a
    different mode of collapse
  - **No architectural tricks**: Symmetric architecture, no stop-gradient,
    no momentum encoder, no negative mining
  - **Interpretable loss**: Each term has a clear geometric meaning

  ## Loss Terms

  1. **Variance** (v): Maintain variance of each embedding dimension above
     a threshold (prevents informational collapse where all embeddings
     become identical)
  2. **Invariance** (i): MSE between embeddings of augmented views
     (ensures representations are view-invariant)
  3. **Covariance** (c): Decorrelate embedding dimensions
     (prevents dimensional collapse where all dimensions are correlated)

  ```
  L = lambda * invariance(Z, Z')
    + mu * [variance(Z) + variance(Z')]
    + nu * [covariance(Z) + covariance(Z')]
  ```

  ## Architecture

  ```
  Augmented View 1         Augmented View 2
        |                         |
        v                         v
  +------------+           +------------+
  |  Encoder   |           |  Encoder   |  (shared weights)
  +------------+           +------------+
        |                         |
        v                         v
  +------------+           +------------+
  | Projector  |           | Projector  |  (shared weights)
  +------------+           +------------+
        |                         |
        v                         v
       Z                         Z'
        |                         |
        +------> VICReg Loss <----+
  ```

  ## Usage

      model = VICReg.build(encoder_dim: 287, projection_dim: 256)

      # Compute loss between two batches of projections
      loss = VICReg.vicreg_loss(z, z_prime,
        lambda_inv: 25.0,
        mu_var: 25.0,
        nu_cov: 1.0
      )

  ## References
  - Paper: https://arxiv.org/abs/2105.04906
  """

  require Axon
  import Nx.Defn

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default projection head output dimension"
  def default_projection_dim, do: 256

  @doc "Default encoder/projector hidden dimension"
  def default_hidden_dim, do: 512

  @doc "Default invariance loss coefficient"
  def default_lambda_inv, do: 25.0

  @doc "Default variance loss coefficient"
  def default_mu_var, do: 25.0

  @doc "Default covariance loss coefficient"
  def default_nu_cov, do: 1.0

  @doc "Default variance target threshold"
  def default_variance_target, do: 1.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a VICReg model (encoder + projector).

  ## Options
    - `:encoder_dim` - Input feature dimension (required)
    - `:projection_dim` - Projector output dimension (default: 256)
    - `:hidden_dim` - Hidden dimension for encoder and projector (default: 512)

  ## Returns
    An Axon model mapping inputs to projection embeddings.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    encoder_dim = Keyword.fetch!(opts, :encoder_dim)
    projection_dim = Keyword.get(opts, :projection_dim, default_projection_dim())
    hidden_dim = Keyword.get(opts, :hidden_dim, default_hidden_dim())

    input = Axon.input("features", shape: {nil, encoder_dim})

    # Encoder
    encoded =
      input
      |> Axon.dense(hidden_dim, name: "encoder_fc1")
      |> Axon.activation(:relu, name: "encoder_relu1")
      |> Axon.layer_norm(name: "encoder_norm1")
      |> Axon.dense(hidden_dim, name: "encoder_fc2")
      |> Axon.activation(:relu, name: "encoder_relu2")
      |> Axon.layer_norm(name: "encoder_norm2")

    # Projector (3-layer MLP)
    encoded
    |> Axon.dense(hidden_dim, name: "proj_fc1")
    |> Axon.activation(:relu, name: "proj_relu1")
    |> Axon.layer_norm(name: "proj_norm1")
    |> Axon.dense(hidden_dim, name: "proj_fc2")
    |> Axon.activation(:relu, name: "proj_relu2")
    |> Axon.dense(projection_dim, name: "proj_fc3")
  end

  # ============================================================================
  # VICReg Loss
  # ============================================================================

  @doc """
  Compute the full VICReg loss.

  ## Parameters
    - `z` - Embeddings from view 1: [batch, projection_dim]
    - `z_prime` - Embeddings from view 2: [batch, projection_dim]

  ## Options
    - `:lambda_inv` - Invariance loss weight (default: 25.0)
    - `:mu_var` - Variance loss weight (default: 25.0)
    - `:nu_cov` - Covariance loss weight (default: 1.0)
    - `:variance_target` - Target standard deviation (default: 1.0)

  ## Returns
    Scalar loss tensor.
  """
  @spec vicreg_loss(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn vicreg_loss(z, z_prime, opts \\ []) do
    lambda_inv =
      case opts[:lambda_inv] do
        nil -> 25.0
        val -> val
      end

    mu_var =
      case opts[:mu_var] do
        nil -> 25.0
        val -> val
      end

    nu_cov =
      case opts[:nu_cov] do
        nil -> 1.0
        val -> val
      end

    variance_target =
      case opts[:variance_target] do
        nil -> 1.0
        val -> val
      end

    # Invariance: MSE between views
    inv_loss = invariance_loss(z, z_prime)

    # Variance: keep std dev above threshold for both views
    var_loss = variance_loss(z, variance_target) + variance_loss(z_prime, variance_target)

    # Covariance: decorrelate dimensions for both views
    cov_loss = covariance_loss(z) + covariance_loss(z_prime)

    lambda_inv * inv_loss + mu_var * var_loss + nu_cov * cov_loss
  end

  @doc """
  Invariance term: MSE between the two views.

  Encourages representations of augmented views to be similar.
  """
  @spec invariance_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn invariance_loss(z, z_prime) do
    Nx.mean(Nx.pow(z - z_prime, 2))
  end

  @doc """
  Variance term: hinge loss on standard deviation.

  Prevents informational collapse by ensuring each dimension maintains
  variance above a target threshold across the batch.
  """
  @spec variance_loss(Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  defn variance_loss(z, target \\ 1.0) do
    # Standard deviation of each dimension across the batch
    std = Nx.sqrt(Nx.variance(z, axes: [0]) + 1.0e-4)

    # Hinge loss: max(0, target - std)
    Nx.mean(Nx.max(0.0, target - std))
  end

  @doc """
  Covariance term: decorrelate embedding dimensions.

  Prevents dimensional collapse by pushing the off-diagonal elements
  of the covariance matrix toward zero.
  """
  @spec covariance_loss(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn covariance_loss(z) do
    batch_size = Nx.axis_size(z, 0)
    dim = Nx.axis_size(z, 1)

    # Center the embeddings
    z_centered = z - Nx.mean(z, axes: [0], keep_axes: true)

    # Covariance matrix: [dim, dim]
    cov = Nx.dot(Nx.transpose(z_centered), z_centered) / (batch_size - 1)

    # Zero out the diagonal (we only penalize off-diagonal)
    identity = Nx.eye(dim)
    off_diag = cov * (1.0 - identity)

    # Sum of squared off-diagonal elements, normalized
    Nx.sum(Nx.pow(off_diag, 2)) / dim
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of the VICReg model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :projection_dim, default_projection_dim())
  end
end
