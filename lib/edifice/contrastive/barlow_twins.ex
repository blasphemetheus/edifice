defmodule Edifice.Contrastive.BarlowTwins do
  @moduledoc """
  Barlow Twins - Redundancy Reduction for Self-Supervised Learning.

  Implements Barlow Twins from "Barlow Twins: Self-Supervised Learning via
  Redundancy Reduction" (Zbontar et al., ICML 2021). Barlow Twins prevents
  representation collapse by pushing the cross-correlation matrix of two
  augmented views toward the identity matrix.

  ## Key Innovation

  The loss has two terms on the cross-correlation matrix C:
  1. **Invariance**: Diagonal elements should be 1 (views agree per dimension)
  2. **Redundancy reduction**: Off-diagonal elements should be 0
     (dimensions should be independent)

  ```
  L = SUM_i (1 - C_ii)^2 + lambda * SUM_{i!=j} C_ij^2
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
       Z_A        Cross-Corr     Z_B
        |                         |
        +-------> Loss <---------+
  ```

  ## Usage

      model = BarlowTwins.build(encoder_dim: 287, projection_dim: 256)

      loss = BarlowTwins.barlow_twins_loss(z_a, z_b, lambda_param: 0.005)

  ## References
  - Paper: https://arxiv.org/abs/2103.03230
  """
  import Nx.Defn

  @doc "Default projection dimension"
  @spec default_projection_dim() :: pos_integer()
  def default_projection_dim, do: 256

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 512

  @doc "Default redundancy reduction coefficient"
  @spec default_lambda() :: float()
  def default_lambda, do: 0.005

  @doc """
  Build a Barlow Twins model (encoder + projector).

  ## Options
    - `:encoder_dim` - Input feature dimension (required)
    - `:projection_dim` - Projector output dimension (default: 256)
    - `:hidden_size` - Hidden dimension (default: 512)

  ## Returns
    An Axon model mapping inputs to projection embeddings.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:encoder_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:projection_dim, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    encoder_dim = Keyword.fetch!(opts, :encoder_dim)
    projection_dim = Keyword.get(opts, :projection_dim, default_projection_dim())
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())

    input = Axon.input("features", shape: {nil, encoder_dim})

    # Encoder
    encoded =
      input
      |> Axon.dense(hidden_size, name: "encoder_fc1")
      |> Axon.activation(:relu, name: "encoder_relu1")
      |> Axon.layer_norm(name: "encoder_norm1")
      |> Axon.dense(hidden_size, name: "encoder_fc2")
      |> Axon.activation(:relu, name: "encoder_relu2")
      |> Axon.layer_norm(name: "encoder_norm2")

    # Projector (3-layer MLP with BN)
    encoded
    |> Axon.dense(hidden_size, name: "proj_fc1")
    |> Axon.activation(:relu, name: "proj_relu1")
    |> Axon.layer_norm(name: "proj_norm1")
    |> Axon.dense(hidden_size, name: "proj_fc2")
    |> Axon.activation(:relu, name: "proj_relu2")
    |> Axon.dense(projection_dim, name: "proj_fc3")
  end

  @doc """
  Compute the Barlow Twins loss.

  ## Parameters
    - `z_a` - Embeddings from view A: [batch, projection_dim]
    - `z_b` - Embeddings from view B: [batch, projection_dim]

  ## Options
    - `:lambda_param` - Off-diagonal penalty weight (default: 0.005)

  ## Returns
    Scalar loss tensor.
  """
  @spec barlow_twins_loss(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn barlow_twins_loss(z_a, z_b, opts \\ []) do
    lambda_param =
      case opts[:lambda_param] do
        nil -> 0.005
        val -> val
      end

    dim = Nx.axis_size(z_a, 1)

    # Normalize along batch dimension (zero mean, unit std)
    z_a_norm = batch_normalize(z_a)
    z_b_norm = batch_normalize(z_b)

    batch_size = Nx.axis_size(z_a, 0)

    # Cross-correlation matrix: [dim, dim]
    cc = Nx.dot(Nx.transpose(z_a_norm), z_b_norm) / batch_size

    # Loss: invariance term (diagonal) + redundancy reduction (off-diagonal)
    identity = Nx.eye(dim)

    # Invariance: sum of (1 - C_ii)^2
    on_diag = Nx.sum(Nx.pow(1.0 - cc * identity, 2) * identity)

    # Redundancy reduction: sum of C_ij^2 for i != j
    off_diag = Nx.sum(Nx.pow(cc * (1.0 - identity), 2))

    on_diag + lambda_param * off_diag
  end

  defnp batch_normalize(z) do
    mean = Nx.mean(z, axes: [0], keep_axes: true)
    std = Nx.sqrt(Nx.variance(z, axes: [0], keep_axes: true) + 1.0e-4)
    (z - mean) / std
  end

  @doc """
  Get the output size of the Barlow Twins model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :projection_dim, default_projection_dim())
  end
end
