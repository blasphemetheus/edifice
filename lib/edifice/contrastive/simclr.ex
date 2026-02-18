defmodule Edifice.Contrastive.SimCLR do
  @moduledoc """
  SimCLR - Simple Contrastive Learning of Representations.

  Implements SimCLR from "A Simple Framework for Contrastive Learning of
  Visual Representations" (Chen et al., ICML 2020). SimCLR learns
  representations by maximizing agreement between differently augmented
  views of the same data example via a contrastive loss (NT-Xent).

  ## Key Components

  - **Augmentation**: Two random augmentations of each example
  - **Encoder**: Shared backbone that extracts representations
  - **Projection Head**: MLP that maps representations to contrastive space
  - **NT-Xent Loss**: Normalized temperature-scaled cross-entropy

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
  | Projection |           | Projection |  (shared weights)
  |    Head    |           |    Head    |
  +------------+           +------------+
        |                         |
        v                         v
       z_i        NT-Xent        z_j
        |                         |
        +-------> Loss <---------+
  ```

  ## Usage

      model = SimCLR.build(encoder_dim: 287, projection_dim: 128)

      # Compute NT-Xent loss between projections
      loss = SimCLR.nt_xent_loss(z_i, z_j, temperature: 0.5)

  ## References
  - Paper: https://arxiv.org/abs/2002.05709
  """

  require Axon
  import Nx.Defn

  @doc "Default projection head output dimension"
  @spec default_projection_dim() :: pos_integer()
  def default_projection_dim, do: 128

  @doc "Default hidden dimension for encoder and projection head"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default temperature for NT-Xent loss"
  @spec default_temperature() :: float()
  def default_temperature, do: 0.5

  @doc """
  Build a SimCLR model (encoder + projection head).

  ## Options
    - `:encoder_dim` - Input feature dimension (required)
    - `:projection_dim` - Projection head output dimension (default: 128)
    - `:hidden_size` - Hidden dimension (default: 256)

  ## Returns
    An Axon model mapping inputs to projection embeddings.
  """
  @spec build(keyword()) :: Axon.t()
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

    # Projection head (2-layer MLP)
    encoded
    |> Axon.dense(hidden_size, name: "proj_fc1")
    |> Axon.activation(:relu, name: "proj_relu")
    |> Axon.dense(projection_dim, name: "proj_fc2")
  end

  @doc """
  Compute the NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

  Given embeddings from two views of the same batch, treats (z_i[k], z_j[k])
  as positive pairs and all other combinations as negatives.

  ## Parameters
    - `z_i` - Embeddings from view 1: [batch, projection_dim]
    - `z_j` - Embeddings from view 2: [batch, projection_dim]

  ## Options
    - `:temperature` - Temperature scaling (default: 0.5)

  ## Returns
    Scalar loss tensor.
  """
  @spec nt_xent_loss(Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  defn nt_xent_loss(z_i, z_j, opts \\ []) do
    temperature =
      case opts[:temperature] do
        nil -> 0.5
        val -> val
      end

    # L2 normalize
    z_i = l2_normalize(z_i)
    z_j = l2_normalize(z_j)

    batch_size = Nx.axis_size(z_i, 0)

    # Cosine similarity matrix between all pairs
    # Concatenate: [2*batch, dim]
    z = Nx.concatenate([z_i, z_j], axis: 0)

    # sim: [2*batch, 2*batch]
    sim = Nx.dot(z, [1], z, [1]) / temperature

    # Positive pairs: (i, i+batch) and (i+batch, i)
    # Create labels: for row i, positive is at column i+batch
    # For row i+batch, positive is at column i
    labels_top = Nx.iota({batch_size}) |> Nx.add(batch_size)
    labels_bottom = Nx.iota({batch_size})
    labels = Nx.concatenate([labels_top, labels_bottom])

    # Mask out self-similarity (diagonal)
    n = 2 * batch_size
    mask = 1.0 - Nx.eye(n)
    sim = sim * mask + Nx.eye(n) * -1.0e9

    # Cross-entropy loss
    # log_softmax along rows, then gather the positive positions
    log_probs = log_softmax(sim)

    # Gather positive log-probs using one-hot encoding
    one_hot_labels = Nx.equal(Nx.new_axis(Nx.iota({n}), 1), Nx.new_axis(labels, 0))
    one_hot_labels = Nx.as_type(one_hot_labels, Nx.type(log_probs))

    positive_log_probs = Nx.sum(log_probs * one_hot_labels, axes: [1])

    -Nx.mean(positive_log_probs)
  end

  defnp l2_normalize(x) do
    norm = Nx.sqrt(Nx.sum(x * x, axes: [1], keep_axes: true) + 1.0e-8)
    x / norm
  end

  defnp log_softmax(x) do
    max_val = Nx.reduce_max(x, axes: [1], keep_axes: true)
    shifted = x - max_val
    shifted - Nx.log(Nx.sum(Nx.exp(shifted), axes: [1], keep_axes: true))
  end

  @doc """
  Get the output size of the SimCLR model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :projection_dim, default_projection_dim())
  end
end
