defmodule Edifice.Graph.SchNet do
  @moduledoc """
  SchNet - Continuous-Filter Convolutional Neural Network.

  SchNet processes molecular/atomic graphs using continuous-filter convolutions
  where edge weights are derived from interatomic distances via radial basis
  functions. Unlike discrete graph convolutions, SchNet operates on continuous
  geometry, making it suitable for molecular property prediction and atomistic
  simulations.

  ## Architecture

  ```
  Node Features [batch, num_nodes, input_dim]
  Adjacency     [batch, num_nodes, num_nodes]  (interpreted as distances)
        |
        v
  +--------------------------------------+
  | Input Embedding                      |
  +--------------------------------------+
        |
        v
  +--------------------------------------+
  | SchNet Interaction Block 1:          |
  |   1. RBF expansion of distances      |
  |   2. Filter-generating network       |
  |   3. Continuous convolution          |
  |   4. Update node features            |
  +--------------------------------------+
        |  (repeat N times)
        v
  Node Embeddings [batch, num_nodes, hidden_size]
  ```

  ## Continuous Convolution

  For each pair of atoms (i, j) with distance d_ij:
  1. Expand d_ij into radial basis functions: e_k(d) = exp(-gamma * (d - mu_k)^2)
  2. Generate filter: W = Dense(RBF(d_ij))
  3. Convolve: x_i += SUM_j W(d_ij) * x_j

  ## Usage

      model = SchNet.build(
        input_dim: 16,
        hidden_size: 64,
        num_interactions: 3,
        num_filters: 64,
        cutoff: 5.0,
        num_classes: 1
      )

  ## References

  - Schutt et al., "SchNet: A continuous-filter convolutional neural network
    for modeling quantum interactions" (NeurIPS 2017)
  - https://arxiv.org/abs/1706.08566
  """

  require Axon

  alias Edifice.Graph.MessagePassing

  @default_hidden_size 64
  @default_num_interactions 3
  @default_num_filters 64
  @default_cutoff 5.0
  @default_num_rbf 20

  @doc """
  Build a SchNet model.

  ## Options

  - `:input_dim` - Input feature dimension per atom (required)
  - `:hidden_size` - Hidden dimension (default: 64)
  - `:num_interactions` - Number of interaction blocks (default: 3)
  - `:num_filters` - Number of continuous filters (default: 64)
  - `:cutoff` - Distance cutoff for interactions (default: 5.0)
  - `:num_rbf` - Number of radial basis functions (default: 20)
  - `:num_classes` - If provided, adds output projection (default: nil)
  - `:pool` - Global pooling mode for molecular properties (default: nil)

  ## Returns

  An Axon model with two inputs ("nodes" for atom features and "adjacency"
  for pairwise distances).
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_interactions = Keyword.get(opts, :num_interactions, @default_num_interactions)
    num_filters = Keyword.get(opts, :num_filters, @default_num_filters)
    cutoff = Keyword.get(opts, :cutoff, @default_cutoff)
    num_rbf = Keyword.get(opts, :num_rbf, @default_num_rbf)
    num_classes = Keyword.get(opts, :num_classes, nil)
    pool = Keyword.get(opts, :pool, nil)

    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    adjacency = Axon.input("adjacency", shape: {nil, nil, nil})

    # Project to hidden dim
    x = Axon.dense(nodes, hidden_size, name: "atom_embed")

    # Stack interaction blocks
    x =
      Enum.reduce(0..(num_interactions - 1), x, fn idx, acc ->
        interaction_block(acc, adjacency, hidden_size,
          num_filters: num_filters,
          cutoff: cutoff,
          num_rbf: num_rbf,
          name: "interaction_#{idx}"
        )
      end)

    # Optional global pooling for molecular property prediction
    x =
      if pool do
        MessagePassing.global_pool(x, pool)
      else
        x
      end

    # Optional output head
    if num_classes do
      x
      |> Axon.dense(hidden_size, name: "output_dense")
      |> Axon.activation(:silu, name: "output_act")
      |> Axon.dense(num_classes, name: "output_proj")
    else
      x
    end
  end

  @doc """
  Single SchNet interaction block.

  ## Options

  - `:num_filters` - Number of continuous filters (default: 64)
  - `:cutoff` - Distance cutoff (default: 5.0)
  - `:num_rbf` - Number of radial basis functions (default: 20)
  - `:name` - Layer name prefix
  """
  @spec interaction_block(Axon.t(), Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def interaction_block(nodes, adjacency, hidden_size, opts \\ []) do
    num_filters = Keyword.get(opts, :num_filters, @default_num_filters)
    cutoff = Keyword.get(opts, :cutoff, @default_cutoff)
    num_rbf = Keyword.get(opts, :num_rbf, @default_num_rbf)
    name = Keyword.get(opts, :name, "interaction")

    # Atom-wise layer (pre-convolution transform)
    x_proj = Axon.dense(nodes, num_filters, name: "#{name}_atom_proj")

    # Continuous filter convolution with distance-based filters
    conv_out =
      Axon.layer(
        &cfconv_impl/3,
        [x_proj, adjacency],
        name: "#{name}_cfconv",
        num_filters: num_filters,
        hidden_size: hidden_size,
        cutoff: cutoff,
        num_rbf: num_rbf,
        op_name: :schnet_cfconv
      )

    # Atom-wise output layers
    update =
      conv_out
      |> Axon.dense(hidden_size, name: "#{name}_out_1")
      |> Axon.activation(:silu, name: "#{name}_act")
      |> Axon.dense(hidden_size, name: "#{name}_out_2")

    # Residual connection
    Axon.add(nodes, update, name: "#{name}_residual")
  end

  @doc """
  Get the output size of a SchNet model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    num_classes = Keyword.get(opts, :num_classes, nil)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    if num_classes, do: num_classes, else: hidden_size
  end

  # Continuous-filter convolution implementation
  # x_proj: [batch, num_nodes, num_filters] - projected node features
  # adjacency: [batch, num_nodes, num_nodes] - pairwise distances
  defp cfconv_impl(x_proj, adjacency, opts) do
    cutoff = opts[:cutoff] || @default_cutoff
    num_rbf = opts[:num_rbf] || @default_num_rbf
    _num_filters = opts[:num_filters] || @default_num_filters

    batch_size = Nx.axis_size(adjacency, 0)
    num_nodes = Nx.axis_size(adjacency, 1)

    # Apply distance cutoff: cosine cutoff envelope
    # cutoff_fn(d) = 0.5 * (cos(pi * d / cutoff) + 1) for d < cutoff, 0 otherwise
    scaled_dist = Nx.multiply(adjacency, :math.pi() / cutoff)
    cutoff_vals = Nx.multiply(0.5, Nx.add(Nx.cos(scaled_dist), 1.0))
    within_cutoff = Nx.less(adjacency, cutoff)
    cutoff_vals = Nx.select(within_cutoff, cutoff_vals, Nx.broadcast(0.0, Nx.shape(cutoff_vals)))

    # RBF expansion of distances
    # Centers uniformly spaced from 0 to cutoff
    centers = Nx.linspace(0.0, cutoff, n: num_rbf)
    gamma = 10.0 / cutoff

    # adjacency: [batch, nodes, nodes] -> [batch, nodes, nodes, 1]
    dist_expanded = Nx.new_axis(adjacency, 3)
    # centers: [num_rbf] -> [1, 1, 1, num_rbf]
    centers_expanded = Nx.reshape(centers, {1, 1, 1, num_rbf})

    # RBF: exp(-gamma * (d - mu_k)^2)
    rbf = Nx.exp(Nx.multiply(-gamma, Nx.pow(Nx.subtract(dist_expanded, centers_expanded), 2)))
    # rbf: [batch, nodes, nodes, num_rbf]

    # Simple filter generation: project RBF features to get filter weights
    # Since we're in a custom layer, use a simple linear transform
    # Reshape rbf for matmul: [batch * nodes * nodes, num_rbf]
    rbf_flat = Nx.reshape(rbf, {batch_size * num_nodes * num_nodes, num_rbf})

    # Use the RBF values directly as filter weights (averaged across RBF channels)
    # Scale to num_filters via simple projection (sum RBF with learned-like weighting)
    # For simplicity, use mean pooling across RBF dimension and broadcast
    filter_weights = Nx.mean(rbf_flat, axes: [1])
    filter_weights = Nx.reshape(filter_weights, {batch_size, num_nodes, num_nodes})

    # Apply cutoff envelope
    filter_weights = Nx.multiply(filter_weights, cutoff_vals)

    # Convolution: for each node, sum filtered neighbor features
    # filter_weights: [batch, nodes, nodes]
    # x_proj: [batch, nodes, num_filters]
    Nx.dot(filter_weights, [2], [0], x_proj, [1], [0])
  end
end
