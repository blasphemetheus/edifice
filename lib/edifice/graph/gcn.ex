defmodule Edifice.Graph.GCN do
  @moduledoc """
  Graph Convolutional Network (Kipf & Welling, 2017).

  Implements spectral graph convolutions approximated by first-order Chebyshev
  polynomials. Each GCN layer propagates and transforms node features using the
  graph structure via the normalized adjacency matrix.

  ## Architecture

  ```
  Node Features [batch, num_nodes, input_dim]
  Adjacency     [batch, num_nodes, num_nodes]
        |
        v
  +------------------------------------+
  | GCN Layer 1:                       |
  |   H' = sigma(D^-1/2 A D^-1/2 H W) |
  +------------------------------------+
        |
        v
  +------------------------------------+
  | GCN Layer 2:                       |
  |   H' = sigma(D^-1/2 A D^-1/2 H W) |
  +------------------------------------+
        |
        v
  Node Embeddings [batch, num_nodes, hidden_dim]
  ```

  The normalized adjacency `D^{-1/2} A D^{-1/2}` ensures symmetric
  normalization that prevents feature magnitudes from scaling with node degree.

  ## Graph Classification

  For graph-level tasks, use `build_classifier/1` which adds global pooling
  and a dense classification head on top of the GCN layers.

  ## Usage

      # Node classification
      model = GCN.build(input_dim: 16, hidden_dims: [64, 32], num_classes: 7)

      # Graph classification
      model = GCN.build_classifier(
        input_dim: 16,
        hidden_dims: [64, 64],
        num_classes: 2,
        pool: :mean
      )

  ## References

  - "Semi-Supervised Classification with Graph Convolutional Networks"
    (Kipf & Welling, ICLR 2017)
  """

  alias Edifice.Graph.MessagePassing

  @default_hidden_dims [64, 64]
  @default_activation :relu
  @default_dropout 0.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Graph Convolutional Network.

  Stacks multiple GCN layers that propagate features along edges. The final
  output is per-node embeddings suitable for node classification or as input
  to a graph-level pooling layer.

  ## Options

  - `:input_dim` - Input feature dimension per node (required)
  - `:hidden_dims` - List of hidden dimensions for each GCN layer (default: [64, 64])
  - `:num_classes` - If provided, adds a final classification layer (default: nil)
  - `:activation` - Activation function (default: :relu)
  - `:dropout` - Dropout rate between layers (default: 0.0)

  ## Returns

  An Axon model with two inputs ("nodes" and "adjacency"). Output shape is
  `{batch, num_nodes, last_hidden_dim}` or `{batch, num_nodes, num_classes}`
  if `:num_classes` is set.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:dropout, float()}
          | {:hidden_dims, pos_integer()}
          | {:input_dim, pos_integer()}
          | {:num_classes, pos_integer() | nil}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    num_classes = Keyword.get(opts, :num_classes, nil)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Graph inputs
    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    adjacency = Axon.input("adjacency", shape: {nil, nil, nil})

    # Stack GCN layers
    output =
      hidden_dims
      |> Enum.with_index()
      |> Enum.reduce(nodes, fn {hidden_dim, idx}, acc ->
        gcn_layer(acc, adjacency, hidden_dim,
          name: "gcn_#{idx}",
          activation: activation,
          dropout: dropout
        )
      end)

    # Optional classification head (per-node)
    if num_classes do
      Axon.dense(output, num_classes, name: "gcn_classifier")
    else
      output
    end
  end

  @doc """
  Build a GCN with global pooling and dense classifier for graph classification.

  Adds a global pooling layer after the GCN layers to produce a single vector
  per graph, followed by a dense classification head.

  ## Options

  All options from `build/1`, plus:

  - `:pool` - Global pooling mode: :sum, :mean, :max (default: :mean)
  - `:classifier_dims` - Hidden dims for classification MLP (default: [64])

  ## Returns

  An Axon model outputting `{batch, num_classes}`.
  """
  @spec build_classifier(keyword()) :: Axon.t()
  def build_classifier(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    num_classes = Keyword.fetch!(opts, :num_classes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    pool_mode = Keyword.get(opts, :pool, :mean)
    classifier_dims = Keyword.get(opts, :classifier_dims, [64])

    # Graph inputs
    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    adjacency = Axon.input("adjacency", shape: {nil, nil, nil})

    # GCN layers for node embeddings
    node_embeddings =
      hidden_dims
      |> Enum.with_index()
      |> Enum.reduce(nodes, fn {hidden_dim, idx}, acc ->
        gcn_layer(acc, adjacency, hidden_dim,
          name: "gcn_#{idx}",
          activation: activation,
          dropout: dropout
        )
      end)

    # Global pooling: [batch, num_nodes, dim] -> [batch, dim]
    graph_repr = MessagePassing.global_pool(node_embeddings, pool_mode)

    # Classification MLP
    output =
      classifier_dims
      |> Enum.with_index()
      |> Enum.reduce(graph_repr, fn {dim, idx}, acc ->
        layer =
          acc
          |> Axon.dense(dim, name: "classifier_dense_#{idx}")
          |> Axon.activation(activation, name: "classifier_act_#{idx}")

        if dropout > 0.0 do
          Axon.dropout(layer, rate: dropout, name: "classifier_drop_#{idx}")
        else
          layer
        end
      end)

    Axon.dense(output, num_classes, name: "classifier_output")
  end

  # ============================================================================
  # GCN Layer
  # ============================================================================

  @doc """
  Single Graph Convolutional layer.

  Implements the spectral convolution rule:

      H' = sigma(D^{-1/2} A D^{-1/2} H W)

  where A is the adjacency matrix (with self-loops added), D is the degree
  matrix, H is the node feature matrix, and W is a learnable weight matrix.

  ## Parameters

  - `nodes` - Node features Axon node `{batch, num_nodes, in_dim}`
  - `adjacency` - Adjacency matrix Axon node `{batch, num_nodes, num_nodes}`
  - `output_dim` - Output feature dimension

  ## Options

  - `:name` - Layer name prefix (default: "gcn")
  - `:activation` - Activation function (default: :relu)
  - `:dropout` - Dropout rate (default: 0.0)
  - `:add_self_loops` - Add self-loops to adjacency (default: true)

  ## Returns

  Axon node with shape `{batch, num_nodes, output_dim}`.
  """
  @spec gcn_layer(Axon.t(), Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def gcn_layer(nodes, adjacency, output_dim, opts \\ []) do
    name = Keyword.get(opts, :name, "gcn")
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    add_self_loops = Keyword.get(opts, :add_self_loops, true)

    # Linear transform: H W
    transformed = Axon.dense(nodes, output_dim, name: "#{name}_weight")

    # Apply normalized adjacency: D^{-1/2} A D^{-1/2} (H W)
    propagated =
      Axon.layer(
        &gcn_propagate_impl/3,
        [transformed, adjacency],
        name: "#{name}_propagate",
        add_self_loops: add_self_loops,
        op_name: :gcn_propagate
      )

    # Activation
    activated = Axon.activation(propagated, activation, name: "#{name}_activation")

    # Optional dropout
    if dropout > 0.0 do
      Axon.dropout(activated, rate: dropout, name: "#{name}_dropout")
    else
      activated
    end
  end

  # GCN propagation: D^{-1/2} A D^{-1/2} H
  # transformed: [batch, num_nodes, dim]
  # adjacency: [batch, num_nodes, num_nodes]
  defp gcn_propagate_impl(transformed, adjacency, opts) do
    add_self_loops = opts[:add_self_loops] != false

    # Add self-loops: A_hat = A + I
    adj =
      if add_self_loops do
        num_nodes = Nx.axis_size(adjacency, 1)
        eye = Nx.eye(num_nodes)
        Nx.add(adjacency, eye)
      else
        adjacency
      end

    # Compute degree matrix: D_ii = sum_j A_hat_ij
    degree = Nx.sum(adj, axes: [2])

    # D^{-1/2}: inverse square root of degree
    # Add epsilon to avoid division by zero for isolated nodes
    d_inv_sqrt = Nx.rsqrt(Nx.max(degree, 1.0e-6))

    # Apply symmetric normalization: D^{-1/2} A D^{-1/2} H
    # Step 1: D^{-1/2} on the right side of A: scale columns of A
    # D^{-1/2} is diagonal, so A * D^{-1/2} = A .* d_inv_sqrt[j] per column
    # [batch, 1, num_nodes]
    d_right = Nx.new_axis(d_inv_sqrt, 1)
    adj_scaled = Nx.multiply(adj, d_right)

    # Step 2: D^{-1/2} on the left side: scale rows of result
    # [batch, num_nodes, 1]
    d_left = Nx.new_axis(d_inv_sqrt, 2)
    adj_norm = Nx.multiply(adj_scaled, d_left)

    # Propagate: A_norm @ H (batched matrix multiply)
    # adj_norm: [batch, num_nodes, num_nodes], transformed: [batch, num_nodes, dim]
    # Contract adj axis 2 with transformed axis 1, keeping batch axis 0 aligned
    Nx.dot(adj_norm, [2], [0], transformed, [1], [0])
  end
end
