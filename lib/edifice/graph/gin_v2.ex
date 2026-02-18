defmodule Edifice.Graph.GINv2 do
  @moduledoc """
  GINv2: Graph Isomorphism Network with edge features (Hu et al., 2020).

  Extends the original GIN architecture to incorporate edge features in the
  message passing step. For each edge (u, v), the message incorporates both
  the source node features and the edge features, allowing the network to
  learn from edge attributes such as bond types, distances, or relationship
  properties.

  ## Architecture

  ```
  Node Features  [batch, num_nodes, input_dim]
  Adjacency      [batch, num_nodes, num_nodes]
  Edge Features  [batch, num_nodes, num_nodes, edge_dim]
        |
        v
  +----------------------------------------------+
  | GINv2 Layer:                                  |
  |   1. Edge messages: Dense(h_u + edge_proj)    |
  |   2. Aggregate: sum over neighbors (via adj)  |
  |   3. Combine: (1+eps)*h_v + aggregated        |
  |   4. Transform: MLP(combined)                 |
  +----------------------------------------------+
        |
        v
  Node Embeddings [batch, num_nodes, hidden_dim]
  ```

  ## Differences from GIN

  - Takes a third input "edge_features" with shape `[batch, nodes, nodes, edge_dim]`
  - Edge features are projected and combined with neighbor features before aggregation
  - More expressive for graphs with rich edge information (molecules, knowledge graphs)

  ## Usage

      model = GINv2.build(
        input_dim: 16,
        edge_dim: 4,
        hidden_dims: [64, 64],
        num_classes: 2
      )

  ## References

  - Hu et al., "Strategies for Pre-training Graph Neural Networks" (ICLR 2020)
  - https://arxiv.org/abs/1905.12265
  """

  alias Edifice.Graph.MessagePassing

  @default_hidden_dims [64, 64]
  @default_activation :relu
  @default_dropout 0.0

  @doc """
  Build a GINv2 model with edge features.

  ## Options

    - `:input_dim` - Input feature dimension per node (required)
    - `:edge_dim` - Edge feature dimension (required)
    - `:hidden_dims` - List of hidden dimensions for each layer (default: [64, 64])
    - `:num_classes` - If provided, adds a classification head (default: nil)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:activation` - Activation for MLPs (default: :relu)
    - `:pool` - Global pooling for graph classification: :mean, :sum, :max (default: nil)

  ## Returns

    An Axon model with three inputs: "nodes", "adjacency", and "edge_features".
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:dropout, float()}
          | {:edge_dim, pos_integer()}
          | {:hidden_dims, [pos_integer()]}
          | {:input_dim, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:pool, :mean | :sum | :max | nil}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    edge_dim = Keyword.fetch!(opts, :edge_dim)
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    num_classes = Keyword.get(opts, :num_classes, nil)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    activation = Keyword.get(opts, :activation, @default_activation)
    pool = Keyword.get(opts, :pool, nil)

    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    adjacency = Axon.input("adjacency", shape: {nil, nil, nil})
    edge_features = Axon.input("edge_features", shape: {nil, nil, nil, edge_dim})

    # Stack GINv2 layers
    output =
      hidden_dims
      |> Enum.with_index()
      |> Enum.reduce(nodes, fn {hidden_dim, idx}, acc ->
        ginv2_layer(acc, adjacency, edge_features, hidden_dim,
          name: "ginv2_#{idx}",
          dropout: dropout,
          activation: activation
        )
      end)

    # Optional global pooling
    output =
      if pool do
        MessagePassing.global_pool(output, pool)
      else
        output
      end

    # Optional classification head
    if num_classes do
      Axon.dense(output, num_classes, name: "ginv2_classifier")
    else
      output
    end
  end

  @doc """
  Get the output size of a GINv2 model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    num_classes = Keyword.get(opts, :num_classes, nil)
    if num_classes, do: num_classes, else: List.last(hidden_dims)
  end

  @doc """
  Single GINv2 layer with edge features.

  ## Options

    - `:name` - Layer name prefix (default: "ginv2")
    - `:dropout` - Dropout rate (default: 0.0)
    - `:activation` - Activation for MLP (default: :relu)
  """
  @spec ginv2_layer(Axon.t(), Axon.t(), Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def ginv2_layer(nodes, adjacency, edge_features, output_dim, opts \\ []) do
    name = Keyword.get(opts, :name, "ginv2")
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    activation = Keyword.get(opts, :activation, @default_activation)

    # Project edge features to match node feature dimension
    edge_proj = Axon.dense(edge_features, output_dim, name: "#{name}_edge_proj")

    # Aggregate neighbor features incorporating edge features:
    # For each node v, sum over neighbors u: (h_u_proj + edge_{u,v}_proj) * A[v,u]
    node_proj = Axon.dense(nodes, output_dim, name: "#{name}_node_proj")

    aggregated =
      Axon.layer(
        &ginv2_aggregate_impl/4,
        [node_proj, edge_proj, adjacency],
        name: "#{name}_aggregate",
        op_name: :ginv2_aggregate
      )

    # Combine: (1 + eps) * h_v_proj + aggregated
    # Project self features to output_dim first, then apply learnable epsilon
    self_proj = Axon.dense(nodes, output_dim, name: "#{name}_self_proj")
    eps_proj = Axon.dense(nodes, 1, name: "#{name}_eps", use_bias: false)

    scaled_self =
      Axon.layer(
        fn self_feats, eps_val, _opts ->
          eps = Nx.add(1.0, eps_val)
          Nx.multiply(self_feats, eps)
        end,
        [self_proj, eps_proj],
        name: "#{name}_scale_self",
        op_name: :ginv2_scale
      )

    combined = Axon.add(scaled_self, aggregated, name: "#{name}_combine")

    # MLP: 2-layer with activation
    mlp_output =
      combined
      |> Axon.dense(output_dim, name: "#{name}_mlp_1")
      |> Axon.activation(activation, name: "#{name}_mlp_act_1")
      |> Axon.dense(output_dim, name: "#{name}_mlp_2")
      |> Axon.activation(activation, name: "#{name}_mlp_act_2")

    if dropout > 0.0 do
      Axon.dropout(mlp_output, rate: dropout, name: "#{name}_dropout")
    else
      mlp_output
    end
  end

  # Aggregate neighbor features with edge features
  # node_proj: [batch, num_nodes, dim]
  # edge_proj: [batch, num_nodes, num_nodes, dim]
  # adjacency: [batch, num_nodes, num_nodes]
  defp ginv2_aggregate_impl(node_proj, edge_proj, adjacency, _opts) do
    # Expand node features for each potential source: [batch, 1, num_nodes, dim]
    node_expanded = Nx.new_axis(node_proj, 1)

    # Combine node and edge features: [batch, num_nodes, num_nodes, dim]
    combined = Nx.add(node_expanded, edge_proj)

    # Mask by adjacency: expand adj to [batch, num_nodes, num_nodes, 1]
    adj_mask = Nx.new_axis(adjacency, 3)
    masked = Nx.multiply(combined, adj_mask)

    # Sum over source nodes (axis 2): [batch, num_nodes, dim]
    Nx.sum(masked, axes: [2])
  end
end
