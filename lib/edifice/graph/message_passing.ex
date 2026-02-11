defmodule Edifice.Graph.MessagePassing do
  @moduledoc """
  Generic Message Passing Neural Network (MPNN) framework.

  Implements the message passing paradigm from "Neural Message Passing for
  Quantum Chemistry" (Gilmer et al., 2017). This module provides the building
  blocks for constructing graph neural networks where information propagates
  along edges.

  ## Architecture

  ```
  Node Features [batch, num_nodes, feature_dim]
  Adjacency     [batch, num_nodes, num_nodes]
        |
        v
  +----------------------------+
  | For each edge (i,j):       |
  |   m_ij = msg_fn(h_i, h_j) |
  +----------------------------+
        |
        v
  +----------------------------+
  | Aggregate per node:        |
  |   M_i = AGG({m_ij : j->i}) |
  +----------------------------+
        |
        v
  +----------------------------+
  | Update node features:      |
  |   h_i' = update(h_i, M_i) |
  +----------------------------+
        |
        v
  Updated Node Features [batch, num_nodes, feature_dim']
  ```

  ## Graph Representation

  Graphs are represented as dense adjacency matrices since Nx does not support
  sparse tensors. For large graphs, consider batching subgraphs.

  - Adjacency: `{batch, num_nodes, num_nodes}` - binary or weighted
  - Node features: `{batch, num_nodes, feature_dim}`

  ## Usage

      # Generic message passing step
      nodes = Axon.input("nodes", shape: {nil, 10, 64})
      adj = Axon.input("adjacency", shape: {nil, 10, 10})

      updated = MessagePassing.message_passing_layer(nodes, adj, 64,
        name: "mpnn_1", activation: :relu)

      # Graph-level output via pooling
      graph_repr = MessagePassing.global_pool(updated, :mean)
  """

  require Axon

  @type aggregation :: :sum | :mean | :max

  # ============================================================================
  # Message Passing Layer
  # ============================================================================

  @doc """
  Generic message passing step.

  For each edge (i, j) in the graph, computes a message from the features of
  nodes i and j, aggregates incoming messages per node, and updates node
  features via a learned transformation.

  The message function concatenates sender and receiver features, then applies
  a dense layer. The update function concatenates the node's current features
  with aggregated messages, then applies a dense layer.

  ## Parameters

  - `nodes` - Node features Axon node `{batch, num_nodes, feature_dim}`
  - `adjacency` - Adjacency matrix Axon node `{batch, num_nodes, num_nodes}`
  - `output_dim` - Output feature dimension per node
  - `opts` - Options

  ## Options

  - `:name` - Layer name prefix (default: "mpnn")
  - `:activation` - Activation function (default: :relu)
  - `:aggregation` - Message aggregation: :sum, :mean, :max (default: :sum)
  - `:dropout` - Dropout rate (default: 0.0)

  ## Returns

  Axon node with shape `{batch, num_nodes, output_dim}`.
  """
  @spec message_passing_layer(Axon.t(), Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def message_passing_layer(nodes, adjacency, output_dim, opts \\ []) do
    name = Keyword.get(opts, :name, "mpnn")
    activation = Keyword.get(opts, :activation, :relu)
    aggregation = Keyword.get(opts, :aggregation, :sum)
    dropout = Keyword.get(opts, :dropout, 0.0)

    # Message function: for each edge (i,j), compute message from h_i, h_j
    # We use adjacency-masked matrix operations for efficiency.
    #
    # Strategy: transform node features, then use adjacency to gather/aggregate.
    # msg_ij = W_msg * [h_i || h_j] approximated as W1*h_i + W2*h_j masked by A

    # Transform features for message computation
    sender_proj = Axon.dense(nodes, output_dim, name: "#{name}_sender_proj")
    receiver_proj = Axon.dense(nodes, output_dim, name: "#{name}_receiver_proj")

    # Compute messages and aggregate using adjacency matrix
    aggregated_messages =
      Axon.layer(
        &message_aggregate_impl/4,
        [sender_proj, receiver_proj, adjacency],
        name: "#{name}_aggregate",
        aggregation: aggregation,
        op_name: :message_aggregate
      )

    # Update: combine node features with aggregated messages
    combined =
      Axon.concatenate([nodes, aggregated_messages], axis: 2, name: "#{name}_combine")

    updated =
      combined
      |> Axon.dense(output_dim, name: "#{name}_update")
      |> Axon.activation(activation, name: "#{name}_activation")

    if dropout > 0.0 do
      Axon.dropout(updated, rate: dropout, name: "#{name}_dropout")
    else
      updated
    end
  end

  # Message computation and aggregation implementation.
  # sender_proj: [batch, num_nodes, dim] - projected sender features
  # receiver_proj: [batch, num_nodes, dim] - projected receiver features
  # adjacency: [batch, num_nodes, num_nodes] - edge mask
  defp message_aggregate_impl(sender_proj, receiver_proj, adjacency, opts) do
    aggregation = opts[:aggregation] || :sum

    # For each node i, aggregate messages from all neighbors j:
    # messages_to_i = sum_j( A[i,j] * (sender_proj[j] + receiver_proj[i]) )
    #
    # Using matmul: A @ sender_proj gives sum of sender features weighted by edges.
    # This is the standard GNN aggregation: aggregate neighbor features.

    # Neighbor messages: A @ sender_proj => [batch, num_nodes, dim]
    # Each row i gets the sum of sender_proj[j] for all j where A[i,j] > 0
    neighbor_msgs = Nx.dot(adjacency, [2], [0], sender_proj, [1], [0])

    # Add self (receiver) contribution
    messages = Nx.add(neighbor_msgs, receiver_proj)

    case aggregation do
      :sum ->
        messages

      :mean ->
        # Normalize by node degree (number of neighbors)
        degree = Nx.sum(adjacency, axes: [2], keep_axes: true)
        # Avoid division by zero
        degree = Nx.max(degree, 1.0)
        Nx.divide(neighbor_msgs, degree) |> Nx.add(receiver_proj)

      :max ->
        # For max aggregation, we mask non-neighbors with large negative values
        # and take max over the neighbor dimension
        mask = Nx.greater(adjacency, 0)
        # Expand sender_proj for broadcasting: [batch, 1, num_nodes, dim]
        sender_expanded = Nx.new_axis(sender_proj, 1)
        # Expand mask: [batch, num_nodes, num_nodes, 1]
        mask_expanded = Nx.new_axis(mask, 3)

        # Broadcast sender features to all potential receivers
        # [batch, num_nodes, num_nodes, dim]
        broadcast_msgs = Nx.multiply(sender_expanded, mask_expanded)

        # Replace masked positions with very negative values
        neg_inf = Nx.broadcast(-1.0e9, Nx.shape(broadcast_msgs))

        masked =
          Nx.select(
            Nx.broadcast(mask_expanded, Nx.shape(broadcast_msgs)),
            broadcast_msgs,
            neg_inf
          )

        # Max over neighbors (axis 2)
        max_msgs = Nx.reduce_max(masked, axes: [2])
        Nx.add(max_msgs, receiver_proj)
    end
  end

  # ============================================================================
  # Aggregation Functions
  # ============================================================================

  @doc """
  Aggregate messages from neighbors using the specified method.

  This is a standalone aggregation function that operates on pre-computed
  messages. For most use cases, `message_passing_layer/4` handles aggregation
  internally.

  ## Parameters

  - `messages` - Message tensor `{batch, num_nodes, num_neighbors, msg_dim}`
  - `adjacency` - Adjacency matrix `{batch, num_nodes, num_nodes}`
  - `mode` - Aggregation mode: :sum, :mean, or :max

  ## Returns

  Aggregated messages `{batch, num_nodes, msg_dim}`.
  """
  @spec aggregate(Axon.t(), Axon.t(), aggregation()) :: Axon.t()
  def aggregate(node_features, adjacency, mode) do
    Axon.layer(
      &aggregate_impl/3,
      [node_features, adjacency],
      mode: mode,
      op_name: :graph_aggregate
    )
  end

  defp aggregate_impl(node_features, adjacency, opts) do
    mode = opts[:mode] || :sum

    # A @ H gives neighbor feature sums per node
    aggregated = Nx.dot(adjacency, [2], node_features, [1])

    case mode do
      :sum ->
        aggregated

      :mean ->
        degree = Nx.sum(adjacency, axes: [2], keep_axes: true)
        degree = Nx.max(degree, 1.0)
        Nx.divide(aggregated, degree)

      :max ->
        # Use masking approach for max
        mask = Nx.greater(adjacency, 0)
        features_expanded = Nx.new_axis(node_features, 1)
        mask_expanded = Nx.new_axis(mask, 3)

        broadcast_feats = Nx.multiply(features_expanded, mask_expanded)
        neg_inf = Nx.broadcast(-1.0e9, Nx.shape(broadcast_feats))

        masked =
          Nx.select(
            Nx.broadcast(mask_expanded, Nx.shape(broadcast_feats)),
            broadcast_feats,
            neg_inf
          )

        Nx.reduce_max(masked, axes: [2])
    end
  end

  # ============================================================================
  # Global Pooling
  # ============================================================================

  @doc """
  Pool node features to a graph-level representation.

  Reduces the node dimension by aggregating all node features in each graph
  into a single vector. Essential for graph classification tasks.

  ## Parameters

  - `node_features` - Axon node with shape `{batch, num_nodes, feature_dim}`
  - `mode` - Pooling mode: :sum, :mean, or :max (default: :mean)

  ## Returns

  Axon node with shape `{batch, feature_dim}`.
  """
  @spec global_pool(Axon.t(), aggregation()) :: Axon.t()
  def global_pool(node_features, mode \\ :mean) do
    Axon.nx(
      node_features,
      fn features ->
        # features: [batch, num_nodes, feature_dim]
        case mode do
          :sum -> Nx.sum(features, axes: [1])
          :mean -> Nx.mean(features, axes: [1])
          :max -> Nx.reduce_max(features, axes: [1])
        end
      end,
      name: "global_pool_#{mode}"
    )
  end
end
