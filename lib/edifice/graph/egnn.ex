defmodule Edifice.Graph.EGNN do
  @moduledoc """
  E(n) Equivariant Graph Neural Network.

  EGNN processes graphs with 3D (or n-D) coordinates while preserving Euclidean
  symmetries: rotation, translation, and reflection. This makes it ideal for
  molecular simulations, protein structure prediction, and physical systems
  where the laws are invariant to these transformations.

  ## Architecture

  ```
  Node Features [batch, num_nodes, node_dim]
  Coordinates   [batch, num_nodes, coord_dim]  (typically 3D positions)
  Edge Index    [batch, num_edges, 2]          (source, target pairs)
  Edge Features [batch, num_edges, edge_dim]   (optional)
        |
        v
  +--------------------------------------+
  | EGNN Layer 1:                        |
  |   1. Compute squared distances       |
  |   2. Edge message: φ_e(h_i, h_j, d²) |
  |   3. Coordinate update: equivariant  |
  |   4. Feature update: invariant       |
  +--------------------------------------+
        |  (repeat N times)
        v
  Updated Node Features [batch, num_nodes, hidden_dim]
  Updated Coordinates   [batch, num_nodes, coord_dim]
  ```

  ## Key Equations

  For each layer, given node features h and coordinates x:

  1. **Edge message**: `m_ij = φ_e(h_i, h_j, ||x_i - x_j||², a_ij)`
     - Uses squared distance (invariant scalar), not raw positions
     - φ_e is a small MLP

  2. **Coordinate update**: `x_i' = x_i + Σ_j (x_i - x_j) · φ_x(m_ij)`
     - The direction (x_i - x_j) is equivariant
     - Scaling by φ_x(m_ij) preserves equivariance

  3. **Feature update**: `h_i' = φ_h(h_i, Σ_j m_ij)`
     - Aggregated messages are invariant
     - φ_h is a small MLP

  ## Usage

      model = EGNN.build(
        in_node_features: 16,
        in_edge_features: 4,
        hidden_dim: 64,
        num_layers: 4,
        out_features: 32
      )

  ## References

  - Satorras et al., "E(n) Equivariant Graph Neural Networks" (NeurIPS 2021)
  - https://arxiv.org/abs/2102.09844
  """

  @default_hidden_dim 64
  @default_num_layers 4
  @default_coord_dim 3

  @doc """
  Build an EGNN model.

  ## Options

    - `:in_node_features` - Input node feature dimension (required)
    - `:in_edge_features` - Input edge feature dimension (default: 0)
    - `:hidden_dim` - Hidden dimension (default: 64)
    - `:num_layers` - Number of EGNN layers (default: 4)
    - `:out_features` - Output feature dimension (default: hidden_dim)
    - `:coord_dim` - Coordinate dimension, e.g., 3 for 3D (default: 3)
    - `:update_coords` - Whether to update coordinates (default: true)

  ## Returns

    An Axon model with inputs:
    - "nodes": Node features [batch, num_nodes, in_node_features]
    - "coords": Node coordinates [batch, num_nodes, coord_dim]
    - "edge_index": Edge indices [batch, num_edges, 2]
    - "edge_features": Edge features [batch, num_edges, in_edge_features] (optional)

    Returns a container with `node_features` and `coords`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:coord_dim, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:in_edge_features, non_neg_integer()}
          | {:in_node_features, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:out_features, pos_integer()}
          | {:update_coords, boolean()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    in_node_features = Keyword.fetch!(opts, :in_node_features)
    in_edge_features = Keyword.get(opts, :in_edge_features, 0)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    out_features = Keyword.get(opts, :out_features, hidden_dim)
    coord_dim = Keyword.get(opts, :coord_dim, @default_coord_dim)
    update_coords = Keyword.get(opts, :update_coords, true)

    # Inputs
    nodes = Axon.input("nodes", shape: {nil, nil, in_node_features})
    coords = Axon.input("coords", shape: {nil, nil, coord_dim})
    edge_index = Axon.input("edge_index", shape: {nil, nil, 2})

    edge_features =
      if in_edge_features > 0 do
        Axon.input("edge_features", shape: {nil, nil, in_edge_features})
      else
        nil
      end

    # Project node features to hidden dimension
    h = Axon.dense(nodes, hidden_dim, name: "node_embed")

    # Stack EGNN layers
    {h, x} =
      Enum.reduce(0..(num_layers - 1), {h, coords}, fn idx, {h_acc, x_acc} ->
        egnn_layer(h_acc, x_acc, edge_index, edge_features,
          hidden_dim: hidden_dim,
          in_edge_features: in_edge_features,
          update_coords: update_coords,
          name: "egnn_layer_#{idx}"
        )
      end)

    # Project to output dimension if different
    h =
      if out_features != hidden_dim do
        Axon.dense(h, out_features, name: "output_proj")
      else
        h
      end

    Axon.container(%{node_features: h, coords: x})
  end

  @doc """
  Single E(n)-equivariant graph neural network layer.

  ## Parameters

    - `node_feats` - Node features [batch, num_nodes, hidden_dim]
    - `coords` - Node coordinates [batch, num_nodes, coord_dim]
    - `edge_index` - Edge indices [batch, num_edges, 2]
    - `edge_features` - Optional edge features [batch, num_edges, edge_dim]

  ## Options

    - `:hidden_dim` - Hidden dimension
    - `:in_edge_features` - Edge feature dimension
    - `:update_coords` - Whether to update coordinates
    - `:name` - Layer name prefix

  ## Returns

    Tuple of {updated_node_feats, updated_coords}.
  """
  @spec egnn_layer(Axon.t(), Axon.t(), Axon.t(), Axon.t() | nil, keyword()) ::
          {Axon.t(), Axon.t()}
  def egnn_layer(node_feats, coords, edge_index, edge_features \\ nil, opts \\ []) do
    hidden_dim = Keyword.fetch!(opts, :hidden_dim)
    in_edge_features = Keyword.get(opts, :in_edge_features, 0)
    update_coords = Keyword.get(opts, :update_coords, true)
    name = Keyword.get(opts, :name, "egnn_layer")

    # Edge message MLP parameters
    # Input: h_i || h_j || d² || edge_features
    edge_mlp_in = hidden_dim * 2 + 1 + in_edge_features

    edge_w1 =
      Axon.param("#{name}_edge_w1", {edge_mlp_in, hidden_dim}, initializer: :glorot_uniform)

    edge_b1 = Axon.param("#{name}_edge_b1", {hidden_dim}, initializer: :zeros)
    edge_w2 = Axon.param("#{name}_edge_w2", {hidden_dim, hidden_dim}, initializer: :glorot_uniform)
    edge_b2 = Axon.param("#{name}_edge_b2", {hidden_dim}, initializer: :zeros)

    # Coordinate update MLP (scalar output)
    coord_w1 = Axon.param("#{name}_coord_w1", {hidden_dim, hidden_dim}, initializer: :glorot_uniform)
    coord_b1 = Axon.param("#{name}_coord_b1", {hidden_dim}, initializer: :zeros)
    coord_w2 = Axon.param("#{name}_coord_w2", {hidden_dim, 1}, initializer: :zeros)
    coord_b2 = Axon.param("#{name}_coord_b2", {1}, initializer: :zeros)

    # Node update MLP
    node_w1 =
      Axon.param("#{name}_node_w1", {hidden_dim * 2, hidden_dim}, initializer: :glorot_uniform)

    node_b1 = Axon.param("#{name}_node_b1", {hidden_dim}, initializer: :zeros)
    node_w2 = Axon.param("#{name}_node_w2", {hidden_dim, hidden_dim}, initializer: :glorot_uniform)
    node_b2 = Axon.param("#{name}_node_b2", {hidden_dim}, initializer: :zeros)

    # Build edge features input (or nil placeholder)
    edge_feat_input =
      if edge_features do
        edge_features
      else
        # Create a dummy zero tensor for the layer
        Axon.nx(node_feats, fn _ -> Nx.tensor(0.0) end, name: "#{name}_no_edge_feats")
      end

    # Compute EGNN layer
    layer_inputs =
      [
        node_feats,
        coords,
        edge_index,
        edge_feat_input,
        edge_w1,
        edge_b1,
        edge_w2,
        edge_b2,
        coord_w1,
        coord_b1,
        coord_w2,
        coord_b2,
        node_w1,
        node_b1,
        node_w2,
        node_b2
      ]

    result =
      Axon.layer(
        &egnn_layer_impl/17,
        layer_inputs,
        name: name,
        hidden_dim: hidden_dim,
        in_edge_features: in_edge_features,
        update_coords: update_coords,
        op_name: :egnn_layer
      )

    # Split result into node features and coordinates
    h_out =
      Axon.nx(result, fn r -> r.node_features end, name: "#{name}_h_out")

    x_out =
      Axon.nx(result, fn r -> r.coords end, name: "#{name}_x_out")

    {h_out, x_out}
  end

  # EGNN layer implementation
  defp egnn_layer_impl(
         node_feats,
         coords,
         edge_index,
         edge_features,
         edge_w1,
         edge_b1,
         edge_w2,
         edge_b2,
         coord_w1,
         coord_b1,
         coord_w2,
         coord_b2,
         node_w1,
         node_b1,
         node_w2,
         node_b2,
         opts
       ) do
    hidden_dim = opts[:hidden_dim]
    in_edge_features = opts[:in_edge_features] || 0
    update_coords = opts[:update_coords]

    batch_size = Nx.axis_size(node_feats, 0)
    num_nodes = Nx.axis_size(node_feats, 1)
    num_edges = Nx.axis_size(edge_index, 1)
    coord_dim = Nx.axis_size(coords, 2)

    # Get source and target indices
    # edge_index: [batch, num_edges, 2] -> src, tgt each [batch, num_edges]
    src_idx = Nx.slice_along_axis(edge_index, 0, 1, axis: 2) |> Nx.squeeze(axes: [2])
    tgt_idx = Nx.slice_along_axis(edge_index, 1, 1, axis: 2) |> Nx.squeeze(axes: [2])

    # Gather source and target node features and coordinates
    # node_feats: [batch, num_nodes, hidden_dim]
    # We need to gather for each edge
    h_src = gather_nodes(node_feats, src_idx, batch_size, num_edges, hidden_dim)
    h_tgt = gather_nodes(node_feats, tgt_idx, batch_size, num_edges, hidden_dim)
    x_src = gather_nodes(coords, src_idx, batch_size, num_edges, coord_dim)
    x_tgt = gather_nodes(coords, tgt_idx, batch_size, num_edges, coord_dim)

    # Compute squared distances (invariant)
    # ||x_i - x_j||² = sum((x_src - x_tgt)²)
    coord_diff = Nx.subtract(x_src, x_tgt)
    sq_dist = Nx.sum(Nx.pow(coord_diff, 2), axes: [2], keep_axes: true)

    # Build edge message input: [h_i, h_j, d², edge_feat]
    edge_input =
      if in_edge_features > 0 do
        Nx.concatenate([h_src, h_tgt, sq_dist, edge_features], axis: 2)
      else
        Nx.concatenate([h_src, h_tgt, sq_dist], axis: 2)
      end

    # Edge MLP: φ_e
    # [batch, num_edges, edge_mlp_in] -> [batch, num_edges, hidden_dim]
    m = edge_mlp(edge_input, edge_w1, edge_b1, edge_w2, edge_b2, batch_size, num_edges)

    # Coordinate update: x_i' = x_i + Σ_j (x_i - x_j) · φ_x(m_ij)
    coords_out =
      if update_coords do
        # φ_x: message -> scalar weight
        coord_weight = coord_mlp(m, coord_w1, coord_b1, coord_w2, coord_b2, batch_size, num_edges)
        # coord_weight: [batch, num_edges, 1]

        # Weighted coordinate difference: (x_src - x_tgt) * weight
        weighted_diff = Nx.multiply(coord_diff, coord_weight)

        # Scatter-add to source nodes
        coord_update = scatter_add(weighted_diff, src_idx, num_nodes, batch_size, coord_dim)

        Nx.add(coords, coord_update)
      else
        coords
      end

    # Aggregate messages to nodes
    # m_agg[i] = Σ_j m_ij where i is the source
    m_agg = scatter_add(m, src_idx, num_nodes, batch_size, hidden_dim)

    # Node update: h_i' = φ_h(h_i, m_agg)
    node_input = Nx.concatenate([node_feats, m_agg], axis: 2)
    h_update = node_mlp(node_input, node_w1, node_b1, node_w2, node_b2, batch_size, num_nodes)

    # Residual connection for node features
    h_out = Nx.add(node_feats, h_update)

    %{node_features: h_out, coords: coords_out}
  end

  # Gather node features for each edge
  defp gather_nodes(features, indices, _batch_size, _num_edges, _feat_dim) do
    # features: [batch, num_nodes, feat_dim]
    # indices: [batch, num_edges]
    # output: [batch, num_edges, feat_dim]

    # Use one-hot gather
    num_nodes = Nx.axis_size(features, 1)

    # One-hot: [batch, num_edges, num_nodes]
    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(indices, :s32), 2),
        Nx.reshape(Nx.iota({num_nodes}), {1, 1, num_nodes})
      )
      |> Nx.as_type(:f32)

    # Gather: [batch, num_edges, num_nodes] @ [batch, num_nodes, feat_dim]
    # -> [batch, num_edges, feat_dim]
    Nx.dot(one_hot, [2], [0], features, [1], [0])
  end

  # Scatter-add values to nodes
  defp scatter_add(values, indices, num_nodes, _batch_size, _feat_dim) do
    # values: [batch, num_edges, feat_dim]
    # indices: [batch, num_edges]
    # output: [batch, num_nodes, feat_dim]

    # One-hot for scatter: [batch, num_edges, num_nodes]
    one_hot =
      Nx.equal(
        Nx.new_axis(Nx.as_type(indices, :s32), 2),
        Nx.reshape(Nx.iota({num_nodes}), {1, 1, num_nodes})
      )
      |> Nx.as_type(:f32)

    # Scatter-add: transpose one_hot to [batch, num_nodes, num_edges]
    # then matmul with values [batch, num_edges, feat_dim]
    one_hot_t = Nx.transpose(one_hot, axes: [0, 2, 1])
    Nx.dot(one_hot_t, [2], [0], values, [1], [0])
  end

  # 2-layer MLP with SiLU activation
  defp edge_mlp(x, w1, b1, w2, b2, batch_size, num_items) do
    input_dim = Nx.axis_size(x, 2)
    hidden = Nx.axis_size(w1, 1)

    x_flat = Nx.reshape(x, {batch_size * num_items, input_dim})
    h = Nx.add(Nx.dot(x_flat, w1), b1)
    h = Nx.multiply(h, Nx.sigmoid(h))
    out = Nx.add(Nx.dot(h, w2), b2)
    Nx.reshape(out, {batch_size, num_items, hidden})
  end

  defp coord_mlp(x, w1, b1, w2, b2, batch_size, num_items) do
    hidden_dim = Nx.axis_size(x, 2)

    x_flat = Nx.reshape(x, {batch_size * num_items, hidden_dim})
    h = Nx.add(Nx.dot(x_flat, w1), b1)
    h = Nx.multiply(h, Nx.sigmoid(h))
    out = Nx.add(Nx.dot(h, w2), b2)
    Nx.reshape(out, {batch_size, num_items, 1})
  end

  defp node_mlp(x, w1, b1, w2, b2, batch_size, num_nodes) do
    input_dim = Nx.axis_size(x, 2)
    hidden = Nx.axis_size(w1, 1)

    x_flat = Nx.reshape(x, {batch_size * num_nodes, input_dim})
    h = Nx.add(Nx.dot(x_flat, w1), b1)
    h = Nx.multiply(h, Nx.sigmoid(h))
    out = Nx.add(Nx.dot(h, w2), b2)
    Nx.reshape(out, {batch_size, num_nodes, hidden})
  end

  @doc """
  Get the output size of an EGNN model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    out_features = Keyword.get(opts, :out_features, nil)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    out_features || hidden_dim
  end
end
