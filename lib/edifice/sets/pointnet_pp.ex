defmodule Edifice.Sets.PointNetPP do
  @moduledoc """
  PointNet++ — Deep Hierarchical Feature Learning on Point Sets.

  <!-- verified: true, date: 2026-02-27 -->

  Extends PointNet with hierarchical processing via Set Abstraction (SA)
  layers. Each SA layer downsamples points using Farthest Point Sampling (FPS),
  groups neighbors via ball query, and applies a mini-PointNet to each group.
  This captures local geometric structure at multiple scales.

  ## Architecture

  ```
  Point Cloud [batch, N, 3]
        |
        v
  +-------------------------------+
  | SA Layer 1 (N -> N1)          |
  |   FPS -> Ball Query -> MLP    |
  |   -> Max Pool per group       |
  +-------------------------------+
        |
        v
  +-------------------------------+
  | SA Layer 2 (N1 -> N2)         |
  |   FPS -> Ball Query -> MLP    |
  |   -> Max Pool per group       |
  +-------------------------------+
        |
        v
  +-------------------------------+
  | Global SA (N2 -> 1)           |
  |   Group all -> MLP -> Pool    |
  +-------------------------------+
        |
        v
  +-------------------------------+
  | Classification Head           |
  |   FC -> ReLU -> Drop -> FC    |
  +-------------------------------+
        |
        v
  [batch, num_classes]
  ```

  ## Usage

      model = PointNetPP.build(
        num_classes: 40,
        sa_configs: [
          %{num_points: 32, radius: 0.2, max_neighbors: 16, mlp: [32, 32, 64]},
          %{num_points: 16, radius: 0.4, max_neighbors: 16, mlp: [64, 64, 128]}
        ]
      )

  ## References

  - Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets
    in a Metric Space" (NeurIPS 2017)
  - https://arxiv.org/abs/1706.02413
  """

  @default_input_dim 3
  @default_activation :relu
  @default_dropout 0.3

  @default_sa_configs [
    %{num_points: 32, radius: 0.2, max_neighbors: 16, mlp: [32, 32, 64]},
    %{num_points: 16, radius: 0.4, max_neighbors: 16, mlp: [64, 64, 128]}
  ]

  @default_global_mlp [128, 256, 512]
  @default_fc_dims [256, 128]

  @doc """
  Build a PointNet++ classification model.

  ## Options

    - `:num_classes` - Number of output classes (required)
    - `:input_dim` - Point feature dimension (default: 3 for xyz)
    - `:sa_configs` - List of SA layer configs (see below)
    - `:global_mlp` - MLP dims for global SA layer (default: [128, 256, 512])
    - `:fc_dims` - FC head dims after global pooling (default: [256, 128])
    - `:activation` - Activation function (default: :relu)
    - `:dropout` - Dropout rate in FC head (default: 0.3)

  ### SA Config

  Each SA config is a map with:
    - `:num_points` - Number of centroids after FPS
    - `:radius` - Ball query radius
    - `:max_neighbors` - Maximum neighbors per centroid
    - `:mlp` - List of MLP hidden dims for mini-PointNet

  ## Returns

  An Axon model. Input: "points" `[batch, N, input_dim]`.
  Output: `[batch, num_classes]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:dropout, float()}
          | {:fc_dims, [pos_integer()]}
          | {:global_mlp, [pos_integer()]}
          | {:input_dim, pos_integer()}
          | {:num_classes, pos_integer()}
          | {:sa_configs, [map()]}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    num_classes = Keyword.fetch!(opts, :num_classes)
    input_dim = Keyword.get(opts, :input_dim, @default_input_dim)
    sa_configs = Keyword.get(opts, :sa_configs, @default_sa_configs)
    global_mlp = Keyword.get(opts, :global_mlp, @default_global_mlp)
    fc_dims = Keyword.get(opts, :fc_dims, @default_fc_dims)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    points = Axon.input("points", shape: {nil, nil, input_dim})

    # SA layers: progressively downsample and enrich features
    {xyz, features, _feat_dim} =
      sa_configs
      |> Enum.with_index()
      |> Enum.reduce({points, points, input_dim}, fn {config, idx}, {xyz_in, feat_in, feat_dim} ->
        sa_layer(xyz_in, feat_in, config, feat_dim, activation, idx)
      end)

    # Global SA: pool all remaining points
    last_mlp_dim = List.last(global_mlp)
    global_features = global_sa(xyz, features, global_mlp, activation)

    # FC classification head
    output =
      fc_dims
      |> Enum.with_index()
      |> Enum.reduce(global_features, fn {dim, idx}, acc ->
        layer =
          acc
          |> Axon.dense(dim, name: "fc_#{idx}")
          |> Axon.layer_norm(name: "fc_bn_#{idx}")
          |> Axon.activation(activation, name: "fc_act_#{idx}")

        if dropout > 0.0 do
          Axon.dropout(layer, rate: dropout, name: "fc_drop_#{idx}")
        else
          layer
        end
      end)

    _ = last_mlp_dim
    Axon.dense(output, num_classes, name: "classifier")
  end

  @doc "Get the output size of a PointNet++ model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :num_classes)
  end

  # ===========================================================================
  # Set Abstraction Layer
  # ===========================================================================

  defp sa_layer(xyz, features, config, _input_feat_dim, activation, idx) do
    %{num_points: num_points, radius: radius, max_neighbors: k, mlp: mlp_dims} = config
    prefix = "sa_#{idx}"

    # FPS + Ball Query + Grouping: all in one custom layer
    # Output: grouped features [batch, num_points, k, input_feat_dim]
    #         centroid xyz [batch, num_points, 3]
    #         centroid-relative grouped xyz [batch, num_points, k, 3]
    grouped =
      Axon.layer(
        &fps_ball_query_group_impl/3,
        [xyz, features],
        name: "#{prefix}_group",
        num_points: num_points,
        radius: radius,
        max_neighbors: k,
        op_name: :sa_group
      )

    # Split out the centroid xyz for the next layer
    centroid_xyz =
      Axon.layer(
        &extract_centroids_impl/2,
        [xyz],
        name: "#{prefix}_centroids",
        num_points: num_points,
        op_name: :extract_centroids
      )

    # Mini-PointNet: shared MLP over grouped points, then max pool
    # grouped shape: [batch, num_points, k, 3 + input_feat_dim]
    # Dense layers broadcast over leading dims
    point_features =
      mlp_dims
      |> Enum.with_index()
      |> Enum.reduce(grouped, fn {dim, i}, acc ->
        acc
        |> Axon.dense(dim, name: "#{prefix}_mlp_#{i}")
        |> Axon.layer_norm(name: "#{prefix}_bn_#{i}")
        |> Axon.activation(activation, name: "#{prefix}_act_#{i}")
      end)

    # Max pool over neighbors: [batch, num_points, k, last_dim] -> [batch, num_points, last_dim]
    sa_output =
      Axon.nx(
        point_features,
        fn t -> Nx.reduce_max(t, axes: [2]) end,
        name: "#{prefix}_pool"
      )

    last_dim = List.last(mlp_dims)
    {centroid_xyz, sa_output, last_dim}
  end

  # ===========================================================================
  # Global Set Abstraction (group all points)
  # ===========================================================================

  defp global_sa(xyz, features, mlp_dims, activation) do
    # Combine xyz + features: [batch, N, 3+feat_dim]
    combined =
      Axon.layer(
        fn xyz_t, feat_t, _opts ->
          # Get centroid-relative coords (subtract mean)
          centroid = Nx.mean(xyz_t, axes: [1], keep_axes: true)
          relative_xyz = Nx.subtract(xyz_t, centroid)
          Nx.concatenate([relative_xyz, feat_t], axis: -1)
        end,
        [xyz, features],
        name: "global_sa_concat",
        op_name: :concatenate
      )

    # Shared MLP
    point_features =
      mlp_dims
      |> Enum.with_index()
      |> Enum.reduce(combined, fn {dim, i}, acc ->
        acc
        |> Axon.dense(dim, name: "global_mlp_#{i}")
        |> Axon.layer_norm(name: "global_bn_#{i}")
        |> Axon.activation(activation, name: "global_act_#{i}")
      end)

    # Global max pool: [batch, N, dim] -> [batch, dim]
    Axon.nx(
      point_features,
      fn t -> Nx.reduce_max(t, axes: [1]) end,
      name: "global_pool"
    )
  end

  # ===========================================================================
  # Custom Layer Implementations
  # ===========================================================================

  # FPS + Ball Query + Grouping
  # Returns: [batch, num_points, k, 3 + feat_dim] (relative coords + features)
  defp fps_ball_query_group_impl(xyz, features, opts) do
    num_points = opts[:num_points]
    radius = opts[:radius]
    k = opts[:max_neighbors]
    {batch, n, coord_dim} = Nx.shape(xyz)
    {_batch, _n, feat_dim} = Nx.shape(features)

    # Step 1: Farthest Point Sampling
    centroid_idx = farthest_point_sampling(xyz, num_points, batch, n)

    # Step 2: Gather centroid coordinates
    centroid_xyz = batch_gather(xyz, centroid_idx, batch, num_points, coord_dim)

    # Step 3: Ball query — find k nearest neighbors within radius
    # Pairwise distances: [batch, num_points, N]
    # centroids: [batch, num_points, 1, 3], xyz: [batch, 1, N, 3]
    diff =
      Nx.subtract(
        Nx.new_axis(centroid_xyz, 2),
        Nx.new_axis(xyz, 1)
      )

    sq_dist = Nx.sum(Nx.multiply(diff, diff), axes: [-1])

    # Mask out-of-radius: set to large value
    radius_sq = Nx.tensor(radius * radius)
    in_radius = Nx.less_equal(sq_dist, radius_sq) |> Nx.as_type(:f32)
    masked_dist = Nx.add(sq_dist, Nx.multiply(Nx.subtract(1.0, in_radius), 1.0e10))

    # Get top-K nearest indices
    sorted_idx = Nx.argsort(masked_dist, axis: -1)
    neighbor_idx = Nx.slice_along_axis(sorted_idx, 0, k, axis: 2)

    # Step 4: Gather neighbor features and coordinates
    neighbor_xyz = batch_gather_neighbors(xyz, neighbor_idx, batch, num_points, k, coord_dim)
    neighbor_feat = batch_gather_neighbors(features, neighbor_idx, batch, num_points, k, feat_dim)

    # Center-relative coordinates
    relative_xyz = Nx.subtract(neighbor_xyz, Nx.new_axis(centroid_xyz, 2))

    # Concatenate relative xyz + features
    Nx.concatenate([relative_xyz, neighbor_feat], axis: -1)
  end

  # Extract centroid coordinates after FPS
  defp extract_centroids_impl(xyz, opts) do
    num_points = opts[:num_points]
    {batch, n, coord_dim} = Nx.shape(xyz)
    centroid_idx = farthest_point_sampling(xyz, num_points, batch, n)
    batch_gather(xyz, centroid_idx, batch, num_points, coord_dim)
  end

  # Farthest Point Sampling: iteratively select the point farthest from
  # all previously selected points
  defp farthest_point_sampling(xyz, num_points, batch, n) do
    # Initialize distances to infinity
    distances = Nx.broadcast(1.0e10, {batch, n})
    # Start with first point (index 0)
    farthest = Nx.broadcast(0, {batch}) |> Nx.as_type(:s32)
    centroids = Nx.broadcast(0, {batch, num_points}) |> Nx.as_type(:s32)

    {_distances, _farthest, centroids} =
      Enum.reduce(0..(num_points - 1), {distances, farthest, centroids}, fn i,
                                                                            {dist, far, cents} ->
        # Record current farthest point
        cents = put_column(cents, i, far)

        # Get coordinates of selected point: [batch, 3]
        selected = gather_points(xyz, far, batch)

        # Compute distances to selected point: [batch, N]
        diff = Nx.subtract(xyz, Nx.new_axis(selected, 1))
        new_dist = Nx.sum(Nx.multiply(diff, diff), axes: [-1])

        # Update minimum distances
        dist = Nx.min(dist, new_dist)

        # Select next farthest point
        far = Nx.argmax(dist, axis: 1) |> Nx.as_type(:s32)

        {dist, far, cents}
      end)

    centroids
  end

  # Gather a single point per batch element
  # xyz: [batch, N, D], indices: [batch] -> [batch, D]
  defp gather_points(xyz, indices, batch) do
    # Create batch indices
    batch_idx = Nx.iota({batch}) |> Nx.as_type(:s32)
    # Stack [batch_idx, point_idx] -> [batch, 2]
    gather_idx = Nx.stack([batch_idx, indices], axis: 1)
    Nx.gather(xyz, gather_idx)
  end

  # Put a value into column i of a 2D tensor
  defp put_column(tensor, col_idx, values) do
    {batch, num_cols} = Nx.shape(tensor)
    # Create a one-hot mask for the column
    mask = Nx.equal(Nx.iota({num_cols}), col_idx) |> Nx.as_type(:s32)
    mask = Nx.broadcast(mask, {batch, num_cols})
    # Broadcast values to [batch, num_cols]
    val_expanded = Nx.new_axis(values, 1) |> Nx.broadcast({batch, num_cols})
    # tensor * (1 - mask) + values * mask
    Nx.add(
      Nx.multiply(tensor, Nx.subtract(1, mask)),
      Nx.multiply(val_expanded, mask)
    )
  end

  # Batch gather: gather rows from a [batch, N, D] tensor using indices [batch, M]
  # Returns [batch, M, D]
  defp batch_gather(tensor, indices, batch, m, d) do
    # Flatten batch dim for gathering
    flat_tensor = Nx.reshape(tensor, {batch * Nx.axis_size(tensor, 1), d})
    # Offset indices by batch
    offsets = Nx.iota({batch, 1}) |> Nx.multiply(Nx.axis_size(tensor, 1))
    flat_indices = Nx.add(indices, offsets) |> Nx.reshape({batch * m})
    # Gather and reshape
    gathered = Nx.take(flat_tensor, flat_indices, axis: 0)
    Nx.reshape(gathered, {batch, m, d})
  end

  # Batch gather neighbors: [batch, N, D] with indices [batch, M, K] -> [batch, M, K, D]
  defp batch_gather_neighbors(tensor, indices, batch, m, k, d) do
    n = Nx.axis_size(tensor, 1)
    flat_tensor = Nx.reshape(tensor, {batch * n, d})
    offsets = Nx.iota({batch, 1, 1}) |> Nx.multiply(n)
    flat_indices = Nx.add(indices, offsets) |> Nx.reshape({batch * m * k})
    gathered = Nx.take(flat_tensor, flat_indices, axis: 0)
    Nx.reshape(gathered, {batch, m, k, d})
  end
end
