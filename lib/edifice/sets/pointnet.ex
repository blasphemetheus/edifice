defmodule Edifice.Sets.PointNet do
  @moduledoc """
  Point cloud processing network (Qi et al., 2017).

  PointNet processes unordered 3D point clouds for classification and
  segmentation. It achieves permutation invariance through a symmetric function
  (max pooling) applied after per-point feature extraction.

  ## Architecture

  ```
  Point Cloud [batch, num_points, point_dim]
        |
        v
  +------------------------------+
  | Optional T-Net:              |
  |   Predict 3x3 transform     |
  |   Apply to input points     |
  +------------------------------+
        |
        v
  +------------------------------+
  | Shared MLP (per-point):      |
  |   64 -> 64                   |
  +------------------------------+
        |
        v
  +------------------------------+
  | Optional Feature T-Net:      |
  |   Predict 64x64 transform   |
  |   Apply to point features   |
  +------------------------------+
        |
        v
  +------------------------------+
  | Shared MLP (per-point):      |
  |   64 -> 128 -> 1024         |
  +------------------------------+
        |
        v
  +------------------------------+
  | Max Pool (symmetric fn):     |
  |   Global feature vector      |
  +------------------------------+
        |
        v
  +------------------------------+
  | Global MLP + Classifier:     |
  |   512 -> 256 -> num_classes  |
  +------------------------------+
        |
        v
  Output [batch, num_classes]
  ```

  ## Key Insight

  The max pooling over points acts as a symmetric function, ensuring the
  network output is invariant to point ordering. The T-Net learns spatial
  transformations to canonicalize the input, improving robustness to
  geometric transformations.

  ## Usage

      # Basic PointNet for 3D classification
      model = PointNet.build(
        input_dim: 3,
        num_classes: 40,
        hidden_dims: [64, 128, 1024]
      )

      # With input transformation network
      model = PointNet.build(
        input_dim: 3,
        num_classes: 40,
        use_t_net: true
      )

  ## References

  - "PointNet: Deep Learning on Point Sets for 3D Classification and
    Segmentation" (Qi et al., CVPR 2017)
  """

  require Axon

  @default_hidden_dims [64, 128, 1024]
  @default_global_dims [512, 256]
  @default_activation :relu
  @default_dropout 0.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a PointNet model for point cloud classification.

  ## Options

  - `:input_dim` - Dimension of each point (default: 3 for 3D xyz)
  - `:num_classes` - Number of output classes (required)
  - `:hidden_dims` - Per-point MLP hidden sizes (default: [64, 128, 1024])
  - `:global_dims` - Global MLP sizes after pooling (default: [512, 256])
  - `:activation` - Activation function (default: :relu)
  - `:dropout` - Dropout rate for global MLP (default: 0.0)
  - `:use_t_net` - Use input transformation network (default: false)
  - `:use_feature_t_net` - Use feature transformation network (default: false)

  ## Returns

  An Axon model. Input shape: `{batch, num_points, input_dim}`.
  Output shape: `{batch, num_classes}`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.get(opts, :input_dim, 3)
    num_classes = Keyword.fetch!(opts, :num_classes)
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    global_dims = Keyword.get(opts, :global_dims, @default_global_dims)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    use_t_net = Keyword.get(opts, :use_t_net, false)
    use_feature_t_net = Keyword.get(opts, :use_feature_t_net, false)

    # Input: [batch, num_points, point_dim]
    input = Axon.input("input", shape: {nil, nil, input_dim})

    # Optional input transformation (T-Net for spatial alignment)
    points =
      if use_t_net do
        transform = t_net(input, input_dim, name: "input_t_net")
        apply_transform(input, transform, "input_transform")
      else
        input
      end

    # First shared MLP block (per-point features)
    # Dense layers applied to last dim are shared across points
    first_block_dim = List.first(hidden_dims) || 64
    first_block_dims = Enum.take(hidden_dims, div(length(hidden_dims), 2) + 1)
    remaining_dims = Enum.drop(hidden_dims, div(length(hidden_dims), 2) + 1)

    point_features =
      first_block_dims
      |> Enum.with_index()
      |> Enum.reduce(points, fn {dim, idx}, acc ->
        acc
        |> Axon.dense(dim, name: "shared_mlp1_#{idx}")
        |> Axon.activation(activation, name: "shared_act1_#{idx}")
      end)

    # Optional feature transformation (T-Net for feature alignment)
    feature_dim = List.last(first_block_dims) || first_block_dim

    point_features =
      if use_feature_t_net do
        feat_transform = t_net(point_features, feature_dim, name: "feature_t_net")
        apply_transform(point_features, feat_transform, "feature_transform")
      else
        point_features
      end

    # Second shared MLP block
    point_features =
      remaining_dims
      |> Enum.with_index()
      |> Enum.reduce(point_features, fn {dim, idx}, acc ->
        acc
        |> Axon.dense(dim, name: "shared_mlp2_#{idx}")
        |> Axon.activation(activation, name: "shared_act2_#{idx}")
      end)

    # Symmetric function: max pool over points
    # [batch, num_points, feature_dim] -> [batch, feature_dim]
    global_feature =
      Axon.nx(
        point_features,
        fn features ->
          Nx.reduce_max(features, axes: [1])
        end,
        name: "max_pool"
      )

    # Global MLP for classification
    output =
      global_dims
      |> Enum.with_index()
      |> Enum.reduce(global_feature, fn {dim, idx}, acc ->
        layer =
          acc
          |> Axon.dense(dim, name: "global_mlp_#{idx}")
          |> Axon.activation(activation, name: "global_act_#{idx}")

        if dropout > 0.0 do
          Axon.dropout(layer, rate: dropout, name: "global_drop_#{idx}")
        else
          layer
        end
      end)

    # Classification head
    Axon.dense(output, num_classes, name: "classifier")
  end

  # ============================================================================
  # T-Net (Transformation Network)
  # ============================================================================

  @doc """
  Build a T-Net (Transformation Network) that predicts a transformation matrix.

  The T-Net is a mini-PointNet that processes the input to predict a KxK
  transformation matrix. This matrix is applied to the input to achieve
  spatial invariance.

  ## Parameters

  - `input` - Axon node with shape `{batch, num_points, k}`
  - `k` - Dimension of the transformation matrix (k x k)
  - `opts` - Options

  ## Options

  - `:name` - Layer name prefix (default: "t_net")
  - `:hidden_dims` - T-Net MLP sizes (default: [64, 128, 256])

  ## Returns

  Axon node with shape `{batch, k, k}` representing the predicted
  transformation matrix, initialized near identity.
  """
  @spec t_net(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def t_net(input, k, opts \\ []) do
    name = Keyword.get(opts, :name, "t_net")
    hidden_dims = Keyword.get(opts, :hidden_dims, [64, 128, 256])

    # Per-point feature extraction
    features =
      hidden_dims
      |> Enum.with_index()
      |> Enum.reduce(input, fn {dim, idx}, acc ->
        acc
        |> Axon.dense(dim, name: "#{name}_mlp_#{idx}")
        |> Axon.activation(:relu, name: "#{name}_act_#{idx}")
      end)

    # Max pool over points: [batch, num_points, dim] -> [batch, dim]
    pooled =
      Axon.nx(
        features,
        fn f -> Nx.reduce_max(f, axes: [1]) end,
        name: "#{name}_pool"
      )

    # Predict k*k transformation matrix values
    matrix_values =
      pooled
      |> Axon.dense(128, name: "#{name}_fc1")
      |> Axon.activation(:relu, name: "#{name}_fc1_act")
      |> Axon.dense(k * k, name: "#{name}_matrix")

    # Reshape to [batch, k, k] and add identity for initialization near I
    Axon.nx(
      matrix_values,
      fn values ->
        batch_size = Nx.axis_size(values, 0)
        matrix = Nx.reshape(values, {batch_size, k, k})

        # Add identity matrix so the network starts as identity transform
        eye = Nx.eye(k)
        Nx.add(matrix, eye)
      end,
      name: "#{name}_reshape"
    )
  end

  # ============================================================================
  # Internal Helpers
  # ============================================================================

  # Apply a transformation matrix to point features
  # input: [batch, num_points, k]
  # transform: [batch, k, k]
  # output: [batch, num_points, k]
  defp apply_transform(input, transform, name) do
    Axon.layer(
      &transform_impl/3,
      [input, transform],
      name: name,
      op_name: :point_transform
    )
  end

  defp transform_impl(input, transform, _opts) do
    # input: [batch, num_points, k]
    # transform: [batch, k, k]
    # output: input @ transform^T = [batch, num_points, k]
    transform_t = Nx.transpose(transform, axes: [0, 2, 1])
    Nx.dot(input, [2], transform_t, [1])
  end
end
