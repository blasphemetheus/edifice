defmodule Edifice.Graph.GraphSAGE do
  @moduledoc """
  GraphSAGE - Inductive Representation Learning on Large Graphs.

  GraphSAGE learns node embeddings by sampling and aggregating features from
  a node's local neighborhood. Unlike transductive methods (GCN), GraphSAGE
  can generate embeddings for previously unseen nodes, making it suitable for
  evolving graphs and inductive settings.

  ## Architecture

  ```
  Node Features [batch, num_nodes, input_dim]
  Adjacency     [batch, num_nodes, num_nodes]
        |
        v
  +--------------------------------------+
  | GraphSAGE Layer 1:                   |
  |   1. Aggregate neighbor features     |
  |      h_N(v) = AGG({h_u : u in N(v)})|
  |   2. Concatenate with self           |
  |      h_v' = [h_v || h_N(v)]         |
  |   3. Project and normalize           |
  |      h_v' = W * h_v' / ||h_v'||     |
  +--------------------------------------+
        |
        v
  +--------------------------------------+
  | GraphSAGE Layer 2                    |
  +--------------------------------------+
        |
        v
  Node Embeddings [batch, num_nodes, hidden_dim]
  ```

  ## Aggregation Strategies

  | Aggregator | Description |
  |------------|-------------|
  | `:mean`    | Mean of neighbor features (default) |
  | `:max`     | Element-wise max of neighbor features |
  | `:pool`    | Max-pooled through a dense layer |

  ## Usage

      model = GraphSAGE.build(
        input_dim: 16,
        hidden_dims: [64, 64],
        aggregator: :mean,
        num_classes: 7
      )

  ## References

  - Hamilton et al., "Inductive Representation Learning on Large Graphs" (NeurIPS 2017)
  - https://arxiv.org/abs/1706.02216
  """

  alias Edifice.Graph.MessagePassing

  @default_hidden_dims [64, 64]
  @default_aggregator :mean
  @default_activation :relu
  @default_dropout 0.0

  @doc """
  Build a GraphSAGE model.

  ## Options

  - `:input_dim` - Input feature dimension per node (required)
  - `:hidden_dims` - List of hidden dimensions for each layer (default: [64, 64])
  - `:aggregator` - Aggregation strategy: :mean, :max, :pool (default: :mean)
  - `:num_classes` - If provided, adds a classification head (default: nil)
  - `:activation` - Activation function (default: :relu)
  - `:dropout` - Dropout rate (default: 0.0)
  - `:pool` - Global pooling mode for graph classification: :mean, :sum, :max (default: nil)

  ## Returns

  An Axon model with two inputs ("nodes" and "adjacency").
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:aggregator, :mean | :max | :pool}
          | {:dropout, float()}
          | {:hidden_dims, pos_integer()}
          | {:input_dim, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:pool, atom()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    aggregator = Keyword.get(opts, :aggregator, @default_aggregator)
    num_classes = Keyword.get(opts, :num_classes, nil)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    pool = Keyword.get(opts, :pool, nil)

    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    adjacency = Axon.input("adjacency", shape: {nil, nil, nil})

    # Stack GraphSAGE layers
    output =
      hidden_dims
      |> Enum.with_index()
      |> Enum.reduce(nodes, fn {hidden_dim, idx}, acc ->
        sage_layer(acc, adjacency, hidden_dim,
          name: "sage_#{idx}",
          aggregator: aggregator,
          activation: activation,
          dropout: dropout
        )
      end)

    # Optional global pooling for graph classification
    output =
      if pool do
        MessagePassing.global_pool(output, pool)
      else
        output
      end

    # Optional classification head
    if num_classes do
      Axon.dense(output, num_classes, name: "sage_classifier")
    else
      output
    end
  end

  @doc """
  Single GraphSAGE layer: aggregate neighbors, concatenate with self, project.

  ## Options

  - `:name` - Layer name prefix (default: "sage")
  - `:aggregator` - Aggregation strategy (default: :mean)
  - `:activation` - Activation function (default: :relu)
  - `:dropout` - Dropout rate (default: 0.0)
  """
  @spec sage_layer(Axon.t(), Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def sage_layer(nodes, adjacency, output_dim, opts \\ []) do
    name = Keyword.get(opts, :name, "sage")
    aggregator = Keyword.get(opts, :aggregator, @default_aggregator)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # For pool aggregator, transform neighbors before aggregation
    neighbor_features =
      case aggregator do
        :pool ->
          pool_dim = output_dim

          nodes
          |> Axon.dense(pool_dim, name: "#{name}_pool_proj")
          |> Axon.activation(:relu, name: "#{name}_pool_act")

        _ ->
          nodes
      end

    # Aggregate neighbor features using adjacency matrix
    aggregated =
      Axon.layer(
        &sage_aggregate_impl/3,
        [neighbor_features, adjacency],
        name: "#{name}_aggregate",
        aggregator: aggregator,
        op_name: :sage_aggregate
      )

    # Concatenate self features with aggregated neighbor features
    combined =
      Axon.concatenate([nodes, aggregated], axis: 2, name: "#{name}_concat")

    # Project concatenated features
    projected =
      combined
      |> Axon.dense(output_dim, name: "#{name}_proj")
      |> Axon.activation(activation, name: "#{name}_act")

    # L2 normalize
    normalized =
      Axon.nx(
        projected,
        fn x ->
          norm = Nx.sqrt(Nx.sum(Nx.pow(x, 2), axes: [2], keep_axes: true))
          Nx.divide(x, Nx.max(norm, 1.0e-6))
        end,
        name: "#{name}_l2_norm"
      )

    if dropout > 0.0 do
      Axon.dropout(normalized, rate: dropout, name: "#{name}_dropout")
    else
      normalized
    end
  end

  @doc """
  Get the output size of a GraphSAGE model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    num_classes = Keyword.get(opts, :num_classes, nil)
    if num_classes, do: num_classes, else: List.last(hidden_dims)
  end

  # Aggregate neighbor features via adjacency matrix
  defp sage_aggregate_impl(neighbor_features, adjacency, opts) do
    aggregator = opts[:aggregator] || :mean

    case aggregator do
      :mean ->
        # Mean aggregation: A @ H / degree
        aggregated = Nx.dot(adjacency, [2], [0], neighbor_features, [1], [0])
        degree = Nx.sum(adjacency, axes: [2], keep_axes: true)
        degree = Nx.max(degree, 1.0)
        Nx.divide(aggregated, degree)

      :max ->
        # Max aggregation: element-wise max over neighbors
        mask = Nx.greater(adjacency, 0)
        features_expanded = Nx.new_axis(neighbor_features, 1)
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

      :pool ->
        # Pool aggregator: features already projected, use max aggregation
        mask = Nx.greater(adjacency, 0)
        features_expanded = Nx.new_axis(neighbor_features, 1)
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
end
