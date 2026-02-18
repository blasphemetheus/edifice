defmodule Edifice.Graph.PNA do
  @moduledoc """
  Principal Neighbourhood Aggregation (Corso et al., 2020).

  PNA combines multiple aggregation functions with degree-based scalers to
  create a maximally expressive message passing scheme. By using diverse
  aggregators (mean, max, sum, std) and scalers (identity, amplification,
  attenuation), PNA captures a richer set of graph structural properties
  than any single aggregation method.

  ## Architecture

  ```
  Node Features [batch, num_nodes, input_dim]
  Adjacency     [batch, num_nodes, num_nodes]
        |
        v
  +------------------------------------------+
  | PNA Layer:                               |
  |   1. Apply all aggregators:              |
  |      [mean(N(v)), max(N(v)),             |
  |       sum(N(v)), std(N(v))]              |
  |   2. Apply scalers to each:              |
  |      [identity, amplification]           |
  |   3. Concatenate all combinations        |
  |   4. Project to output_dim               |
  +------------------------------------------+
        |
        v
  Node Embeddings [batch, num_nodes, hidden_dim]
  ```

  ## Usage

      model = PNA.build(
        input_dim: 16,
        hidden_dims: [64, 64],
        aggregators: [:mean, :max, :sum, :std],
        scalers: [:identity, :amplification],
        num_classes: 3
      )

  ## References

  - Corso et al., "Principal Neighbourhood Aggregation for Graph Nets" (NeurIPS 2020)
  - https://arxiv.org/abs/2004.05718
  """

  alias Edifice.Graph.MessagePassing

  @default_hidden_dims [64, 64]
  @default_aggregators [:mean, :max, :sum, :std]
  @default_scalers [:identity, :amplification]
  @default_activation :relu
  @default_dropout 0.0

  @doc """
  Build a PNA model.

  ## Options

  - `:input_dim` - Input feature dimension per node (required)
  - `:hidden_dims` - List of hidden dimensions (default: [64, 64])
  - `:aggregators` - List of aggregation functions (default: [:mean, :max, :sum, :std])
  - `:scalers` - List of degree scalers (default: [:identity, :amplification])
  - `:num_classes` - If provided, adds a classification head (default: nil)
  - `:activation` - Activation function (default: :relu)
  - `:dropout` - Dropout rate (default: 0.0)
  - `:pool` - Global pooling for graph classification (default: nil)

  ## Returns

  An Axon model with two inputs ("nodes" and "adjacency").
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:aggregators, [atom()]}
          | {:dropout, float()}
          | {:hidden_dims, [pos_integer()]}
          | {:input_dim, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:pool, atom()}
          | {:scalers, [atom()]}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    aggregators = Keyword.get(opts, :aggregators, @default_aggregators)
    scalers = Keyword.get(opts, :scalers, @default_scalers)
    num_classes = Keyword.get(opts, :num_classes, nil)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    pool = Keyword.get(opts, :pool, nil)

    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    adjacency = Axon.input("adjacency", shape: {nil, nil, nil})

    # Stack PNA layers
    output =
      hidden_dims
      |> Enum.with_index()
      |> Enum.reduce(nodes, fn {hidden_dim, idx}, acc ->
        pna_layer(acc, adjacency, hidden_dim,
          name: "pna_#{idx}",
          aggregators: aggregators,
          scalers: scalers,
          activation: activation,
          dropout: dropout
        )
      end)

    output =
      if pool do
        MessagePassing.global_pool(output, pool)
      else
        output
      end

    if num_classes do
      Axon.dense(output, num_classes, name: "pna_classifier")
    else
      output
    end
  end

  @doc """
  Single PNA layer: multiple aggregators + degree scalers + projection.

  ## Options

  - `:name` - Layer name prefix (default: "pna")
  - `:aggregators` - List of aggregation functions (default: [:mean, :max, :sum, :std])
  - `:scalers` - List of degree scalers (default: [:identity, :amplification])
  - `:activation` - Activation function (default: :relu)
  - `:dropout` - Dropout rate (default: 0.0)
  """
  @spec pna_layer(Axon.t(), Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def pna_layer(nodes, adjacency, output_dim, opts \\ []) do
    name = Keyword.get(opts, :name, "pna")
    aggregators = Keyword.get(opts, :aggregators, @default_aggregators)
    scalers = Keyword.get(opts, :scalers, @default_scalers)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    # Compute all aggregator + scaler combinations
    aggregated =
      Axon.layer(
        &pna_multi_aggregate_impl/3,
        [nodes, adjacency],
        name: "#{name}_multi_agg",
        aggregators: aggregators,
        scalers: scalers,
        op_name: :pna_aggregate
      )

    # Project the concatenated aggregations down to output_dim
    projected =
      aggregated
      |> Axon.dense(output_dim, name: "#{name}_proj")
      |> Axon.layer_norm(name: "#{name}_ln")
      |> Axon.activation(activation, name: "#{name}_act")

    if dropout > 0.0 do
      Axon.dropout(projected, rate: dropout, name: "#{name}_dropout")
    else
      projected
    end
  end

  @doc """
  Get the output size of a PNA model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    num_classes = Keyword.get(opts, :num_classes, nil)
    if num_classes, do: num_classes, else: List.last(hidden_dims)
  end

  # Compute all aggregator-scaler combinations
  defp pna_multi_aggregate_impl(nodes, adjacency, opts) do
    aggregators = opts[:aggregators] || @default_aggregators
    scalers = opts[:scalers] || @default_scalers

    feature_dim = Nx.axis_size(nodes, 2)

    # Compute degree for scalers
    degree = Nx.sum(adjacency, axes: [2], keep_axes: true)
    # Log degree for amplification scaler (log(1 + degree))
    log_degree = Nx.log1p(degree)

    # Compute each aggregation
    aggregations =
      Enum.flat_map(aggregators, fn agg ->
        base = compute_aggregation(nodes, adjacency, degree, agg)

        # Apply scalers to each aggregation
        Enum.map(scalers, fn scaler ->
          apply_scaler(base, degree, log_degree, scaler)
        end)
      end)

    # Concatenate all aggregation-scaler combinations along feature axis
    # Each is [batch, num_nodes, feature_dim], total is [batch, num_nodes, N * feature_dim]
    num_combinations = length(aggregators) * length(scalers)
    batch_size = Nx.axis_size(nodes, 0)
    num_nodes = Nx.axis_size(nodes, 1)

    stacked = Nx.stack(aggregations, axis: 3)
    Nx.reshape(stacked, {batch_size, num_nodes, feature_dim * num_combinations})
  end

  defp compute_aggregation(nodes, adjacency, degree, :mean) do
    aggregated = Nx.dot(adjacency, [2], [0], nodes, [1], [0])
    Nx.divide(aggregated, Nx.max(degree, 1.0))
  end

  defp compute_aggregation(nodes, adjacency, _degree, :sum) do
    Nx.dot(adjacency, [2], [0], nodes, [1], [0])
  end

  defp compute_aggregation(nodes, adjacency, _degree, :max) do
    mask = Nx.greater(adjacency, 0)
    features_expanded = Nx.new_axis(nodes, 1)
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

  defp compute_aggregation(nodes, adjacency, degree, :std) do
    # Standard deviation: sqrt(E[x^2] - E[x]^2)
    safe_degree = Nx.max(degree, 1.0)
    mean = Nx.divide(Nx.dot(adjacency, [2], [0], nodes, [1], [0]), safe_degree)
    sq_mean = Nx.divide(Nx.dot(adjacency, [2], [0], Nx.pow(nodes, 2), [1], [0]), safe_degree)
    variance = Nx.max(Nx.subtract(sq_mean, Nx.pow(mean, 2)), 0.0)
    Nx.sqrt(Nx.add(variance, 1.0e-6))
  end

  defp apply_scaler(aggregated, _degree, _log_degree, :identity) do
    aggregated
  end

  defp apply_scaler(aggregated, _degree, log_degree, :amplification) do
    # Scale by log(1 + degree) to amplify high-degree nodes
    Nx.multiply(aggregated, log_degree)
  end

  defp apply_scaler(aggregated, _degree, log_degree, :attenuation) do
    # Scale by 1 / log(1 + degree) to attenuate high-degree nodes
    Nx.divide(aggregated, Nx.max(log_degree, 1.0e-6))
  end
end
