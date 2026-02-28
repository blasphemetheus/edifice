defmodule Edifice.Graph.KAGNN do
  @moduledoc """
  KA-GNN: Kolmogorov-Arnold Graph Neural Network.

  Replaces MLP transformations in GNN message passing with Fourier KAN layers
  (learnable edge activations parameterized as Fourier series). Each edge
  activation phi_{j,i}(x) = sum_k (A_k cos(kx) + B_k sin(kx)) provides more
  expressivity than fixed ReLU/GELU for molecular property prediction.

  ```
  Node Features [batch, num_nodes, input_dim]
  Adjacency     [batch, num_nodes, num_nodes]
        |
  +------------------------------------------+
  | KA-GNN Layer (x num_layers)              |
  |                                          |
  | 1. Mean-aggregate neighbor features      |
  | 2. Concat self + aggregated: [2*dim]     |
  | 3. Fourier KAN transform: [2*dim] -> dim |
  | 4. Residual connection                   |
  +------------------------------------------+
        |
  Global Mean Pool -> KAN Readout -> [batch, num_classes]
  ```

  ## Usage

      model = KAGNN.build(
        input_dim: 16,
        hidden_dim: 64,
        num_layers: 3,
        num_harmonics: 4,
        num_classes: 1
      )

  ## Reference

  - Liu et al., "KA-GNN: Kolmogorov-Arnold Graph Neural Networks" (Nature MI, 2025)
  """

  alias Edifice.Graph.MessagePassing

  @default_hidden_dim 64
  @default_num_layers 3
  @default_num_harmonics 4
  @default_num_classes 1
  @default_dropout 0.0

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_dim, pos_integer()}
          | {:hidden_dim, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_harmonics, pos_integer()}
          | {:num_classes, pos_integer()}
          | {:dropout, float()}
          | {:pool, :mean | :sum | :max}

  @doc """
  Build a KA-GNN model.

  ## Options

    - `:input_dim` - Input feature dimension per node (required)
    - `:hidden_dim` - Hidden dimension per layer (default: 64)
    - `:num_layers` - Number of message passing layers (default: 3)
    - `:num_harmonics` - Fourier basis terms per activation (default: 4)
    - `:num_classes` - Output classes (default: 1)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:pool` - Global pooling: :mean, :sum, :max (default: :mean)

  ## Returns

    An Axon model with inputs "nodes" and "adjacency".
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_harmonics = Keyword.get(opts, :num_harmonics, @default_num_harmonics)
    num_classes = Keyword.get(opts, :num_classes, @default_num_classes)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    pool = Keyword.get(opts, :pool, :mean)

    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    adjacency = Axon.input("adjacency", shape: {nil, nil, nil})

    # Initial projection via Fourier KAN
    x = fourier_kan_layer(nodes, hidden_dim, num_harmonics, "kagnn_init")

    # Message passing layers
    x =
      Enum.reduce(0..(num_layers - 1), x, fn i, acc ->
        kagnn_layer(acc, adjacency, hidden_dim, num_harmonics, dropout, "kagnn_mp_#{i}")
      end)

    # Global pooling
    x = MessagePassing.global_pool(x, pool)

    # Readout via Fourier KAN
    x = fourier_kan_layer(x, hidden_dim, num_harmonics, "kagnn_readout")
    Axon.dense(x, num_classes, name: "kagnn_classifier")
  end

  # Single KA-GNN message passing layer with residual
  defp kagnn_layer(nodes, adjacency, hidden_dim, num_harmonics, dropout, name) do
    # Mean-aggregate neighbor features: (A @ H) / degree
    neighbor_agg =
      Axon.layer(
        &mean_aggregate_impl/3,
        [nodes, adjacency],
        name: "#{name}_agg",
        op_name: :kagnn_aggregate
      )

    # Concatenate self + aggregated: [batch, nodes, 2*dim]
    concat =
      Axon.layer(
        fn self_feats, agg_feats, _opts ->
          Nx.concatenate([self_feats, agg_feats], axis: -1)
        end,
        [nodes, neighbor_agg],
        name: "#{name}_concat",
        op_name: :kagnn_concat
      )

    # Fourier KAN transform: [2*dim] -> dim
    transformed = fourier_kan_layer(concat, hidden_dim, num_harmonics, "#{name}_kan")

    transformed =
      if dropout > 0.0 do
        Axon.dropout(transformed, rate: dropout, name: "#{name}_drop")
      else
        transformed
      end

    # Residual connection
    Axon.add(nodes, transformed)
  end

  # Fourier KAN layer: phi(x) = base_silu(x) + linear(fourier_features(x))
  # Fourier features are computed in Axon.nx, then projected by Axon.dense
  defp fourier_kan_layer(input, output_dim, num_harmonics, name) do
    # Base: standard linear + SiLU
    base = Axon.dense(input, output_dim, name: "#{name}_base")
    base = Axon.activation(base, :silu)

    # Fourier features: compute cos(k*x), sin(k*x) for k=1..K
    fourier_feats =
      Axon.nx(input, fn x ->
        feats =
          Enum.map(1..num_harmonics, fn k ->
            scaled = Nx.multiply(x, k)
            Nx.concatenate([Nx.cos(scaled), Nx.sin(scaled)], axis: -1)
          end)

        Nx.concatenate(feats, axis: -1)
      end, name: "#{name}_fourier_feats")

    # Project fourier features to output_dim via learned weights
    fourier_proj = Axon.dense(fourier_feats, output_dim, name: "#{name}_fourier_proj")

    # Combine base + Fourier
    Axon.add(base, fourier_proj)
  end

  # Mean aggregation: (A @ H) / max(degree, 1)
  defp mean_aggregate_impl(nodes, adjacency, _opts) do
    agg = Nx.dot(adjacency, [2], [0], nodes, [1], [0])
    degree = Nx.sum(adjacency, axes: [2], keep_axes: true)
    degree = Nx.max(degree, 1.0)
    Nx.divide(agg, degree)
  end

  @doc """
  Get the output size of a KA-GNN model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :num_classes, @default_num_classes)
  end

  @doc """
  Recommended default configuration.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_dim: 64,
      num_layers: 3,
      num_harmonics: 4,
      num_classes: 1,
      dropout: 0.0,
      pool: :mean
    ]
  end
end
