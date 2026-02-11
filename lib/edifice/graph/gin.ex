defmodule Edifice.Graph.GIN do
  @moduledoc """
  Graph Isomorphism Network (Xu et al., 2019).

  GIN is provably the most expressive GNN architecture under the message passing
  framework, achieving the same discriminative power as the Weisfeiler-Lehman
  graph isomorphism test. Each GIN layer applies:

      h_v' = MLP((1 + eps) * h_v + SUM(h_u for u in N(v)))

  where eps is a learnable parameter that weights self-features relative to
  neighbor aggregation.

  ## Architecture

  ```
  Node Features [batch, num_nodes, input_dim]
  Adjacency     [batch, num_nodes, num_nodes]
        |
        v
  +--------------------------------------+
  | GIN Layer 1:                         |
  |   1. Aggregate: sum neighbor feats   |
  |   2. Combine: (1+eps)*h_v + agg     |
  |   3. Transform: MLP(combined)        |
  +--------------------------------------+
        |
        v
  +--------------------------------------+
  | GIN Layer N                          |
  +--------------------------------------+
        |
        v
  Node Embeddings [batch, num_nodes, hidden_dim]
  ```

  ## Usage

      model = GIN.build(
        input_dim: 16,
        hidden_dims: [64, 64],
        num_classes: 2,
        epsilon_learnable: true
      )

  ## References

  - Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019)
  - https://arxiv.org/abs/1810.00826
  """

  require Axon

  alias Edifice.Graph.MessagePassing

  @default_hidden_dims [64, 64]
  @default_activation :relu
  @default_dropout 0.0

  @doc """
  Build a Graph Isomorphism Network.

  ## Options

  - `:input_dim` - Input feature dimension per node (required)
  - `:hidden_dims` - List of hidden dimensions for each GIN layer (default: [64, 64])
  - `:num_classes` - If provided, adds a classification head (default: nil)
  - `:epsilon_learnable` - Whether eps is a learnable parameter (default: true)
  - `:dropout` - Dropout rate (default: 0.0)
  - `:activation` - Activation for MLPs (default: :relu)
  - `:pool` - Global pooling for graph classification: :mean, :sum, :max (default: nil)

  ## Returns

  An Axon model with two inputs ("nodes" and "adjacency").
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    num_classes = Keyword.get(opts, :num_classes, nil)
    epsilon_learnable = Keyword.get(opts, :epsilon_learnable, true)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    activation = Keyword.get(opts, :activation, @default_activation)
    pool = Keyword.get(opts, :pool, nil)

    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    adjacency = Axon.input("adjacency", shape: {nil, nil, nil})

    # Stack GIN layers
    output =
      hidden_dims
      |> Enum.with_index()
      |> Enum.reduce(nodes, fn {hidden_dim, idx}, acc ->
        gin_layer(acc, adjacency, hidden_dim,
          name: "gin_#{idx}",
          epsilon_learnable: epsilon_learnable,
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
      Axon.dense(output, num_classes, name: "gin_classifier")
    else
      output
    end
  end

  @doc """
  Single GIN layer: aggregate neighbors via sum, combine with self, apply MLP.

  ## Options

  - `:name` - Layer name prefix (default: "gin")
  - `:epsilon_learnable` - Whether eps is learnable (default: true)
  - `:dropout` - Dropout rate (default: 0.0)
  - `:activation` - Activation for MLP (default: :relu)
  """
  @spec gin_layer(Axon.t(), Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def gin_layer(nodes, adjacency, output_dim, opts \\ []) do
    name = Keyword.get(opts, :name, "gin")
    epsilon_learnable = Keyword.get(opts, :epsilon_learnable, true)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    activation = Keyword.get(opts, :activation, @default_activation)

    # Aggregate neighbor features via sum: A @ H
    neighbor_sum =
      Axon.layer(
        &gin_aggregate_impl/3,
        [nodes, adjacency],
        name: "#{name}_aggregate",
        op_name: :gin_aggregate
      )

    # Combine: (1 + eps) * h_v + sum(h_u)
    combined =
      if epsilon_learnable do
        # Use a dense layer with 1 output as learnable epsilon per feature
        eps_proj = Axon.dense(nodes, 1, name: "#{name}_eps", use_bias: false)
        # eps_proj: [batch, nodes, 1], broadcast to scale self features
        scaled_self =
          Axon.layer(
            fn self_feats, eps_val, _opts ->
              eps = Nx.add(1.0, eps_val)
              Nx.multiply(self_feats, eps)
            end,
            [nodes, eps_proj],
            name: "#{name}_scale_self",
            op_name: :gin_scale
          )

        Axon.add(scaled_self, neighbor_sum, name: "#{name}_combine")
      else
        # Fixed eps = 0: just self + neighbors
        Axon.add(nodes, neighbor_sum, name: "#{name}_combine")
      end

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

  @doc """
  Get the output size of a GIN model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    num_classes = Keyword.get(opts, :num_classes, nil)
    if num_classes, do: num_classes, else: List.last(hidden_dims)
  end

  # Sum aggregation via adjacency matrix
  defp gin_aggregate_impl(nodes, adjacency, _opts) do
    # Sum of neighbor features: A @ H
    Nx.dot(adjacency, [2], [0], nodes, [1], [0])
  end
end
