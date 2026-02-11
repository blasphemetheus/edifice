defmodule Edifice.Graph.GAT do
  @moduledoc """
  Graph Attention Network (Velickovic et al., 2018).

  Implements attention-based message passing where each node attends to its
  neighbors with learned attention weights. Unlike GCN which uses fixed
  normalization, GAT learns to weight neighbor contributions adaptively.

  ## Architecture

  ```
  Node Features [batch, num_nodes, input_dim]
  Adjacency     [batch, num_nodes, num_nodes]
        |
        v
  +--------------------------------------+
  | GAT Layer (K heads):                 |
  |                                      |
  |   For each head k:                   |
  |     1. Project: z_i = W_k h_i        |
  |     2. Attention: e_ij =             |
  |        LeakyReLU(a^T [z_i || z_j])   |
  |     3. Normalize: alpha_ij =         |
  |        softmax_j(e_ij) * A_ij        |
  |     4. Aggregate: h_i' =             |
  |        sigma(SUM_j alpha_ij z_j)     |
  |                                      |
  |   Concatenate heads: [h1 || ... hK]  |
  +--------------------------------------+
        |
        v
  Node Embeddings [batch, num_nodes, num_heads * hidden_dim]
  ```

  Multi-head attention allows the model to jointly attend to information from
  different representation subspaces at different positions.

  ## Usage

      # Build a GAT for node classification
      model = GAT.build(
        input_dim: 16,
        hidden_dim: 8,
        num_heads: 8,
        num_classes: 7,
        dropout: 0.6
      )

  ## References

  - "Graph Attention Networks" (Velickovic et al., ICLR 2018)
  """

  require Axon

  @default_hidden_dim 8
  @default_num_heads 8
  @default_activation :elu
  @default_dropout 0.0
  @default_negative_slope 0.2

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Graph Attention Network.

  Constructs a two-layer GAT with multi-head attention in the first layer
  (heads concatenated) and single-head attention in the output layer
  (heads averaged), following the original paper's design.

  ## Options

  - `:input_dim` - Input feature dimension per node (required)
  - `:hidden_dim` - Hidden dimension per attention head (default: 8)
  - `:num_heads` - Number of attention heads (default: 8)
  - `:num_classes` - Number of output classes (required)
  - `:activation` - Activation function (default: :elu)
  - `:dropout` - Dropout rate for features and attention (default: 0.0)
  - `:num_layers` - Number of GAT layers (default: 2)

  ## Returns

  An Axon model with two inputs ("nodes" and "adjacency"). Output shape is
  `{batch, num_nodes, num_classes}` for node classification.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_dim = Keyword.get(opts, :hidden_dim, @default_hidden_dim)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_classes = Keyword.fetch!(opts, :num_classes)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    num_layers = Keyword.get(opts, :num_layers, 2)

    # Graph inputs
    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    adjacency = Axon.input("adjacency", shape: {nil, nil, nil})

    # Hidden GAT layers: multi-head with concatenation
    output =
      if num_layers > 1 do
        Enum.reduce(0..(num_layers - 2), nodes, fn idx, acc ->
          gat_layer(acc, adjacency, hidden_dim,
            num_heads: num_heads,
            name: "gat_#{idx}",
            activation: activation,
            dropout: dropout,
            concat_heads: true
          )
        end)
      else
        nodes
      end

    # Output layer: single head (or average over heads) for classification
    gat_layer(output, adjacency, num_classes,
      num_heads: 1,
      name: "gat_output",
      activation: nil,
      dropout: 0.0,
      concat_heads: false
    )
  end

  # ============================================================================
  # GAT Layer
  # ============================================================================

  @doc """
  Single Graph Attention layer with multi-head attention.

  Each attention head independently computes attention coefficients over
  neighbors and produces an output. Heads are either concatenated (hidden
  layers) or averaged (output layer).

  ## Parameters

  - `nodes` - Node features Axon node `{batch, num_nodes, in_dim}`
  - `adjacency` - Adjacency matrix Axon node `{batch, num_nodes, num_nodes}`
  - `output_dim` - Output dimension per head

  ## Options

  - `:num_heads` - Number of attention heads (default: 8)
  - `:name` - Layer name prefix (default: "gat")
  - `:activation` - Activation function, nil for none (default: :elu)
  - `:dropout` - Dropout rate (default: 0.0)
  - `:concat_heads` - Concatenate heads (true) or average (false) (default: true)
  - `:negative_slope` - LeakyReLU negative slope (default: 0.2)

  ## Returns

  If `concat_heads` is true: `{batch, num_nodes, num_heads * output_dim}`
  If `concat_heads` is false: `{batch, num_nodes, output_dim}`
  """
  @spec gat_layer(Axon.t(), Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def gat_layer(nodes, adjacency, output_dim, opts \\ []) do
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    name = Keyword.get(opts, :name, "gat")
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    concat_heads = Keyword.get(opts, :concat_heads, true)
    negative_slope = Keyword.get(opts, :negative_slope, @default_negative_slope)

    # Apply dropout to input features
    input_dropped =
      if dropout > 0.0 do
        Axon.dropout(nodes, rate: dropout, name: "#{name}_input_drop")
      else
        nodes
      end

    # Project to all heads at once: [batch, num_nodes, num_heads * output_dim]
    total_dim = num_heads * output_dim
    projected = Axon.dense(input_dropped, total_dim, name: "#{name}_proj")

    # Attention parameters: one vector per head for source and target
    # a_src, a_tgt each [num_heads, output_dim] flattened to [num_heads * output_dim]
    attn_src = Axon.dense(projected, num_heads, name: "#{name}_attn_src", use_bias: false)
    attn_tgt = Axon.dense(projected, num_heads, name: "#{name}_attn_tgt", use_bias: false)

    # Compute attention-weighted aggregation
    result =
      Axon.layer(
        &gat_attention_impl/5,
        [projected, attn_src, attn_tgt, adjacency],
        name: "#{name}_attention",
        num_heads: num_heads,
        output_dim: output_dim,
        concat_heads: concat_heads,
        negative_slope: negative_slope,
        op_name: :gat_attention
      )

    # Activation
    if activation do
      Axon.activation(result, activation, name: "#{name}_act")
    else
      result
    end
  end

  # GAT attention implementation
  # projected: [batch, num_nodes, num_heads * output_dim]
  # attn_src: [batch, num_nodes, num_heads] - source attention scores
  # attn_tgt: [batch, num_nodes, num_heads] - target attention scores
  # adjacency: [batch, num_nodes, num_nodes]
  defp gat_attention_impl(projected, attn_src, attn_tgt, adjacency, opts) do
    num_heads = opts[:num_heads]
    output_dim = opts[:output_dim]
    concat_heads = opts[:concat_heads]
    negative_slope = opts[:negative_slope] || @default_negative_slope

    batch_size = Nx.axis_size(projected, 0)
    num_nodes = Nx.axis_size(projected, 1)

    # Reshape projected to [batch, num_nodes, num_heads, output_dim]
    z = Nx.reshape(projected, {batch_size, num_nodes, num_heads, output_dim})

    # Compute attention coefficients: e_ij = LeakyReLU(a_src_i + a_tgt_j)
    # attn_src: [batch, num_nodes, num_heads] -> [batch, num_nodes, 1, num_heads]
    # attn_tgt: [batch, num_nodes, num_heads] -> [batch, 1, num_nodes, num_heads]
    # [batch, num_nodes, 1, num_heads]
    src_scores = Nx.new_axis(attn_src, 2)
    # [batch, 1, num_nodes, num_heads]
    tgt_scores = Nx.new_axis(attn_tgt, 1)

    # e_ij: [batch, num_nodes, num_nodes, num_heads]
    e = Nx.add(src_scores, tgt_scores)

    # LeakyReLU
    e = leaky_relu(e, negative_slope)

    # Mask non-edges with large negative values for softmax
    # adjacency: [batch, num_nodes, num_nodes] -> [batch, num_nodes, num_nodes, 1]
    mask = Nx.new_axis(Nx.greater(adjacency, 0), 3)
    neg_inf = Nx.broadcast(-1.0e9, Nx.shape(e))
    e_masked = Nx.select(Nx.broadcast(mask, Nx.shape(e)), e, neg_inf)

    # Softmax over neighbors (axis 2) per head
    # Subtract max for numerical stability
    e_max = Nx.reduce_max(e_masked, axes: [2], keep_axes: true)
    e_stable = Nx.subtract(e_masked, e_max)
    e_exp = Nx.exp(e_stable)

    # Zero out non-edges after exp
    e_exp = Nx.multiply(e_exp, Nx.broadcast(mask, Nx.shape(e_exp)))

    # Normalize
    e_sum = Nx.sum(e_exp, axes: [2], keep_axes: true)
    alpha = Nx.divide(e_exp, Nx.max(e_sum, 1.0e-9))

    # alpha: [batch, num_nodes, num_nodes, num_heads]
    # z: [batch, num_nodes, num_heads, output_dim]
    # We need: for each node i, head k: sum_j alpha[i,j,k] * z[j,k,:]

    # Transpose z for matmul: [batch, num_heads, num_nodes, output_dim]
    z_t = Nx.transpose(z, axes: [0, 2, 1, 3])

    # Transpose alpha: [batch, num_heads, num_nodes, num_nodes]
    alpha_t = Nx.transpose(alpha, axes: [0, 3, 1, 2])

    # Attention-weighted aggregation: [batch, num_heads, num_nodes, output_dim]
    h_prime = Nx.dot(alpha_t, [3], [0, 1], z_t, [2], [0, 1])

    # Transpose back: [batch, num_nodes, num_heads, output_dim]
    h_prime = Nx.transpose(h_prime, axes: [0, 2, 1, 3])

    if concat_heads do
      # Concatenate heads: [batch, num_nodes, num_heads * output_dim]
      Nx.reshape(h_prime, {batch_size, num_nodes, num_heads * output_dim})
    else
      # Average over heads: [batch, num_nodes, output_dim]
      Nx.mean(h_prime, axes: [2])
    end
  end

  # ============================================================================
  # Attention Coefficients
  # ============================================================================

  @doc """
  Compute attention coefficients between connected nodes.

  Returns the raw (pre-softmax) attention scores for visualization or analysis.
  The attention mechanism is:

      e_ij = LeakyReLU(a^T [W h_i || W h_j])

  ## Parameters

  - `nodes` - Node features Axon node `{batch, num_nodes, feature_dim}`
  - `adjacency` - Adjacency matrix Axon node `{batch, num_nodes, num_nodes}`
  - `hidden_dim` - Projection dimension
  - `opts` - Options

  ## Options

  - `:name` - Layer name prefix (default: "gat_attn")
  - `:negative_slope` - LeakyReLU slope (default: 0.2)

  ## Returns

  Axon node with attention coefficients `{batch, num_nodes, num_nodes}`.
  """
  @spec attention_coefficients(Axon.t(), Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def attention_coefficients(nodes, adjacency, hidden_dim, opts \\ []) do
    name = Keyword.get(opts, :name, "gat_attn")
    negative_slope = Keyword.get(opts, :negative_slope, @default_negative_slope)

    # Project node features
    projected = Axon.dense(nodes, hidden_dim, name: "#{name}_proj")

    # Attention score projections (single head)
    attn_src = Axon.dense(projected, 1, name: "#{name}_src", use_bias: false)
    attn_tgt = Axon.dense(projected, 1, name: "#{name}_tgt", use_bias: false)

    Axon.layer(
      &attention_coefficients_impl/4,
      [attn_src, attn_tgt, adjacency],
      name: "#{name}_coefficients",
      negative_slope: negative_slope,
      op_name: :gat_coefficients
    )
  end

  # Compute normalized attention coefficients (single-head)
  # attn_src: [batch, num_nodes, 1]
  # attn_tgt: [batch, num_nodes, 1]
  # adjacency: [batch, num_nodes, num_nodes]
  defp attention_coefficients_impl(attn_src, attn_tgt, adjacency, opts) do
    negative_slope = opts[:negative_slope] || @default_negative_slope

    # e_ij = LeakyReLU(attn_src_i + attn_tgt_j)
    # attn_src: [batch, num_nodes, 1] -> [batch, num_nodes, 1]
    # attn_tgt: [batch, num_nodes, 1] -> [batch, 1, num_nodes]
    src = attn_src
    tgt = Nx.transpose(attn_tgt, axes: [0, 2, 1])

    # [batch, num_nodes, num_nodes]
    e = Nx.add(src, tgt)
    e = leaky_relu(e, negative_slope)

    # Mask non-edges
    mask = Nx.greater(adjacency, 0)
    neg_inf = Nx.broadcast(-1.0e9, Nx.shape(e))
    e_masked = Nx.select(mask, e, neg_inf)

    # Softmax over neighbors
    e_max = Nx.reduce_max(e_masked, axes: [2], keep_axes: true)
    e_stable = Nx.subtract(e_masked, e_max)
    e_exp = Nx.exp(e_stable)
    e_exp = Nx.multiply(e_exp, mask)
    e_sum = Nx.sum(e_exp, axes: [2], keep_axes: true)
    Nx.divide(e_exp, Nx.max(e_sum, 1.0e-9))
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  # LeakyReLU: max(x, negative_slope * x)
  defp leaky_relu(x, negative_slope) do
    Nx.select(Nx.greater(x, 0), x, Nx.multiply(x, negative_slope))
  end
end
