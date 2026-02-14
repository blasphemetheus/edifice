defmodule Edifice.Graph.GraphTransformer do
  @moduledoc """
  Graph Transformer with structural encoding.

  Applies transformer-style multi-head attention to graph-structured data,
  using the adjacency matrix as an attention bias/mask to incorporate graph
  structure. Includes graph positional encoding via random walk structural
  encoding (RWSE) or Laplacian eigenvectors approximated via the adjacency
  matrix powers.

  ## Architecture

  ```
  Node Features [batch, num_nodes, input_dim]
  Adjacency     [batch, num_nodes, num_nodes]
        |
        v
  +--------------------------------------+
  | Input Projection + Positional Enc    |
  +--------------------------------------+
        |
        v
  +--------------------------------------+
  | Graph Transformer Layer 1:           |
  |   Pre-Norm -> Multi-Head Attention   |
  |   (adjacency as attention bias)      |
  |   + Residual                         |
  |   Pre-Norm -> FFN + Residual         |
  +--------------------------------------+
        |
        v
  +--------------------------------------+
  | Graph Transformer Layer N            |
  +--------------------------------------+
        |
        v
  Node Embeddings [batch, num_nodes, hidden_size]
  ```

  ## Usage

      model = GraphTransformer.build(
        input_dim: 16,
        hidden_size: 64,
        num_heads: 4,
        num_layers: 4,
        num_classes: 7
      )

  ## References

  - Dwivedi & Bresson, "A Generalization of Transformer Networks to Graphs" (AAAI 2021)
  - Ying et al., "Do Transformers Really Perform Bad for Graph Representation?" (NeurIPS 2021)
  """

  require Axon

  alias Edifice.Graph.MessagePassing

  @default_hidden_size 64
  @default_num_heads 4
  @default_num_layers 4
  @default_dropout 0.0
  @default_ffn_multiplier 4

  @doc """
  Build a Graph Transformer.

  ## Options

  - `:input_dim` - Input feature dimension per node (required)
  - `:hidden_size` - Hidden dimension (default: 64)
  - `:num_heads` - Number of attention heads (default: 4)
  - `:num_layers` - Number of transformer layers (default: 4)
  - `:num_classes` - If provided, adds a classification head (default: nil)
  - `:dropout` - Dropout rate (default: 0.0)
  - `:pool` - Global pooling for graph classification (default: nil)

  ## Returns

  An Axon model with two inputs ("nodes" and "adjacency").
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_classes = Keyword.get(opts, :num_classes, nil)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    pool = Keyword.get(opts, :pool, nil)

    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    adjacency = Axon.input("adjacency", shape: {nil, nil, nil})

    # Project input to hidden_size
    x = Axon.dense(nodes, hidden_size, name: "input_proj")

    # Add graph positional encoding derived from adjacency
    x =
      Axon.layer(
        &add_graph_pe_impl/3,
        [x, adjacency],
        name: "graph_pe",
        hidden_size: hidden_size,
        op_name: :graph_pe
      )

    # Stack transformer layers
    x =
      Enum.reduce(0..(num_layers - 1), x, fn idx, acc ->
        graph_transformer_layer(acc, adjacency, hidden_size,
          num_heads: num_heads,
          dropout: dropout,
          name: "gt_layer_#{idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_ln")

    # Optional global pooling
    x =
      if pool do
        MessagePassing.global_pool(x, pool)
      else
        x
      end

    # Optional classification head
    if num_classes do
      Axon.dense(x, num_classes, name: "gt_classifier")
    else
      x
    end
  end

  @doc """
  Single Graph Transformer layer with pre-norm attention + FFN.

  ## Options

  - `:num_heads` - Number of attention heads (default: 4)
  - `:dropout` - Dropout rate (default: 0.0)
  - `:name` - Layer name prefix
  """
  @spec graph_transformer_layer(Axon.t(), Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def graph_transformer_layer(nodes, adjacency, hidden_size, opts \\ []) do
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "gt")
    head_dim = div(hidden_size, num_heads)

    # Pre-norm multi-head attention with graph bias
    normed = Axon.layer_norm(nodes, name: "#{name}_attn_ln")

    q = Axon.dense(normed, hidden_size, name: "#{name}_q")
    k = Axon.dense(normed, hidden_size, name: "#{name}_k")
    v = Axon.dense(normed, hidden_size, name: "#{name}_v")

    attended =
      Axon.layer(
        &graph_attention_impl/5,
        [q, k, v, adjacency],
        name: "#{name}_attn",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :graph_transformer_attn
      )

    attended = Axon.dense(attended, hidden_size, name: "#{name}_attn_proj")

    attended =
      if dropout > 0.0 do
        Axon.dropout(attended, rate: dropout, name: "#{name}_attn_drop")
      else
        attended
      end

    # Residual
    x = Axon.add(nodes, attended, name: "#{name}_attn_res")

    # Pre-norm FFN
    ffn_dim = hidden_size * @default_ffn_multiplier

    normed2 = Axon.layer_norm(x, name: "#{name}_ffn_ln")

    ffn =
      normed2
      |> Axon.dense(ffn_dim, name: "#{name}_ffn_up")
      |> Axon.gelu()
      |> Axon.dense(hidden_size, name: "#{name}_ffn_down")

    ffn =
      if dropout > 0.0 do
        Axon.dropout(ffn, rate: dropout, name: "#{name}_ffn_drop")
      else
        ffn
      end

    # Residual
    Axon.add(x, ffn, name: "#{name}_ffn_res")
  end

  @doc """
  Get the output size of a Graph Transformer.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    num_classes = Keyword.get(opts, :num_classes, nil)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    if num_classes, do: num_classes, else: hidden_size
  end

  # Add graph positional encoding based on adjacency powers
  defp add_graph_pe_impl(x, adjacency, _opts) do
    # Simple random walk structural encoding: use normalized adjacency
    # as a source of positional information. We add A^1 * H as positional signal.
    degree = Nx.sum(adjacency, axes: [2], keep_axes: true)
    safe_degree = Nx.max(degree, 1.0)
    norm_adj = Nx.divide(adjacency, safe_degree)

    # One-step random walk encoding
    pe = Nx.dot(norm_adj, [2], [0], x, [1], [0])

    # Add positional encoding to node features
    Nx.add(x, Nx.multiply(0.1, pe))
  end

  # Multi-head attention with adjacency bias
  defp graph_attention_impl(q, k, v, adjacency, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch_size = Nx.axis_size(q, 0)
    num_nodes = Nx.axis_size(q, 1)

    # Reshape to multi-head: [batch, nodes, heads, head_dim]
    q = Nx.reshape(q, {batch_size, num_nodes, num_heads, head_dim})
    k = Nx.reshape(k, {batch_size, num_nodes, num_heads, head_dim})
    v = Nx.reshape(v, {batch_size, num_nodes, num_heads, head_dim})

    # Transpose: [batch, heads, nodes, head_dim]
    q = Nx.transpose(q, axes: [0, 2, 1, 3])
    k = Nx.transpose(k, axes: [0, 2, 1, 3])
    v = Nx.transpose(v, axes: [0, 2, 1, 3])

    # Scaled dot-product attention: [batch, heads, nodes, nodes]
    scale = Nx.rsqrt(Nx.tensor(head_dim, type: :f32))
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1])
    scores = Nx.multiply(scores, scale)

    # Add adjacency bias: connected nodes get bias, unconnected get masked
    # adjacency: [batch, nodes, nodes] -> [batch, 1, nodes, nodes]
    adj_bias = Nx.new_axis(adjacency, 1)
    # Where adjacency is 0, apply large negative bias
    mask = Nx.greater(adj_bias, 0)
    neg_inf = Nx.broadcast(-1.0e9, Nx.shape(scores))

    scores_masked =
      Nx.select(
        Nx.broadcast(mask, Nx.shape(scores)),
        scores,
        neg_inf
      )

    # Softmax
    max_scores = Nx.reduce_max(scores_masked, axes: [3], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores_masked, max_scores))
    exp_scores = Nx.multiply(exp_scores, Nx.broadcast(mask, Nx.shape(exp_scores)))
    sum_scores = Nx.sum(exp_scores, axes: [3], keep_axes: true)
    attn_weights = Nx.divide(exp_scores, Nx.max(sum_scores, 1.0e-9))

    # Weighted sum: [batch, heads, nodes, head_dim]
    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    # Transpose and reshape back: [batch, nodes, hidden_size]
    output = Nx.transpose(output, axes: [0, 2, 1, 3])
    Nx.reshape(output, {batch_size, num_nodes, num_heads * head_dim})
  end
end
