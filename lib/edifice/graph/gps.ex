defmodule Edifice.Graph.GPS do
  @moduledoc """
  GPS — General, Powerful, Scalable Graph Transformer.

  <!-- verified: true, date: 2026-02-27 -->

  Combines local message passing (GIN) with global multi-head attention in
  a dual-branch architecture. Each GPS layer runs MPNN and attention in
  parallel on the same input, sums the normalized branch outputs, then
  applies a feed-forward network. This captures both local graph structure
  (via GIN's WL-test expressive power) and long-range dependencies (via
  global attention without adjacency masking).

  ## Architecture

  ```
  Node Features [batch, N, input_dim]
  Adjacency     [batch, N, N]
        |
        v
  +-----------------------------+
  | Input Projection + RWSE PE  |
  +-----------------------------+
        |
        v
  +-----------------------------+
  | GPS Layer 1..L:             |
  |  branch1: GIN(x, A)        |
  |  branch2: MHA(x)           |
  |  h = Norm(drop(b1)+x)      |
  |    + Norm(drop(b2)+x)      |
  |  out = Norm(FFN(h)+h)      |
  +-----------------------------+
        |
        v
  [batch, N, hidden_size]
  ```

  ## Key Design Choices

  - **Post-norm** residual pattern (norm after residual addition)
  - **BatchNorm** by default (unlike typical Transformers which use LayerNorm)
  - **GIN MPNN** provides WL-test expressivity for local structure
  - **Global attention** without adjacency masking for long-range information flow
  - **FFN** with 2x hidden multiplier (not 4x) and ReLU activation
  - **RWSE** (Random Walk Structural Encoding) as positional information

  ## Usage

      model = GPS.build(
        input_dim: 16,
        hidden_size: 64,
        num_heads: 4,
        num_layers: 5
      )

  ## References

  - Rampasek et al., "Recipe for a General, Powerful, Scalable Graph
    Transformer" (NeurIPS 2022)
  - https://arxiv.org/abs/2205.12454
  """

  alias Edifice.Graph.GIN

  @default_hidden_size 64
  @default_num_heads 4
  @default_num_layers 5
  @default_dropout 0.0
  @default_ffn_multiplier 2
  @default_activation :relu
  @default_pe_dim 8
  @default_rwse_walk_length 16

  @doc """
  Build a GPS model.

  ## Options

    - `:input_dim` - Input feature dimension per node (required)
    - `:hidden_size` - Hidden dimension (default: 64)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:num_layers` - Number of GPS layers (default: 5)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:ffn_multiplier` - FFN inner dim multiplier (default: 2)
    - `:activation` - Activation function (default: :relu)
    - `:pe_dim` - RWSE positional encoding dimension (default: 8)
    - `:rwse_walk_length` - RWSE random walk steps (default: 16)
    - `:num_classes` - If provided, adds classification head (default: nil)
    - `:pool` - Graph-level pooling `:mean` or `:sum` (default: nil)

  ## Inputs

    - "nodes": `[batch, N, input_dim]`
    - "adjacency": `[batch, N, N]` (binary adjacency matrix)

  ## Returns

  Node embeddings `[batch, N, hidden_size]`, or `[batch, num_classes]` with pooling.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:dropout, float()}
          | {:ffn_multiplier, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:input_dim, pos_integer()}
          | {:num_classes, pos_integer()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:pe_dim, pos_integer()}
          | {:pool, :mean | :sum}
          | {:rwse_walk_length, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    ffn_mult = Keyword.get(opts, :ffn_multiplier, @default_ffn_multiplier)
    activation = Keyword.get(opts, :activation, @default_activation)
    pe_dim = Keyword.get(opts, :pe_dim, @default_pe_dim)
    rwse_walk_length = Keyword.get(opts, :rwse_walk_length, @default_rwse_walk_length)
    num_classes = Keyword.get(opts, :num_classes, nil)
    pool = Keyword.get(opts, :pool, nil)

    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    adjacency = Axon.input("adjacency", shape: {nil, nil, nil})

    # RWSE: compute random walk structural encoding from adjacency
    rwse =
      Axon.layer(
        &rwse_impl/2,
        [adjacency],
        name: "rwse",
        walk_length: rwse_walk_length,
        op_name: :rwse
      )

    # Project RWSE: [batch, N, walk_length] -> [batch, N, pe_dim]
    rwse_proj =
      rwse
      |> Axon.layer_norm(name: "rwse_norm")
      |> Axon.dense(pe_dim, name: "rwse_proj")

    # Project node features: [batch, N, input_dim] -> [batch, N, hidden_size - pe_dim]
    node_proj = Axon.dense(nodes, hidden_size - pe_dim, name: "node_proj")

    # Concatenate node features + RWSE
    x =
      Axon.layer(
        fn node_feat, pe_feat, _opts ->
          Nx.concatenate([node_feat, pe_feat], axis: -1)
        end,
        [node_proj, rwse_proj],
        name: "feature_pe_concat",
        op_name: :concatenate
      )

    # Stack GPS layers
    x =
      Enum.reduce(0..(num_layers - 1), x, fn i, acc ->
        gps_layer(acc, adjacency, hidden_size, num_heads, dropout, ffn_mult, activation, i)
      end)

    # Optional pooling for graph-level tasks
    x =
      case pool do
        :mean -> Axon.nx(x, fn t -> Nx.mean(t, axes: [1]) end, name: "mean_pool")
        :sum -> Axon.nx(x, fn t -> Nx.sum(t, axes: [1]) end, name: "sum_pool")
        nil -> x
      end

    # Optional classification head
    if num_classes do
      x
      |> Axon.dense(hidden_size, name: "cls_dense")
      |> Axon.activation(activation, name: "cls_act")
      |> Axon.dense(num_classes, name: "cls_out")
    else
      x
    end
  end

  @doc "Get the output size of a GPS model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    num_classes = Keyword.get(opts, :num_classes, nil)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    if num_classes, do: num_classes, else: hidden_size
  end

  # ===========================================================================
  # GPS Layer
  # ===========================================================================

  defp gps_layer(x, adjacency, hidden_size, num_heads, dropout, ffn_mult, activation, idx) do
    prefix = "gps_#{idx}"

    # Branch 1: Local MPNN (GIN)
    h_local = GIN.gin_layer(x, adjacency, hidden_size, name: "#{prefix}_gin")

    h_local =
      if dropout > 0.0 do
        Axon.dropout(h_local, rate: dropout, name: "#{prefix}_gin_drop")
      else
        h_local
      end

    h_local_res = Axon.add(h_local, x, name: "#{prefix}_gin_res")
    h_local_norm = Axon.layer_norm(h_local_res, name: "#{prefix}_gin_norm")

    # Branch 2: Global multi-head attention (no adjacency masking)
    head_dim = div(hidden_size, num_heads)

    q = Axon.dense(x, hidden_size, name: "#{prefix}_attn_q")
    k = Axon.dense(x, hidden_size, name: "#{prefix}_attn_k")
    v = Axon.dense(x, hidden_size, name: "#{prefix}_attn_v")

    h_attn =
      Axon.layer(
        &global_attention_impl/4,
        [q, k, v],
        name: "#{prefix}_attn",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :gps_attention
      )

    h_attn = Axon.dense(h_attn, hidden_size, name: "#{prefix}_attn_out")

    h_attn =
      if dropout > 0.0 do
        Axon.dropout(h_attn, rate: dropout, name: "#{prefix}_attn_drop")
      else
        h_attn
      end

    h_attn_res = Axon.add(h_attn, x, name: "#{prefix}_attn_res")
    h_attn_norm = Axon.layer_norm(h_attn_res, name: "#{prefix}_attn_norm")

    # Combine branches by summation
    h = Axon.add(h_local_norm, h_attn_norm, name: "#{prefix}_combine")

    # FFN: Dense -> Act -> Dense
    ffn_dim = hidden_size * ffn_mult

    h_ffn =
      h
      |> Axon.dense(ffn_dim, name: "#{prefix}_ffn_1")
      |> Axon.activation(activation, name: "#{prefix}_ffn_act")

    h_ffn =
      if dropout > 0.0 do
        Axon.dropout(h_ffn, rate: dropout, name: "#{prefix}_ffn_drop")
      else
        h_ffn
      end

    h_ffn = Axon.dense(h_ffn, hidden_size, name: "#{prefix}_ffn_2")

    h_ffn_res = Axon.add(h_ffn, h, name: "#{prefix}_ffn_res")
    Axon.layer_norm(h_ffn_res, name: "#{prefix}_ffn_norm")
  end

  # ===========================================================================
  # Custom Layer Implementations
  # ===========================================================================

  # RWSE: Random Walk Structural Encoding
  # Computes diag((D^{-1}A)^k) for k = 1..walk_length
  defp rwse_impl(adjacency, opts) do
    walk_length = opts[:walk_length]
    {_batch, n, _n} = Nx.shape(adjacency)

    # Compute degree-normalized adjacency: D^{-1} A
    degree = Nx.sum(adjacency, axes: [2])
    # Safe inverse: avoid division by zero
    safe_degree = Nx.max(degree, Nx.tensor(1.0e-8))
    inv_degree = Nx.divide(1.0, safe_degree)
    # Zero out where degree was actually 0
    mask = Nx.greater(degree, 0) |> Nx.as_type(:f32)
    inv_degree = Nx.multiply(inv_degree, mask)
    # [batch, N, 1] * [batch, N, N] = row-normalized adjacency
    rw_matrix = Nx.multiply(Nx.new_axis(inv_degree, -1), adjacency)

    # Eye matrix for extracting diagonals
    eye = Nx.eye(n)

    # Compute powers and extract diagonals
    {_final_power, rwse} =
      Enum.reduce(1..walk_length, {rw_matrix, []}, fn _k, {power, diags} ->
        # Extract diagonal: [batch, N]
        diag = Nx.sum(Nx.multiply(power, eye), axes: [2])
        # Next power: power @ rw_matrix
        next_power = Nx.dot(power, [2], [0], rw_matrix, [1], [0])
        {next_power, diags ++ [diag]}
      end)

    # Stack diagonals: [batch, N, walk_length]
    Nx.stack(rwse, axis: 2)
  end

  # Global multi-head self-attention (no adjacency masking)
  defp global_attention_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    {batch, seq_len, _dim} = Nx.shape(q)

    # Reshape to multi-head: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
    q_h =
      q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    k_h =
      k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    v_h =
      v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = :math.sqrt(head_dim)
    scores = Nx.dot(q_h, [3], [0, 1], k_h, [3], [0, 1]) |> Nx.divide(scale)
    scores_stable = Nx.subtract(scores, Nx.reduce_max(scores, axes: [-1], keep_axes: true))
    weights = Nx.exp(scores_stable)
    weights = Nx.divide(weights, Nx.sum(weights, axes: [-1], keep_axes: true))

    # Weighted sum of values
    attn_out = Nx.dot(weights, [3], [0, 1], v_h, [2], [0, 1])

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    attn_out
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end
end
