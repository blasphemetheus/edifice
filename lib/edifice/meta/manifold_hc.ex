defmodule Edifice.Meta.ManifoldHC do
  @moduledoc """
  mHC: Manifold-Constrained Hyper-Connections.

  Implements mHC from "mHC: Manifold-Constrained Hyper-Connections"
  (DeepSeek, arXiv:2512.24880). Replaces standard residual connections
  with multi-rate residual streams constrained to the Birkhoff polytope,
  restoring the identity mapping property while diversifying connectivity.

  ## Key Ideas

  - **Multi-stream residual**: Expands hidden state from C to n*C (n streams
    of C dimensions). Each transformer sublayer operates on an aggregated
    single stream, then broadcasts back to all n streams.
  - **Data-dependent routing**: Three learnable matrices H^pre, H^post, H^res
    are computed dynamically from the current state at each position.
  - **Birkhoff constraint**: H^res (stream mixer) is projected onto the set of
    doubly stochastic matrices via Sinkhorn-Knopp normalization, ensuring
    spectral norm <= 1 and stable signal propagation across layers.

  ## Architecture

  ```
  Standard Residual:     x_{l+1} = x_l + F(x_l)

  mHC Residual:          x_{l+1} = H^res · x_l + H^post ⊗ F(H^pre · x_l)

  Where:
    x_l ∈ [batch, seq, n, C]     (n parallel streams)
    H^pre  ∈ [batch, seq, n]     (stream aggregation weights, sigmoid)
    H^post ∈ [batch, seq, n]     (stream expansion weights, 2·sigmoid)
    H^res  ∈ [batch, seq, n, n]  (stream mixing, doubly stochastic)
  ```

  ## Usage

      model = ManifoldHC.build(
        hidden_size: 256,
        expansion_rate: 4,
        num_layers: 6,
        num_heads: 4
      )

      # Input:  [batch, seq, hidden_size]
      # Output: [batch, seq, hidden_size]

  ## References

  - "mHC: Manifold-Constrained Hyper-Connections" (DeepSeek, 2024)
    https://arxiv.org/abs/2512.24880
  - Sinkhorn & Knopp, "Concerning nonnegative matrices and doubly
    stochastic matrices" (1967)
  """

  alias Edifice.Blocks.SwiGLU

  @default_hidden_size 256
  @default_expansion_rate 4
  @default_num_layers 4
  @default_num_heads 4
  @default_sinkhorn_iters 5
  @default_mlp_ratio 4.0

  @doc """
  Build a transformer model with mHC residual connections.

  ## Options

    - `:hidden_size` - Base hidden dimension C (default: 256)
    - `:expansion_rate` - Number of parallel streams n (default: 4)
    - `:num_layers` - Number of transformer blocks (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:sinkhorn_iters` - Sinkhorn-Knopp iterations for H^res (default: 5)
    - `:mlp_ratio` - FFN expansion ratio (default: 4.0)
    - `:seq_len` - Sequence length (default: nil for dynamic)
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:expansion_rate, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:seq_len, pos_integer() | nil}
          | {:sinkhorn_iters, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    expansion_rate = Keyword.get(opts, :expansion_rate, @default_expansion_rate)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    sinkhorn_iters = Keyword.get(opts, :sinkhorn_iters, @default_sinkhorn_iters)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    seq_len = Keyword.get(opts, :seq_len, nil)

    input = Axon.input("sequence", shape: {nil, seq_len, hidden_size})

    # Expand: [batch, seq, C] -> [batch, seq, n*C] by replication
    x =
      Axon.nx(
        input,
        fn t ->
          batch = Nx.axis_size(t, 0)
          seq = Nx.axis_size(t, 1)
          c = Nx.axis_size(t, 2)
          # Replicate C dimensions n times: tile along last axis
          Nx.reshape(t, {batch, seq, 1, c})
          |> Nx.broadcast({batch, seq, expansion_rate, c})
          |> Nx.reshape({batch, seq, expansion_rate * c})
        end,
        name: "expand_streams"
      )

    # Transformer blocks with mHC residuals
    x =
      Enum.reduce(0..(num_layers - 1), x, fn idx, acc ->
        mhc_block(acc,
          hidden_size: hidden_size,
          expansion_rate: expansion_rate,
          num_heads: num_heads,
          sinkhorn_iters: sinkhorn_iters,
          mlp_ratio: mlp_ratio,
          name: "block_#{idx}"
        )
      end)

    # Collapse: [batch, seq, n*C] -> [batch, seq, C] by averaging streams
    x =
      Axon.nx(
        x,
        fn t ->
          batch = Nx.axis_size(t, 0)
          seq = Nx.axis_size(t, 1)

          Nx.reshape(t, {batch, seq, expansion_rate, hidden_size})
          |> Nx.mean(axes: [2])
        end,
        name: "collapse_streams"
      )

    Axon.layer_norm(x, name: "final_norm")
  end

  # ============================================================================
  # mHC Transformer Block
  # ============================================================================

  # Each block has two mHC-wrapped sublayers: attention and FFN.
  defp mhc_block(x, opts) do
    name = Keyword.fetch!(opts, :name)
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    expansion_rate = Keyword.fetch!(opts, :expansion_rate)
    num_heads = Keyword.fetch!(opts, :num_heads)
    sinkhorn_iters = Keyword.fetch!(opts, :sinkhorn_iters)
    mlp_ratio = Keyword.fetch!(opts, :mlp_ratio)

    head_dim = div(hidden_size, num_heads)
    mlp_hidden = round(hidden_size * mlp_ratio)

    # === Attention sublayer with mHC ===
    # Compute H matrices from expanded state
    {h_pre_attn, h_post_attn, h_res_attn} =
      compute_h_matrices(x, expansion_rate, "#{name}_attn")

    # Pre-aggregate: weighted sum of streams -> [batch, seq, C]
    x_agg_attn =
      Axon.layer(
        &pre_aggregate_impl/3,
        [x, h_pre_attn],
        name: "#{name}_attn_agg",
        hidden_size: hidden_size,
        expansion_rate: expansion_rate,
        op_name: :pre_aggregate
      )

    # Standard self-attention on aggregated input
    normed = Axon.layer_norm(x_agg_attn, name: "#{name}_attn_norm")
    q = Axon.dense(normed, hidden_size, name: "#{name}_attn_q")
    k = Axon.dense(normed, hidden_size, name: "#{name}_attn_k")
    v = Axon.dense(normed, hidden_size, name: "#{name}_attn_v")

    attn_out =
      Axon.layer(
        &mha_impl/4,
        [q, k, v],
        name: "#{name}_attn_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :multi_head_attention
      )

    y_attn = Axon.dense(attn_out, hidden_size, name: "#{name}_attn_proj")

    # mHC merge: H^res * x + H^post * y
    x =
      Axon.layer(
        &mhc_merge_impl/5,
        [x, y_attn, h_post_attn, h_res_attn],
        name: "#{name}_attn_merge",
        hidden_size: hidden_size,
        expansion_rate: expansion_rate,
        sinkhorn_iters: sinkhorn_iters,
        op_name: :mhc_merge
      )

    # === FFN sublayer with mHC ===
    {h_pre_ffn, h_post_ffn, h_res_ffn} =
      compute_h_matrices(x, expansion_rate, "#{name}_ffn")

    x_agg_ffn =
      Axon.layer(
        &pre_aggregate_impl/3,
        [x, h_pre_ffn],
        name: "#{name}_ffn_agg",
        hidden_size: hidden_size,
        expansion_rate: expansion_rate,
        op_name: :pre_aggregate
      )

    normed2 = Axon.layer_norm(x_agg_ffn, name: "#{name}_ffn_norm")

    y_ffn =
      SwiGLU.layer(normed2, hidden_size: hidden_size, inner_size: mlp_hidden, name: "#{name}_ffn")

    Axon.layer(
      &mhc_merge_impl/5,
      [x, y_ffn, h_post_ffn, h_res_ffn],
      name: "#{name}_ffn_merge",
      hidden_size: hidden_size,
      expansion_rate: expansion_rate,
      sinkhorn_iters: sinkhorn_iters,
      op_name: :mhc_merge
    )
  end

  # ============================================================================
  # H Matrix Computation
  # ============================================================================

  # Compute the three routing matrices from the expanded state.
  # Returns {h_pre_logits, h_post_logits, h_res_logits} as Axon nodes.
  defp compute_h_matrices(x, n, name_prefix) do
    # Layer norm on the expanded state
    normed = Axon.layer_norm(x, name: "#{name_prefix}_h_norm")

    # Project to get H coefficient logits
    # Small kernel init approximates the gating factor alpha ≈ 0.01
    h_pre = Axon.dense(normed, n, name: "#{name_prefix}_h_pre")
    h_post = Axon.dense(normed, n, name: "#{name_prefix}_h_post")
    h_res = Axon.dense(normed, n * n, name: "#{name_prefix}_h_res")

    {h_pre, h_post, h_res}
  end

  # ============================================================================
  # Runtime Implementations
  # ============================================================================

  # Pre-aggregate: weighted sum of n streams into one.
  # x: [batch, seq, n*C], h_pre: [batch, seq, n] -> [batch, seq, C]
  defp pre_aggregate_impl(x, h_pre_logits, opts) do
    hidden_size = opts[:hidden_size]
    n = opts[:expansion_rate]
    batch = Nx.axis_size(x, 0)
    seq = Nx.axis_size(x, 1)

    # Apply sigmoid constraint
    h_pre = Nx.sigmoid(h_pre_logits)

    # Reshape x to streams: [batch, seq, n, C]
    streams = Nx.reshape(x, {batch, seq, n, hidden_size})

    # Weighted sum: [batch, seq, n] dot [batch, seq, n, C] -> [batch, seq, C]
    # Contract on axis 2 (n) with batch axes [0, 1]
    Nx.dot(h_pre, [2], [0, 1], streams, [2], [0, 1])
  end

  # mHC merge: H^res * x_streams + H^post * y
  # x: [batch, seq, n*C], y: [batch, seq, C],
  # h_post: [batch, seq, n], h_res: [batch, seq, n*n]
  defp mhc_merge_impl(x, y, h_post_logits, h_res_logits, opts) do
    hidden_size = opts[:hidden_size]
    n = opts[:expansion_rate]
    sinkhorn_iters = opts[:sinkhorn_iters]
    batch = Nx.axis_size(x, 0)
    seq = Nx.axis_size(x, 1)

    # Apply constraints
    h_post = Nx.multiply(2.0, Nx.sigmoid(h_post_logits))
    h_res_raw = Nx.reshape(h_res_logits, {batch, seq, n, n})
    h_res = batch_sinkhorn(h_res_raw, sinkhorn_iters)

    # Reshape x to streams: [batch, seq, n, C]
    streams = Nx.reshape(x, {batch, seq, n, hidden_size})

    # Residual mixing: H^res * streams
    # [batch, seq, n, n] x [batch, seq, n, C] -> [batch, seq, n, C]
    x_mixed = Nx.dot(h_res, [3], [0, 1], streams, [2], [0, 1])

    # Post-expansion: H^post * y -> broadcast to n streams
    # h_post: [batch, seq, n] -> [batch, seq, n, 1]
    # y: [batch, seq, C] -> [batch, seq, 1, C]
    h_post_exp = Nx.reshape(h_post, {batch, seq, n, 1})
    y_exp = Nx.reshape(y, {batch, seq, 1, hidden_size})
    post_contrib = Nx.multiply(h_post_exp, y_exp)

    # Combine and flatten
    result = Nx.add(x_mixed, post_contrib)
    Nx.reshape(result, {batch, seq, n * hidden_size})
  end

  # Multi-head attention (standard SDPA)
  defp mha_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))

    weights =
      Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # Sinkhorn-Knopp Normalization
  # ============================================================================

  # Project matrices onto the Birkhoff polytope (doubly stochastic).
  # Input: [batch, seq, n, n] -> Output: [batch, seq, n, n]
  # Each [n, n] slice becomes doubly stochastic (rows and columns sum to 1).
  defp batch_sinkhorn(h_raw, iters) do
    # Ensure positivity
    m = Nx.exp(h_raw)

    # Alternating row/column normalization
    Enum.reduce(1..iters, m, fn _, m_acc ->
      # Row normalize: each row sums to 1
      row_sum = Nx.sum(m_acc, axes: [3], keep_axes: true)
      m_acc = Nx.divide(m_acc, Nx.add(row_sum, 1.0e-8))

      # Column normalize: each column sums to 1
      col_sum = Nx.sum(m_acc, axes: [2], keep_axes: true)
      Nx.divide(m_acc, Nx.add(col_sum, 1.0e-8))
    end)
  end

  @doc "Get the output size (hidden_size)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Get recommended defaults for different model sizes.
  """
  @spec recommended_defaults(atom()) :: keyword()
  def recommended_defaults(size \\ :small) do
    case size do
      :small ->
        [hidden_size: 256, expansion_rate: 4, num_layers: 4, num_heads: 4]

      :medium ->
        [hidden_size: 512, expansion_rate: 4, num_layers: 8, num_heads: 8]

      :large ->
        [hidden_size: 1024, expansion_rate: 4, num_layers: 12, num_heads: 16]
    end
  end
end
