defmodule Edifice.Graph.SE3Transformer do
  @moduledoc """
  SE(3)-Transformer — 3D Roto-Translation Equivariant Attention Network.

  <!-- verified: true, date: 2026-02-27 -->

  The SE(3)-Transformer applies self-attention to 3D point clouds while
  maintaining equivariance under rotations and translations. It uses fiber
  features: type-0 (scalar, rotation-invariant) and type-1 (vector, 3D
  equivariant) features that are processed through equivariant attention
  and tensor field network (TFN) convolutions.

  ## Architecture

  ```
  Node Features [batch, N, input_dim]
  Positions     [batch, N, 3]
        |
        v
  +----------------------------------------------+
  | SE(3)-Equivariant Attention Layer 1:         |
  |   1. Compute pairwise distances + directions |
  |   2. Radial basis expansion                  |
  |   3. Invariant attention (from type-0)       |
  |   4. Equivariant value aggregation           |
  |   5. Type-0 + Type-1 feature updates         |
  +----------------------------------------------+
        |  (repeat N times)
        v
  Per-atom Embeddings [batch, N, out_size]
  ```

  ## Fiber Features

  - **Type-0 (scalars)**: `[batch, N, channels_0]` — rotation invariant
  - **Type-1 (vectors)**: `[batch, N, channels_1, 3]` — transform equivariantly

  Attention weights are computed from type-0 features only (guaranteeing
  invariance), while values include both types (preserving equivariance).

  ## Usage

      model = SE3Transformer.build(
        input_dim: 16,
        hidden_size: 64,
        num_layers: 4,
        num_heads: 4,
        channels_1: 16,
        cutoff: 5.0
      )

  ## References

  - Fuchs et al., "SE(3)-Transformers: 3D Roto-Translation Equivariant
    Attention Networks" (NeurIPS 2020)
  - https://arxiv.org/abs/2006.10503
  """

  @default_hidden_size 64
  @default_num_layers 4
  @default_num_heads 4
  @default_channels_1 16
  @default_cutoff 5.0
  @default_num_radial 16
  @default_out_size 64

  @doc """
  Build an SE(3)-Transformer model.

  ## Options

    - `:input_dim` - Input feature dimension per node (required)
    - `:hidden_size` - Type-0 scalar channels (default: 64)
    - `:num_layers` - Number of equivariant attention layers (default: 4)
    - `:num_heads` - Number of attention heads (default: 4)
    - `:channels_1` - Type-1 vector channels (default: 16)
    - `:cutoff` - Distance cutoff for interactions (default: 5.0)
    - `:num_radial` - Number of radial basis functions (default: 16)
    - `:out_size` - Output feature dimension (default: 64)
    - `:num_classes` - If provided, adds classification head (default: nil)
    - `:pool` - Global pooling mode `:sum` or `:mean` (default: nil)

  ## Returns

  An Axon model with inputs "nodes" `[batch, N, input_dim]` and
  "positions" `[batch, N, 3]`. Returns `[batch, N, out_size]` per-node
  features, or `[batch, out_size]` with pooling, or `[batch, num_classes]`
  with classification head.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:channels_1, pos_integer()}
          | {:cutoff, float()}
          | {:hidden_size, pos_integer()}
          | {:input_dim, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:num_heads, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:num_radial, pos_integer()}
          | {:out_size, pos_integer()}
          | {:pool, :sum | :mean | nil}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    channels_1 = Keyword.get(opts, :channels_1, @default_channels_1)
    cutoff = Keyword.get(opts, :cutoff, @default_cutoff)
    num_radial = Keyword.get(opts, :num_radial, @default_num_radial)
    out_size = Keyword.get(opts, :out_size, @default_out_size)
    num_classes = Keyword.get(opts, :num_classes, nil)
    pool = Keyword.get(opts, :pool, nil)

    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    positions = Axon.input("positions", shape: {nil, nil, 3})

    # Compute geometric features: distances, directions, RBF
    geo =
      Axon.layer(
        &compute_geometry_impl/2,
        [positions],
        name: "geometry",
        cutoff: cutoff,
        num_radial: num_radial,
        op_name: :compute_geometry
      )

    # Embed type-0 scalars
    h0 =
      nodes
      |> Axon.dense(hidden_size, name: "type0_embed")
      |> Axon.layer_norm(name: "type0_embed_ln")

    # Initialize type-1 vectors from relative directions weighted by node features
    h1 =
      Axon.layer(
        &init_type1_impl/3,
        [nodes, geo],
        name: "type1_init",
        channels_1: channels_1,
        op_name: :init_type1
      )

    h1_proj =
      Axon.param("type1_init_proj", {input_dim, channels_1}, initializer: :glorot_uniform)

    h1 =
      Axon.layer(
        fn feat, proj, _opts ->
          # feat: [batch, N, input_dim, 3], proj: [input_dim, channels_1]
          {batch, n, _d, c3} = Nx.shape(feat)
          flat = Nx.reshape(feat, {batch * n, Nx.axis_size(feat, 2), c3})
          # [batch*N, 3, input_dim] @ [input_dim, channels_1] → [batch*N, 3, channels_1]
          flat_t = Nx.transpose(flat, axes: [0, 2, 1])
          result = Nx.dot(flat_t, [2], proj, [0])
          # → [batch*N, channels_1, 3]
          result = Nx.transpose(result, axes: [0, 2, 1])
          Nx.reshape(result, {batch, n, Nx.axis_size(proj, 1), 3})
        end,
        [h1, h1_proj],
        name: "type1_project",
        op_name: :project_type1
      )

    # Stack SE(3)-equivariant attention layers
    {h0, _h1} =
      Enum.reduce(0..(num_layers - 1), {h0, h1}, fn idx, {s0, s1} ->
        se3_attention_layer(s0, s1, geo, positions,
          hidden_size: hidden_size,
          channels_1: channels_1,
          num_heads: num_heads,
          cutoff: cutoff,
          num_radial: num_radial,
          name: "se3_layer_#{idx}"
        )
      end)

    # Output projection (type-0 scalar features only — invariant)
    x =
      if out_size != hidden_size do
        Axon.dense(h0, out_size, name: "output_proj")
      else
        h0
      end

    # Optional pooling
    x =
      case pool do
        :sum ->
          Axon.layer(
            fn t, _opts -> Nx.sum(t, axes: [1]) end,
            [x],
            name: "global_sum_pool",
            op_name: :sum_pool
          )

        :mean ->
          Axon.layer(
            fn t, _opts -> Nx.mean(t, axes: [1]) end,
            [x],
            name: "global_mean_pool",
            op_name: :mean_pool
          )

        nil ->
          x
      end

    # Optional classification head
    if num_classes do
      x
      |> Axon.dense(out_size, name: "cls_hidden")
      |> Axon.layer_norm(name: "cls_ln")
      |> Axon.activation(:silu, name: "cls_act")
      |> Axon.dense(num_classes, name: "cls_head")
    else
      x
    end
  end

  @doc "Get the output size of an SE(3)-Transformer model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    num_classes = Keyword.get(opts, :num_classes, nil)
    out_size = Keyword.get(opts, :out_size, @default_out_size)
    if num_classes, do: num_classes, else: out_size
  end

  # ============================================================================
  # SE(3)-Equivariant Attention Layer
  # ============================================================================

  defp se3_attention_layer(h0, h1, geo, _positions, opts) do
    hidden = opts[:hidden_size]
    channels_1 = opts[:channels_1]
    num_heads = opts[:num_heads]
    cutoff = opts[:cutoff]
    num_radial = opts[:num_radial]
    name = opts[:name]

    head_dim = div(hidden, num_heads)

    # --- Type-0 attention (invariant) ---

    # Q, K from type-0: [batch, N, hidden]
    q0 = Axon.dense(h0, hidden, name: "#{name}_q0")
    k0 = Axon.dense(h0, hidden, name: "#{name}_k0")
    v0 = Axon.dense(h0, hidden, name: "#{name}_v0")

    # Radial bias: RBF → [batch, N, N, num_heads] distance-dependent bias
    radial_bias_w =
      Axon.param("#{name}_radial_bias", {num_radial, num_heads}, initializer: :glorot_uniform)

    # Equivariant attention computation
    attn_out =
      Axon.layer(
        &equivariant_attention_impl/7,
        [q0, k0, v0, h1, geo, radial_bias_w],
        name: "#{name}_attn",
        num_heads: num_heads,
        head_dim: head_dim,
        channels_1: channels_1,
        cutoff: cutoff,
        op_name: :se3_attention
      )

    # Split out type-0 and type-1 results
    new_h0 =
      Axon.nx(attn_out, fn r -> r.type0 end, name: "#{name}_attn_h0")

    new_h1 =
      Axon.nx(attn_out, fn r -> r.type1 end, name: "#{name}_attn_h1")

    # Type-0 residual + norm
    h0_out = Axon.add(h0, new_h0, name: "#{name}_res0")
    h0_out = Axon.layer_norm(h0_out, name: "#{name}_ln0")

    # Type-0 FFN
    h0_ffn =
      h0_out
      |> Axon.dense(hidden * 2, name: "#{name}_ffn1")
      |> Axon.layer_norm(name: "#{name}_ffn_ln")
      |> Axon.activation(:silu, name: "#{name}_ffn_act")
      |> Axon.dense(hidden, name: "#{name}_ffn2")

    h0_out = Axon.add(h0_out, h0_ffn, name: "#{name}_ffn_res")
    h0_out = Axon.layer_norm(h0_out, name: "#{name}_ffn_ln2")

    # Type-1 residual + equivariant norm
    h1_out =
      Axon.layer(
        &type1_residual_norm_impl/3,
        [h1, new_h1],
        name: "#{name}_type1_resnorm",
        op_name: :type1_residual_norm
      )

    # Type-1 gating from type-0 (scalar gates vector channels — equivariant)
    h1_gate =
      Axon.dense(h0_out, channels_1, name: "#{name}_type1_gate")

    h1_out =
      Axon.layer(
        &type1_gate_impl/3,
        [h1_out, h1_gate],
        name: "#{name}_type1_gated",
        op_name: :type1_gate
      )

    # Type-1 direction update from positions (equivariant message)
    h1_msg_w =
      Axon.param(
        "#{name}_type1_msg_w",
        {num_radial, channels_1},
        initializer: :glorot_uniform
      )

    h1_dir =
      Axon.layer(
        &type1_direction_message_impl/4,
        [h0_out, geo, h1_msg_w],
        name: "#{name}_type1_dir",
        cutoff: cutoff,
        channels_1: channels_1,
        op_name: :type1_direction_msg
      )

    h1_out =
      Axon.layer(
        fn a, b, _opts -> Nx.add(a, b) end,
        [h1_out, h1_dir],
        name: "#{name}_type1_add",
        op_name: :add
      )

    {h0_out, h1_out}
  end

  # ============================================================================
  # Runtime Implementations
  # ============================================================================

  # Compute geometric features: pairwise distances, directions, RBF
  defp compute_geometry_impl(positions, opts) do
    cutoff = opts[:cutoff]
    num_radial = opts[:num_radial]

    # Pairwise differences: [batch, N, N, 3]
    pos_i = Nx.new_axis(positions, 2)
    pos_j = Nx.new_axis(positions, 1)
    diff = Nx.subtract(pos_i, pos_j)

    # Distances: [batch, N, N]
    dist = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(diff, diff), axes: [3]), 1.0e-8))

    # Unit directions: [batch, N, N, 3]
    dir = Nx.divide(diff, Nx.new_axis(dist, -1))

    # Mask self-interactions and beyond-cutoff
    n = Nx.axis_size(dist, 1)
    diag_mask = Nx.equal(Nx.iota({n, 1}), Nx.iota({1, n}))
    not_self = Nx.logical_not(diag_mask)
    within = Nx.logical_and(Nx.less(dist, cutoff), not_self)
    mask_f = Nx.as_type(within, Nx.type(dist))

    # Gaussian RBF: [batch, N, N, num_radial]
    centers = Nx.linspace(0.0, cutoff, n: num_radial)
    gamma = Nx.divide(10.0, Nx.multiply(cutoff, cutoff))
    dist_exp = Nx.new_axis(dist, -1)
    rbf = Nx.exp(Nx.negate(Nx.multiply(gamma, Nx.pow(Nx.subtract(dist_exp, centers), 2))))

    # Apply mask to RBF
    rbf = Nx.multiply(rbf, Nx.new_axis(mask_f, -1))

    %{dist: dist, dir: dir, rbf: rbf, mask: mask_f}
  end

  # Initialize type-1 features from neighbor directions
  # Returns: [batch, N, input_dim, 3]
  defp init_type1_impl(nodes, geo, _opts) do
    # nodes: [batch, N, input_dim], dir: [batch, N, N, 3], mask: [batch, N, N]
    dir = geo.dir
    mask = geo.mask

    # Aggregate direction vectors weighted by mask: [batch, N, 3]
    masked_dir = Nx.multiply(dir, Nx.new_axis(mask, -1))

    # For each node, average neighbor directions: [batch, N, 3]
    avg_dir = Nx.mean(masked_dir, axes: [2])

    # Outer product with node features: [batch, N, input_dim, 3]
    # nodes: [batch, N, input_dim] → [batch, N, input_dim, 1]
    # avg_dir: [batch, N, 3] → [batch, N, 1, 3]
    Nx.multiply(Nx.new_axis(nodes, 3), Nx.new_axis(avg_dir, 2))
  end

  # SE(3)-equivariant attention computation
  # Returns container with :type0 and :type1
  # credo:disable-for-next-line Credo.Check.Refactor.FunctionArity
  defp equivariant_attention_impl(q0, k0, v0, h1, geo, radial_bias_w, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    cutoff = opts[:cutoff]

    {batch, n, _hidden} = Nx.shape(q0)

    # Reshape for multi-head: [batch, N, heads, head_dim]
    q = Nx.reshape(q0, {batch, n, num_heads, head_dim})
    k = Nx.reshape(k0, {batch, n, num_heads, head_dim})
    v = Nx.reshape(v0, {batch, n, num_heads, head_dim})

    # Attention scores: [batch, i, heads, j] via q_i . k_j
    # q: [batch, N, heads, head_dim] → [batch, N, 1, heads, head_dim]
    # k: [batch, N, heads, head_dim] → [batch, 1, N, heads, head_dim]
    q_exp = Nx.new_axis(q, 2)
    k_exp = Nx.new_axis(k, 1)
    scores = Nx.sum(Nx.multiply(q_exp, k_exp), axes: [4])
    # scores: [batch, N, N, heads]

    scale = Nx.sqrt(Nx.tensor(head_dim, type: :f32))
    scores = Nx.divide(scores, scale)

    # Add radial bias: rbf @ radial_bias_w → [batch, N, N, heads]
    rbf = geo.rbf
    {_b, _n1, _n2, nr} = Nx.shape(rbf)
    rbf_flat = Nx.reshape(rbf, {batch * n * n, nr})
    bias = Nx.dot(rbf_flat, radial_bias_w)
    bias = Nx.reshape(bias, {batch, n, n, num_heads})
    scores = Nx.add(scores, bias)

    # Mask: apply cutoff mask [batch, N, N] → [batch, N, N, 1]
    mask = geo.mask
    mask_exp = Nx.new_axis(mask, -1)
    large_neg = Nx.broadcast(-1.0e9, Nx.shape(scores))
    mask_f = Nx.as_type(mask_exp, Nx.type(scores))
    scores = Nx.add(Nx.multiply(scores, mask_f), Nx.multiply(large_neg, Nx.subtract(1.0, mask_f)))

    # Softmax over j (axis 2): [batch, N, N, heads]
    scores_max = Nx.reduce_max(scores, axes: [2], keep_axes: true)
    scores_exp = Nx.exp(Nx.subtract(scores, scores_max))
    scores_exp = Nx.multiply(scores_exp, mask_f)
    attn = Nx.divide(scores_exp, Nx.add(Nx.sum(scores_exp, axes: [2], keep_axes: true), 1.0e-8))

    # --- Type-0 output ---
    # attn: [batch, i, j, heads], v: [batch, j, heads, head_dim]
    # → sum over j: [batch, i, heads, head_dim]
    attn_v = Nx.new_axis(attn, -1)
    v_exp = Nx.new_axis(v, 1)
    out0 = Nx.sum(Nx.multiply(attn_v, v_exp), axes: [2])
    # out0: [batch, N, heads, head_dim] → [batch, N, hidden]
    out0 = Nx.reshape(out0, {batch, n, num_heads * head_dim})

    # --- Type-1 output ---
    # h1: [batch, N, channels_1, 3]
    # Use scalar attention (averaged over heads) to weight type-1 neighbor features
    # attn_mean: [batch, i, j] (mean over heads)
    attn_mean = Nx.mean(attn, axes: [3])

    # Direction-weighted type-1 aggregation
    # For equivariance: weight type-1 features by invariant attention
    # h1_j: [batch, 1, N, C1, 3] * attn: [batch, N, N, 1, 1] → sum over j
    h1_exp = Nx.new_axis(h1, 1)
    attn_1 = Nx.reshape(attn_mean, {batch, n, n, 1, 1})
    out1 = Nx.sum(Nx.multiply(h1_exp, attn_1), axes: [2])
    # out1: [batch, N, C1, 3]

    # Add direction component weighted by distance
    dir = geo.dir
    dist = geo.dist

    # Smooth cutoff weight
    dist_clamp = Nx.min(dist, cutoff)
    env = Nx.subtract(1.0, Nx.divide(dist_clamp, cutoff))
    env = Nx.multiply(env, Nx.as_type(mask, Nx.type(env)))

    # Weighted directions: [batch, N, N, 3] * [batch, N, N] → [batch, N, 3]
    weighted_dir = Nx.multiply(dir, Nx.new_axis(Nx.multiply(env, attn_mean), -1))
    agg_dir = Nx.sum(weighted_dir, axes: [2])
    # agg_dir: [batch, N, 3] → [batch, N, 1, 3]
    agg_dir_exp = Nx.new_axis(agg_dir, 2)

    # Scale by learned channel weights (from type-0 projection, approximated here)
    channels_1 = Nx.axis_size(h1, 2)
    dir_contrib = Nx.broadcast(agg_dir_exp, {batch, n, channels_1, 3})
    out1 = Nx.add(out1, Nx.multiply(0.1, dir_contrib))

    %{type0: out0, type1: out1}
  end

  # Type-1 equivariant layer norm: normalize by per-channel vector magnitude
  defp type1_residual_norm_impl(h1, new_h1, _opts) do
    # h1, new_h1: [batch, N, C1, 3]
    residual = Nx.add(h1, new_h1)

    # Per-channel norm: [batch, N, C1]
    norms = Nx.sqrt(Nx.add(Nx.sum(Nx.multiply(residual, residual), axes: [3]), 1.0e-8))
    # Normalize: divide each 3-vector by its norm
    Nx.divide(residual, Nx.new_axis(norms, -1))
  end

  # Gate type-1 features with scalar gates (equivariant: scalar * vector = vector)
  defp type1_gate_impl(h1, gate, _opts) do
    # h1: [batch, N, C1, 3], gate: [batch, N, C1]
    gate_sig = Nx.sigmoid(gate)
    Nx.multiply(h1, Nx.new_axis(gate_sig, -1))
  end

  # Type-1 direction message: aggregate neighbor directions weighted by scalar features
  defp type1_direction_message_impl(h0, geo, msg_w, opts) do
    cutoff = opts[:cutoff]
    channels_1 = opts[:channels_1]

    rbf = geo.rbf
    dir = geo.dir
    mask = geo.mask

    {batch, n, _hidden} = Nx.shape(h0)

    # RBF → channel weights: [batch, N, N, C1]
    {_b, _n1, _n2, nr} = Nx.shape(rbf)
    rbf_flat = Nx.reshape(rbf, {batch * n * n, nr})
    weights = Nx.dot(rbf_flat, msg_w)
    weights = Nx.reshape(weights, {batch, n, n, channels_1})

    # Scalar feature similarity between nodes (invariant)
    # h0: [batch, N, hidden] → norm then dot product
    h0_norm =
      Nx.divide(
        h0,
        Nx.add(Nx.sqrt(Nx.sum(Nx.multiply(h0, h0), axes: [2], keep_axes: true)), 1.0e-8)
      )

    # sim: [batch, i, j] = h0_norm_i . h0_norm_j (batched)
    sim = Nx.dot(h0_norm, [2], [0], h0_norm, [2], [0])
    # Apply mask
    sim = Nx.multiply(sim, mask)

    # Cutoff envelope
    dist = geo.dist
    dist_clamp = Nx.min(dist, cutoff)
    env = Nx.subtract(1.0, Nx.divide(dist_clamp, cutoff))

    # Combined scalar weight: [batch, N, N]
    scalar_w = Nx.multiply(sim, Nx.multiply(env, Nx.as_type(mask, Nx.type(env))))

    # Direction * weights * scalar: [batch, N, N, C1, 3]
    # dir: [batch, N, N, 3] → [batch, N, N, 1, 3]
    # weights: [batch, N, N, C1] → [batch, N, N, C1, 1]
    # scalar_w: [batch, N, N] → [batch, N, N, 1, 1]
    dir_exp = Nx.new_axis(dir, 3)
    w_exp = Nx.new_axis(weights, -1)
    sw_exp = scalar_w |> Nx.new_axis(-1) |> Nx.new_axis(-1)

    msg = Nx.multiply(Nx.multiply(dir_exp, w_exp), sw_exp)
    # Sum over neighbors (axis 2): [batch, N, C1, 3]
    Nx.sum(msg, axes: [2])
  end
end
