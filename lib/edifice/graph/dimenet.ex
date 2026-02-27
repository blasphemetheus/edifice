defmodule Edifice.Graph.DimeNet do
  @moduledoc """
  DimeNet++ — Directional Message Passing Neural Network.

  Implements DimeNet++ from "Fast and Uncertainty-Aware Directional Message
  Passing for Non-Equilibrium Molecules" (Gasteiger et al., NeurIPS-W 2020).
  Extends standard GNNs by incorporating angular (directional) information
  between atom triplets, enabling richer geometric representations for
  molecular property prediction.

  ## Key Innovations

  - **Edge-level messages**: Operates on atom pairs (edges) rather than
    individual atoms, capturing pairwise interactions directly.
  - **Angular information**: Models angles between bond triplets (k, j, i)
    using Chebyshev angular basis, capturing directional geometry.
  - **Radial Bessel basis**: Expands interatomic distances into smooth
    orthogonal basis functions with learned frequencies.
  - **Envelope cutoff**: Smooth polynomial cutoff ensures zero contribution
    at the interaction boundary.

  ## Architecture

  ```
  Inputs: "nodes" [batch, N, input_dim], "positions" [batch, N, 3]
        |
        v
  [Compute distances, RBF, envelope from positions]
        |
        v
  [Embedding Block: atom features + RBF → edge messages]
        |
        v
  [Output Block 0: edge → atom aggregation] → P_0
        |
        v
  [Interaction Block 1: angular aggregate + residual] → [Output Block 1] → P += P_1
        |  (repeat num_blocks - 1 times)
        v
  Atom Predictions [batch, N, out_emb_size]
        |
        v
  [Optional pooling + output head]
  ```

  ## Usage

      model = DimeNet.build(
        input_dim: 16,
        hidden_size: 128,
        num_blocks: 4,
        cutoff: 5.0,
        num_classes: 1,
        pool: :sum
      )

      # Input: atom features + 3D coordinates
      # Output: per-atom or per-molecule predictions

  ## References

  - Gasteiger et al., "Directional Message Passing for Molecular Graphs"
    (ICLR 2020) — https://arxiv.org/abs/2003.03123
  - Gasteiger et al., "Fast and Uncertainty-Aware Directional Message
    Passing" (NeurIPS-W 2020) — https://arxiv.org/abs/2011.14115
  """

  alias Edifice.Graph.MessagePassing

  @default_hidden_size 128
  @default_num_blocks 4
  @default_num_radial 6
  @default_num_spherical 7
  @default_cutoff 5.0
  @default_envelope_exponent 5
  @default_int_emb_size 64
  @default_basis_emb_size 8
  @default_out_emb_size 256
  @default_num_output_layers 3

  @doc """
  Build a DimeNet++ model.

  ## Options

    - `:input_dim` - Input atom feature dimension (required)
    - `:hidden_size` - Hidden dimension for edge messages (default: 128)
    - `:num_blocks` - Number of interaction blocks (default: 4)
    - `:num_radial` - Number of radial basis functions (default: 6)
    - `:num_spherical` - Number of angular basis functions (default: 7)
    - `:cutoff` - Distance cutoff in Angstroms (default: 5.0)
    - `:envelope_exponent` - Smooth cutoff exponent (default: 5)
    - `:int_emb_size` - Bottleneck dimension in interaction (default: 64)
    - `:basis_emb_size` - Basis embedding dimension (default: 8)
    - `:out_emb_size` - Output embedding dimension (default: 256)
    - `:num_output_layers` - Dense layers in output block (default: 3)
    - `:num_classes` - Output classes; nil for embeddings (default: nil)
    - `:pool` - Global pooling (:sum, :mean, nil) (default: nil)

  ## Returns

  An Axon model taking two inputs:
  - `"nodes"` — atom features `[batch, num_atoms, input_dim]`
  - `"positions"` — 3D coordinates `[batch, num_atoms, 3]`
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:basis_emb_size, pos_integer()}
          | {:cutoff, float()}
          | {:envelope_exponent, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:input_dim, pos_integer()}
          | {:int_emb_size, pos_integer()}
          | {:num_blocks, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:num_output_layers, pos_integer()}
          | {:num_radial, pos_integer()}
          | {:num_spherical, pos_integer()}
          | {:out_emb_size, pos_integer()}
          | {:pool, :sum | :mean | nil}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_blocks = Keyword.get(opts, :num_blocks, @default_num_blocks)
    num_radial = Keyword.get(opts, :num_radial, @default_num_radial)
    num_spherical = Keyword.get(opts, :num_spherical, @default_num_spherical)
    cutoff = Keyword.get(opts, :cutoff, @default_cutoff)
    envelope_exponent = Keyword.get(opts, :envelope_exponent, @default_envelope_exponent)
    int_emb_size = Keyword.get(opts, :int_emb_size, @default_int_emb_size)
    basis_emb_size = Keyword.get(opts, :basis_emb_size, @default_basis_emb_size)
    out_emb_size = Keyword.get(opts, :out_emb_size, @default_out_emb_size)
    num_output_layers = Keyword.get(opts, :num_output_layers, @default_num_output_layers)
    num_classes = Keyword.get(opts, :num_classes, nil)
    pool = Keyword.get(opts, :pool, nil)

    nodes = Axon.input("nodes", shape: {nil, nil, input_dim})
    positions = Axon.input("positions", shape: {nil, nil, 3})

    # Atom embedding: [batch, N, input_dim] → [batch, N, hidden_size]
    x = Axon.dense(nodes, hidden_size, name: "atom_embed")

    # Compute RBF from positions: [batch, N, N, num_radial]
    rbf =
      Axon.layer(
        &compute_rbf_impl/2,
        [positions],
        name: "rbf",
        cutoff: cutoff,
        exponent: envelope_exponent,
        num_radial: num_radial,
        op_name: :radial_bessel_basis
      )

    # Create edge embeddings: [batch, N, N, hidden_size]
    edge_features =
      Axon.layer(
        &broadcast_edge_features_impl/3,
        [x, rbf],
        name: "edge_broadcast",
        op_name: :broadcast_concat
      )

    edge_emb =
      edge_features
      |> Axon.dense(hidden_size, name: "edge_proj")
      |> Axon.layer_norm(name: "edge_ln")
      |> Axon.activation(:silu, name: "edge_act")

    # First output block (before any interaction)
    atom_pred =
      output_block(edge_emb, rbf, "output_0",
        hidden_size: hidden_size,
        out_emb_size: out_emb_size,
        num_output_layers: num_output_layers
      )

    # Interaction + Output blocks
    {_edge_emb, atom_pred} =
      if num_blocks > 1 do
        Enum.reduce(1..(num_blocks - 1), {edge_emb, atom_pred}, fn idx, {emb, pred} ->
          new_emb =
            interaction_block(emb, rbf, positions, "block_#{idx}",
              hidden_size: hidden_size,
              int_emb_size: int_emb_size,
              basis_emb_size: basis_emb_size,
              num_spherical: num_spherical,
              cutoff: cutoff
            )

          new_pred_contrib =
            output_block(new_emb, rbf, "output_#{idx}",
              hidden_size: hidden_size,
              out_emb_size: out_emb_size,
              num_output_layers: num_output_layers
            )

          new_pred = Axon.add(pred, new_pred_contrib, name: "pred_add_#{idx}")
          {new_emb, new_pred}
        end)
      else
        {edge_emb, atom_pred}
      end

    # atom_pred: [batch, N, out_emb_size]

    # Optional global pooling
    x =
      if pool do
        MessagePassing.global_pool(atom_pred, pool)
      else
        atom_pred
      end

    # Optional output head
    if num_classes do
      Axon.dense(x, num_classes, name: "output_proj")
    else
      x
    end
  end

  @doc "Get the output size of a DimeNet model."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    num_classes = Keyword.get(opts, :num_classes, nil)
    out_emb_size = Keyword.get(opts, :out_emb_size, @default_out_emb_size)
    if num_classes, do: num_classes, else: out_emb_size
  end

  # ============================================================================
  # Interaction Block (DimeNet++ style)
  # ============================================================================

  defp interaction_block(edge_emb, rbf, positions, name, opts) do
    hidden = opts[:hidden_size]
    int_emb = opts[:int_emb_size]
    basis_emb = opts[:basis_emb_size]
    num_spherical = opts[:num_spherical]
    cutoff = opts[:cutoff]

    # Edge transforms: [batch, N, N, hidden]
    x_ji = Axon.dense(edge_emb, hidden, name: "#{name}_ji")
    x_kj = Axon.dense(edge_emb, hidden, name: "#{name}_kj")

    # RBF transform: [batch, N, N, num_radial] → [batch, N, N, hidden]
    rbf_t =
      rbf
      |> Axon.dense(basis_emb, use_bias: false, name: "#{name}_rbf1")
      |> Axon.dense(hidden, use_bias: false, name: "#{name}_rbf2")

    # Distance-weighted: x_kj * rbf_transform
    x_kj_weighted =
      Axon.layer(
        fn a, b, _opts -> Nx.multiply(a, b) end,
        [x_kj, rbf_t],
        name: "#{name}_dist_weight",
        op_name: :multiply
      )

    # Down-project: [batch, N, N, hidden] → [batch, N, N, int_emb]
    x_kj_down = Axon.dense(x_kj_weighted, int_emb, use_bias: false, name: "#{name}_down")

    # Angular aggregate with SBF
    sbf_w1 =
      Axon.param("#{name}_sbf_w1", {num_spherical, basis_emb}, initializer: :glorot_uniform)

    sbf_w2 =
      Axon.param("#{name}_sbf_w2", {basis_emb, int_emb}, initializer: :glorot_uniform)

    aggregated =
      Axon.layer(
        &angular_aggregate_impl/5,
        [x_kj_down, positions, sbf_w1, sbf_w2],
        name: "#{name}_angular",
        cutoff: cutoff,
        num_spherical: num_spherical,
        op_name: :angular_aggregate
      )

    # Up-project: [batch, N, N, int_emb] → [batch, N, N, hidden]
    x_kj_up = Axon.dense(aggregated, hidden, use_bias: false, name: "#{name}_up")

    # Combine
    h = Axon.add(x_ji, x_kj_up, name: "#{name}_combine")

    # Residual layer before skip
    h = residual_dense(h, hidden, "#{name}_res_before")

    # Skip connection
    h_skip =
      h
      |> Axon.dense(hidden, name: "#{name}_skip_proj")
      |> Axon.layer_norm(name: "#{name}_skip_ln")
      |> Axon.activation(:silu, name: "#{name}_skip_act")

    h = Axon.add(h_skip, edge_emb, name: "#{name}_skip")

    # Residual layers after skip
    h = residual_dense(h, hidden, "#{name}_res_after_0")
    residual_dense(h, hidden, "#{name}_res_after_1")
  end

  # ============================================================================
  # Output Block
  # ============================================================================

  defp output_block(edge_emb, rbf, name, opts) do
    hidden = opts[:hidden_size]
    out_emb = opts[:out_emb_size]
    num_output_layers = opts[:num_output_layers]

    # RBF transform
    rbf_t = Axon.dense(rbf, hidden, use_bias: false, name: "#{name}_rbf")

    # Distance-weighted edge features
    x =
      Axon.layer(
        fn emb, rbf_feat, _opts -> Nx.multiply(emb, rbf_feat) end,
        [edge_emb, rbf_t],
        name: "#{name}_weight",
        op_name: :multiply
      )

    # Aggregate to atoms: sum over source dimension → [batch, N, hidden]
    x =
      Axon.layer(
        fn t, _opts -> Nx.sum(t, axes: [1]) end,
        [x],
        name: "#{name}_agg",
        op_name: :aggregate
      )

    # Up-project
    x =
      x
      |> Axon.dense(out_emb, name: "#{name}_up")
      |> Axon.layer_norm(name: "#{name}_up_ln")
      |> Axon.activation(:silu, name: "#{name}_up_act")

    # Dense output layers
    x =
      Enum.reduce(0..(num_output_layers - 2), x, fn idx, acc ->
        acc
        |> Axon.dense(out_emb, name: "#{name}_out_#{idx}")
        |> Axon.layer_norm(name: "#{name}_out_ln_#{idx}")
        |> Axon.activation(:silu, name: "#{name}_out_act_#{idx}")
      end)

    Axon.dense(x, out_emb, name: "#{name}_out_final")
  end

  # ============================================================================
  # Residual Dense Layer
  # ============================================================================

  defp residual_dense(x, hidden, name) do
    h =
      x
      |> Axon.dense(hidden, name: "#{name}_dense1")
      |> Axon.layer_norm(name: "#{name}_ln1")
      |> Axon.activation(:silu, name: "#{name}_act1")
      |> Axon.dense(hidden, name: "#{name}_dense2")

    Axon.add(x, h, name: "#{name}_res")
  end

  # ============================================================================
  # Runtime Implementations
  # ============================================================================

  # Compute radial Bessel basis with envelope from positions.
  # positions: [batch, N, 3] → [batch, N, N, num_radial]
  defp compute_rbf_impl(positions, opts) do
    cutoff = opts[:cutoff]
    exponent = opts[:exponent]
    num_radial = opts[:num_radial]

    # Pairwise distances
    pos_i = Nx.new_axis(positions, 2)
    pos_j = Nx.new_axis(positions, 1)
    diff = Nx.subtract(pos_i, pos_j)
    dist = Nx.sqrt(Nx.add(Nx.sum(Nx.pow(diff, 2), axes: [3]), 1.0e-8))

    # Envelope
    env = envelope(dist, cutoff, exponent)

    # Bessel basis: sin(freq_n * d/c) / d * envelope
    freqs =
      Nx.multiply(
        Nx.add(Nx.iota({num_radial}, type: Nx.type(dist)), 1),
        :math.pi()
      )

    freqs = Nx.reshape(freqs, {1, 1, 1, num_radial})
    d_scaled = Nx.new_axis(Nx.divide(dist, cutoff), 3)

    rbf = Nx.sin(Nx.multiply(freqs, d_scaled))
    d_safe = Nx.new_axis(Nx.add(dist, 1.0e-8), 3)
    rbf = Nx.divide(rbf, d_safe)

    env_expanded = Nx.new_axis(env, 3)
    Nx.multiply(rbf, env_expanded)
  end

  # Smooth polynomial envelope for cutoff
  defp envelope(dist, cutoff, exponent) do
    p = exponent + 1
    x = Nx.divide(dist, cutoff)

    a = -(p + 1) * (p + 2) / 2
    b = p * (p + 2)
    c = -p * (p + 1) / 2

    inv_x = Nx.divide(1.0, Nx.add(x, 1.0e-8))
    x_pm1 = Nx.pow(x, p - 1)
    x_p = Nx.multiply(x_pm1, x)
    x_pp1 = Nx.multiply(x_p, x)

    u = Nx.add(inv_x, Nx.multiply(a, x_pm1))
    u = Nx.add(u, Nx.multiply(b, x_p))
    u = Nx.add(u, Nx.multiply(c, x_pp1))

    # Zero outside cutoff AND on diagonal (self-interactions)
    n = Nx.axis_size(dist, 1)
    diag_mask = Nx.equal(Nx.iota({n, 1}), Nx.iota({1, n}))
    not_self = Nx.logical_not(diag_mask)
    within = Nx.logical_and(Nx.less(dist, cutoff), not_self)
    Nx.select(within, u, Nx.broadcast(0.0, Nx.shape(u)))
  end

  # Broadcast atom features to edge level and concatenate with RBF.
  # x: [batch, N, hidden], rbf: [batch, N, N, num_radial]
  # → [batch, N, N, 2*hidden + num_radial]
  defp broadcast_edge_features_impl(x, rbf, _opts) do
    batch = Nx.axis_size(x, 0)
    n = Nx.axis_size(x, 1)
    h = Nx.axis_size(x, 2)

    x_j = Nx.new_axis(x, 2) |> Nx.broadcast({batch, n, n, h})
    x_i = Nx.new_axis(x, 1) |> Nx.broadcast({batch, n, n, h})

    Nx.concatenate([x_j, x_i, rbf], axis: 3)
  end

  # Angular aggregate: compute angles, Chebyshev basis, Hadamard product, sum.
  # x_kj_down: [batch, N, N, int_emb], positions: [batch, N, 3]
  # sbf_w1: [num_spherical, basis_emb], sbf_w2: [basis_emb, int_emb]
  # → [batch, N, N, int_emb]
  # credo:disable-for-next-line Credo.Check.Refactor.ABCSize
  defp angular_aggregate_impl(x_kj_down, positions, sbf_w1, sbf_w2, opts) do
    cutoff = opts[:cutoff]
    num_spherical = opts[:num_spherical]

    batch = Nx.axis_size(positions, 0)
    n = Nx.axis_size(positions, 1)
    int_emb = Nx.axis_size(x_kj_down, 3)

    # Pairwise difference vectors: diff[b, a, c, :] = pos[a] - pos[c]
    pos_row = Nx.new_axis(positions, 2)
    pos_col = Nx.new_axis(positions, 1)
    diff = Nx.subtract(pos_row, pos_col)

    # Pairwise distances: [batch, N, N]
    dist = Nx.sqrt(Nx.add(Nx.sum(Nx.pow(diff, 2), axes: [3]), 1.0e-8))

    # Cosine of angles: cos(angle at j between k→j and j→i directions)
    # diff_kj[b, k, j, xyz], diff_ij[b, i, j, xyz]
    # cos_angle[b, k, j, i] = sum_xyz diff[b,k,j,xyz] * diff[b,i,j,xyz] / norms
    #
    # Compute via element-wise broadcasting (Nx.dot batch axes must be successive):
    # diff_for_k: [batch, k, j, 1, 3]
    # diff_for_i: transpose j/i → [batch, j, i, 3] → expand → [batch, 1, j, i, 3]
    diff_for_k = Nx.new_axis(diff, 3)

    diff_for_i =
      diff
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.new_axis(1)

    dot_prod = Nx.sum(Nx.multiply(diff_for_k, diff_for_i), axes: [4])
    # dot_prod: [batch, k, j, i]

    # Distance products for normalization
    dist_kj_exp = Nx.new_axis(dist, 3)

    dist_ij_exp =
      dist
      |> Nx.transpose(axes: [0, 2, 1])
      |> Nx.new_axis(1)

    denom = Nx.add(Nx.multiply(dist_kj_exp, dist_ij_exp), 1.0e-8)

    cos_angle = Nx.clip(Nx.divide(dot_prod, denom), -1.0, 1.0)

    # Chebyshev angular basis: T_l(cos_angle) for l = 0..num_spherical-1
    # T_0 = 1, T_1 = x, T_l = 2*x*T_{l-1} - T_{l-2}
    t0 = Nx.broadcast(Nx.tensor(1.0, type: Nx.type(cos_angle)), Nx.shape(cos_angle))
    t1 = cos_angle

    chebyshev_list =
      if num_spherical <= 1 do
        [t0]
      else
        if num_spherical == 2 do
          [t0, t1]
        else
          Enum.reduce(2..(num_spherical - 1), [t1, t0], fn _l, [tp, tpp | _] = acc ->
            t_new = Nx.subtract(Nx.multiply(2.0, Nx.multiply(cos_angle, tp)), tpp)
            [t_new | acc]
          end)
          |> Enum.reverse()
        end
      end

    # Stack: [batch, k, j, i, num_spherical]
    chebyshev = Nx.stack(chebyshev_list, axis: -1)

    # SBF transform: chebyshev @ sbf_w1 @ sbf_w2 → [batch, k, j, i, int_emb]
    flat = Nx.reshape(chebyshev, {batch * n * n * n, num_spherical})
    sbf_t = flat |> Nx.dot(sbf_w1) |> Nx.dot(sbf_w2)
    sbf_t = Nx.reshape(sbf_t, {batch, n, n, n, int_emb})

    # Hadamard: x_kj_down[b, k, j, :] * sbf_t[b, k, j, i, :]
    x_exp = Nx.new_axis(x_kj_down, 3)
    product = Nx.multiply(x_exp, sbf_t)

    # Cutoff mask on k-j edges: [batch, k, j, 1, 1] → broadcast to [batch, k, j, i, int_emb]
    mask = Nx.less(dist, cutoff)
    mask_f = Nx.new_axis(Nx.new_axis(mask, -1), -1) |> Nx.as_type(Nx.type(product))
    product = Nx.multiply(product, mask_f)

    # Sum over k (axis 1): [batch, j, i, int_emb]
    Nx.sum(product, axes: [1])
  end
end
