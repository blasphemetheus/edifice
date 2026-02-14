defmodule Edifice.Graph.CorrectnessTest do
  @moduledoc """
  Correctness tests for graph neural network architectures.
  Verifies mathematical properties: permutation equivariance,
  adjacency handling, and degree-zero robustness.
  """
  use ExUnit.Case, async: true

  import Edifice.TestHelpers

  @batch 2
  @num_nodes 4
  @node_dim 8
  @hidden 8

  # ── Permutation Equivariance ──────────────────────────────────
  # GCN(Px, PAP^T) = P * GCN(x, A) — permuting inputs permutes outputs

  for arch <- [:gcn, :gat, :graph_sage, :gin] do
    @tag timeout: 120_000
    test "#{arch} is permutation equivariant" do
      model =
        Edifice.build(unquote(arch),
          input_dim: @node_dim,
          hidden_size: @hidden,
          num_classes: @hidden,
          num_layers: 1,
          num_heads: 2,
          dropout: 0.0
        )

      nodes = random_tensor({@batch, @num_nodes, @node_dim})
      adj = random_tensor({@batch, @num_nodes, @num_nodes})
      # Symmetrize adjacency
      adj = Nx.add(adj, Nx.transpose(adj, axes: [0, 2, 1]))

      # Forward pass with original ordering
      input_map = %{"nodes" => nodes, "adjacency" => adj}
      {predict_fn, params} = build_and_init(model, input_map)
      out_orig = predict_fn.(params, input_map)

      # Permutation: swap nodes 0 and 1
      perm = [1, 0, 2, 3]
      nodes_perm = Nx.take(nodes, Nx.tensor(perm), axis: 1)
      adj_perm = Nx.take(Nx.take(adj, Nx.tensor(perm), axis: 1), Nx.tensor(perm), axis: 2)

      perm_map = %{"nodes" => nodes_perm, "adjacency" => adj_perm}
      out_perm = predict_fn.(params, perm_map)

      # Permute original output to match
      out_orig_perm = Nx.take(out_orig, Nx.tensor(perm), axis: 1)

      # Should be close (within floating point tolerance)
      diff = Nx.mean(Nx.abs(Nx.subtract(out_perm, out_orig_perm))) |> Nx.to_number()
      assert diff < 0.01, "#{unquote(arch)} not equivariant: mean diff = #{diff}"
    end
  end

  # ── Zero Adjacency (Isolated Nodes) ──────────────────────────
  # With zero adjacency, each node should be processed independently

  @tag timeout: 120_000
  test "gcn handles isolated nodes (zero adjacency)" do
    model =
      Edifice.build(:gcn,
        input_dim: @node_dim,
        hidden_size: @hidden,
        num_classes: @hidden,
        num_layers: 1,
        dropout: 0.0
      )

    nodes = random_tensor({@batch, @num_nodes, @node_dim})
    zero_adj = Nx.broadcast(Nx.tensor(0.0), {@batch, @num_nodes, @num_nodes})

    input_map = %{"nodes" => nodes, "adjacency" => zero_adj}
    {predict_fn, params} = build_and_init(model, input_map)
    output = predict_fn.(params, input_map)

    assert_finite!(output, "gcn isolated nodes")
    assert {@batch, @num_nodes, _} = Nx.shape(output)
  end

  # ── Self-Loop Adjacency ──────────────────────────────────────
  # Identity adjacency (self-loops only) should give valid output

  @tag timeout: 120_000
  test "gcn handles identity adjacency (self-loops)" do
    model =
      Edifice.build(:gcn,
        input_dim: @node_dim,
        hidden_size: @hidden,
        num_classes: @hidden,
        num_layers: 1,
        dropout: 0.0
      )

    nodes = random_tensor({@batch, @num_nodes, @node_dim})
    identity = Nx.broadcast(Nx.eye(@num_nodes), {@batch, @num_nodes, @num_nodes})

    input_map = %{"nodes" => nodes, "adjacency" => identity}
    {predict_fn, params} = build_and_init(model, input_map)
    output = predict_fn.(params, input_map)

    assert_finite!(output, "gcn self-loops")
    assert {@batch, @num_nodes, _} = Nx.shape(output)
  end

  # ── SchNet Distance Invariance ──────────────────────────────
  # SchNet with uniform distances should give valid output

  @tag timeout: 120_000
  test "schnet handles uniform distances" do
    model =
      Edifice.build(:schnet,
        input_dim: @node_dim,
        hidden_size: @hidden,
        num_interactions: 1,
        num_filters: @hidden,
        num_rbf: 8
      )

    nodes = random_tensor({@batch, @num_nodes, @node_dim})
    uniform_dist = Nx.broadcast(Nx.tensor(1.0), {@batch, @num_nodes, @num_nodes})

    input_map = %{"nodes" => nodes, "adjacency" => uniform_dist}
    {predict_fn, params} = build_and_init(model, input_map)
    output = predict_fn.(params, input_map)

    assert_finite!(output, "schnet uniform distances")
  end
end
