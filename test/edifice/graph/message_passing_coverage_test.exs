defmodule Edifice.Graph.MessagePassingCoverageTest do
  @moduledoc """
  Additional coverage tests for Edifice.Graph.MessagePassing.
  Targets forward pass execution of all aggregation modes in
  message_aggregate_impl, aggregate_impl, and global_pool,
  plus edge cases like sparse adjacency, disconnected nodes, and
  weighted adjacency matrices.
  """
  use ExUnit.Case, async: true

  alias Edifice.Graph.MessagePassing

  @batch 2
  @num_nodes 4
  @feature_dim 8
  @output_dim 16

  # Helper to build and run a message passing model
  defp build_and_run_mpnn(aggregation, opts \\ []) do
    dropout = Keyword.get(opts, :dropout, 0.0)
    activation = Keyword.get(opts, :activation, :relu)
    adj_matrix = Keyword.get(opts, :adj_matrix, nil)
    node_feats = Keyword.get(opts, :node_feats, nil)

    nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
    adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

    model =
      MessagePassing.message_passing_layer(nodes, adj, @output_dim,
        aggregation: aggregation,
        dropout: dropout,
        activation: activation,
        name: "mpnn_#{aggregation}_test"
      )

    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
      "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
    }

    params = init_fn.(template, Axon.ModelState.empty())

    adj_data =
      adj_matrix ||
        build_chain_adjacency(@batch, @num_nodes)

    node_data =
      node_feats ||
        Nx.broadcast(0.3, {@batch, @num_nodes, @feature_dim})

    output =
      predict_fn.(params, %{
        "nodes" => node_data,
        "adjacency" => adj_data
      })

    {output, params}
  end

  # Build a chain adjacency: 0->1->2->3
  defp build_chain_adjacency(batch, num_nodes) do
    adj =
      for i <- 0..(num_nodes - 2), into: [] do
        {i, i + 1}
      end

    matrix = Nx.broadcast(0.0, {num_nodes, num_nodes})

    matrix =
      Enum.reduce(adj, matrix, fn {i, j}, acc ->
        Nx.put_slice(acc, [i, j], Nx.tensor([[1.0]]))
      end)

    Nx.broadcast(matrix, {batch, num_nodes, num_nodes})
  end

  # ============================================================================
  # message_passing_layer forward passes with different aggregations
  # ============================================================================
  describe "message_passing_layer forward pass - :sum aggregation" do
    test "produces correct shape with chain adjacency" do
      {output, _params} = build_and_run_mpnn(:sum)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "produces finite values" do
      {output, _params} = build_and_run_mpnn(:sum)
      refute Nx.any(Nx.is_infinity(output)) |> Nx.to_number() == 1
    end
  end

  describe "message_passing_layer forward pass - :mean aggregation" do
    test "produces correct shape with chain adjacency" do
      {output, _params} = build_and_run_mpnn(:mean)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "with identity adjacency (self-loops only)" do
      adj = Nx.eye(@num_nodes) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})
      {output, _params} = build_and_run_mpnn(:mean, adj_matrix: adj)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "with fully connected adjacency" do
      adj = Nx.broadcast(1.0, {@batch, @num_nodes, @num_nodes})
      {output, _params} = build_and_run_mpnn(:mean, adj_matrix: adj)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
    end
  end

  describe "message_passing_layer forward pass - :max aggregation" do
    test "produces correct shape with chain adjacency" do
      {output, _params} = build_and_run_mpnn(:max)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
    end

    test "with dense adjacency" do
      adj = Nx.broadcast(1.0, {@batch, @num_nodes, @num_nodes})
      {output, _params} = build_and_run_mpnn(:max, adj_matrix: adj)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "with sparse adjacency (single edge)" do
      # Only edge 0 -> 1
      adj_single = Nx.broadcast(0.0, {@batch, @num_nodes, @num_nodes})

      adj_single =
        Nx.put_slice(
          adj_single,
          [0, 0, 1],
          Nx.broadcast(1.0, {1, 1, 1})
        )

      adj_single =
        Nx.put_slice(
          adj_single,
          [1, 0, 1],
          Nx.broadcast(1.0, {1, 1, 1})
        )

      {output, _params} = build_and_run_mpnn(:max, adj_matrix: adj_single)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
    end
  end

  # ============================================================================
  # message_passing_layer with dropout
  # ============================================================================
  describe "message_passing_layer with dropout" do
    test "dropout > 0 builds and runs forward pass" do
      {output, _params} = build_and_run_mpnn(:sum, dropout: 0.5)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
    end

    test "dropout = 0 builds without dropout layer" do
      {output, _params} = build_and_run_mpnn(:sum, dropout: 0.0)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
    end
  end

  # ============================================================================
  # message_passing_layer with different activations
  # ============================================================================
  describe "message_passing_layer with different activations" do
    test "sigmoid activation" do
      {output, _params} = build_and_run_mpnn(:sum, activation: :sigmoid)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}

      # Sigmoid outputs are between 0 and 1
      min_val = Nx.reduce_min(output) |> Nx.to_number()
      max_val = Nx.reduce_max(output) |> Nx.to_number()
      assert min_val >= 0.0
      assert max_val <= 1.0
    end

    test "tanh activation" do
      {output, _params} = build_and_run_mpnn(:sum, activation: :tanh)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
    end
  end

  # ============================================================================
  # aggregate/3 forward pass with all modes
  # ============================================================================
  describe "aggregate/3 forward pass" do
    setup do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})
      %{nodes: nodes, adj: adj}
    end

    test ":sum aggregation forward pass", %{nodes: nodes, adj: adj} do
      model = MessagePassing.aggregate(nodes, adj, :sum)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      # Identity adjacency: each node only gets its own features
      adj_matrix = Nx.eye(@num_nodes) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})
      node_feats = Nx.broadcast(1.0, {@batch, @num_nodes, @feature_dim})

      output =
        predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})

      assert Nx.shape(output) == {@batch, @num_nodes, @feature_dim}

      # With identity adjacency and sum, each node gets its own features
      expected_val = 1.0
      actual = Nx.to_number(output[0][0][0])
      assert_in_delta actual, expected_val, 1.0e-5
    end

    test ":mean aggregation forward pass with varying degree", %{nodes: nodes, adj: adj} do
      model = MessagePassing.aggregate(nodes, adj, :mean)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      # Fully connected: each node connects to all (degree = num_nodes)
      adj_matrix = Nx.broadcast(1.0, {@batch, @num_nodes, @num_nodes})
      node_feats = Nx.broadcast(2.0, {@batch, @num_nodes, @feature_dim})

      output =
        predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})

      assert Nx.shape(output) == {@batch, @num_nodes, @feature_dim}

      # Mean of constant 2.0 features with degree=4 should be 2.0
      actual = Nx.to_number(output[0][0][0])
      assert_in_delta actual, 2.0, 1.0e-4
    end

    test ":mean aggregation with disconnected nodes", %{nodes: nodes, adj: adj} do
      model = MessagePassing.aggregate(nodes, adj, :mean)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      # Adjacency with all zeros (disconnected graph)
      adj_matrix = Nx.broadcast(0.0, {@batch, @num_nodes, @num_nodes})
      node_feats = Nx.broadcast(5.0, {@batch, @num_nodes, @feature_dim})

      output =
        predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})

      assert Nx.shape(output) == {@batch, @num_nodes, @feature_dim}
      # Division by max(degree, 1) prevents NaN
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test ":max aggregation forward pass", %{nodes: nodes, adj: adj} do
      model = MessagePassing.aggregate(nodes, adj, :max)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      # Fully connected adjacency
      adj_matrix = Nx.broadcast(1.0, {@batch, @num_nodes, @num_nodes})
      node_feats = Nx.broadcast(0.3, {@batch, @num_nodes, @feature_dim})

      output =
        predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})

      assert Nx.shape(output) == {@batch, @num_nodes, @feature_dim}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test ":max aggregation with sparse adjacency", %{nodes: nodes, adj: adj} do
      model = MessagePassing.aggregate(nodes, adj, :max)
      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      # Only one edge per batch item
      adj_matrix = Nx.broadcast(0.0, {@batch, @num_nodes, @num_nodes})

      adj_matrix =
        Nx.put_slice(adj_matrix, [0, 0, 1], Nx.broadcast(1.0, {1, 1, 1}))

      adj_matrix =
        Nx.put_slice(adj_matrix, [1, 2, 3], Nx.broadcast(1.0, {1, 1, 1}))

      node_feats = Nx.broadcast(0.3, {@batch, @num_nodes, @feature_dim})

      output =
        predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})

      assert Nx.shape(output) == {@batch, @num_nodes, @feature_dim}
    end
  end

  # ============================================================================
  # global_pool/2 forward pass with all modes
  # ============================================================================
  describe "global_pool/2 forward pass" do
    test ":sum pool computes correct sum" do
      input = Axon.input("input", shape: {nil, @num_nodes, @feature_dim})
      model = MessagePassing.global_pool(input, :sum)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
          Axon.ModelState.empty()
        )

      # All 1s: sum over num_nodes should give num_nodes
      data = Nx.broadcast(1.0, {@batch, @num_nodes, @feature_dim})
      output = predict_fn.(params, data)

      assert Nx.shape(output) == {@batch, @feature_dim}
      val = Nx.to_number(output[0][0])
      assert_in_delta val, @num_nodes * 1.0, 1.0e-5
    end

    test ":mean pool computes correct mean" do
      input = Axon.input("input", shape: {nil, @num_nodes, @feature_dim})
      model = MessagePassing.global_pool(input, :mean)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
          Axon.ModelState.empty()
        )

      # All 3.0: mean should be 3.0
      data = Nx.broadcast(3.0, {@batch, @num_nodes, @feature_dim})
      output = predict_fn.(params, data)

      assert Nx.shape(output) == {@batch, @feature_dim}
      val = Nx.to_number(output[0][0])
      assert_in_delta val, 3.0, 1.0e-5
    end

    test ":max pool computes correct max" do
      input = Axon.input("input", shape: {nil, @num_nodes, @feature_dim})
      model = MessagePassing.global_pool(input, :max)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
          Axon.ModelState.empty()
        )

      # Create data where node 2 has the largest values
      data = Nx.broadcast(1.0, {@batch, @num_nodes, @feature_dim})
      big_node = Nx.broadcast(10.0, {1, 1, @feature_dim})
      data = Nx.put_slice(data, [0, 2, 0], big_node)

      output = predict_fn.(params, data)

      assert Nx.shape(output) == {@batch, @feature_dim}
      # For batch 0, max should be 10.0
      val = Nx.to_number(output[0][0])
      assert_in_delta val, 10.0, 1.0e-5
    end

    test "default pool mode is :mean" do
      input = Axon.input("input", shape: {nil, @num_nodes, @feature_dim})
      model = MessagePassing.global_pool(input)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
          Axon.ModelState.empty()
        )

      data = Nx.broadcast(2.0, {@batch, @num_nodes, @feature_dim})
      output = predict_fn.(params, data)

      assert Nx.shape(output) == {@batch, @feature_dim}
      val = Nx.to_number(output[0][0])
      assert_in_delta val, 2.0, 1.0e-5
    end
  end

  # ============================================================================
  # message_passing_layer with weighted adjacency
  # ============================================================================
  describe "message_passing_layer with weighted adjacency" do
    test ":sum aggregation with weighted edges" do
      adj = Nx.broadcast(0.0, {@batch, @num_nodes, @num_nodes})
      # Edge 0->1 with weight 2.0, edge 1->2 with weight 0.5
      adj = Nx.put_slice(adj, [0, 0, 1], Nx.broadcast(2.0, {1, 1, 1}))
      adj = Nx.put_slice(adj, [0, 1, 2], Nx.broadcast(0.5, {1, 1, 1}))
      adj = Nx.put_slice(adj, [1, 0, 1], Nx.broadcast(2.0, {1, 1, 1}))
      adj = Nx.put_slice(adj, [1, 1, 2], Nx.broadcast(0.5, {1, 1, 1}))

      {output, _params} = build_and_run_mpnn(:sum, adj_matrix: adj)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test ":mean aggregation with weighted edges" do
      adj = Nx.broadcast(0.0, {@batch, @num_nodes, @num_nodes})
      adj = Nx.put_slice(adj, [0, 0, 1], Nx.broadcast(3.0, {1, 1, 1}))
      adj = Nx.put_slice(adj, [0, 0, 2], Nx.broadcast(1.0, {1, 1, 1}))
      adj = Nx.put_slice(adj, [1, 0, 1], Nx.broadcast(3.0, {1, 1, 1}))
      adj = Nx.put_slice(adj, [1, 0, 2], Nx.broadcast(1.0, {1, 1, 1}))

      {output, _params} = build_and_run_mpnn(:mean, adj_matrix: adj)
      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # Multiple message passing layers stacked
  # ============================================================================
  describe "stacked message passing layers" do
    test "two layers of message passing" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      layer1 =
        MessagePassing.message_passing_layer(nodes, adj, @output_dim,
          aggregation: :sum,
          name: "mpnn_layer1"
        )

      model =
        MessagePassing.message_passing_layer(layer1, adj, @output_dim,
          aggregation: :mean,
          name: "mpnn_layer2"
        )

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj_matrix = Nx.eye(@num_nodes) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})
      node_feats = Nx.broadcast(0.3, {@batch, @num_nodes, @feature_dim})

      output =
        predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})

      assert Nx.shape(output) == {@batch, @num_nodes, @output_dim}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end

  # ============================================================================
  # message_passing_layer + global_pool pipeline
  # ============================================================================
  describe "message passing + global pool pipeline" do
    test "graph classification pipeline with sum pool" do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})

      node_updated =
        MessagePassing.message_passing_layer(nodes, adj, @output_dim,
          aggregation: :sum,
          name: "mpnn_clf"
        )

      model = MessagePassing.global_pool(node_updated, :sum)

      {init_fn, predict_fn} = Axon.build(model)

      template = %{
        "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      params = init_fn.(template, Axon.ModelState.empty())

      adj_matrix = build_chain_adjacency(@batch, @num_nodes)
      node_feats = Nx.broadcast(0.3, {@batch, @num_nodes, @feature_dim})

      output =
        predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})

      # After global pool, shape is {batch, output_dim}
      assert Nx.shape(output) == {@batch, @output_dim}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end
end
