defmodule Edifice.Graph.MessagePassingTest do
  use ExUnit.Case, async: true
  @moduletag :graph

  alias Edifice.Graph.MessagePassing

  @batch 2
  @num_nodes 4
  @feature_dim 8

  describe "message_passing_layer/4" do
    setup do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})
      %{nodes: nodes, adj: adj}
    end

    test "builds with sum aggregation", %{nodes: nodes, adj: adj} do
      model = MessagePassing.message_passing_layer(nodes, adj, 16, aggregation: :sum)
      assert %Axon{} = model

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "nodes" => Nx.template({@batch, @num_nodes, @feature_dim}, :f32),
            "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
          },
          Axon.ModelState.empty()
        )

      # Simple adjacency: node 0 connects to node 1, node 1 to node 2, etc.
      adj_matrix = Nx.eye(@num_nodes) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})
      node_feats = Nx.broadcast(0.5, {@batch, @num_nodes, @feature_dim})

      output = predict_fn.(params, %{"nodes" => node_feats, "adjacency" => adj_matrix})
      assert Nx.shape(output) == {@batch, @num_nodes, 16}
    end

    test "builds with mean aggregation", %{nodes: nodes, adj: adj} do
      model = MessagePassing.message_passing_layer(nodes, adj, 16, aggregation: :mean)
      assert %Axon{} = model
    end

    test "builds with max aggregation", %{nodes: nodes, adj: adj} do
      model = MessagePassing.message_passing_layer(nodes, adj, 16, aggregation: :max)
      assert %Axon{} = model
    end

    test "supports dropout", %{nodes: nodes, adj: adj} do
      model = MessagePassing.message_passing_layer(nodes, adj, 16, dropout: 0.2)
      assert %Axon{} = model
    end
  end

  describe "aggregate/3" do
    setup do
      nodes = Axon.input("nodes", shape: {nil, @num_nodes, @feature_dim})
      adj = Axon.input("adjacency", shape: {nil, @num_nodes, @num_nodes})
      %{nodes: nodes, adj: adj}
    end

    for mode <- [:sum, :mean, :max] do
      test "#{mode} aggregation produces correct shape", %{nodes: nodes, adj: adj} do
        model = MessagePassing.aggregate(nodes, adj, unquote(mode))
        assert %Axon{} = model
      end
    end
  end

  describe "global_pool/2" do
    for mode <- [:sum, :mean, :max] do
      test "#{mode} pool reduces to graph-level representation" do
        input = Axon.input("input", shape: {nil, @num_nodes, @feature_dim})
        model = MessagePassing.global_pool(input, unquote(mode))

        {init_fn, predict_fn} = Axon.build(model)

        params =
          init_fn.(Nx.template({@batch, @num_nodes, @feature_dim}, :f32), Axon.ModelState.empty())

        output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @num_nodes, @feature_dim}))
        assert Nx.shape(output) == {@batch, @feature_dim}
      end
    end
  end
end
