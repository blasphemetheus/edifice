defmodule Edifice.Graph.GCNTest do
  use ExUnit.Case, async: true

  alias Edifice.Graph.GCN

  @batch_size 2
  @num_nodes 5
  @input_dim 8

  defp build_graph_inputs(batch_size \\ @batch_size, num_nodes \\ @num_nodes) do
    nodes = Nx.broadcast(0.5, {batch_size, num_nodes, @input_dim})

    # Create a simple adjacency matrix (ring graph + self-loops)
    adjacency =
      Nx.tensor([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
      ])
      |> Nx.broadcast({batch_size, num_nodes, num_nodes})

    {nodes, adjacency}
  end

  describe "build/1" do
    test "produces node embeddings with correct shape" do
      model = GCN.build(input_dim: @input_dim, hidden_dims: [16, 8])

      {init_fn, predict_fn} = Axon.build(model)
      {nodes, adjacency} = build_graph_inputs()

      params =
        init_fn.(
          %{
            "nodes" => Nx.template({@batch_size, @num_nodes, @input_dim}, :f32),
            "adjacency" => Nx.template({@batch_size, @num_nodes, @num_nodes}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{"nodes" => nodes, "adjacency" => adjacency})

      assert Nx.shape(output) == {@batch_size, @num_nodes, 8}
    end

    test "with num_classes adds classification head" do
      model =
        GCN.build(input_dim: @input_dim, hidden_dims: [16], num_classes: 3)

      {init_fn, predict_fn} = Axon.build(model)
      {nodes, adjacency} = build_graph_inputs()

      params =
        init_fn.(
          %{
            "nodes" => Nx.template({@batch_size, @num_nodes, @input_dim}, :f32),
            "adjacency" => Nx.template({@batch_size, @num_nodes, @num_nodes}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{"nodes" => nodes, "adjacency" => adjacency})

      assert Nx.shape(output) == {@batch_size, @num_nodes, 3}
    end

    test "respects default hidden_dims of [64, 64]" do
      model = GCN.build(input_dim: @input_dim)

      {init_fn, predict_fn} = Axon.build(model)
      {nodes, adjacency} = build_graph_inputs()

      params =
        init_fn.(
          %{
            "nodes" => Nx.template({@batch_size, @num_nodes, @input_dim}, :f32),
            "adjacency" => Nx.template({@batch_size, @num_nodes, @num_nodes}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{"nodes" => nodes, "adjacency" => adjacency})

      # Default hidden_dims is [64, 64], so last dim is 64
      assert Nx.shape(output) == {@batch_size, @num_nodes, 64}
    end
  end

  describe "build_classifier/1" do
    test "produces graph-level classification output" do
      model =
        GCN.build_classifier(
          input_dim: @input_dim,
          hidden_dims: [16],
          num_classes: 4,
          classifier_dims: [32]
        )

      {init_fn, predict_fn} = Axon.build(model)
      {nodes, adjacency} = build_graph_inputs()

      params =
        init_fn.(
          %{
            "nodes" => Nx.template({@batch_size, @num_nodes, @input_dim}, :f32),
            "adjacency" => Nx.template({@batch_size, @num_nodes, @num_nodes}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{"nodes" => nodes, "adjacency" => adjacency})

      # Graph classification: pooled to [batch, num_classes]
      assert Nx.shape(output) == {@batch_size, 4}
    end
  end

  describe "gcn_layer/4" do
    test "transforms node features through propagation" do
      nodes_input = Axon.input("nodes", shape: {nil, nil, @input_dim})
      adj_input = Axon.input("adjacency", shape: {nil, nil, nil})

      output = GCN.gcn_layer(nodes_input, adj_input, 16, name: "test_gcn")

      {init_fn, predict_fn} = Axon.build(output)
      {nodes, adjacency} = build_graph_inputs()

      params =
        init_fn.(
          %{
            "nodes" => Nx.template({@batch_size, @num_nodes, @input_dim}, :f32),
            "adjacency" => Nx.template({@batch_size, @num_nodes, @num_nodes}, :f32)
          },
          Axon.ModelState.empty()
        )

      result =
        predict_fn.(params, %{"nodes" => nodes, "adjacency" => adjacency})

      assert Nx.shape(result) == {@batch_size, @num_nodes, 16}
    end
  end
end
