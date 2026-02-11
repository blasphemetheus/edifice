defmodule Edifice.Graph.GraphNewTest do
  use ExUnit.Case, async: true

  alias Edifice.Graph.GraphSAGE
  alias Edifice.Graph.GIN
  alias Edifice.Graph.PNA
  alias Edifice.Graph.GraphTransformer
  alias Edifice.Graph.SchNet

  @batch_size 2
  @num_nodes 4
  @input_dim 8
  @hidden_dim 16

  defp build_graph_inputs(batch_size \\ @batch_size, num_nodes \\ @num_nodes) do
    nodes = Nx.broadcast(0.5, {batch_size, num_nodes, @input_dim})

    # Simple ring graph with self-loops
    adjacency =
      Nx.tensor([
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 1, 1]
      ])
      |> Nx.broadcast({batch_size, num_nodes, num_nodes})

    {nodes, adjacency}
  end

  defp build_and_predict(model) do
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

    predict_fn.(params, %{"nodes" => nodes, "adjacency" => adjacency})
  end

  # ============================================================================
  # GraphSAGE Tests
  # ============================================================================

  describe "GraphSAGE.build/1" do
    test "produces node embeddings with correct shape" do
      model = GraphSAGE.build(input_dim: @input_dim, hidden_dims: [@hidden_dim])
      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "with num_classes adds classification head" do
      model = GraphSAGE.build(input_dim: @input_dim, hidden_dims: [@hidden_dim], num_classes: 3)
      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, 3}
    end

    test "with mean aggregator" do
      model =
        GraphSAGE.build(
          input_dim: @input_dim,
          hidden_dims: [@hidden_dim],
          aggregator: :mean
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "with max aggregator" do
      model =
        GraphSAGE.build(
          input_dim: @input_dim,
          hidden_dims: [@hidden_dim],
          aggregator: :max
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "with pool aggregator" do
      model =
        GraphSAGE.build(
          input_dim: @input_dim,
          hidden_dims: [@hidden_dim],
          aggregator: :pool
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "default hidden_dims [64, 64]" do
      model = GraphSAGE.build(input_dim: @input_dim)
      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, 64}
    end

    test "output_size/1 returns correct value" do
      assert GraphSAGE.output_size(hidden_dims: [32, 16]) == 16
      assert GraphSAGE.output_size(hidden_dims: [32], num_classes: 5) == 5
    end
  end

  # ============================================================================
  # GIN Tests
  # ============================================================================

  describe "GIN.build/1" do
    test "produces node embeddings with correct shape" do
      model = GIN.build(input_dim: @input_dim, hidden_dims: [@hidden_dim])
      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "with num_classes adds classification head" do
      model = GIN.build(input_dim: @input_dim, hidden_dims: [@hidden_dim], num_classes: 4)
      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, 4}
    end

    test "with learnable epsilon" do
      model =
        GIN.build(
          input_dim: @input_dim,
          hidden_dims: [@hidden_dim],
          epsilon_learnable: true
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "with fixed epsilon" do
      model =
        GIN.build(
          input_dim: @input_dim,
          hidden_dims: [@hidden_dim],
          epsilon_learnable: false
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "default hidden_dims [64, 64]" do
      model = GIN.build(input_dim: @input_dim)
      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, 64}
    end

    test "output_size/1 returns correct value" do
      assert GIN.output_size(hidden_dims: [32, 16]) == 16
      assert GIN.output_size(hidden_dims: [32], num_classes: 7) == 7
    end
  end

  # ============================================================================
  # PNA Tests
  # ============================================================================

  describe "PNA.build/1" do
    test "produces node embeddings with correct shape" do
      model = PNA.build(input_dim: @input_dim, hidden_dims: [@hidden_dim])
      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "with num_classes adds classification head" do
      model = PNA.build(input_dim: @input_dim, hidden_dims: [@hidden_dim], num_classes: 5)
      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, 5}
    end

    test "with custom aggregators and scalers" do
      model =
        PNA.build(
          input_dim: @input_dim,
          hidden_dims: [@hidden_dim],
          aggregators: [:mean, :max],
          scalers: [:identity]
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "with all aggregators and scalers" do
      model =
        PNA.build(
          input_dim: @input_dim,
          hidden_dims: [@hidden_dim],
          aggregators: [:mean, :max, :sum, :std],
          scalers: [:identity, :amplification, :attenuation]
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "default hidden_dims [64, 64]" do
      model = PNA.build(input_dim: @input_dim)
      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, 64}
    end

    test "output_size/1 returns correct value" do
      assert PNA.output_size(hidden_dims: [32, 16]) == 16
      assert PNA.output_size(hidden_dims: [32], num_classes: 3) == 3
    end
  end

  # ============================================================================
  # GraphTransformer Tests
  # ============================================================================

  describe "GraphTransformer.build/1" do
    test "produces node embeddings with correct shape" do
      model =
        GraphTransformer.build(
          input_dim: @input_dim,
          hidden_dim: @hidden_dim,
          num_heads: 4,
          num_layers: 2
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "with num_classes adds classification head" do
      model =
        GraphTransformer.build(
          input_dim: @input_dim,
          hidden_dim: @hidden_dim,
          num_heads: 4,
          num_layers: 1,
          num_classes: 6
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, 6}
    end

    test "defaults produce correct shape" do
      model = GraphTransformer.build(input_dim: @input_dim)
      output = build_and_predict(model)
      # Default hidden_dim=64, num_heads=4, num_layers=4
      assert Nx.shape(output) == {@batch_size, @num_nodes, 64}
    end

    test "output_size/1 returns correct value" do
      assert GraphTransformer.output_size(hidden_dim: 32) == 32
      assert GraphTransformer.output_size(hidden_dim: 32, num_classes: 5) == 5
    end
  end

  # ============================================================================
  # SchNet Tests
  # ============================================================================

  describe "SchNet.build/1" do
    test "produces node embeddings with correct shape" do
      model =
        SchNet.build(
          input_dim: @input_dim,
          hidden_dim: @hidden_dim,
          num_interactions: 2,
          num_filters: @hidden_dim
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "with num_classes adds output projection" do
      model =
        SchNet.build(
          input_dim: @input_dim,
          hidden_dim: @hidden_dim,
          num_interactions: 1,
          num_filters: @hidden_dim,
          num_classes: 3
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, 3}
    end

    test "with distance-based adjacency (non-binary)" do
      model =
        SchNet.build(
          input_dim: @input_dim,
          hidden_dim: @hidden_dim,
          num_interactions: 1,
          num_filters: @hidden_dim,
          cutoff: 3.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      nodes = Nx.broadcast(0.5, {@batch_size, @num_nodes, @input_dim})

      # Distance matrix (continuous values, not binary)
      adjacency =
        Nx.tensor([
          [0.0, 1.5, 2.8, 4.0],
          [1.5, 0.0, 1.2, 2.5],
          [2.8, 1.2, 0.0, 1.8],
          [4.0, 2.5, 1.8, 0.0]
        ])
        |> Nx.broadcast({@batch_size, @num_nodes, @num_nodes})

      params =
        init_fn.(
          %{
            "nodes" => Nx.template({@batch_size, @num_nodes, @input_dim}, :f32),
            "adjacency" => Nx.template({@batch_size, @num_nodes, @num_nodes}, :f32)
          },
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"nodes" => nodes, "adjacency" => adjacency})
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "defaults produce correct shape" do
      model = SchNet.build(input_dim: @input_dim)
      output = build_and_predict(model)
      # Default hidden_dim=64
      assert Nx.shape(output) == {@batch_size, @num_nodes, 64}
    end

    test "output_size/1 returns correct value" do
      assert SchNet.output_size(hidden_dim: 32) == 32
      assert SchNet.output_size(hidden_dim: 32, num_classes: 1) == 1
    end
  end
end
