defmodule Edifice.Graph.GINv2Test do
  use ExUnit.Case, async: true
  @moduletag :graph

  alias Edifice.Graph.GINv2

  @batch_size 2
  @num_nodes 4
  @input_dim 8
  @edge_dim 4
  @hidden_dim 16

  defp build_inputs do
    nodes = Nx.broadcast(0.5, {@batch_size, @num_nodes, @input_dim})

    adjacency =
      Nx.tensor([
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 1, 1]
      ])
      |> Nx.broadcast({@batch_size, @num_nodes, @num_nodes})

    edge_features =
      Nx.broadcast(0.1, {@batch_size, @num_nodes, @num_nodes, @edge_dim})

    {nodes, adjacency, edge_features}
  end

  defp build_and_predict(model) do
    {init_fn, predict_fn} = Axon.build(model)
    {nodes, adjacency, edge_features} = build_inputs()

    params =
      init_fn.(
        %{
          "nodes" => Nx.template({@batch_size, @num_nodes, @input_dim}, :f32),
          "adjacency" => Nx.template({@batch_size, @num_nodes, @num_nodes}, :f32),
          "edge_features" => Nx.template({@batch_size, @num_nodes, @num_nodes, @edge_dim}, :f32)
        },
        Axon.ModelState.empty()
      )

    predict_fn.(params, %{
      "nodes" => nodes,
      "adjacency" => adjacency,
      "edge_features" => edge_features
    })
  end

  describe "build/1" do
    test "produces node embeddings with correct shape" do
      model =
        GINv2.build(input_dim: @input_dim, edge_dim: @edge_dim, hidden_dims: [@hidden_dim])

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, @hidden_dim}
    end

    test "with num_classes adds classification head" do
      model =
        GINv2.build(
          input_dim: @input_dim,
          edge_dim: @edge_dim,
          hidden_dims: [@hidden_dim],
          num_classes: 3
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, 3}
    end

    test "with global mean pooling" do
      model =
        GINv2.build(
          input_dim: @input_dim,
          edge_dim: @edge_dim,
          hidden_dims: [@hidden_dim],
          pool: :mean
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @hidden_dim}
    end

    test "with global sum pooling and num_classes" do
      model =
        GINv2.build(
          input_dim: @input_dim,
          edge_dim: @edge_dim,
          hidden_dims: [@hidden_dim],
          pool: :sum,
          num_classes: 5
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, 5}
    end

    test "with multiple layers" do
      model =
        GINv2.build(
          input_dim: @input_dim,
          edge_dim: @edge_dim,
          hidden_dims: [@hidden_dim, 32]
        )

      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, 32}
    end

    test "output contains finite values" do
      model =
        GINv2.build(input_dim: @input_dim, edge_dim: @edge_dim, hidden_dims: [@hidden_dim])

      output = build_and_predict(model)
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "default hidden_dims [64, 64]" do
      model = GINv2.build(input_dim: @input_dim, edge_dim: @edge_dim)
      output = build_and_predict(model)
      assert Nx.shape(output) == {@batch_size, @num_nodes, 64}
    end
  end

  describe "output_size/1" do
    test "returns last hidden dim" do
      assert GINv2.output_size(hidden_dims: [32, 16]) == 16
    end

    test "returns num_classes when set" do
      assert GINv2.output_size(hidden_dims: [32], num_classes: 7) == 7
    end
  end
end
