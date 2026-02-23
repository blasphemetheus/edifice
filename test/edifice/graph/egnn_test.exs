defmodule Edifice.Graph.EGNNTest do
  use ExUnit.Case, async: true

  alias Edifice.Graph.EGNN

  @batch_size 2
  @num_nodes 5
  @num_edges 8
  @in_node_features 16
  @hidden_dim 32
  @coord_dim 3

  describe "EGNN.build/1" do
    test "produces correct output shape" do
      model =
        EGNN.build(
          in_node_features: @in_node_features,
          hidden_dim: @hidden_dim,
          num_layers: 2,
          out_features: @hidden_dim
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      templates = %{
        "nodes" => Nx.template({@batch_size, @num_nodes, @in_node_features}, :f32),
        "coords" => Nx.template({@batch_size, @num_nodes, @coord_dim}, :f32),
        "edge_index" => Nx.template({@batch_size, @num_edges, 2}, :s32)
      }

      params = init_fn.(templates, Axon.ModelState.empty())

      # Create inputs with valid edge indices
      nodes = Nx.broadcast(0.5, {@batch_size, @num_nodes, @in_node_features})
      coords = Nx.broadcast(1.0, {@batch_size, @num_nodes, @coord_dim})
      # Create simple edge indices (all pointing to valid nodes)
      edge_index = create_edge_index(@batch_size, @num_edges, @num_nodes)

      output =
        predict_fn.(params, %{
          "nodes" => nodes,
          "coords" => coords,
          "edge_index" => edge_index
        })

      assert Nx.shape(output.node_features) == {@batch_size, @num_nodes, @hidden_dim}
      assert Nx.shape(output.coords) == {@batch_size, @num_nodes, @coord_dim}
    end

    test "with edge features" do
      in_edge_features = 4

      model =
        EGNN.build(
          in_node_features: @in_node_features,
          in_edge_features: in_edge_features,
          hidden_dim: @hidden_dim,
          num_layers: 2
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      templates = %{
        "nodes" => Nx.template({@batch_size, @num_nodes, @in_node_features}, :f32),
        "coords" => Nx.template({@batch_size, @num_nodes, @coord_dim}, :f32),
        "edge_index" => Nx.template({@batch_size, @num_edges, 2}, :s32),
        "edge_features" => Nx.template({@batch_size, @num_edges, in_edge_features}, :f32)
      }

      params = init_fn.(templates, Axon.ModelState.empty())

      nodes = Nx.broadcast(0.5, {@batch_size, @num_nodes, @in_node_features})
      coords = Nx.broadcast(1.0, {@batch_size, @num_nodes, @coord_dim})
      edge_index = create_edge_index(@batch_size, @num_edges, @num_nodes)
      edge_features = Nx.broadcast(0.1, {@batch_size, @num_edges, in_edge_features})

      output =
        predict_fn.(params, %{
          "nodes" => nodes,
          "coords" => coords,
          "edge_index" => edge_index,
          "edge_features" => edge_features
        })

      assert Nx.shape(output.node_features) == {@batch_size, @num_nodes, @hidden_dim}
      assert Nx.shape(output.coords) == {@batch_size, @num_nodes, @coord_dim}
    end

    test "with update_coords: false" do
      model =
        EGNN.build(
          in_node_features: @in_node_features,
          hidden_dim: @hidden_dim,
          num_layers: 2,
          update_coords: false
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      templates = %{
        "nodes" => Nx.template({@batch_size, @num_nodes, @in_node_features}, :f32),
        "coords" => Nx.template({@batch_size, @num_nodes, @coord_dim}, :f32),
        "edge_index" => Nx.template({@batch_size, @num_edges, 2}, :s32)
      }

      params = init_fn.(templates, Axon.ModelState.empty())

      nodes = Nx.broadcast(0.5, {@batch_size, @num_nodes, @in_node_features})
      coords = Nx.broadcast(1.0, {@batch_size, @num_nodes, @coord_dim})
      edge_index = create_edge_index(@batch_size, @num_edges, @num_nodes)

      output =
        predict_fn.(params, %{
          "nodes" => nodes,
          "coords" => coords,
          "edge_index" => edge_index
        })

      # Coords should be unchanged
      assert Nx.shape(output.coords) == {@batch_size, @num_nodes, @coord_dim}
    end

    test "with different output features" do
      out_features = 64

      model =
        EGNN.build(
          in_node_features: @in_node_features,
          hidden_dim: @hidden_dim,
          num_layers: 2,
          out_features: out_features
        )

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      templates = %{
        "nodes" => Nx.template({@batch_size, @num_nodes, @in_node_features}, :f32),
        "coords" => Nx.template({@batch_size, @num_nodes, @coord_dim}, :f32),
        "edge_index" => Nx.template({@batch_size, @num_edges, 2}, :s32)
      }

      params = init_fn.(templates, Axon.ModelState.empty())

      nodes = Nx.broadcast(0.5, {@batch_size, @num_nodes, @in_node_features})
      coords = Nx.broadcast(1.0, {@batch_size, @num_nodes, @coord_dim})
      edge_index = create_edge_index(@batch_size, @num_edges, @num_nodes)

      output =
        predict_fn.(params, %{
          "nodes" => nodes,
          "coords" => coords,
          "edge_index" => edge_index
        })

      assert Nx.shape(output.node_features) == {@batch_size, @num_nodes, out_features}
    end
  end

  describe "EGNN.output_size/1" do
    test "returns out_features when specified" do
      assert EGNN.output_size(out_features: 128) == 128
    end

    test "returns hidden_dim when out_features not specified" do
      assert EGNN.output_size(hidden_dim: 64) == 64
    end

    test "returns default when nothing specified" do
      assert EGNN.output_size() == 64
    end
  end

  # Helper to create valid edge indices
  defp create_edge_index(batch_size, num_edges, num_nodes) do
    # Create random but valid edge indices
    src =
      Nx.iota({batch_size, num_edges, 1})
      |> Nx.remainder(num_nodes)
      |> Nx.as_type(:s32)

    tgt =
      Nx.add(Nx.iota({batch_size, num_edges, 1}), 1)
      |> Nx.remainder(num_nodes)
      |> Nx.as_type(:s32)

    Nx.concatenate([src, tgt], axis: 2)
  end
end
