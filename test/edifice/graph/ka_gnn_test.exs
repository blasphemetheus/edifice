defmodule Edifice.Graph.KAGNNTest do
  use ExUnit.Case, async: true

  alias Edifice.Graph.KAGNN

  import Edifice.TestHelpers

  @moduletag :graph

  @batch 2
  @num_nodes 6
  @input_dim 8
  @hidden_dim 16
  @num_classes 1

  describe "build/1" do
    test "builds model with default options" do
      model = KAGNN.build(input_dim: @input_dim)
      assert %Axon{} = model
    end

    test "forward pass produces correct shape" do
      model =
        KAGNN.build(
          input_dim: @input_dim,
          hidden_dim: @hidden_dim,
          num_layers: 2,
          num_harmonics: 2,
          num_classes: @num_classes,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      nodes = random_tensor({@batch, @num_nodes, @input_dim}, 42)
      adj = Nx.broadcast(Nx.tensor(0.0), {@batch, @num_nodes, @num_nodes})
      # Simple ring adjacency
      adj = Nx.put_slice(adj, [0, 0, 1], Nx.broadcast(1.0, {1, 1, 1}))
      adj = Nx.put_slice(adj, [0, 1, 0], Nx.broadcast(1.0, {1, 1, 1}))

      input = %{"nodes" => nodes, "adjacency" => adj}
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch, @num_classes}
    end

    test "batch=1 forward pass" do
      model =
        KAGNN.build(
          input_dim: @input_dim,
          hidden_dim: @hidden_dim,
          num_layers: 1,
          num_harmonics: 2,
          num_classes: 2,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      nodes = random_tensor({1, 4, @input_dim}, 99)
      adj = Nx.eye(4) |> Nx.new_axis(0) |> Nx.broadcast({1, 4, 4})

      input = %{"nodes" => nodes, "adjacency" => adj}
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {1, 2}
    end

    test "output values are finite" do
      model =
        KAGNN.build(
          input_dim: @input_dim,
          hidden_dim: @hidden_dim,
          num_layers: 2,
          num_harmonics: 2,
          num_classes: @num_classes,
          dropout: 0.0
        )

      {init_fn, predict_fn} = Axon.build(model)

      nodes = random_tensor({@batch, @num_nodes, @input_dim}, 77)
      adj = Nx.eye(@num_nodes) |> Nx.new_axis(0) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})

      input = %{"nodes" => nodes, "adjacency" => adj}
      params = init_fn.(input, %{})
      output = predict_fn.(params, input)

      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns num_classes" do
      assert KAGNN.output_size(num_classes: 5) == 5
    end

    test "returns default" do
      assert KAGNN.output_size() == 1
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list" do
      defaults = KAGNN.recommended_defaults()
      assert Keyword.keyword?(defaults)
      assert defaults[:hidden_dim] == 64
      assert defaults[:num_harmonics] == 4
    end
  end

  describe "registry integration" do
    test "builds via Edifice.build/2" do
      model =
        Edifice.build(:ka_gnn,
          input_dim: 8,
          hidden_dim: 16,
          num_layers: 1,
          num_harmonics: 2,
          dropout: 0.0
        )

      assert %Axon{} = model
    end
  end
end
