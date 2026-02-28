defmodule Edifice.Graph.GPSTest do
  use ExUnit.Case, async: true
  @moduletag :graph

  import Edifice.TestHelpers

  alias Edifice.Graph.GPS

  @moduletag timeout: 120_000

  @batch 2
  @num_nodes 6
  @input_dim 8
  @hidden_size 16

  @small_opts [
    input_dim: @input_dim,
    hidden_size: @hidden_size,
    num_heads: 4,
    num_layers: 2,
    pe_dim: 4,
    rwse_walk_length: 4
  ]

  defp random_inputs(batch \\ @batch) do
    key = Nx.Random.key(42)
    {nodes, key} = Nx.Random.normal(key, shape: {batch, @num_nodes, @input_dim})

    # Binary adjacency (symmetric)
    adj =
      Nx.tensor([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
      ])
      |> Nx.broadcast({batch, @num_nodes, @num_nodes})
      |> Nx.as_type(:f32)

    {_key, _} = {key, nil}
    %{"nodes" => nodes, "adjacency" => adj}
  end

  defp build_and_predict(opts, inputs \\ nil) do
    inputs = inputs || random_inputs()
    model = GPS.build(opts)
    {init_fn, predict_fn} = Axon.build(model)

    template = %{
      "nodes" => Nx.template(Nx.shape(inputs["nodes"]), :f32),
      "adjacency" => Nx.template(Nx.shape(inputs["adjacency"]), :f32)
    }

    params = init_fn.(template, %{})
    output = predict_fn.(params, inputs)
    {output, params}
  end

  describe "build/1" do
    test "produces correct output shape" do
      {output, _} = build_and_predict(@small_opts)
      assert Nx.shape(output) == {@batch, @num_nodes, @hidden_size}
    end

    test "output values are finite" do
      {output, _} = build_and_predict(@small_opts)
      assert_finite!(output)
    end

    test "works with batch_size=1" do
      inputs = random_inputs(1)
      {output, _} = build_and_predict(@small_opts, inputs)
      assert Nx.shape(output) == {1, @num_nodes, @hidden_size}
      assert_finite!(output)
    end

    test "with mean pooling" do
      opts = Keyword.put(@small_opts, :pool, :mean)
      {output, _} = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "with sum pooling" do
      opts = Keyword.put(@small_opts, :pool, :sum)
      {output, _} = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, @hidden_size}
      assert_finite!(output)
    end

    test "with classification head" do
      opts = Keyword.merge(@small_opts, pool: :mean, num_classes: 5)
      {output, _} = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, 5}
      assert_finite!(output)
    end

    test "single layer" do
      opts = Keyword.put(@small_opts, :num_layers, 1)
      {output, _} = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, @num_nodes, @hidden_size}
      assert_finite!(output)
    end

    test "with dropout" do
      opts = Keyword.put(@small_opts, :dropout, 0.1)
      {output, _} = build_and_predict(opts)
      assert Nx.shape(output) == {@batch, @num_nodes, @hidden_size}
      assert_finite!(output)
    end
  end

  describe "output_size/1" do
    test "returns hidden_size by default" do
      assert GPS.output_size(hidden_size: 32) == 32
    end

    test "returns num_classes when specified" do
      assert GPS.output_size(hidden_size: 32, num_classes: 7) == 7
    end

    test "returns default hidden_size with no opts" do
      assert GPS.output_size([]) == 64
    end
  end
end
