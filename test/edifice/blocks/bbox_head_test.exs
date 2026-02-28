defmodule Edifice.Blocks.BBoxHeadTest do
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.Blocks.BBoxHead

  @batch 2
  @num_queries 10
  @hidden_dim 32

  describe "layer/3" do
    test "produces 4-element bbox output" do
      input = Axon.input("input", shape: {nil, @num_queries, @hidden_dim})
      model = BBoxHead.layer(input, @hidden_dim, "bbox")

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @num_queries, @hidden_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, @num_queries, @hidden_dim}))

      assert Nx.shape(output) == {@batch, @num_queries, 4}
    end

    test "output values in [0, 1] due to sigmoid" do
      input = Axon.input("input", shape: {nil, @num_queries, @hidden_dim})
      model = BBoxHead.layer(input, @hidden_dim, "bbox")

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @num_queries, @hidden_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @num_queries, @hidden_dim})
      output = predict_fn.(params, test_input)

      min_val = Nx.to_number(Nx.reduce_min(output))
      max_val = Nx.to_number(Nx.reduce_max(output))

      assert min_val >= 0.0
      assert max_val <= 1.0
    end

    test "handles batch_size=1" do
      input = Axon.input("input", shape: {nil, @num_queries, @hidden_dim})
      model = BBoxHead.layer(input, @hidden_dim, "bbox")

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({1, @num_queries, @hidden_dim}, :f32), Axon.ModelState.empty())

      output = predict_fn.(params, Nx.broadcast(0.5, {1, @num_queries, @hidden_dim}))

      assert Nx.shape(output) == {1, @num_queries, 4}
    end

    test "handles single query" do
      input = Axon.input("input", shape: {nil, 1, @hidden_dim})
      model = BBoxHead.layer(input, @hidden_dim, "bbox")

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, 1, @hidden_dim}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.5, {@batch, 1, @hidden_dim}))

      assert Nx.shape(output) == {@batch, 1, 4}
    end

    test "output is finite for random input" do
      input = Axon.input("input", shape: {nil, @num_queries, @hidden_dim})
      model = BBoxHead.layer(input, @hidden_dim, "bbox")

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch, @num_queries, @hidden_dim}, :f32), Axon.ModelState.empty())

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @num_queries, @hidden_dim})
      output = predict_fn.(params, test_input)

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end
end
