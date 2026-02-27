defmodule Edifice.Blocks.CrossAttentionTest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.CrossAttention

  @batch 2

  describe "layer/3 (shared KV)" do
    test "output shape matches query sequence length" do
      queries = Axon.input("queries", shape: {nil, 8, 16})
      context = Axon.input("context", shape: {nil, 12, 16})

      model =
        CrossAttention.layer(queries, context,
          hidden_size: 32,
          name: "test_cross"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "queries" => Nx.template({@batch, 8, 16}, :f32),
            "context" => Nx.template({@batch, 12, 16}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "queries" => Nx.broadcast(0.5, {@batch, 8, 16}),
          "context" => Nx.broadcast(0.5, {@batch, 12, 16})
        })

      assert Nx.shape(output) == {@batch, 8, 32}
    end

    test "different context produces different output" do
      queries = Axon.input("queries", shape: {nil, 4, 16})
      context = Axon.input("context", shape: {nil, 8, 16})

      model =
        CrossAttention.layer(queries, context,
          hidden_size: 16,
          name: "test_cross"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "queries" => Nx.template({1, 4, 16}, :f32),
            "context" => Nx.template({1, 8, 16}, :f32)
          },
          Axon.ModelState.empty()
        )

      query_input = Nx.broadcast(1.0, {1, 4, 16})

      out1 =
        predict_fn.(params, %{
          "queries" => query_input,
          "context" => Nx.broadcast(1.0, {1, 8, 16})
        })

      key = Nx.Random.key(42)
      {rand_ctx, _} = Nx.Random.normal(key, shape: {1, 8, 16})

      out2 =
        predict_fn.(params, %{
          "queries" => query_input,
          "context" => rand_ctx
        })

      diff = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(out1, out2))))
      assert diff > 0.01
    end

    test "multi-head attention with num_heads option" do
      queries = Axon.input("queries", shape: {nil, 4, 32})
      context = Axon.input("context", shape: {nil, 8, 32})

      model =
        CrossAttention.layer(queries, context,
          hidden_size: 32,
          num_heads: 4,
          name: "test_mh_cross"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "queries" => Nx.template({@batch, 4, 32}, :f32),
            "context" => Nx.template({@batch, 8, 32}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "queries" => Nx.broadcast(0.5, {@batch, 4, 32}),
          "context" => Nx.broadcast(0.5, {@batch, 8, 32})
        })

      assert Nx.shape(output) == {@batch, 4, 32}
    end

    test "handles batch_size=1" do
      queries = Axon.input("queries", shape: {nil, 4, 16})
      context = Axon.input("context", shape: {nil, 8, 16})

      model =
        CrossAttention.layer(queries, context,
          hidden_size: 16,
          name: "test_cross"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "queries" => Nx.template({1, 4, 16}, :f32),
            "context" => Nx.template({1, 8, 16}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "queries" => Nx.broadcast(0.5, {1, 4, 16}),
          "context" => Nx.broadcast(0.5, {1, 8, 16})
        })

      assert Nx.shape(output) == {1, 4, 16}
    end

    test "output is finite" do
      queries = Axon.input("queries", shape: {nil, 4, 16})
      context = Axon.input("context", shape: {nil, 8, 16})

      model =
        CrossAttention.layer(queries, context,
          hidden_size: 16,
          name: "test_cross"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "queries" => Nx.template({@batch, 4, 16}, :f32),
            "context" => Nx.template({@batch, 8, 16}, :f32)
          },
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.normal(key, shape: {@batch, 4, 16})
      {c, _} = Nx.Random.normal(key, shape: {@batch, 8, 16})

      output = predict_fn.(params, %{"queries" => q, "context" => c})

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end

  describe "layer/4 (separate KV)" do
    test "output shape with separate key and value inputs" do
      queries = Axon.input("queries", shape: {nil, 4, 16})
      keys = Axon.input("keys", shape: {nil, 8, 16})
      values = Axon.input("values", shape: {nil, 8, 16})

      model =
        CrossAttention.layer(queries, keys, values,
          hidden_size: 16,
          name: "test_sep_cross"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "queries" => Nx.template({@batch, 4, 16}, :f32),
            "keys" => Nx.template({@batch, 8, 16}, :f32),
            "values" => Nx.template({@batch, 8, 16}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "queries" => Nx.broadcast(0.5, {@batch, 4, 16}),
          "keys" => Nx.broadcast(0.5, {@batch, 8, 16}),
          "values" => Nx.broadcast(0.5, {@batch, 8, 16})
        })

      assert Nx.shape(output) == {@batch, 4, 16}
    end
  end
end
