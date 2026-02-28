defmodule Edifice.Blocks.AdaptiveNormTest do
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.Blocks.AdaptiveNorm

  @batch 2
  @seq_len 8
  @hidden 32

  describe "layer/3 with :adaln mode" do
    test "produces correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      condition = Axon.input("condition", shape: {nil, @hidden})

      model =
        AdaptiveNorm.layer(input, condition,
          hidden_size: @hidden,
          mode: :adaln,
          name: "test_adaln"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
            "condition" => Nx.template({@batch, @hidden}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "input" => Nx.broadcast(0.5, {@batch, @seq_len, @hidden}),
          "condition" => Nx.broadcast(0.1, {@batch, @hidden})
        })

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end

    test "output is finite" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      condition = Axon.input("condition", shape: {nil, @hidden})

      model =
        AdaptiveNorm.layer(input, condition,
          hidden_size: @hidden,
          mode: :adaln,
          name: "test_adaln"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
            "condition" => Nx.template({@batch, @hidden}, :f32)
          },
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {@batch, @seq_len, @hidden})

      output =
        predict_fn.(params, %{
          "input" => test_input,
          "condition" => Nx.broadcast(0.0, {@batch, @hidden})
        })

      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end

    test "responds to different conditioning" do
      h = 64
      input = Axon.input("input", shape: {nil, 4, h})
      condition = Axon.input("condition", shape: {nil, h})

      model =
        AdaptiveNorm.layer(input, condition,
          hidden_size: h,
          mode: :adaln,
          name: "test_adaln"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input" => Nx.template({1, 4, h}, :f32),
            "condition" => Nx.template({1, h}, :f32)
          },
          Axon.ModelState.empty()
        )

      key = Nx.Random.key(42)
      {test_input, _} = Nx.Random.normal(key, shape: {1, 4, h})

      out_zeros =
        predict_fn.(params, %{
          "input" => test_input,
          "condition" => Nx.broadcast(0.0, {1, h})
        })

      out_large =
        predict_fn.(params, %{
          "input" => test_input,
          "condition" => Nx.broadcast(100.0, {1, h})
        })

      diff = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(out_zeros, out_large))))
      assert diff > 0.01, "Different conditions should produce different outputs"
    end
  end

  describe "layer/3 with :adaln_zero mode" do
    test "produces correct shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      condition = Axon.input("condition", shape: {nil, @hidden})

      model =
        AdaptiveNorm.layer(input, condition,
          hidden_size: @hidden,
          mode: :adaln_zero,
          name: "test_adaln_zero"
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{
            "input" => Nx.template({@batch, @seq_len, @hidden}, :f32),
            "condition" => Nx.template({@batch, @hidden}, :f32)
          },
          Axon.ModelState.empty()
        )

      output =
        predict_fn.(params, %{
          "input" => Nx.broadcast(1.0, {@batch, @seq_len, @hidden}),
          "condition" => Nx.broadcast(0.0, {@batch, @hidden})
        })

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
      refute Nx.any(Nx.is_nan(output)) |> Nx.to_number() == 1
    end
  end
end
