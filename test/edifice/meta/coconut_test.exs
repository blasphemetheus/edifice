defmodule Edifice.Meta.CoconutTest do
  use ExUnit.Case, async: true
  @moduletag :meta

  alias Edifice.Meta.Coconut

  @opts [
    embed_dim: 32,
    hidden_size: 32,
    num_heads: 4,
    num_thoughts: 2,
    num_layers: 1,
    dropout: 0.0,
    window_size: 8
  ]

  describe "build/1" do
    test "produces correct output shape" do
      model = Coconut.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), Axon.ModelState.empty())
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end

    test "outputs are finite" do
      model = Coconut.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), Axon.ModelState.empty())
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      model = Coconut.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 8, 32}, :f32), Axon.ModelState.empty())
      out = predict_fn.(params, Nx.divide(Nx.iota({1, 8, 32}, type: :f32), 256))
      assert Nx.shape(out) == {1, 32}
    end

    test "num_thoughts=1 works" do
      opts = Keyword.put(@opts, :num_thoughts, 1)
      model = Coconut.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), Axon.ModelState.empty())
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end

    test "num_thoughts=5 works" do
      opts = Keyword.put(@opts, :num_thoughts, 5)
      model = Coconut.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), Axon.ModelState.empty())
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end

    test "different embed_dim and hidden_size" do
      opts = Keyword.merge(@opts, embed_dim: 24, hidden_size: 32)
      model = Coconut.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 24}, :f32), Axon.ModelState.empty())
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 24}, type: :f32), 384))
      assert Nx.shape(out) == {2, 32}
    end

    test "multiple layers per thought step" do
      opts = Keyword.put(@opts, :num_layers, 2)
      model = Coconut.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), Axon.ModelState.empty())
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Coconut.output_size(hidden_size: 128) == 128
    end

    test "uses default" do
      assert Coconut.output_size([]) == 256
    end
  end

  describe "Edifice.build/2" do
    test "builds via registry" do
      model = Edifice.build(:coconut, @opts)
      assert %Axon{} = model
    end
  end
end
