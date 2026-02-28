defmodule Edifice.Attention.SPLATest do
  use ExUnit.Case, async: true
  @moduletag :attention

  alias Edifice.Attention.SPLA

  @opts [
    embed_dim: 32,
    hidden_size: 32,
    num_heads: 4,
    head_dim: 8,
    block_stride: 8,
    block_window: 8,
    selection_ratio: 0.5,
    num_layers: 2,
    dropout: 0.0,
    seq_len: 16
  ]

  describe "build/1" do
    test "produces correct output shape" do
      model = SPLA.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 16, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 16, 32}, type: :f32), 1024))
      assert Nx.shape(out) == {2, 32}
    end

    test "outputs are finite" do
      model = SPLA.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 16, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 16, 32}, type: :f32), 1024))
      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      model = SPLA.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 16, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({1, 16, 32}, type: :f32), 512))
      assert Nx.shape(out) == {1, 32}
    end

    test "longer sequence" do
      opts = Keyword.merge(@opts, seq_len: 32, block_stride: 8)
      model = SPLA.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 32, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 32, 32}, type: :f32), 2048))
      assert Nx.shape(out) == {2, 32}
    end

    test "different embed_dim and hidden_size" do
      opts = Keyword.merge(@opts, embed_dim: 24, hidden_size: 32)
      model = SPLA.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 16, 24}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 16, 24}, type: :f32), 768))
      assert Nx.shape(out) == {2, 32}
    end

    test "single layer" do
      opts = Keyword.put(@opts, :num_layers, 1)
      model = SPLA.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 16, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 16, 32}, type: :f32), 1024))
      assert Nx.shape(out) == {2, 32}
    end

    test "low selection ratio" do
      opts = Keyword.put(@opts, :selection_ratio, 0.25)
      model = SPLA.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 16, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 16, 32}, type: :f32), 1024))
      assert Nx.shape(out) == {2, 32}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert SPLA.output_size(hidden_size: 128) == 128
    end

    test "uses default" do
      assert SPLA.output_size([]) == 256
    end
  end

  describe "Edifice.build/2" do
    test "builds via registry" do
      model = Edifice.build(:spla, @opts)
      assert %Axon{} = model
    end
  end
end
