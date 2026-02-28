defmodule Edifice.Meta.MixtureOfRecursionsTest do
  use ExUnit.Case, async: true

  alias Edifice.Meta.MixtureOfRecursions

  @opts [
    embed_dim: 32,
    hidden_size: 32,
    num_heads: 4,
    num_recursions: 2,
    num_layers: 4,
    dropout: 0.0,
    window_size: 8
  ]

  describe "build/1 expert-choice" do
    test "produces correct output shape" do
      model = MixtureOfRecursions.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end

    test "outputs are finite" do
      model = MixtureOfRecursions.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      model = MixtureOfRecursions.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({1, 8, 32}, type: :f32), 256))
      assert Nx.shape(out) == {1, 32}
    end

    test "3 recursions work" do
      opts = Keyword.put(@opts, :num_recursions, 3)
      model = MixtureOfRecursions.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end

    test "different embed_dim and hidden_size" do
      opts = Keyword.merge(@opts, embed_dim: 24, hidden_size: 32)
      model = MixtureOfRecursions.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 24}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 24}, type: :f32), 384))
      assert Nx.shape(out) == {2, 32}
    end
  end

  describe "build/1 token-choice" do
    test "produces correct output shape" do
      opts = Keyword.put(@opts, :routing, :token_choice)
      model = MixtureOfRecursions.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end

    test "outputs are finite" do
      opts = Keyword.put(@opts, :routing, :token_choice)
      model = MixtureOfRecursions.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      opts = Keyword.put(@opts, :routing, :token_choice)
      model = MixtureOfRecursions.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({1, 8, 32}, type: :f32), 256))
      assert Nx.shape(out) == {1, 32}
    end

    test "3 recursions work" do
      opts = @opts |> Keyword.put(:routing, :token_choice) |> Keyword.put(:num_recursions, 3)
      model = MixtureOfRecursions.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert MixtureOfRecursions.output_size(hidden_size: 128) == 128
    end

    test "uses default" do
      assert MixtureOfRecursions.output_size([]) == 256
    end
  end

  describe "Edifice.build/2" do
    test "builds via registry" do
      model = Edifice.build(:mixture_of_recursions, @opts)
      assert %Axon{} = model
    end
  end
end
