defmodule Edifice.Meta.MixtureOfExpertDepthsTest do
  use ExUnit.Case, async: true
  @moduletag :meta

  alias Edifice.Meta.MixtureOfExpertDepths

  @opts [
    embed_dim: 32,
    hidden_size: 32,
    num_heads: 4,
    num_experts: 2,
    num_layers: 2,
    dropout: 0.0,
    window_size: 8
  ]

  describe "build/1" do
    test "produces correct output shape" do
      model = MixtureOfExpertDepths.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      input = Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512)
      out = predict_fn.(params, input)
      assert Nx.shape(out) == {2, 32}
    end

    test "outputs are finite" do
      model = MixtureOfExpertDepths.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      input = Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512)
      out = predict_fn.(params, input)
      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      model = MixtureOfExpertDepths.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 8, 32}, :f32), %{})
      input = Nx.divide(Nx.iota({1, 8, 32}, type: :f32), 256)
      out = predict_fn.(params, input)
      assert Nx.shape(out) == {1, 32}
    end

    test "3 experts work" do
      opts = Keyword.put(@opts, :num_experts, 3)
      model = MixtureOfExpertDepths.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      input = Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512)
      out = predict_fn.(params, input)
      assert Nx.shape(out) == {2, 32}
    end

    test "4 experts work" do
      opts = Keyword.put(@opts, :num_experts, 4)
      model = MixtureOfExpertDepths.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      input = Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512)
      out = predict_fn.(params, input)
      assert Nx.shape(out) == {2, 32}
    end

    test "different embed_dim and hidden_size" do
      opts = Keyword.merge(@opts, embed_dim: 24, hidden_size: 32)
      model = MixtureOfExpertDepths.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 24}, :f32), %{})
      input = Nx.divide(Nx.iota({2, 8, 24}, type: :f32), 384)
      out = predict_fn.(params, input)
      assert Nx.shape(out) == {2, 32}
    end

    test "custom expert hidden multiplier" do
      opts = Keyword.put(@opts, :expert_hidden_multiplier, 2)
      model = MixtureOfExpertDepths.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      input = Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512)
      out = predict_fn.(params, input)
      assert Nx.shape(out) == {2, 32}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert MixtureOfExpertDepths.output_size(hidden_size: 128) == 128
    end

    test "uses default" do
      assert MixtureOfExpertDepths.output_size([]) == 256
    end
  end

  describe "Edifice.build/2" do
    test "builds via registry" do
      model = Edifice.build(:mixture_of_expert_depths, @opts)
      assert %Axon{} = model
    end
  end
end
