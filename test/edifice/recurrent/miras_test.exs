defmodule Edifice.Recurrent.MIRASTest do
  use ExUnit.Case, async: true
  @moduletag :recurrent

  alias Edifice.Recurrent.MIRAS

  @base_opts [
    embed_dim: 32,
    hidden_size: 32,
    memory_size: 16,
    num_layers: 2,
    dropout: 0.0,
    window_size: 8
  ]

  describe "build/1 moneta variant" do
    test "produces correct output shape" do
      model = MIRAS.build(Keyword.put(@base_opts, :variant, :moneta))
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end

    test "outputs are finite" do
      model = MIRAS.build(Keyword.put(@base_opts, :variant, :moneta))
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      model = MIRAS.build(Keyword.put(@base_opts, :variant, :moneta))
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({1, 8, 32}, type: :f32), 256))
      assert Nx.shape(out) == {1, 32}
    end

    test "custom p_norm" do
      opts = @base_opts |> Keyword.put(:variant, :moneta) |> Keyword.put(:p_norm, 3.0)
      model = MIRAS.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end
  end

  describe "build/1 yaad variant" do
    test "produces correct output shape" do
      model = MIRAS.build(Keyword.put(@base_opts, :variant, :yaad))
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end

    test "outputs are finite" do
      model = MIRAS.build(Keyword.put(@base_opts, :variant, :yaad))
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      model = MIRAS.build(Keyword.put(@base_opts, :variant, :yaad))
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({1, 8, 32}, type: :f32), 256))
      assert Nx.shape(out) == {1, 32}
    end
  end

  describe "build/1 memora variant" do
    test "produces correct output shape" do
      model = MIRAS.build(Keyword.put(@base_opts, :variant, :memora))
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end

    test "outputs are finite" do
      model = MIRAS.build(Keyword.put(@base_opts, :variant, :memora))
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      model = MIRAS.build(Keyword.put(@base_opts, :variant, :memora))
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({1, 8, 32}, type: :f32), 256))
      assert Nx.shape(out) == {1, 32}
    end
  end

  describe "build/1 default variant" do
    test "defaults to moneta" do
      model = MIRAS.build(@base_opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 32}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 32}, type: :f32), 512))
      assert Nx.shape(out) == {2, 32}
    end

    test "different embed_dim and hidden_size" do
      opts = Keyword.merge(@base_opts, embed_dim: 24, hidden_size: 32)
      model = MIRAS.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 8, 24}, :f32), %{})
      out = predict_fn.(params, Nx.divide(Nx.iota({2, 8, 24}, type: :f32), 384))
      assert Nx.shape(out) == {2, 32}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert MIRAS.output_size(hidden_size: 128) == 128
    end

    test "uses default" do
      assert MIRAS.output_size([]) == 256
    end
  end

  describe "Edifice.build/2" do
    test "builds moneta via registry" do
      model = Edifice.build(:miras, Keyword.put(@base_opts, :variant, :moneta))
      assert %Axon{} = model
    end

    test "builds yaad via registry" do
      model = Edifice.build(:miras, Keyword.put(@base_opts, :variant, :yaad))
      assert %Axon{} = model
    end

    test "builds memora via registry" do
      model = Edifice.build(:miras, Keyword.put(@base_opts, :variant, :memora))
      assert %Axon{} = model
    end
  end
end
