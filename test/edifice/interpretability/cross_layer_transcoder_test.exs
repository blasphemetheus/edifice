defmodule Edifice.Interpretability.CrossLayerTranscoderTest do
  use ExUnit.Case, async: true

  alias Edifice.Interpretability.CrossLayerTranscoder

  @opts [hidden_size: 16, num_layers: 3, dict_size: 32, top_k: 4]

  describe "build/1" do
    test "produces correct output shape" do
      model = CrossLayerTranscoder.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      # 3 layers * 16 hidden = 48 total
      params = init_fn.(Nx.template({2, 48}, :f32), %{})
      input = Nx.iota({2, 48}, type: :f32) |> Nx.divide(100)
      out = predict_fn.(params, input)

      assert Nx.shape(out) == {2, 48}
    end

    test "outputs are finite" do
      model = CrossLayerTranscoder.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 48}, :f32), %{})
      input = Nx.iota({2, 48}, type: :f32) |> Nx.divide(100)
      out = predict_fn.(params, input)

      assert out |> Nx.is_nan() |> Nx.any() |> Nx.to_number() == 0
      assert out |> Nx.is_infinity() |> Nx.any() |> Nx.to_number() == 0
    end

    test "batch=1 works" do
      model = CrossLayerTranscoder.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, 48}, :f32), %{})
      out = predict_fn.(params, Nx.iota({1, 48}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out) == {1, 48}
    end

    test "different hidden size" do
      opts = Keyword.put(@opts, :hidden_size, 32)
      model = CrossLayerTranscoder.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      # 3 layers * 32 = 96
      params = init_fn.(Nx.template({2, 96}, :f32), %{})
      out = predict_fn.(params, Nx.iota({2, 96}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out) == {2, 96}
    end

    test "different number of layers" do
      opts = Keyword.put(@opts, :num_layers, 4)
      model = CrossLayerTranscoder.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      # 4 layers * 16 = 64
      params = init_fn.(Nx.template({2, 64}, :f32), %{})
      out = predict_fn.(params, Nx.iota({2, 64}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out) == {2, 64}
    end

    test "different dict size and top_k" do
      opts = @opts |> Keyword.put(:dict_size, 64) |> Keyword.put(:top_k, 8)
      model = CrossLayerTranscoder.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 48}, :f32), %{})
      out = predict_fn.(params, Nx.iota({2, 48}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out) == {2, 48}
    end

    test "single layer degenerates to standard transcoder shape" do
      opts = Keyword.put(@opts, :num_layers, 1)
      model = CrossLayerTranscoder.build(opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 16}, :f32), %{})
      out = predict_fn.(params, Nx.iota({2, 16}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out) == {2, 16}
    end
  end

  describe "output_size/1" do
    test "returns num_layers * hidden_size" do
      assert CrossLayerTranscoder.output_size(hidden_size: 16, num_layers: 3) == 48
    end

    test "uses default num_layers" do
      assert CrossLayerTranscoder.output_size(hidden_size: 16) == 96
    end
  end

  describe "Edifice.build/2" do
    test "builds cross_layer_transcoder via registry" do
      model = Edifice.build(:cross_layer_transcoder, @opts)
      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({2, 48}, :f32), %{})
      out = predict_fn.(params, Nx.iota({2, 48}, type: :f32) |> Nx.divide(100))
      assert Nx.shape(out) == {2, 48}
    end
  end
end
