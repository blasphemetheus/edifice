defmodule Edifice.Interpretability.CrosscoderTest do
  use ExUnit.Case, async: true
  @moduletag :interpretability

  alias Edifice.Interpretability.Crosscoder

  @batch 4
  @input_size 32
  @dict_size 64
  @top_k 8
  @num_sources 3

  @opts [
    input_size: @input_size,
    dict_size: @dict_size,
    top_k: @top_k,
    num_sources: @num_sources
  ]

  defp templates do
    for i <- 0..(@num_sources - 1), into: %{} do
      {"crosscoder_source_#{i}", Nx.template({@batch, @input_size}, :f32)}
    end
  end

  defp random_inputs do
    key = Nx.Random.key(42)

    for i <- 0..(@num_sources - 1), into: %{} do
      {input, _key} = Nx.Random.uniform(key, shape: {@batch, @input_size})
      {"crosscoder_source_#{i}", input}
    end
  end

  describe "build/1 symmetric" do
    test "builds an Axon container model" do
      model = Crosscoder.build(@opts)
      assert %Axon{} = model
    end

    test "produces one output per source" do
      model = Crosscoder.build(@opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(templates(), Axon.ModelState.empty())
      output = predict_fn.(params, random_inputs())

      assert is_map(output)
      assert map_size(output) == @num_sources

      for i <- 0..(@num_sources - 1) do
        key = :"source_#{i}"
        assert Map.has_key?(output, key)
        assert Nx.shape(output[key]) == {@batch, @input_size}
      end
    end

    test "has separate encoder and decoder per source" do
      model = Crosscoder.build(@opts)
      {init_fn, _predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(templates(), Axon.ModelState.empty())
      data = params.data

      for i <- 0..(@num_sources - 1) do
        assert Map.has_key?(data, "crosscoder_encoder_#{i}")
        assert Map.has_key?(data, "crosscoder_decoder_#{i}")
      end
    end
  end

  describe "build/1 asymmetric (cross-layer transcoder)" do
    test "supports different output_size" do
      output_size = 48

      model =
        Crosscoder.build(
          input_size: @input_size,
          output_size: output_size,
          dict_size: @dict_size,
          num_sources: 2,
          top_k: @top_k
        )

      templates =
        for i <- 0..1, into: %{} do
          {"crosscoder_source_#{i}", Nx.template({@batch, @input_size}, :f32)}
        end

      inputs =
        for i <- 0..1, into: %{} do
          key = Nx.Random.key(i)
          {v, _} = Nx.Random.uniform(key, shape: {@batch, @input_size})
          {"crosscoder_source_#{i}", v}
        end

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      params = init_fn.(templates, Axon.ModelState.empty())
      output = predict_fn.(params, inputs)

      assert Nx.shape(output.source_0) == {@batch, output_size}
      assert Nx.shape(output.source_1) == {@batch, output_size}
    end
  end

  describe "build_encoder/1" do
    test "returns shared sparse activations" do
      encoder = Crosscoder.build_encoder(@opts)
      {init_fn, predict_fn} = Axon.build(encoder, mode: :inference)
      params = init_fn.(templates(), Axon.ModelState.empty())
      hidden = predict_fn.(params, random_inputs())

      assert Nx.shape(hidden) == {@batch, @dict_size}
    end
  end

  describe "output_size/1" do
    test "returns input_size for symmetric" do
      assert Crosscoder.output_size(@opts) == @input_size
    end

    test "returns output_size for asymmetric" do
      assert Crosscoder.output_size(input_size: 32, output_size: 48) == 48
    end
  end

  describe "Edifice.build/2" do
    test "builds crosscoder via registry" do
      model = Edifice.build(:crosscoder, @opts)
      assert %Axon{} = model
    end
  end
end
