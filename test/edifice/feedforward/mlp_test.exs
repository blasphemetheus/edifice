defmodule Edifice.Feedforward.MLPTest do
  use ExUnit.Case, async: true

  alias Edifice.Feedforward.MLP

  @batch_size 2

  describe "build/1" do
    test "builds model with correct output shape" do
      model = MLP.build(input_size: 64, hidden_sizes: [128, 64])

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 64}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, 64}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "output size matches last hidden size" do
      model = MLP.build(input_size: 32, hidden_sizes: [256, 128])

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 32}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, 32}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 128}
    end

    @tag :slow
    test "uses default hidden sizes when not specified" do
      model = MLP.build(input_size: 64)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 64}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, 64}, type: :f32)
      output = predict_fn.(params, input)

      # Default hidden sizes are [512, 512]
      assert Nx.shape(output) == {@batch_size, 512}
    end

    test "supports layer_norm option" do
      model = MLP.build(input_size: 64, hidden_sizes: [128, 64], layer_norm: true)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 64}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, 64}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 64}
    end

    test "supports residual option" do
      model = MLP.build(input_size: 64, hidden_sizes: [128, 64], residual: true)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, 64}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, 64}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 64}
    end
  end

  describe "build_temporal/1" do
    test "processes sequence input and outputs last frame hidden" do
      seq_len = 12
      embed_size = 64

      model = MLP.build_temporal(embed_size: embed_size, seq_len: seq_len)

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, seq_len, embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, seq_len, embed_size}, type: :f32)
      output = predict_fn.(params, input)

      # Default hidden_sizes [512, 512], output is last hidden size
      assert Nx.shape(output) == {@batch_size, 512}
    end

    test "respects custom hidden sizes" do
      seq_len = 12
      embed_size = 64

      model =
        MLP.build_temporal(embed_size: embed_size, seq_len: seq_len, hidden_sizes: [128, 32])

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, seq_len, embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, seq_len, embed_size}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, 32}
    end
  end

  describe "output_size/1" do
    test "returns last hidden size" do
      assert MLP.output_size(hidden_sizes: [128, 64]) == 64
    end

    test "returns default when no hidden sizes given" do
      assert MLP.output_size() == 512
    end
  end
end
