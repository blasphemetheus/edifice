defmodule Edifice.RecurrentTest do
  use ExUnit.Case, async: true

  alias Edifice.Recurrent

  @batch_size 2
  @seq_len 8

  describe "build/1 with LSTM" do
    test "builds LSTM model with correct output shape" do
      embed_dim = 64
      hidden_size = 32

      model =
        Recurrent.build(
          embed_dim: embed_dim,
          hidden_size: hidden_size,
          cell_type: :lstm,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, embed_dim}, type: :f32)
      output = predict_fn.(params, input)

      # Default return_sequences: false, so output is [batch, hidden_size]
      assert Nx.shape(output) == {@batch_size, hidden_size}
    end

    test "supports multi-layer LSTM" do
      embed_dim = 64
      hidden_size = 32

      model =
        Recurrent.build(
          embed_dim: embed_dim,
          hidden_size: hidden_size,
          cell_type: :lstm,
          num_layers: 2,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, embed_dim}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, hidden_size}
    end

    test "supports return_sequences option" do
      embed_dim = 64
      hidden_size = 32

      model =
        Recurrent.build(
          embed_dim: embed_dim,
          hidden_size: hidden_size,
          cell_type: :lstm,
          return_sequences: true,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, embed_dim}, type: :f32)
      output = predict_fn.(params, input)

      # return_sequences: true -> [batch, seq_len, hidden_size]
      assert Nx.shape(output) == {@batch_size, @seq_len, hidden_size}
    end
  end

  describe "build/1 with GRU" do
    test "builds GRU model with correct output shape" do
      embed_dim = 64
      hidden_size = 32

      model =
        Recurrent.build(
          embed_dim: embed_dim,
          hidden_size: hidden_size,
          cell_type: :gru,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, embed_dim}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, hidden_size}
    end

    test "supports return_sequences with GRU" do
      embed_dim = 64
      hidden_size = 32

      model =
        Recurrent.build(
          embed_dim: embed_dim,
          hidden_size: hidden_size,
          cell_type: :gru,
          return_sequences: true,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, embed_dim}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, @seq_len, hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert Recurrent.output_size(hidden_size: 128) == 128
    end

    test "returns default when not specified" do
      assert Recurrent.output_size() == 256
    end
  end

  describe "initial_hidden/2" do
    test "returns LSTM hidden state tuple" do
      {h, c} = Recurrent.initial_hidden(4, hidden_size: 32, cell_type: :lstm)
      assert Nx.shape(h) == {4, 32}
      assert Nx.shape(c) == {4, 32}
    end

    test "returns GRU hidden state tensor" do
      h = Recurrent.initial_hidden(4, hidden_size: 32, cell_type: :gru)
      assert Nx.shape(h) == {4, 32}
    end
  end

  describe "cell_types/0" do
    test "returns supported cell types" do
      assert Recurrent.cell_types() == [:lstm, :gru]
    end
  end
end
