defmodule Edifice.Recurrent.XLSTMTest do
  use ExUnit.Case, async: true
  @moduletag :recurrent

  alias Edifice.Recurrent.XLSTM

  @batch_size 2
  @seq_len 8

  describe "build/1 with sLSTM variant" do
    test "builds sLSTM model with correct output shape" do
      embed_dim = 64
      hidden_size = 32

      model =
        XLSTM.build(
          embed_dim: embed_dim,
          hidden_size: hidden_size,
          num_layers: 2,
          variant: :slstm,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, embed_dim}, type: :f32)
      output = predict_fn.(params, input)

      # Output is last timestep: [batch, hidden_size]
      assert Nx.shape(output) == {@batch_size, hidden_size}
    end
  end

  describe "build/1 with mLSTM variant" do
    test "builds mLSTM model with correct output shape" do
      embed_dim = 64
      hidden_size = 32
      num_heads = 2
      head_dim = 16

      model =
        XLSTM.build(
          embed_dim: embed_dim,
          hidden_size: hidden_size,
          num_layers: 2,
          variant: :mlstm,
          num_heads: num_heads,
          head_dim: head_dim,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, embed_dim}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, hidden_size}
    end
  end

  describe "build/1 with mixed variant" do
    test "builds mixed sLSTM/mLSTM model with correct output shape" do
      embed_dim = 64
      hidden_size = 32

      model =
        XLSTM.build(
          embed_dim: embed_dim,
          hidden_size: hidden_size,
          num_layers: 4,
          variant: :mixed,
          num_heads: 2,
          head_dim: 16,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, embed_dim}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, hidden_size}
    end
  end

  describe "build/1 with default variant" do
    test "uses mixed variant by default" do
      embed_dim = 64
      hidden_size = 32

      model =
        XLSTM.build(
          embed_dim: embed_dim,
          hidden_size: hidden_size,
          num_layers: 2,
          num_heads: 2,
          head_dim: 16,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, embed_dim}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, embed_dim}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, hidden_size}
    end
  end

  describe "build/1 with embed_dim == hidden_size" do
    test "skips input projection" do
      size = 32

      model =
        XLSTM.build(
          embed_dim: size,
          hidden_size: size,
          num_layers: 1,
          variant: :slstm,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, @seq_len, size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, size}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert XLSTM.output_size(hidden_size: 128) == 128
    end

    test "returns default when not specified" do
      assert XLSTM.output_size() == XLSTM.default_hidden_size()
    end
  end

  describe "param_count/1" do
    test "returns a positive integer for sLSTM" do
      count = XLSTM.param_count(embed_dim: 64, hidden_size: 32, num_layers: 2, variant: :slstm)
      assert is_integer(count)
      assert count > 0
    end

    test "returns a positive integer for mLSTM" do
      count =
        XLSTM.param_count(
          embed_dim: 64,
          hidden_size: 32,
          num_layers: 2,
          variant: :mlstm,
          num_heads: 2,
          head_dim: 16
        )

      assert is_integer(count)
      assert count > 0
    end

    test "returns a positive integer for mixed" do
      count = XLSTM.param_count(embed_dim: 64, hidden_size: 32, num_layers: 4, variant: :mixed)
      assert is_integer(count)
      assert count > 0
    end
  end
end
