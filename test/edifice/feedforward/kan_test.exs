defmodule Edifice.Feedforward.KANTest do
  use ExUnit.Case, async: true
  @moduletag timeout: 120_000

  alias Edifice.Feedforward.KAN

  @batch_size 2

  describe "build/1" do
    test "builds model with correct output shape" do
      embed_size = 64
      hidden_size = 32
      seq_len = 8

      model =
        KAN.build(
          embed_size: embed_size,
          hidden_size: hidden_size,
          num_layers: 2,
          seq_len: seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, seq_len, embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, seq_len, embed_size}, type: :f32)
      output = predict_fn.(params, input)

      # Output is last timestep: [batch, hidden_size]
      assert Nx.shape(output) == {@batch_size, hidden_size}
    end

    test "uses default hidden_size when not specified" do
      assert KAN.default_hidden_size() == 256
    end

    test "handles embed_size equal to hidden_size (no projection)" do
      size = 32
      seq_len = 4

      model = KAN.build(embed_size: size, hidden_size: size, num_layers: 1, seq_len: seq_len)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, seq_len, size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, seq_len, size}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, size}
    end

    test "supports custom grid_size" do
      embed_size = 32
      hidden_size = 16
      seq_len = 4

      model =
        KAN.build(
          embed_size: embed_size,
          hidden_size: hidden_size,
          num_layers: 1,
          grid_size: 4,
          seq_len: seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, seq_len, embed_size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, seq_len, embed_size}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, hidden_size}
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert KAN.output_size(hidden_size: 128) == 128
    end

    test "returns default when not specified" do
      assert KAN.output_size() == KAN.default_hidden_size()
    end
  end

  describe "param_count/1" do
    test "returns a positive integer" do
      count = KAN.param_count(embed_size: 64, hidden_size: 32, num_layers: 2, grid_size: 4)
      assert is_integer(count)
      assert count > 0
    end
  end
end
