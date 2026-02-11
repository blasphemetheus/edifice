defmodule Edifice.Convolutional.TCNTest do
  use ExUnit.Case, async: true

  alias Edifice.Convolutional.TCN

  @batch_size 2

  describe "build/1" do
    test "builds TCN with correct output shape" do
      input_size = 64
      channels = [32, 32]
      seq_len = 16

      model = TCN.build(
        input_size: input_size,
        channels: channels,
        seq_len: seq_len
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, seq_len, input_size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, seq_len, input_size}, type: :f32)
      output = predict_fn.(params, input)

      # Output: [batch, seq_len, last_channel]
      assert Nx.shape(output) == {@batch_size, seq_len, 32}
    end

    test "uses default channels when not specified" do
      input_size = 64
      seq_len = 16

      model = TCN.build(input_size: input_size, seq_len: seq_len)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, seq_len, input_size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, seq_len, input_size}, type: :f32)
      output = predict_fn.(params, input)

      # Default channels [64, 64, 64, 64], last is 64
      assert Nx.shape(output) == {@batch_size, seq_len, 64}
    end

    test "preserves sequence length" do
      input_size = 32
      seq_len = 24
      channels = [16, 16, 16]

      model = TCN.build(
        input_size: input_size,
        channels: channels,
        seq_len: seq_len,
        kernel_size: 3
      )

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch_size, seq_len, input_size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, seq_len, input_size}, type: :f32)
      output = predict_fn.(params, input)

      # seq_len should be preserved through causal convolutions
      assert Nx.shape(output) == {@batch_size, seq_len, 16}
    end
  end

  describe "output_size/1" do
    test "returns last channel count" do
      assert TCN.output_size(channels: [128, 64, 32]) == 32
    end

    test "returns default when not specified" do
      assert TCN.output_size() == 64
    end
  end

  describe "receptive_field/1" do
    test "calculates receptive field for default config" do
      rf = TCN.receptive_field(num_layers: 4, kernel_size: 3)
      # 1 + 2 * (3 - 1) * (2^4 - 1) = 1 + 4 * 15 = 61
      assert rf == 61
    end

    test "calculates receptive field for custom config" do
      rf = TCN.receptive_field(num_layers: 6, kernel_size: 3)
      # 1 + 2 * (3 - 1) * (2^6 - 1) = 1 + 4 * 63 = 253
      assert rf == 253
    end
  end

  describe "layers_for_receptive_field/2" do
    test "calculates minimum layers for target receptive field" do
      layers = TCN.layers_for_receptive_field(256, kernel_size: 3)
      assert is_integer(layers)
      assert layers > 0

      # Verify the result achieves the target
      actual_rf = TCN.receptive_field(num_layers: layers, kernel_size: 3)
      assert actual_rf >= 256
    end
  end
end
