defmodule Edifice.Scientific.FNOTest do
  use ExUnit.Case, async: true

  alias Edifice.Scientific.FNO

  describe "build/1" do
    test "builds FNO model with required options" do
      model = FNO.build(
        in_channels: 1,
        out_channels: 1,
        modes: 4,
        hidden_channels: 16,
        num_layers: 2
      )

      assert %Axon{} = model
    end

    test "builds model with multiple channels" do
      model = FNO.build(
        in_channels: 3,
        out_channels: 2,
        modes: 8,
        hidden_channels: 32,
        num_layers: 3
      )

      assert %Axon{} = model
    end

    test "model produces output with correct shape" do
      in_channels = 1
      out_channels = 1
      grid_size = 16
      batch_size = 2

      model = FNO.build(
        in_channels: in_channels,
        out_channels: out_channels,
        modes: 4,
        hidden_channels: 16,
        num_layers: 2
      )

      {init_fn, predict_fn} = Axon.build(model)

      params = init_fn.(
        Nx.template({batch_size, grid_size, in_channels}, :f32),
        %{}
      )

      input = Nx.broadcast(0.1, {batch_size, grid_size, in_channels})
      output = predict_fn.(params, %{"input" => input})

      assert {^batch_size, ^grid_size, ^out_channels} = Nx.shape(output)
    end

    test "model handles different grid sizes" do
      model = FNO.build(
        in_channels: 1,
        out_channels: 1,
        modes: 4,
        hidden_channels: 16,
        num_layers: 2
      )

      {init_fn, predict_fn} = Axon.build(model)

      # Test with grid size 32
      params = init_fn.(Nx.template({2, 32, 1}, :f32), %{})
      input = Nx.broadcast(0.1, {2, 32, 1})
      output = predict_fn.(params, %{"input" => input})
      assert {2, 32, 1} = Nx.shape(output)
    end
  end

  describe "build_fno_block/2" do
    test "builds a single FNO block" do
      hidden_channels = 16
      modes = 4

      input = Axon.input("input", shape: {nil, nil, hidden_channels})

      output = FNO.build_fno_block(input,
        hidden_channels: hidden_channels,
        modes: modes,
        name: "test_block"
      )

      assert %Axon{} = output
    end
  end

  describe "build_spectral_conv/4" do
    test "builds spectral convolution layer" do
      hidden_channels = 16
      modes = 4

      input = Axon.input("input", shape: {nil, nil, hidden_channels})
      output = FNO.build_spectral_conv(input, hidden_channels, modes, "test_spectral")

      assert %Axon{} = output
    end
  end

  describe "param_count/1" do
    test "returns positive count" do
      count = FNO.param_count(
        in_channels: 1,
        out_channels: 1,
        modes: 16,
        hidden_channels: 64,
        num_layers: 4
      )

      assert count > 0
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = FNO.recommended_defaults()

      assert Keyword.has_key?(defaults, :hidden_channels)
      assert Keyword.has_key?(defaults, :num_layers)
      assert Keyword.has_key?(defaults, :modes)
      assert Keyword.has_key?(defaults, :activation)
    end
  end
end
