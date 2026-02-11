defmodule Edifice.Recurrent.ReservoirTest do
  use ExUnit.Case, async: true

  alias Edifice.Recurrent.Reservoir

  @batch_size 2
  @seq_len 8

  describe "build/1" do
    test "builds reservoir with correct output shape" do
      input_size = 32
      reservoir_size = 64
      output_size = 16

      model =
        Reservoir.build(
          input_size: input_size,
          reservoir_size: reservoir_size,
          output_size: output_size,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, input_size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, input_size}, type: :f32)
      output = predict_fn.(params, input)

      # Output: [batch, output_size]
      assert Nx.shape(output) == {@batch_size, output_size}
    end

    test "uses default reservoir_size when output_size not specified" do
      input_size = 32
      reservoir_size = 64

      model =
        Reservoir.build(
          input_size: input_size,
          reservoir_size: reservoir_size,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, input_size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, input_size}, type: :f32)
      output = predict_fn.(params, input)

      # output_size defaults to reservoir_size
      assert Nx.shape(output) == {@batch_size, reservoir_size}
    end

    test "supports custom spectral_radius and sparsity" do
      input_size = 16
      reservoir_size = 32
      output_size = 8

      model =
        Reservoir.build(
          input_size: input_size,
          reservoir_size: reservoir_size,
          output_size: output_size,
          spectral_radius: 0.95,
          sparsity: 0.8,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, input_size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, input_size}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, output_size}
    end

    test "supports custom input_scaling and leak_rate" do
      input_size = 16
      reservoir_size = 32
      output_size = 8

      model =
        Reservoir.build(
          input_size: input_size,
          reservoir_size: reservoir_size,
          output_size: output_size,
          input_scaling: 0.5,
          leak_rate: 0.8,
          seq_len: @seq_len
        )

      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(Nx.template({@batch_size, @seq_len, input_size}, :f32), Axon.ModelState.empty())

      input = Nx.iota({@batch_size, @seq_len, input_size}, type: :f32)
      output = predict_fn.(params, input)

      assert Nx.shape(output) == {@batch_size, output_size}
    end

    test "builds model graph without error" do
      model =
        Reservoir.build(
          input_size: 32,
          reservoir_size: 64,
          output_size: 16,
          seq_len: @seq_len
        )

      # Model is an Axon struct that can be built
      assert %Axon{} = model
    end
  end

  describe "output_size/1" do
    test "returns output_size when specified" do
      assert Reservoir.output_size(output_size: 128) == 128
    end

    test "falls back to reservoir_size" do
      assert Reservoir.output_size(reservoir_size: 256) == 256
    end

    test "returns default when nothing specified" do
      assert Reservoir.output_size() == 500
    end
  end
end
