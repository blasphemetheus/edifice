defmodule Edifice.Blocks.SinusoidalPETest do
  use ExUnit.Case, async: true

  alias Edifice.Blocks.SinusoidalPE

  @batch 2
  @seq_len 8
  @hidden 32

  describe "build_table/1" do
    test "returns correct shape" do
      table = SinusoidalPE.build_table(max_len: 100, dim: 64)
      assert Nx.shape(table) == {100, 64}
    end

    test "values bounded in [-1, 1]" do
      table = SinusoidalPE.build_table(max_len: 100, dim: 64)

      assert Nx.to_number(Nx.reduce_max(table)) <= 1.0 + 1.0e-6
      assert Nx.to_number(Nx.reduce_min(table)) >= -1.0 - 1.0e-6
    end

    test "distinct positions produce distinct encodings" do
      table = SinusoidalPE.build_table(max_len: 100, dim: 64)

      pos0 = table[0]
      pos1 = table[1]
      pos50 = table[50]

      diff_01 = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(pos0, pos1))))
      diff_050 = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(pos0, pos50))))

      assert diff_01 > 0.1
      assert diff_050 > 0.1
    end

    test "position 0 has sin=0 for all sin dimensions" do
      table = SinusoidalPE.build_table(max_len: 100, dim: 64)
      sin_at_pos0 = Nx.slice_along_axis(table[0], 0, 32, axis: 0)

      max_sin_at_0 = Nx.to_number(Nx.reduce_max(Nx.abs(sin_at_pos0)))
      assert max_sin_at_0 < 1.0e-5
    end

    test "deterministic (no learned params)" do
      t1 = SinusoidalPE.build_table(max_len: 50, dim: 32)
      t2 = SinusoidalPE.build_table(max_len: 50, dim: 32)

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(t1, t2))))
      assert diff < 1.0e-6
    end
  end

  describe "layer/2" do
    test "adds positional encoding and preserves shape" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = SinusoidalPE.layer(input, dim: @hidden)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.0, {@batch, @seq_len, @hidden}))

      assert Nx.shape(output) == {@batch, @seq_len, @hidden}
    end

    test "zero input produces non-zero output (PE is additive)" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = SinusoidalPE.layer(input, dim: @hidden)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.0, {@batch, @seq_len, @hidden}))

      max_val = Nx.to_number(Nx.reduce_max(Nx.abs(output)))
      assert max_val > 0.01, "PE should produce non-zero output from zero input"
    end

    test "handles batch_size=1" do
      input = Axon.input("input", shape: {nil, @seq_len, @hidden})
      model = SinusoidalPE.layer(input, dim: @hidden)

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({1, @seq_len, @hidden}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, Nx.broadcast(0.0, {1, @seq_len, @hidden}))

      assert Nx.shape(output) == {1, @seq_len, @hidden}
    end
  end
end
