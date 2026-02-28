defmodule Edifice.Blocks.SinusoidalPE2DTest do
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.Blocks.SinusoidalPE2D

  describe "build_table/2" do
    test "produces correct shape" do
      pe = SinusoidalPE2D.build_table(16, 64)
      assert Nx.shape(pe) == {1, 16, 64}
    end

    test "handles non-square grid (seq_len=12, dim=32)" do
      pe = SinusoidalPE2D.build_table(12, 32)
      assert Nx.shape(pe) == {1, 12, 32}
    end

    test "values are bounded in [-1, 1]" do
      pe = SinusoidalPE2D.build_table(64, 128)

      max_val = Nx.to_number(Nx.reduce_max(pe))
      min_val = Nx.to_number(Nx.reduce_min(pe))

      assert max_val <= 1.0 + 1.0e-5
      assert min_val >= -1.0 - 1.0e-5
    end

    test "distinct positions produce distinct encodings" do
      pe = SinusoidalPE2D.build_table(16, 64)

      pos0 = pe[0][0]
      pos1 = pe[0][1]
      pos8 = pe[0][8]

      diff_01 = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(pos0, pos1))))
      diff_08 = Nx.to_number(Nx.sum(Nx.abs(Nx.subtract(pos0, pos8))))

      assert diff_01 > 0.01, "Adjacent positions should differ"
      assert diff_08 > 0.01, "Distant positions should differ"
    end

    test "output is finite (no NaN or Inf)" do
      pe = SinusoidalPE2D.build_table(100, 256)

      refute Nx.any(Nx.is_nan(pe)) |> Nx.to_number() == 1
      refute Nx.any(Nx.is_infinity(pe)) |> Nx.to_number() == 1
    end

    test "seq_len=1 works" do
      pe = SinusoidalPE2D.build_table(1, 16)
      assert Nx.shape(pe) == {1, 1, 16}
    end

    test "deterministic (same input produces same output)" do
      pe1 = SinusoidalPE2D.build_table(16, 64)
      pe2 = SinusoidalPE2D.build_table(16, 64)

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(pe1, pe2))))
      assert diff < 1.0e-6
    end
  end
end
