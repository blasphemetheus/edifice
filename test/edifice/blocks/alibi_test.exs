defmodule Edifice.Blocks.ALiBiTest do
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.Blocks.ALiBi

  describe "compute_slopes/1" do
    test "returns correct number of slopes" do
      slopes = ALiBi.compute_slopes(8)
      assert Nx.shape(slopes) == {8}
    end

    test "slopes follow geometric sequence for power-of-2 heads" do
      slopes = ALiBi.compute_slopes(8)

      expected =
        Enum.map(1..8, fn i -> :math.pow(2.0, -i) end)
        |> Nx.tensor(type: :f32)

      diff = Nx.to_number(Nx.reduce_max(Nx.abs(Nx.subtract(slopes, expected))))
      assert diff < 1.0e-6
    end

    test "first slope is 0.5, last is 2^-n for n heads" do
      slopes = ALiBi.compute_slopes(8)

      assert_in_delta Nx.to_number(slopes[0]), 0.5, 1.0e-6
      assert_in_delta Nx.to_number(slopes[7]), :math.pow(2.0, -8), 1.0e-6
    end

    test "non-power-of-2 heads" do
      slopes = ALiBi.compute_slopes(6)
      assert Nx.shape(slopes) == {6}
      # Should still produce valid slopes
      refute Nx.any(Nx.is_nan(slopes)) |> Nx.to_number() == 1
    end
  end

  describe "compute_bias/1" do
    test "returns correct shape" do
      bias = ALiBi.compute_bias(seq_len: 16, num_heads: 4)
      assert Nx.shape(bias) == {4, 16, 16}
    end

    test "diagonal is zero (self-distance)" do
      bias = ALiBi.compute_bias(seq_len: 8, num_heads: 4)

      for h <- 0..3 do
        for i <- 0..7 do
          val = Nx.to_number(bias[h][i][i])
          assert_in_delta val, 0.0, 1.0e-5, "Head #{h}, position #{i} diagonal"
        end
      end
    end

    test "bias magnitude increases with distance" do
      bias = ALiBi.compute_bias(seq_len: 8, num_heads: 2, causal: true)

      bias_near = Nx.to_number(bias[0][4][3])
      bias_far = Nx.to_number(bias[0][4][0])

      assert bias_far < bias_near, "Farther positions should have more negative bias"
    end

    test "causal bias has non-positive values on and below diagonal" do
      bias = ALiBi.compute_bias(seq_len: 8, num_heads: 2, causal: true)

      for h <- 0..1, i <- 0..7, j <- 0..i do
        val = Nx.to_number(bias[h][i][j])
        assert val <= 0.0 + 1.0e-6, "Position [#{h}][#{i}][#{j}] = #{val} should be <= 0"
      end
    end
  end
end
