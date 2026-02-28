defmodule Edifice.Blocks.CausalMaskTest do
  use ExUnit.Case, async: true
  @moduletag :blocks

  alias Edifice.Blocks.CausalMask

  describe "causal/1" do
    test "produces correct shape" do
      mask = CausalMask.causal(8)
      assert Nx.shape(mask) == {8, 8}
    end

    test "is lower triangular (true on and below diagonal)" do
      mask = CausalMask.causal(4)

      # Row 0: [true, false, false, false]
      assert Nx.to_number(mask[0][0]) == 1
      assert Nx.to_number(mask[0][1]) == 0
      assert Nx.to_number(mask[0][2]) == 0
      assert Nx.to_number(mask[0][3]) == 0

      # Row 1: [true, true, false, false]
      assert Nx.to_number(mask[1][0]) == 1
      assert Nx.to_number(mask[1][1]) == 1
      assert Nx.to_number(mask[1][2]) == 0

      # Row 3: all true
      assert Nx.to_number(mask[3][0]) == 1
      assert Nx.to_number(mask[3][3]) == 1
    end

    test "diagonal is all true" do
      mask = CausalMask.causal(8)

      for i <- 0..7 do
        assert Nx.to_number(mask[i][i]) == 1, "Position #{i} should attend to itself"
      end
    end

    test "above diagonal is all false" do
      mask = CausalMask.causal(6)

      for i <- 0..4, j <- (i + 1)..5 do
        assert Nx.to_number(mask[i][j]) == 0,
               "Position #{i} should not attend to future position #{j}"
      end
    end

    test "seq_len=1 produces [[true]]" do
      mask = CausalMask.causal(1)
      assert Nx.shape(mask) == {1, 1}
      assert Nx.to_number(mask[0][0]) == 1
    end

    test "returns boolean type" do
      mask = CausalMask.causal(4)
      assert Nx.type(mask) == {:u, 8}
    end
  end

  describe "window/2" do
    test "produces correct shape" do
      mask = CausalMask.window(8, 3)
      assert Nx.shape(mask) == {8, 8}
    end

    test "is stricter than causal (subset of causal mask)" do
      causal = CausalMask.causal(8)
      window = CausalMask.window(8, 3)

      # Window mask should be a subset: wherever window is true, causal must also be true
      # So window AND NOT causal should be all false
      violates = Nx.logical_and(window, Nx.logical_not(causal))
      assert Nx.to_number(Nx.any(violates)) == 0
    end

    test "limits lookback to window_size" do
      mask = CausalMask.window(8, 3)

      # Position 5 should attend to positions 3, 4, 5 (window=3)
      assert Nx.to_number(mask[5][3]) == 1
      assert Nx.to_number(mask[5][4]) == 1
      assert Nx.to_number(mask[5][5]) == 1

      # Position 5 should NOT attend to position 2 (too far back)
      assert Nx.to_number(mask[5][2]) == 0

      # Position 5 should NOT attend to position 6 (future)
      assert Nx.to_number(mask[5][6]) == 0
    end

    test "window=1 means attend only to self" do
      mask = CausalMask.window(6, 1)

      # Should be identity matrix
      for i <- 0..5 do
        for j <- 0..5 do
          expected = if i == j, do: 1, else: 0
          assert Nx.to_number(mask[i][j]) == expected
        end
      end
    end

    test "window >= seq_len is same as causal" do
      causal = CausalMask.causal(6)
      window = CausalMask.window(6, 10)

      diff =
        Nx.to_number(
          Nx.sum(Nx.abs(Nx.subtract(Nx.as_type(causal, :f32), Nx.as_type(window, :f32))))
        )

      assert diff < 1.0e-6
    end
  end

  describe "to_binary_backend/1" do
    test "copies mask to BinaryBackend" do
      mask = CausalMask.causal(4)
      binary_mask = CausalMask.to_binary_backend(mask)

      assert Nx.shape(binary_mask) == {4, 4}
      # Values should be preserved
      assert Nx.to_number(binary_mask[0][0]) == 1
      assert Nx.to_number(binary_mask[0][1]) == 0
    end
  end
end
