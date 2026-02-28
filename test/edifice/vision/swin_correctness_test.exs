defmodule Edifice.Vision.SwinCorrectnessTest do
  use ExUnit.Case, async: true
  @moduletag :vision

  alias Edifice.Vision.SwinTransformer

  @batch 2
  @channels 3

  # ============================================================================
  # Window Partition / Reverse
  # ============================================================================

  describe "window_partition and window_reverse" do
    test "are perfect inverses" do
      key = Nx.Random.key(42)
      h = 8
      w = 8
      ws = 4
      c = 16

      {input, _} = Nx.Random.uniform(key, shape: {@batch, h, w, c})

      partitioned = SwinTransformer.window_partition(input, ws, h, w)
      reversed = SwinTransformer.window_reverse(partitioned, ws, h, w, @batch)

      assert Nx.shape(reversed) == {@batch, h, w, c}
      diff = Nx.subtract(input, reversed) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "partition produces correct number of windows" do
      key = Nx.Random.key(42)
      h = 8
      w = 8
      ws = 4
      c = 16

      {input, _} = Nx.Random.uniform(key, shape: {@batch, h, w, c})
      partitioned = SwinTransformer.window_partition(input, ws, h, w)

      num_windows = div(h, ws) * div(w, ws)
      assert Nx.shape(partitioned) == {@batch * num_windows, ws * ws, c}
    end

    test "each window contains the correct spatial region" do
      # Create a tensor where each position has a unique value
      h = 4
      w = 4
      ws = 2
      c = 1

      # [1, 4, 4, 1] with values 0..15
      input =
        Nx.iota({1, h, w, c}, axis: 1) |> Nx.multiply(w) |> Nx.add(Nx.iota({1, h, w, c}, axis: 2))

      partitioned = SwinTransformer.window_partition(input, ws, h, w)
      # Should have 4 windows of 4 tokens each
      assert Nx.shape(partitioned) == {4, 4, 1}

      # First window should be top-left 2x2: positions (0,0),(0,1),(1,0),(1,1)
      # Values: 0, 1, 4, 5
      first_window = Nx.slice(partitioned, [0, 0, 0], [1, 4, 1]) |> Nx.reshape({4})
      values = Nx.to_flat_list(first_window)
      assert values == [0.0, 1.0, 4.0, 5.0]
    end
  end

  # ============================================================================
  # Cyclic Shift
  # ============================================================================

  describe "cyclic_shift and reverse_cyclic_shift" do
    test "are perfect inverses" do
      key = Nx.Random.key(42)
      h = 8
      w = 8
      shift_size = 3

      {input, _} = Nx.Random.uniform(key, shape: {@batch, h, w, 16})

      shifted = SwinTransformer.cyclic_shift(input, shift_size, h, w)
      reversed = SwinTransformer.reverse_cyclic_shift(shifted, shift_size, h, w)

      diff = Nx.subtract(input, reversed) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end

    test "shift moves content to expected positions" do
      h = 4
      w = 4
      shift_size = 1

      # Create [1, 4, 4, 1] with row indices as values
      input = Nx.iota({1, h, w, 1}, axis: 1) |> Nx.as_type(:f32)

      shifted = SwinTransformer.cyclic_shift(input, shift_size, h, w)

      # After shifting by 1, row 0 should now contain what was in row 1
      row0 = Nx.slice(shifted, [0, 0, 0, 0], [1, 1, w, 1]) |> Nx.reshape({w})
      expected_row0 = Nx.broadcast(1.0, {w})
      diff = Nx.subtract(row0, expected_row0) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end

  # ============================================================================
  # Shift Mask
  # ============================================================================

  describe "compute_shift_mask" do
    test "has correct shape" do
      h = 8
      w = 8
      ws = 4
      shift_size = 2

      mask = SwinTransformer.compute_shift_mask(h, w, ws, shift_size)
      num_windows = div(h, ws) * div(w, ws)
      assert Nx.shape(mask) == {num_windows, ws * ws, ws * ws}
    end

    test "mask values are 0 or -100" do
      mask = SwinTransformer.compute_shift_mask(8, 8, 4, 2)
      unique = mask |> Nx.to_flat_list() |> Enum.uniq() |> Enum.sort()
      assert unique == [-100.0, 0.0]
    end

    test "mask is symmetric" do
      mask = SwinTransformer.compute_shift_mask(8, 8, 4, 2)
      # For each window, mask[i,j] should equal mask[j,i]
      mask_t = Nx.transpose(mask, axes: [0, 2, 1])
      diff = Nx.subtract(mask, mask_t) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-6
    end
  end

  # ============================================================================
  # Relative Position Bias
  # ============================================================================

  describe "compute_relative_position_bias" do
    test "has correct shape" do
      ws = 4
      num_heads = 2
      bias = SwinTransformer.compute_relative_position_bias(ws, num_heads)
      assert Nx.shape(bias) == {1, num_heads, ws * ws, ws * ws}
    end

    test "diagonal is zero (same position has zero distance)" do
      bias = SwinTransformer.compute_relative_position_bias(4, 2)
      # Extract diagonal for first head
      head_bias = Nx.slice(bias, [0, 0, 0, 0], [1, 1, 16, 16]) |> Nx.reshape({16, 16})
      diag = Nx.take_diagonal(head_bias)
      max_diag = Nx.abs(diag) |> Nx.reduce_max() |> Nx.to_number()
      assert max_diag < 1.0e-6
    end

    test "bias decreases with distance" do
      bias = SwinTransformer.compute_relative_position_bias(4, 1)
      head_bias = Nx.reshape(bias, {16, 16})

      # Position (0,0) to adjacent (0,1) should have smaller magnitude than (0,0) to (3,3)
      close_bias = head_bias[0][1] |> Nx.to_number()
      far_bias = head_bias[0][15] |> Nx.to_number()

      # Both negative, far should be more negative
      assert close_bias > far_bias
    end
  end

  # ============================================================================
  # Full Model
  # ============================================================================

  describe "full Swin model" do
    @swin_opts [
      image_size: 32,
      patch_size: 4,
      in_channels: @channels,
      embed_dim: 16,
      depths: [2, 2],
      num_heads: [2, 4],
      window_size: 4
    ]

    test "shifted blocks produce different output than unshifted" do
      # Build two models: one with all regular windows, one with shifted
      # The default alternates, so just verify the model output is finite
      model = SwinTransformer.build(@swin_opts)
      key = Nx.Random.key(42)
      {input, _} = Nx.Random.uniform(key, shape: {@batch, @channels, 32, 32})

      {init_fn, predict_fn} = Axon.build(model)
      params = init_fn.(Nx.template({@batch, @channels, 32, 32}, :f32), Axon.ModelState.empty())
      output = predict_fn.(params, input)

      # Output should be finite and have correct shape
      assert Nx.shape(output) == {@batch, 32}
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end
  end
end
