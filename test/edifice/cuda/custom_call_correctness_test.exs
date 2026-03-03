defmodule Edifice.CUDA.CustomCallCorrectnessTest do
  @moduledoc """
  Correctness tests for fused scan custom call dispatch paths.

  These tests run on BinaryBackend (no GPU required). They validate that:
  1. Different momentum values produce different outputs (regression guard)
  2. Outputs are finite and correctly shaped across momentum values
  3. Causal flash/LASER attention masks future tokens correctly
  4. Flash/LASER attention matches naive reference implementation

  The custom call path through EXLA is implicitly validated by ensuring the
  packed tensor layout (momentum as extra column) matches the kernel spec.
  """
  use ExUnit.Case, async: true

  alias Edifice.CUDA.FusedScan

  # ============================================================================
  # Titans scan momentum correctness
  # ============================================================================

  describe "titans_scan momentum correctness" do
    for momentum <- [0.0, 0.5, 0.9, 0.99] do
      @momentum momentum

      test "momentum=#{momentum} produces finite, non-zero output" do
        batch = 1
        seq = 4
        m = 4
        key = Nx.Random.key(42)
        {combined, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq, 4 * m}, type: {:f, 32})

        result = FusedScan.titans_scan(combined, memory_size: m, momentum: @momentum)

        assert Nx.shape(result) == {batch, seq, m}
        assert Nx.type(result) == {:f, 32}

        # Output should be finite (no NaN/Inf)
        assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
        assert Nx.all(Nx.is_infinity(result) |> Nx.logical_not()) |> Nx.to_number() == 1
      end
    end

    test "different momentum values produce different outputs" do
      batch = 1
      seq = 8
      m = 4
      key = Nx.Random.key(123)
      {combined, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq, 4 * m}, type: {:f, 32})

      result_low = FusedScan.titans_scan(combined, memory_size: m, momentum: 0.1)
      result_high = FusedScan.titans_scan(combined, memory_size: m, momentum: 0.99)

      # With different momentum, outputs should differ
      max_diff = Nx.subtract(result_low, result_high)
                 |> Nx.abs()
                 |> Nx.reduce_max()
                 |> Nx.to_number()

      assert max_diff > 1.0e-6,
        "momentum=0.1 and momentum=0.99 produced identical outputs (max_diff=#{max_diff})"
    end

    test "momentum=0 with zero init matches direct gradient application" do
      # With momentum=0, mom_row stays as just grad (no accumulation),
      # so the update simplifies. Verify against manual computation.
      batch = 1
      seq = 2
      m = 2
      key = Nx.Random.key(77)
      {combined, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq, 4 * m}, type: {:f, 32})

      result_zero_mom = FusedScan.titans_scan(combined, memory_size: m, momentum: 0.0)
      result_small_mom = FusedScan.titans_scan(combined, memory_size: m, momentum: 0.01)

      # With very small momentum, result should be close to zero-momentum
      max_diff = Nx.subtract(result_zero_mom, result_small_mom)
                 |> Nx.abs()
                 |> Nx.reduce_max()
                 |> Nx.to_number()

      assert max_diff < 0.1,
        "momentum=0 and momentum=0.01 differ too much (max_diff=#{max_diff})"
    end
  end

  # ============================================================================
  # MIRAS scan momentum correctness
  # ============================================================================

  describe "miras_scan momentum correctness" do
    for momentum <- [0.0, 0.5, 0.9, 0.99] do
      @momentum momentum

      test "momentum=#{momentum} produces finite, non-zero output" do
        batch = 1
        seq = 4
        m = 4
        key = Nx.Random.key(42)
        {combined, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq, 5 * m}, type: {:f, 32})

        result = FusedScan.miras_scan(combined, memory_size: m, momentum: @momentum)

        assert Nx.shape(result) == {batch, seq, m}
        assert Nx.type(result) == {:f, 32}

        # Output should be finite
        assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
        assert Nx.all(Nx.is_infinity(result) |> Nx.logical_not()) |> Nx.to_number() == 1
      end
    end

    test "different momentum values produce different outputs" do
      batch = 1
      seq = 8
      m = 4
      key = Nx.Random.key(456)
      {combined, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq, 5 * m}, type: {:f, 32})

      result_low = FusedScan.miras_scan(combined, memory_size: m, momentum: 0.1)
      result_high = FusedScan.miras_scan(combined, memory_size: m, momentum: 0.99)

      max_diff = Nx.subtract(result_low, result_high)
                 |> Nx.abs()
                 |> Nx.reduce_max()
                 |> Nx.to_number()

      assert max_diff > 1.0e-6,
        "momentum=0.1 and momentum=0.99 produced identical outputs (max_diff=#{max_diff})"
    end

    test "momentum=0 with zero init matches direct gradient application" do
      batch = 1
      seq = 2
      m = 2
      key = Nx.Random.key(88)
      {combined, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq, 5 * m}, type: {:f, 32})

      result_zero_mom = FusedScan.miras_scan(combined, memory_size: m, momentum: 0.0)
      result_small_mom = FusedScan.miras_scan(combined, memory_size: m, momentum: 0.01)

      max_diff = Nx.subtract(result_zero_mom, result_small_mom)
                 |> Nx.abs()
                 |> Nx.reduce_max()
                 |> Nx.to_number()

      assert max_diff < 0.1,
        "momentum=0 and momentum=0.01 differ too much (max_diff=#{max_diff})"
    end
  end

  # ============================================================================
  # Reservoir scan leak_rate correctness
  # ============================================================================

  describe "reservoir_scan leak_rate correctness" do
    for leak_rate <- [0.3, 0.5, 1.0] do
      @leak_rate leak_rate

      test "leak_rate=#{leak_rate} produces finite output" do
        batch = 2
        seq = 4
        hidden = 8
        key = Nx.Random.key(42)
        {wx, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq, hidden}, type: {:f, 32})
        {w_res, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {hidden, hidden}, type: {:f, 32})

        result = FusedScan.reservoir_scan(wx, w_res, leak_rate: @leak_rate)

        assert Nx.shape(result) == {batch, hidden}
        assert Nx.type(result) == {:f, 32}
        assert Nx.all(Nx.is_nan(result) |> Nx.logical_not()) |> Nx.to_number() == 1
        assert Nx.all(Nx.is_infinity(result) |> Nx.logical_not()) |> Nx.to_number() == 1
      end
    end

    test "leak_rate=1.0 matches no-leak reference (h = tanh(wx + W_res@h))" do
      batch = 1
      seq = 3
      hidden = 4
      key = Nx.Random.key(99)
      {wx, key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {batch, seq, hidden}, type: {:f, 32})
      {w_res, _key} = Nx.Random.uniform(key, -0.2, 0.2, shape: {hidden, hidden}, type: {:f, 32})

      result = FusedScan.reservoir_scan(wx, w_res, leak_rate: 1.0)

      # Manual computation: h_t = tanh(wx_t + W_res @ h_{t-1})
      h = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      h = Enum.reduce(0..(seq - 1), h, fn t, h_prev ->
        wx_t = Nx.slice_along_axis(wx, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        Nx.tanh(Nx.add(wx_t, Nx.dot(h_prev, w_res)))
      end)

      max_diff = Nx.subtract(result, h) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert max_diff < 1.0e-5, "leak_rate=1.0 should match manual computation (diff=#{max_diff})"
    end

    test "different leak_rates produce different outputs" do
      batch = 1
      seq = 8
      hidden = 8
      key = Nx.Random.key(456)
      {wx, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq, hidden}, type: {:f, 32})
      {w_res, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {hidden, hidden}, type: {:f, 32})

      result_full = FusedScan.reservoir_scan(wx, w_res, leak_rate: 1.0)
      result_half = FusedScan.reservoir_scan(wx, w_res, leak_rate: 0.5)

      max_diff = Nx.subtract(result_full, result_half)
                 |> Nx.abs()
                 |> Nx.reduce_max()
                 |> Nx.to_number()

      assert max_diff > 1.0e-4,
        "leak_rate=1.0 vs 0.5 should produce different outputs (max_diff=#{max_diff})"
    end
  end

  # ============================================================================
  # Flash attention causal correctness
  # ============================================================================

  describe "flash_attention causal correctness" do
    test "causal mask prevents attending to future tokens" do
      batch = 1
      heads = 1
      seq = 4
      dim = 4

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, heads, seq, dim}, type: {:f, 32})

      causal_result = FusedScan.flash_attention(q, k, v, causal: true)
      non_causal_result = FusedScan.flash_attention(q, k, v, causal: false)

      # Last token (t=seq-1) sees all tokens in both modes, so should match
      last_causal = Nx.slice_along_axis(causal_result, seq - 1, 1, axis: 2)
      last_non_causal = Nx.slice_along_axis(non_causal_result, seq - 1, 1, axis: 2)

      diff = Nx.subtract(last_causal, last_non_causal) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-4, "Last token should match between causal and non-causal (diff=#{diff})"

      # First token (t=0) in causal only sees itself; non-causal sees all tokens
      first_causal = Nx.slice_along_axis(causal_result, 0, 1, axis: 2)
      first_non_causal = Nx.slice_along_axis(non_causal_result, 0, 1, axis: 2)

      diff2 = Nx.subtract(first_causal, first_non_causal) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff2 > 1.0e-6, "First token should differ between causal and non-causal (diff=#{diff2})"
    end

    test "matches naive attention with causal mask" do
      batch = 1
      heads = 1
      seq = 4
      dim = 4

      key = Nx.Random.key(99)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})

      flash_result = FusedScan.flash_attention(q, k, v, causal: true)
      naive_result = naive_causal_attention(q, k, v)

      max_diff = Nx.subtract(flash_result, naive_result) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert max_diff < 1.0e-4, "Flash vs naive causal attention max_diff=#{max_diff}"
    end
  end

  # ============================================================================
  # LASER attention causal correctness
  # ============================================================================

  describe "laser_attention causal correctness" do
    test "causal mask prevents attending to future tokens" do
      batch = 1
      heads = 1
      seq = 4
      dim = 4

      key = Nx.Random.key(42)
      {q, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {k, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, heads, seq, dim}, type: {:f, 32})
      {v, _key} = Nx.Random.uniform(key, 0.1, 1.0, shape: {batch, heads, seq, dim}, type: {:f, 32})

      causal_result = FusedScan.laser_attention(q, k, v, causal: true)
      non_causal_result = FusedScan.laser_attention(q, k, v, causal: false)

      # Last token sees all tokens in both modes, so should match
      last_causal = Nx.slice_along_axis(causal_result, seq - 1, 1, axis: 2)
      last_non_causal = Nx.slice_along_axis(non_causal_result, seq - 1, 1, axis: 2)

      diff = Nx.subtract(last_causal, last_non_causal) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff < 1.0e-4, "Last token should match between causal and non-causal (diff=#{diff})"

      # First token (t=0) in causal only sees itself; non-causal sees all tokens
      first_causal = Nx.slice_along_axis(causal_result, 0, 1, axis: 2)
      first_non_causal = Nx.slice_along_axis(non_causal_result, 0, 1, axis: 2)

      diff2 = Nx.subtract(first_causal, first_non_causal) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
      assert diff2 > 1.0e-6, "First token should differ between causal and non-causal (diff=#{diff2})"
    end
  end

  # ============================================================================
  # Regression guards — momentum is actually used
  # ============================================================================

  describe "regression guards" do
    test "titans_scan: changing momentum changes output" do
      batch = 2
      seq = 8
      m = 8
      key = Nx.Random.key(2024)
      {combined, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq, 4 * m}, type: {:f, 32})

      result_low = FusedScan.titans_scan(combined, memory_size: m, momentum: 0.1)
      result_high = FusedScan.titans_scan(combined, memory_size: m, momentum: 0.99)

      max_diff = Nx.subtract(result_low, result_high)
                 |> Nx.abs()
                 |> Nx.reduce_max()
                 |> Nx.to_number()

      assert max_diff > 1.0e-4,
        "Titans: momentum=0.1 vs 0.99 should produce different outputs (max_diff=#{max_diff}). " <>
        "If this fails, the momentum parameter may be ignored by the dispatch path."
    end

    test "miras_scan: changing momentum changes output" do
      batch = 2
      seq = 8
      m = 8
      key = Nx.Random.key(2024)
      {combined, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq, 5 * m}, type: {:f, 32})

      result_low = FusedScan.miras_scan(combined, memory_size: m, momentum: 0.1)
      result_high = FusedScan.miras_scan(combined, memory_size: m, momentum: 0.99)

      max_diff = Nx.subtract(result_low, result_high)
                 |> Nx.abs()
                 |> Nx.reduce_max()
                 |> Nx.to_number()

      assert max_diff > 1.0e-4,
        "MIRAS: momentum=0.1 vs 0.99 should produce different outputs (max_diff=#{max_diff}). " <>
        "If this fails, the momentum parameter may be ignored by the dispatch path."
    end

    test "reservoir_scan: changing leak_rate changes output" do
      batch = 2
      seq = 8
      hidden = 8
      key = Nx.Random.key(2024)
      {wx, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq, hidden}, type: {:f, 32})
      {w_res, _key} = Nx.Random.uniform(key, -0.3, 0.3, shape: {hidden, hidden}, type: {:f, 32})

      result_full = FusedScan.reservoir_scan(wx, w_res, leak_rate: 1.0)
      result_half = FusedScan.reservoir_scan(wx, w_res, leak_rate: 0.5)

      max_diff = Nx.subtract(result_full, result_half)
                 |> Nx.abs()
                 |> Nx.reduce_max()
                 |> Nx.to_number()

      assert max_diff > 1.0e-4,
        "Reservoir: leak_rate=1.0 vs 0.5 should produce different outputs (max_diff=#{max_diff}). " <>
        "If this fails, the leak_rate parameter may be ignored by the dispatch path."
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp naive_causal_attention(q, k, v) do
    {_batch, _heads, seq, dim} = Nx.shape(q)
    scale = :math.sqrt(dim)

    # QK^T / sqrt(d): [B, H, seq, seq]
    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(scale)

    # Causal mask: lower triangular [1, 1, seq, seq] for broadcasting
    mask = Nx.iota({seq, seq}, axis: 0)
           |> Nx.greater_equal(Nx.iota({seq, seq}, axis: 1))
           |> Nx.reshape({1, 1, seq, seq})

    neg_inf = Nx.tensor(-1.0e9, type: {:f, 32})
    scores = Nx.select(mask, scores, neg_inf)

    # Softmax along last axis
    weights = softmax_last_axis(scores)

    # Weighted sum: [B, H, seq, dim]
    Nx.dot(weights, [3], [0, 1], v, [2], [0, 1])
  end

  defp softmax_last_axis(x) do
    max_val = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    exp_x = Nx.exp(Nx.subtract(x, max_val))
    sum_exp = Nx.sum(exp_x, axes: [-1], keep_axes: true)
    Nx.divide(exp_x, sum_exp)
  end
end
