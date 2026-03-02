defmodule Edifice.CUDA.BlockScanTest do
  @moduledoc """
  Numerical validation tests for multi-layer block scan kernels.

  Tests run on BinaryBackend (no GPU required) — they validate the Elixir
  fallback paths by comparing block scan output against sequential single-layer
  computation.
  """
  use ExUnit.Case, async: true

  alias Edifice.CUDA.FusedScan

  # ============================================================================
  # Linear block scan tests
  # ============================================================================

  describe "linear_block_fallback" do
    test "matches sequential single-layer linear scan for 2 layers" do
      batch = 2
      seq_len = 4
      hidden = 8
      num_layers = 2

      key = Nx.Random.key(42)
      {input, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {h0, key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {batch, num_layers, hidden}, type: {:f, 32})

      # Build packed weights for each layer
      weights = build_linear_block_weights(key, hidden, num_layers)

      # Block scan (all layers fused)
      block_out = FusedScan.linear_block(input, weights, h0, num_layers)

      # Sequential: run single-layer scans one at a time
      seq_out = sequential_linear_scan(input, weights, h0, hidden, num_layers)

      assert_all_close(block_out, seq_out, atol: 1.0e-5)
    end

    test "matches for 4 layers" do
      batch = 1
      seq_len = 8
      hidden = 4
      num_layers = 4

      key = Nx.Random.key(123)
      {input, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {h0, key} = Nx.Random.uniform(key, 0.0, 0.0, shape: {batch, num_layers, hidden}, type: {:f, 32})

      weights = build_linear_block_weights(key, hidden, num_layers)
      block_out = FusedScan.linear_block(input, weights, h0, num_layers)
      seq_out = sequential_linear_scan(input, weights, h0, hidden, num_layers)

      assert_all_close(block_out, seq_out, atol: 1.0e-5)
    end

    test "single layer matches direct computation" do
      batch = 2
      seq_len = 3
      hidden = 4
      num_layers = 1

      key = Nx.Random.key(77)
      {input, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})
      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_layers, hidden})

      weights = build_linear_block_weights(key, hidden, num_layers)
      block_out = FusedScan.linear_block(input, weights, h0, num_layers)
      seq_out = sequential_linear_scan(input, weights, h0, hidden, num_layers)

      assert_all_close(block_out, seq_out, atol: 1.0e-5)
    end
  end

  # ============================================================================
  # LSTM block scan tests
  # ============================================================================

  describe "lstm_block_fallback" do
    test "matches sequential single-layer LSTM for 2 layers" do
      batch = 2
      seq_len = 4
      hidden = 8
      num_layers = 2

      key = Nx.Random.key(42)
      {input, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {h0, key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {batch, num_layers, hidden}, type: {:f, 32})
      {c0, key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {batch, num_layers, hidden}, type: {:f, 32})

      weights = build_lstm_block_weights(key, hidden, num_layers)
      block_out = FusedScan.lstm_block(input, weights, h0, c0, num_layers)
      seq_out = sequential_lstm_scan(input, weights, h0, c0, hidden, num_layers)

      assert_all_close(block_out, seq_out, atol: 1.0e-5)
    end

    test "matches for 3 layers" do
      batch = 1
      seq_len = 6
      hidden = 4
      num_layers = 3

      key = Nx.Random.key(99)
      {input, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, hidden}, type: {:f, 32})
      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_layers, hidden})
      c0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_layers, hidden})

      weights = build_lstm_block_weights(key, hidden, num_layers)
      block_out = FusedScan.lstm_block(input, weights, h0, c0, num_layers)
      seq_out = sequential_lstm_scan(input, weights, h0, c0, hidden, num_layers)

      assert_all_close(block_out, seq_out, atol: 1.0e-5)
    end
  end

  # ============================================================================
  # GRU block scan tests
  # ============================================================================

  describe "gru_block_fallback" do
    test "matches sequential single-layer GRU for 2 layers" do
      batch = 2
      seq_len = 4
      hidden = 8
      num_layers = 2

      key = Nx.Random.key(42)
      {input, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {h0, key} = Nx.Random.uniform(key, -0.1, 0.1, shape: {batch, num_layers, hidden}, type: {:f, 32})

      weights = build_gru_block_weights(key, hidden, num_layers)
      block_out = FusedScan.gru_block(input, weights, h0, num_layers)
      seq_out = sequential_gru_scan(input, weights, h0, hidden, num_layers)

      assert_all_close(block_out, seq_out, atol: 1.0e-5)
    end

    test "matches for 3 layers" do
      batch = 1
      seq_len = 6
      hidden = 4
      num_layers = 3

      key = Nx.Random.key(55)
      {input, key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, hidden}, type: {:f, 32})
      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, num_layers, hidden})

      weights = build_gru_block_weights(key, hidden, num_layers)
      block_out = FusedScan.gru_block(input, weights, h0, num_layers)
      seq_out = sequential_gru_scan(input, weights, h0, hidden, num_layers)

      assert_all_close(block_out, seq_out, atol: 1.0e-5)
    end
  end

  # ============================================================================
  # Helpers — weight builders
  # ============================================================================

  # Build packed linear block weights: [W_a(H,H) | b_a(H) | W_b(H,H) | b_b(H) | gamma(H) | beta(H)] per layer
  defp build_linear_block_weights(key, hidden, num_layers) do
    layer_stride = 2 * hidden * hidden + 4 * hidden

    {all_weights, _key} =
      Enum.map_reduce(0..(num_layers - 1), key, fn _layer, k ->
        {w_a, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden * hidden}, type: {:f, 32})
        {b_a, k} = Nx.Random.uniform(k, -0.01, 0.01, shape: {hidden}, type: {:f, 32})
        {w_b, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden * hidden}, type: {:f, 32})
        {b_b, k} = Nx.Random.uniform(k, -0.01, 0.01, shape: {hidden}, type: {:f, 32})
        gamma = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {hidden})
        beta = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {hidden})

        layer_w = Nx.concatenate([w_a, b_a, w_b, b_b, gamma, beta])
        {layer_w, k}
      end)

    packed = Nx.concatenate(all_weights)
    ^layer_stride = div(Nx.size(packed), num_layers)
    packed
  end

  # Build packed LSTM block weights: [W_x(H,4H) | b_x(4H) | R(H,4H) | gamma(H) | beta(H)] per layer
  defp build_lstm_block_weights(key, hidden, num_layers) do
    layer_stride = 8 * hidden * hidden + 6 * hidden

    {all_weights, _key} =
      Enum.map_reduce(0..(num_layers - 1), key, fn _layer, k ->
        {w_x, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden * 4 * hidden}, type: {:f, 32})
        {b_x, k} = Nx.Random.uniform(k, -0.01, 0.01, shape: {4 * hidden}, type: {:f, 32})
        {r_w, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden * 4 * hidden}, type: {:f, 32})
        gamma = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {hidden})
        beta = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {hidden})

        layer_w = Nx.concatenate([w_x, b_x, r_w, gamma, beta])
        {layer_w, k}
      end)

    packed = Nx.concatenate(all_weights)
    ^layer_stride = div(Nx.size(packed), num_layers)
    packed
  end

  # Build packed GRU block weights: [W_x(H,3H) | b_x(3H) | R(H,3H) | gamma(H) | beta(H)] per layer
  defp build_gru_block_weights(key, hidden, num_layers) do
    layer_stride = 6 * hidden * hidden + 5 * hidden

    {all_weights, _key} =
      Enum.map_reduce(0..(num_layers - 1), key, fn _layer, k ->
        {w_x, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden * 3 * hidden}, type: {:f, 32})
        {b_x, k} = Nx.Random.uniform(k, -0.01, 0.01, shape: {3 * hidden}, type: {:f, 32})
        {r_w, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {hidden * 3 * hidden}, type: {:f, 32})
        gamma = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {hidden})
        beta = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {hidden})

        layer_w = Nx.concatenate([w_x, b_x, r_w, gamma, beta])
        {layer_w, k}
      end)

    packed = Nx.concatenate(all_weights)
    ^layer_stride = div(Nx.size(packed), num_layers)
    packed
  end

  # ============================================================================
  # Helpers — sequential single-layer reference implementations
  # ============================================================================

  # Run linear scan layers one at a time (reference for block scan validation)
  defp sequential_linear_scan(input, weights, h0, hidden, num_layers) do
    layer_stride = 2 * hidden * hidden + 4 * hidden

    Enum.reduce(0..(num_layers - 1), input, fn layer, x ->
      offset = layer * layer_stride

      w_a = Nx.slice(weights, [offset], [hidden * hidden]) |> Nx.reshape({hidden, hidden})
      offset = offset + hidden * hidden
      b_a = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      w_b = Nx.slice(weights, [offset], [hidden * hidden]) |> Nx.reshape({hidden, hidden})
      offset = offset + hidden * hidden
      b_b = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      gamma = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      beta = Nx.slice(weights, [offset], [hidden])

      # LayerNorm
      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)
      normed = Nx.multiply(Nx.multiply(Nx.subtract(x, mean), Nx.rsqrt(Nx.add(var, 1.0e-5))), gamma)
      normed = Nx.add(normed, beta)

      a_pre = Nx.add(Nx.dot(normed, [2], w_a, [0]), b_a)
      b_pre = Nx.add(Nx.dot(normed, [2], w_b, [0]), b_b)

      h_init = Nx.slice_along_axis(h0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {_batch, seq_len, _hidden} = Nx.shape(a_pre)
      {_, h_list} =
        Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
          a_t = Nx.slice_along_axis(a_pre, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          b_t = Nx.slice_along_axis(b_pre, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          h_t = Nx.add(Nx.multiply(a_t, h_prev), b_t)
          {h_t, [h_t | acc]}
        end)
      scan_out = h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      Nx.add(x, scan_out)
    end)
  end

  # Run LSTM layers one at a time
  defp sequential_lstm_scan(input, weights, h0, c0, hidden, num_layers) do
    layer_stride = 8 * hidden * hidden + 6 * hidden

    Enum.reduce(0..(num_layers - 1), input, fn layer, x ->
      offset = layer * layer_stride

      w_x = Nx.slice(weights, [offset], [hidden * 4 * hidden]) |> Nx.reshape({hidden, 4 * hidden})
      offset = offset + hidden * 4 * hidden
      b_x = Nx.slice(weights, [offset], [4 * hidden])
      offset = offset + 4 * hidden
      r_w = Nx.slice(weights, [offset], [hidden * 4 * hidden]) |> Nx.reshape({hidden, 4 * hidden})
      offset = offset + hidden * 4 * hidden
      gamma = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      beta = Nx.slice(weights, [offset], [hidden])

      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)
      normed = Nx.multiply(Nx.multiply(Nx.subtract(x, mean), Nx.rsqrt(Nx.add(var, 1.0e-5))), gamma)
      normed = Nx.add(normed, beta)

      wx_proj = Nx.add(Nx.dot(normed, [2], w_x, [0]), b_x)
      wx_i = Nx.slice_along_axis(wx_proj, 0, hidden, axis: 2)
      wx_f = Nx.slice_along_axis(wx_proj, hidden, hidden, axis: 2)
      wx_g = Nx.slice_along_axis(wx_proj, 2 * hidden, hidden, axis: 2)
      wx_o = Nx.slice_along_axis(wx_proj, 3 * hidden, hidden, axis: 2)

      h_init = Nx.slice_along_axis(h0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])
      c_init = Nx.slice_along_axis(c0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {_batch, seq_len, _hidden} = Nx.shape(wx_i)
      {_, h_list} =
        Enum.reduce(0..(seq_len - 1), {{h_init, c_init}, []}, fn t, {{h_prev, c_prev}, acc} ->
          rh = Nx.dot(h_prev, [1], r_w, [0])
          rh_i = Nx.slice_along_axis(rh, 0, hidden, axis: 1)
          rh_f = Nx.slice_along_axis(rh, hidden, hidden, axis: 1)
          rh_g = Nx.slice_along_axis(rh, 2 * hidden, hidden, axis: 1)
          rh_o = Nx.slice_along_axis(rh, 3 * hidden, hidden, axis: 1)

          wi_t = Nx.slice_along_axis(wx_i, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          wf_t = Nx.slice_along_axis(wx_f, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          wg_t = Nx.slice_along_axis(wx_g, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          wo_t = Nx.slice_along_axis(wx_o, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

          i_gate = Nx.sigmoid(Nx.add(wi_t, rh_i))
          f_gate = Nx.sigmoid(Nx.add(wf_t, rh_f))
          g_gate = Nx.tanh(Nx.add(wg_t, rh_g))
          o_gate = Nx.sigmoid(Nx.add(wo_t, rh_o))

          c_t = Nx.add(Nx.multiply(f_gate, c_prev), Nx.multiply(i_gate, g_gate))
          h_t = Nx.multiply(o_gate, Nx.tanh(c_t))
          {{h_t, c_t}, [h_t | acc]}
        end)
      scan_out = h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      Nx.add(x, scan_out)
    end)
  end

  # Run GRU layers one at a time
  defp sequential_gru_scan(input, weights, h0, hidden, num_layers) do
    layer_stride = 6 * hidden * hidden + 5 * hidden

    Enum.reduce(0..(num_layers - 1), input, fn layer, x ->
      offset = layer * layer_stride

      w_x = Nx.slice(weights, [offset], [hidden * 3 * hidden]) |> Nx.reshape({hidden, 3 * hidden})
      offset = offset + hidden * 3 * hidden
      b_x = Nx.slice(weights, [offset], [3 * hidden])
      offset = offset + 3 * hidden
      r_w = Nx.slice(weights, [offset], [hidden * 3 * hidden]) |> Nx.reshape({hidden, 3 * hidden})
      offset = offset + hidden * 3 * hidden
      gamma = Nx.slice(weights, [offset], [hidden])
      offset = offset + hidden
      beta = Nx.slice(weights, [offset], [hidden])

      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)
      normed = Nx.multiply(Nx.multiply(Nx.subtract(x, mean), Nx.rsqrt(Nx.add(var, 1.0e-5))), gamma)
      normed = Nx.add(normed, beta)

      wx_proj = Nx.add(Nx.dot(normed, [2], w_x, [0]), b_x)
      wx_r = Nx.slice_along_axis(wx_proj, 0, hidden, axis: 2)
      wx_z = Nx.slice_along_axis(wx_proj, hidden, hidden, axis: 2)
      wx_n = Nx.slice_along_axis(wx_proj, 2 * hidden, hidden, axis: 2)

      h_init = Nx.slice_along_axis(h0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {_batch, seq_len, _hidden} = Nx.shape(wx_r)
      {_, h_list} =
        Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
          rh = Nx.dot(h_prev, [1], r_w, [0])
          rh_r = Nx.slice_along_axis(rh, 0, hidden, axis: 1)
          rh_z = Nx.slice_along_axis(rh, hidden, hidden, axis: 1)
          rh_n = Nx.slice_along_axis(rh, 2 * hidden, hidden, axis: 1)

          wr_t = Nx.slice_along_axis(wx_r, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          wz_t = Nx.slice_along_axis(wx_z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
          wn_t = Nx.slice_along_axis(wx_n, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

          r_gate = Nx.sigmoid(Nx.add(wr_t, rh_r))
          z_gate = Nx.sigmoid(Nx.add(wz_t, rh_z))
          n_gate = Nx.tanh(Nx.add(wn_t, Nx.multiply(r_gate, rh_n)))

          h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_gate), h_prev), Nx.multiply(z_gate, n_gate))
          {h_t, [h_t | acc]}
        end)
      scan_out = h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      Nx.add(x, scan_out)
    end)
  end

  # ============================================================================
  # Helpers — assertion
  # ============================================================================

  defp assert_all_close(left, right, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-5)

    left_flat = Nx.to_flat_list(left)
    right_flat = Nx.to_flat_list(right)

    Enum.zip(left_flat, right_flat)
    |> Enum.each(fn {l, r} ->
      diff = abs(l - r)

      assert diff < atol,
             "Values not close: #{l} vs #{r} (diff=#{diff}, atol=#{atol})"
    end)
  end
end
