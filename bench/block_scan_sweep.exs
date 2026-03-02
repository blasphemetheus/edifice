# Block Scan Multi-Layer Fusion Benchmark
#
# Compares single-layer sequential dispatch vs fused multi-layer block scan
# kernels. Both paths use the Elixir fallback (BinaryBackend) for portability.
#
# Measures wall-clock time for:
#   - Sequential: N separate kernel calls with global memory round-trips
#   - Block: 1 fused kernel call keeping state in registers/shared memory
#
# Usage:
#   nix-shell --run "mix run bench/block_scan_sweep.exs" shell.nix
#
# Environment variables:
#   BENCH_BATCH     - Batch size (default: 2)
#   BENCH_SEQ_LEN   - Sequence length (default: 32)
#   BENCH_HIDDEN    - Hidden dimension (default: 16)
#   BENCH_LAYERS    - Number of layers (default: 4)

Logger.configure(level: :warning)

defmodule BlockScanSweep do
  @moduledoc false

  alias Edifice.CUDA.FusedScan

  @batch String.to_integer(System.get_env("BENCH_BATCH", "2"))
  @seq_len String.to_integer(System.get_env("BENCH_SEQ_LEN", "32"))
  @hidden String.to_integer(System.get_env("BENCH_HIDDEN", "16"))
  @num_layers String.to_integer(System.get_env("BENCH_LAYERS", "4"))

  @warmup 3
  @iters 20

  def run do
    IO.puts("Block Scan Multi-Layer Fusion Benchmark")
    IO.puts("========================================")
    IO.puts("B=#{@batch}, T=#{@seq_len}, H=#{@hidden}, layers=#{@num_layers}")
    IO.puts("")

    bench_linear_block()
    bench_lstm_block()
    bench_gru_block()

    IO.puts("\nDone.")
  end

  # ── Linear Block ─────────────────────────────────────────────

  defp bench_linear_block do
    IO.puts("--- Linear Block Scan (h = a*h + b) ---")

    key = Nx.Random.key(42)
    {input, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch, @seq_len, @hidden}, type: {:f, 32})
    {h0, key} = Nx.Random.uniform(key, 0.0, 0.0, shape: {@batch, @num_layers, @hidden}, type: {:f, 32})
    weights = build_linear_weights(key, @hidden, @num_layers)

    # Block path (fallback — all layers fused in Elixir loop)
    block_us = bench(fn -> FusedScan.linear_block(input, weights, h0, @num_layers) end)

    # Sequential path (same logic, done manually layer by layer)
    seq_us = bench(fn -> sequential_linear(input, weights, h0, @hidden, @num_layers) end)

    report("linear_block", block_us, seq_us)
  end

  # ── LSTM Block ───────────────────────────────────────────────

  defp bench_lstm_block do
    IO.puts("--- LSTM Block Scan (4-gate + R@h) ---")

    key = Nx.Random.key(99)
    {input, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch, @seq_len, @hidden}, type: {:f, 32})
    {h0, key} = Nx.Random.uniform(key, 0.0, 0.0, shape: {@batch, @num_layers, @hidden}, type: {:f, 32})
    {c0, key} = Nx.Random.uniform(key, 0.0, 0.0, shape: {@batch, @num_layers, @hidden}, type: {:f, 32})
    weights = build_lstm_weights(key, @hidden, @num_layers)

    block_us = bench(fn -> FusedScan.lstm_block(input, weights, h0, c0, @num_layers) end)
    seq_us = bench(fn -> sequential_lstm(input, weights, h0, c0, @hidden, @num_layers) end)

    report("lstm_block", block_us, seq_us)
  end

  # ── GRU Block ────────────────────────────────────────────────

  defp bench_gru_block do
    IO.puts("--- GRU Block Scan (3-gate + R@h) ---")

    key = Nx.Random.key(55)
    {input, key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {@batch, @seq_len, @hidden}, type: {:f, 32})
    {h0, key} = Nx.Random.uniform(key, 0.0, 0.0, shape: {@batch, @num_layers, @hidden}, type: {:f, 32})
    weights = build_gru_weights(key, @hidden, @num_layers)

    block_us = bench(fn -> FusedScan.gru_block(input, weights, h0, @num_layers) end)
    seq_us = bench(fn -> sequential_gru(input, weights, h0, @hidden, @num_layers) end)

    report("gru_block", block_us, seq_us)
  end

  # ── Weight builders ──────────────────────────────────────────

  defp build_linear_weights(key, h, n) do
    {all, _} = Enum.map_reduce(1..n, key, fn _, k ->
      {w_a, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {h * h}, type: {:f, 32})
      {b_a, k} = Nx.Random.uniform(k, -0.01, 0.01, shape: {h}, type: {:f, 32})
      {w_b, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {h * h}, type: {:f, 32})
      {b_b, k} = Nx.Random.uniform(k, -0.01, 0.01, shape: {h}, type: {:f, 32})
      g = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {h})
      b = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {h})
      {Nx.concatenate([w_a, b_a, w_b, b_b, g, b]), k}
    end)
    Nx.concatenate(all)
  end

  defp build_lstm_weights(key, h, n) do
    {all, _} = Enum.map_reduce(1..n, key, fn _, k ->
      {w_x, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {h * 4 * h}, type: {:f, 32})
      {b_x, k} = Nx.Random.uniform(k, -0.01, 0.01, shape: {4 * h}, type: {:f, 32})
      {r_w, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {h * 4 * h}, type: {:f, 32})
      g = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {h})
      b = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {h})
      {Nx.concatenate([w_x, b_x, r_w, g, b]), k}
    end)
    Nx.concatenate(all)
  end

  defp build_gru_weights(key, h, n) do
    {all, _} = Enum.map_reduce(1..n, key, fn _, k ->
      {w_x, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {h * 3 * h}, type: {:f, 32})
      {b_x, k} = Nx.Random.uniform(k, -0.01, 0.01, shape: {3 * h}, type: {:f, 32})
      {r_w, k} = Nx.Random.uniform(k, -0.1, 0.1, shape: {h * 3 * h}, type: {:f, 32})
      g = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {h})
      b = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {h})
      {Nx.concatenate([w_x, b_x, r_w, g, b]), k}
    end)
    Nx.concatenate(all)
  end

  # ── Sequential reference implementations ─────────────────────

  defp sequential_linear(input, weights, h0, hidden, num_layers) do
    layer_stride = 2 * hidden * hidden + 4 * hidden
    Enum.reduce(0..(num_layers - 1), input, fn layer, x ->
      offset = layer * layer_stride
      w_a = Nx.slice(weights, [offset], [hidden * hidden]) |> Nx.reshape({hidden, hidden})
      b_a = Nx.slice(weights, [offset + hidden * hidden], [hidden])
      w_b = Nx.slice(weights, [offset + hidden * hidden + hidden], [hidden * hidden]) |> Nx.reshape({hidden, hidden})
      b_b = Nx.slice(weights, [offset + 2 * hidden * hidden + hidden], [hidden])
      gamma = Nx.slice(weights, [offset + 2 * hidden * hidden + 2 * hidden], [hidden])
      beta = Nx.slice(weights, [offset + 2 * hidden * hidden + 3 * hidden], [hidden])

      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)
      normed = Nx.multiply(Nx.multiply(Nx.subtract(x, mean), Nx.rsqrt(Nx.add(var, 1.0e-5))), gamma) |> Nx.add(beta)

      a_pre = Nx.add(Nx.dot(normed, [2], w_a, [0]), b_a)
      b_pre = Nx.add(Nx.dot(normed, [2], w_b, [0]), b_b)
      h_init = Nx.slice_along_axis(h0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {_, seq_len, _} = Nx.shape(a_pre)
      {_, h_list} = Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
        a_t = Nx.slice_along_axis(a_pre, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        b_t = Nx.slice_along_axis(b_pre, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        h_t = Nx.add(Nx.multiply(a_t, h_prev), b_t)
        {h_t, [h_t | acc]}
      end)
      Nx.add(x, h_list |> Enum.reverse() |> Nx.stack(axis: 1))
    end)
  end

  defp sequential_lstm(input, weights, h0, c0, hidden, num_layers) do
    layer_stride = 8 * hidden * hidden + 6 * hidden
    Enum.reduce(0..(num_layers - 1), input, fn layer, x ->
      offset = layer * layer_stride
      w_x = Nx.slice(weights, [offset], [hidden * 4 * hidden]) |> Nx.reshape({hidden, 4 * hidden})
      b_x = Nx.slice(weights, [offset + hidden * 4 * hidden], [4 * hidden])
      r_w = Nx.slice(weights, [offset + hidden * 4 * hidden + 4 * hidden], [hidden * 4 * hidden]) |> Nx.reshape({hidden, 4 * hidden})
      gamma = Nx.slice(weights, [offset + 8 * hidden * hidden + 4 * hidden], [hidden])
      beta = Nx.slice(weights, [offset + 8 * hidden * hidden + 5 * hidden], [hidden])

      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)
      normed = Nx.multiply(Nx.multiply(Nx.subtract(x, mean), Nx.rsqrt(Nx.add(var, 1.0e-5))), gamma) |> Nx.add(beta)

      wx_proj = Nx.add(Nx.dot(normed, [2], w_x, [0]), b_x)
      wx_i = Nx.slice_along_axis(wx_proj, 0, hidden, axis: 2)
      wx_f = Nx.slice_along_axis(wx_proj, hidden, hidden, axis: 2)
      wx_g = Nx.slice_along_axis(wx_proj, 2 * hidden, hidden, axis: 2)
      wx_o = Nx.slice_along_axis(wx_proj, 3 * hidden, hidden, axis: 2)

      h_init = Nx.slice_along_axis(h0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])
      c_init = Nx.slice_along_axis(c0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {_, seq_len, _} = Nx.shape(wx_i)
      {_, h_list} = Enum.reduce(0..(seq_len - 1), {{h_init, c_init}, []}, fn t, {{hp, cp}, acc} ->
        rh = Nx.dot(hp, [1], r_w, [0])
        wi_t = Nx.slice_along_axis(wx_i, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        wf_t = Nx.slice_along_axis(wx_f, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        wg_t = Nx.slice_along_axis(wx_g, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        wo_t = Nx.slice_along_axis(wx_o, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        ig = Nx.sigmoid(Nx.add(wi_t, Nx.slice_along_axis(rh, 0, hidden, axis: 1)))
        fg = Nx.sigmoid(Nx.add(wf_t, Nx.slice_along_axis(rh, hidden, hidden, axis: 1)))
        gg = Nx.tanh(Nx.add(wg_t, Nx.slice_along_axis(rh, 2 * hidden, hidden, axis: 1)))
        og = Nx.sigmoid(Nx.add(wo_t, Nx.slice_along_axis(rh, 3 * hidden, hidden, axis: 1)))

        ct = Nx.add(Nx.multiply(fg, cp), Nx.multiply(ig, gg))
        ht = Nx.multiply(og, Nx.tanh(ct))
        {{ht, ct}, [ht | acc]}
      end)
      Nx.add(x, h_list |> Enum.reverse() |> Nx.stack(axis: 1))
    end)
  end

  defp sequential_gru(input, weights, h0, hidden, num_layers) do
    layer_stride = 6 * hidden * hidden + 5 * hidden
    Enum.reduce(0..(num_layers - 1), input, fn layer, x ->
      offset = layer * layer_stride
      w_x = Nx.slice(weights, [offset], [hidden * 3 * hidden]) |> Nx.reshape({hidden, 3 * hidden})
      b_x = Nx.slice(weights, [offset + hidden * 3 * hidden], [3 * hidden])
      r_w = Nx.slice(weights, [offset + hidden * 3 * hidden + 3 * hidden], [hidden * 3 * hidden]) |> Nx.reshape({hidden, 3 * hidden})
      gamma = Nx.slice(weights, [offset + 6 * hidden * hidden + 3 * hidden], [hidden])
      beta = Nx.slice(weights, [offset + 6 * hidden * hidden + 4 * hidden], [hidden])

      mean = Nx.mean(x, axes: [-1], keep_axes: true)
      var = Nx.variance(x, axes: [-1], keep_axes: true)
      normed = Nx.multiply(Nx.multiply(Nx.subtract(x, mean), Nx.rsqrt(Nx.add(var, 1.0e-5))), gamma) |> Nx.add(beta)

      wx_proj = Nx.add(Nx.dot(normed, [2], w_x, [0]), b_x)
      wx_r = Nx.slice_along_axis(wx_proj, 0, hidden, axis: 2)
      wx_z = Nx.slice_along_axis(wx_proj, hidden, hidden, axis: 2)
      wx_n = Nx.slice_along_axis(wx_proj, 2 * hidden, hidden, axis: 2)

      h_init = Nx.slice_along_axis(h0, layer, 1, axis: 1) |> Nx.squeeze(axes: [1])

      {_, seq_len, _} = Nx.shape(wx_r)
      {_, h_list} = Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {hp, acc} ->
        rh = Nx.dot(hp, [1], r_w, [0])
        wr_t = Nx.slice_along_axis(wx_r, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        wz_t = Nx.slice_along_axis(wx_z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        wn_t = Nx.slice_along_axis(wx_n, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        rg = Nx.sigmoid(Nx.add(wr_t, Nx.slice_along_axis(rh, 0, hidden, axis: 1)))
        zg = Nx.sigmoid(Nx.add(wz_t, Nx.slice_along_axis(rh, hidden, hidden, axis: 1)))
        ng = Nx.tanh(Nx.add(wn_t, Nx.multiply(rg, Nx.slice_along_axis(rh, 2 * hidden, hidden, axis: 1))))

        ht = Nx.add(Nx.multiply(Nx.subtract(1.0, zg), hp), Nx.multiply(zg, ng))
        {ht, [ht | acc]}
      end)
      Nx.add(x, h_list |> Enum.reverse() |> Nx.stack(axis: 1))
    end)
  end

  # ── Timing helpers ───────────────────────────────────────────

  defp bench(fun) do
    # Warmup
    for _ <- 1..@warmup, do: fun.()

    # Timed iterations
    times =
      for _ <- 1..@iters do
        {us, _result} = :timer.tc(fun)
        us
      end

    Enum.sum(times) / length(times)
  end

  defp report(name, block_us, seq_us) do
    ratio = seq_us / max(block_us, 1.0)
    IO.puts("  #{name}:  block=#{Float.round(block_us, 0)}us  sequential=#{Float.round(seq_us, 0)}us  ratio=#{Float.round(ratio, 2)}x")
    IO.puts("")
  end
end

BlockScanSweep.run()
