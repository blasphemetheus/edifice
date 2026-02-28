# Fused CUDA Scan vs Elixir Sequential Scan — Latency Benchmark
#
# Compares the fused CUDA kernel path against the Elixir Enum.reduce
# scan for MinGRU and MinLSTM at various sequence lengths and hidden sizes.
#
# Usage:
#   nix-shell --run "mix run bench/fused_scan_latency.exs" shell.nix
#
# The benchmark measures:
#   1. Raw scan function latency (gates/candidates already computed)
#   2. Full Axon model forward pass latency (end-to-end)

Nx.default_backend(EXLA.Backend)

defmodule FusedScanBench do
  @moduledoc false

  @warmup_iters 10
  @bench_iters 100

  # ── Helpers ──────────────────────────────────────────────────────

  defp rand(shape) do
    key = Nx.Random.key(:rand.uniform(100_000))
    {t, _} = Nx.Random.uniform(key, -2.0, 2.0, shape: shape)
    t
  end

  defp measure(label, fun) do
    # Warmup
    for _ <- 1..@warmup_iters, do: fun.()

    # Timed iterations
    times =
      for _ <- 1..@bench_iters do
        {us, _} = :timer.tc(fun)
        us
      end

    times = Enum.sort(times)
    n = length(times)
    min_us = hd(times)
    max_us = List.last(times)
    mean_us = Enum.sum(times) / n
    median_us = Enum.at(times, div(n, 2))
    p99_us = Enum.at(times, round(n * 0.99) - 1)

    IO.puts(
      "  #{String.pad_trailing(label, 35)}" <>
        "  min=#{pad_ms(min_us)}" <>
        "  med=#{pad_ms(median_us)}" <>
        "  mean=#{pad_ms(mean_us)}" <>
        "  p99=#{pad_ms(p99_us)}" <>
        "  max=#{pad_ms(max_us)}"
    )

    {label, median_us}
  end

  defp pad_ms(us) do
    ms = us / 1000
    String.pad_leading(fmt(ms), 10)
  end

  defp fmt(ms) when ms < 0.01, do: "#{Float.round(ms * 1000, 2)} us"
  defp fmt(ms) when ms < 1, do: "#{Float.round(ms, 3)} ms"
  defp fmt(ms) when ms < 100, do: "#{Float.round(ms, 2)} ms"
  defp fmt(ms), do: "#{Float.round(ms, 0)} ms"

  # ── Raw scan benchmarks ─────────────────────────────────────────

  def bench_raw_scan(batch, seq_len, hidden) do
    IO.puts("\n  Config: batch=#{batch}, seq_len=#{seq_len}, hidden=#{hidden}")
    IO.puts("  " <> String.duplicate("-", 100))

    gates = rand({batch, seq_len, hidden})
    candidates = rand({batch, seq_len, hidden})
    forget_gates = rand({batch, seq_len, hidden})
    input_gates = rand({batch, seq_len, hidden})

    # MinGRU
    {_, elixir_gru} =
      measure("MinGRU Elixir scan", fn ->
        Edifice.Recurrent.MinGRU.min_gru_scan(gates, candidates)
      end)

    {_, fused_gru} =
      measure("MinGRU CUDA fused", fn ->
        Edifice.CUDA.FusedScan.mingru(gates, candidates)
      end)

    speedup_gru = elixir_gru / max(fused_gru, 1)
    IO.puts("  MinGRU speedup: #{Float.round(speedup_gru, 1)}x")
    IO.puts("")

    # MinLSTM
    {_, elixir_lstm} =
      measure("MinLSTM Elixir scan", fn ->
        Edifice.Recurrent.MinLSTM.min_lstm_scan(forget_gates, input_gates, candidates)
      end)

    {_, fused_lstm} =
      measure("MinLSTM CUDA fused", fn ->
        Edifice.CUDA.FusedScan.minlstm(forget_gates, input_gates, candidates)
      end)

    speedup_lstm = elixir_lstm / max(fused_lstm, 1)
    IO.puts("  MinLSTM speedup: #{Float.round(speedup_lstm, 1)}x")

    {speedup_gru, speedup_lstm}
  end

  # ── Full model benchmarks ───────────────────────────────────────

  def bench_full_model(batch, seq_len, hidden) do
    IO.puts("\n  Config: batch=#{batch}, seq_len=#{seq_len}, hidden=#{hidden}")
    IO.puts("  " <> String.duplicate("-", 100))

    input = rand({batch, seq_len, hidden})
    model_opts = [embed_dim: hidden, hidden_size: hidden, num_layers: 2, dropout: 0.0, seq_len: seq_len]

    # MinGRU
    gru_model = Edifice.Recurrent.MinGRU.build(model_opts)
    {init_fn, predict_fn} = Axon.build(gru_model)
    params = init_fn.(Nx.template({batch, seq_len, hidden}, :f32), Axon.ModelState.empty())

    # Warmup (triggers XLA compilation)
    for _ <- 1..5, do: predict_fn.(params, input)

    measure("MinGRU full model (fused)", fn ->
      predict_fn.(params, input)
    end)

    IO.puts("")

    # MinLSTM
    lstm_model = Edifice.Recurrent.MinLSTM.build(model_opts)
    {init_fn, predict_fn} = Axon.build(lstm_model)
    params = init_fn.(Nx.template({batch, seq_len, hidden}, :f32), Axon.ModelState.empty())

    for _ <- 1..5, do: predict_fn.(params, input)

    measure("MinLSTM full model (fused)", fn ->
      predict_fn.(params, input)
    end)
  end

  # ── Main ─────────────────────────────────────────────────────────

  def run do
    IO.puts(String.duplicate("=", 110))
    IO.puts("Fused CUDA Scan vs Elixir Sequential Scan — Latency Benchmark")
    IO.puts("#{@bench_iters} iterations per measurement, #{@warmup_iters} warmup")
    IO.puts(String.duplicate("=", 110))

    # GPU warmup — first EXLA operation absorbs JIT + cuDNN init
    IO.puts("\nPhase 0: GPU warmup...")
    _warmup = Nx.add(rand({2, 2}), rand({2, 2}))
    IO.puts("  Done.\n")

    # ── Raw scan: sweep sequence lengths ───────────────────────────
    IO.puts(String.duplicate("=", 110))
    IO.puts("Phase 1: Raw Scan Latency (gates/candidates pre-computed)")
    IO.puts(String.duplicate("=", 110))

    configs = [
      # {batch, seq_len, hidden}
      {1, 8, 256},
      {1, 32, 256},
      {1, 64, 256},
      {1, 128, 256},
      {4, 32, 256},
      {4, 64, 256},
      {1, 32, 512},
      {1, 32, 1024}
    ]

    results =
      for {b, t, h} <- configs do
        {sg, sl} = bench_raw_scan(b, t, h)
        {b, t, h, sg, sl}
      end

    # ── Summary table ──────────────────────────────────────────────
    IO.puts("\n" <> String.duplicate("=", 110))
    IO.puts("Phase 1 Summary: Speedup (fused vs Elixir)")
    IO.puts(String.duplicate("=", 110))

    IO.puts(
      "  #{String.pad_trailing("Config", 25)}" <>
        "#{String.pad_trailing("MinGRU", 15)}" <>
        "MinLSTM"
    )

    IO.puts("  " <> String.duplicate("-", 50))

    for {b, t, h, sg, sl} <- results do
      label = "B=#{b} T=#{t} H=#{h}"

      IO.puts(
        "  #{String.pad_trailing(label, 25)}" <>
          "#{String.pad_trailing("#{Float.round(sg, 1)}x", 15)}" <>
          "#{Float.round(sl, 1)}x"
      )
    end

    # ── Full model forward pass ────────────────────────────────────
    IO.puts("\n" <> String.duplicate("=", 110))
    IO.puts("Phase 2: Full Axon Model Forward Pass (2 layers, with fused dispatch)")
    IO.puts(String.duplicate("=", 110))

    bench_full_model(1, 32, 256)
    bench_full_model(4, 32, 256)
    bench_full_model(1, 64, 256)

    IO.puts("\n" <> String.duplicate("=", 110))
    IO.puts("Done.")
    IO.puts(String.duplicate("=", 110))
  end
end

FusedScanBench.run()
