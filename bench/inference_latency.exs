# ExPhil Backbone Inference Latency Benchmark
#
# Benchmarks all sequence architectures at ExPhil-realistic dimensions.
# Goal: identify which backbones can hit <16ms inference (60 FPS).
#
# Usage:
#   # GPU (recommended):
#   EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/inference_latency.exs
#
#   # CPU:
#   EXLA_TARGET=host mix run bench/inference_latency.exs
#
# Environment variables:
#   BENCH_EMBED    - Embedding dimension (default: 256)
#   BENCH_SEQ_LEN  - Sequence length / frame context (default: 32)
#   BENCH_BATCH    - Batch size (default: 1, real-time inference)
#   BENCH_LAYERS   - Number of layers (default: 2)
#   BENCH_TIME     - Benchee time per scenario in seconds (default: 5)
#
# Note: Benchee is a :dev dependency. Run with MIX_ENV=dev for Phase 3,
# or it will fall back to manual timing.

Nx.default_backend(EXLA.Backend)

defmodule InferenceLatency do
  @moduledoc false

  # ── Configuration (overridable via env vars) ───────────────────

  @embed String.to_integer(System.get_env("BENCH_EMBED", "256"))
  @seq_len String.to_integer(System.get_env("BENCH_SEQ_LEN", "32"))
  @batch String.to_integer(System.get_env("BENCH_BATCH", "1"))
  @num_layers String.to_integer(System.get_env("BENCH_LAYERS", "2"))
  @bench_time String.to_integer(System.get_env("BENCH_TIME", "5"))

  # Derived dims
  @hidden @embed
  @state_size min(@embed, 16)
  @head_dim max(div(@embed, 8), 8)
  @num_heads max(div(@embed, @head_dim), 1)

  @fps_target_ms 16.0

  # ── Architecture definitions ───────────────────────────────────

  @shared_opts [
    embed_dim: @embed,
    hidden_size: @hidden,
    state_size: @state_size,
    num_layers: @num_layers,
    seq_len: @seq_len,
    window_size: @seq_len,
    head_dim: @head_dim,
    num_heads: @num_heads,
    dropout: 0.0
  ]

  # {name, category, extra_opts_or_builder}
  # Category is for grouping in output: :ssm, :recurrent, :attention, :other
  def architectures do
    ssm_archs() ++ recurrent_archs() ++ attention_archs() ++ other_archs()
  end

  defp ssm_archs do
    standard =
      for arch <- [
            :mamba,
            :mamba_ssd,
            :mamba_cumsum,
            :mamba_hillis_steele,
            :s4,
            :s4d,
            :s5,
            :h3,
            :hyena,
            :bimamba,
            :gated_ssm
          ] do
        {arch, :ssm, @shared_opts}
      end

    hybrid = [
      {:jamba, :ssm, @shared_opts},
      {:zamba, :ssm, @shared_opts},
      {:striped_hyena, :ssm, @shared_opts}
    ]

    standard ++ hybrid
  end

  defp recurrent_archs do
    standard =
      for arch <- [:lstm, :gru, :xlstm, :min_gru, :min_lstm, :delta_net] do
        {arch, :recurrent, @shared_opts}
      end

    special = [
      {:ttt, :recurrent, @shared_opts},
      {:titans, :recurrent, @shared_opts},
      {:liquid, :recurrent, @shared_opts}
    ]

    standard ++ special
  end

  defp attention_archs do
    standard =
      for arch <- [:gqa, :fnet, :linear_transformer, :nystromformer, :performer] do
        {arch, :attention, @shared_opts}
      end

    special = [
      {:retnet, :attention, @shared_opts},
      {:gla, :attention, @shared_opts},
      {:hgrn, :attention, @shared_opts},
      {:griffin, :attention, @shared_opts},
      {:rwkv, :attention, Keyword.merge(@shared_opts, head_size: max(div(@embed, 4), 4))}
    ]

    standard ++ special
  end

  defp other_archs do
    [
      {:kan, :other, @shared_opts},
      {:reservoir, :other,
       [input_size: @embed, reservoir_size: @hidden, output_size: @hidden, seq_len: @seq_len]}
    ]
  end

  # ── Input helper ───────────────────────────────────────────────

  defp rand(shape) do
    key = Nx.Random.key(42)
    {tensor, _key} = Nx.Random.normal(key, shape: shape)
    tensor
  end

  defp input_name(:reservoir), do: "input"
  defp input_name(_), do: "state_sequence"

  # ── Compilation + warmup ───────────────────────────────────────

  defp compile_model(arch, opts) do
    model = Edifice.build(arch, opts)
    input_key = input_name(arch)
    input = rand({@batch, @seq_len, @embed})
    template = %{input_key => Nx.template({@batch, @seq_len, @embed}, :f32)}

    {init_fn, predict_fn} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())

    # Warm up (triggers EXLA compilation)
    input_map = %{input_key => input}
    for _ <- 1..3, do: predict_fn.(params, input_map)

    {predict_fn, params, input_map}
  end

  # ── Runner ─────────────────────────────────────────────────────

  def run do
    IO.puts("=" |> String.duplicate(80))
    IO.puts("ExPhil Backbone Inference Latency — EXLA")

    IO.puts(
      "embed=#{@embed}, seq_len=#{@seq_len}, batch=#{@batch}, " <>
        "layers=#{@num_layers}, heads=#{@num_heads}"
    )

    IO.puts("Target: <#{@fps_target_ms}ms (60 FPS)")
    IO.puts("=" |> String.duplicate(80))
    IO.puts("")

    all_archs = architectures()

    # Phase 0: GPU/EXLA runtime warmup — compile and run the first architecture
    # once to absorb all one-time costs (cuDNN init, BFC allocator, EXLA JIT
    # cache warmup). The first compile is then re-done in Phase 1 for fair timing.
    IO.puts("## Phase 0: GPU Runtime Warmup")
    IO.puts("-" |> String.duplicate(60))

    {arch0, _, opts0} = hd(all_archs)

    {warmup_us, _} =
      :timer.tc(fn ->
        compile_model(arch0, opts0)
      end)

    IO.puts("  Warmed up #{arch0} in #{Float.round(warmup_us / 1_000, 0)} ms (discarded)")
    IO.puts("")

    # Phase 1: Compile all models and report compilation time
    IO.puts("## Phase 1: EXLA Compilation")
    IO.puts("-" |> String.duplicate(60))

    compiled =
      for {arch, category, opts} <- all_archs do
        try do
          {compile_us, {predict_fn, params, input_map}} =
            :timer.tc(fn -> compile_model(arch, opts) end)

          compile_ms = compile_us / 1_000
          IO.puts("  #{String.pad_trailing(to_string(arch), 25)} #{fmt(compile_ms)}")
          {:ok, arch, category, predict_fn, params, input_map, compile_ms}
        rescue
          e ->
            msg = Exception.message(e) |> String.slice(0, 80)
            IO.puts("  #{String.pad_trailing(to_string(arch), 25)} FAIL: #{msg}")
            {:fail, arch, category, msg}
        end
      end

    {successful, failed} = Enum.split_with(compiled, &(elem(&1, 0) == :ok))

    IO.puts("")
    IO.puts("  #{length(successful)}/#{length(all_archs)} compiled successfully")

    if failed != [] do
      IO.puts("  Failed: #{Enum.map_join(failed, ", ", &to_string(elem(&1, 1)))}")
    end

    IO.puts("")

    # Phase 2: Quick latency scan (manual timing, no Benchee overhead)
    # This gives us a fast overview to identify candidates
    IO.puts("## Phase 2: Quick Latency Scan (50 iterations)")
    IO.puts("-" |> String.duplicate(60))

    scan_results =
      for {:ok, arch, category, predict_fn, params, input_map, compile_ms} <- successful do
        {total_us, _} =
          :timer.tc(fn ->
            for _ <- 1..50, do: predict_fn.(params, input_map)
          end)

        avg_ms = total_us / 50 / 1_000

        marker =
          cond do
            avg_ms < @fps_target_ms -> " << 60 FPS"
            avg_ms < @fps_target_ms * 2 -> " ~  30 FPS"
            true -> ""
          end

        IO.puts(
          "  #{String.pad_trailing(to_string(arch), 25)}" <>
            "#{String.pad_trailing(fmt(avg_ms), 12)}" <>
            "#{String.pad_trailing(to_string(category), 12)}" <>
            marker
        )

        {arch, category, avg_ms, predict_fn, params, input_map, compile_ms}
      end
      |> Enum.sort_by(fn {_, _, ms, _, _, _, _} -> ms end)

    IO.puts("")

    # Summary table sorted by latency
    IO.puts("## Latency Ranking")
    IO.puts("-" |> String.duplicate(60))

    IO.puts(
      "  #{String.pad_trailing("Rank", 6)}" <>
        "#{String.pad_trailing("Architecture", 25)}" <>
        "#{String.pad_trailing("Category", 12)}" <>
        "#{String.pad_trailing("Avg (ms)", 12)}" <>
        "FPS"
    )

    IO.puts("  " <> String.duplicate("-", 58))

    scan_results
    |> Enum.with_index(1)
    |> Enum.each(fn {{arch, category, avg_ms, _, _, _, _}, rank} ->
      fps = if avg_ms > 0, do: Float.round(1000.0 / avg_ms, 1), else: 0.0

      marker =
        cond do
          avg_ms < @fps_target_ms -> " *"
          true -> ""
        end

      IO.puts(
        "  #{String.pad_trailing(Integer.to_string(rank), 6)}" <>
          "#{String.pad_trailing(to_string(arch), 25)}" <>
          "#{String.pad_trailing(to_string(category), 12)}" <>
          "#{String.pad_trailing(fmt(avg_ms), 12)}" <>
          "#{fps}#{marker}"
      )
    end)

    IO.puts("")
    fps_viable = Enum.filter(scan_results, fn {_, _, ms, _, _, _, _} -> ms < @fps_target_ms end)

    IO.puts(
      "  #{length(fps_viable)}/#{length(scan_results)} architectures under #{@fps_target_ms}ms (60 FPS viable)"
    )

    IO.puts("")

    # Phase 3: Benchee deep dive on top candidates (fastest 10 or all sub-16ms)
    top_n = max(length(fps_viable), 10) |> min(length(scan_results))
    top_candidates = Enum.take(scan_results, top_n)

    IO.puts("## Phase 3: Benchee Statistical Benchmark (top #{top_n})")
    IO.puts("-" |> String.duplicate(60))
    IO.puts("")

    benchmarks =
      Map.new(top_candidates, fn {arch, _cat, _ms, predict_fn, params, input_map, _} ->
        {to_string(arch), fn -> predict_fn.(params, input_map) end}
      end)

    if Code.ensure_loaded?(Benchee) do
      Benchee.run(benchmarks,
        warmup: 2,
        time: @bench_time,
        memory_time: 2,
        print: [configuration: false],
        formatters: [Benchee.Formatters.Console]
      )
    else
      IO.puts("  (Benchee not available — run with MIX_ENV=dev for statistical benchmarks)")
      IO.puts("  Falling back to manual timing (100 iterations)...")
      IO.puts("")

      benchmarks
      |> Enum.map(fn {name, fun} ->
        {total_us, _} = :timer.tc(fn -> for _ <- 1..100, do: fun.() end)
        avg_ms = total_us / 100 / 1_000
        {name, avg_ms}
      end)
      |> Enum.sort_by(fn {_, ms} -> ms end)
      |> Enum.each(fn {name, avg_ms} ->
        IO.puts("  #{String.pad_trailing(name, 25)} #{fmt(avg_ms)} avg (100 iters)")
      end)
    end

    # Phase 4: Category summary
    IO.puts("")
    IO.puts("## Category Summary (median latency)")
    IO.puts("-" |> String.duplicate(60))

    scan_results
    |> Enum.group_by(fn {_, cat, _, _, _, _, _} -> cat end)
    |> Enum.map(fn {cat, entries} ->
      times = Enum.map(entries, fn {_, _, ms, _, _, _, _} -> ms end) |> Enum.sort()
      median = Enum.at(times, div(length(times), 2))
      best = hd(times)
      worst = List.last(times)
      {cat, median, best, worst, length(entries)}
    end)
    |> Enum.sort_by(fn {_, median, _, _, _} -> median end)
    |> Enum.each(fn {cat, median, best, worst, count} ->
      IO.puts(
        "  #{String.pad_trailing(to_string(cat), 15)}" <>
          "median=#{String.pad_trailing(fmt(median), 10)}" <>
          "best=#{String.pad_trailing(fmt(best), 10)}" <>
          "worst=#{String.pad_trailing(fmt(worst), 10)}" <>
          "(#{count} archs)"
      )
    end)

    IO.puts("")
    IO.puts("Done.")
  end

  defp fmt(ms) when ms < 0.01, do: "#{Float.round(ms * 1000, 1)} us"
  defp fmt(ms) when ms < 1, do: "#{Float.round(ms, 3)} ms"
  defp fmt(ms) when ms < 100, do: "#{Float.round(ms, 2)} ms"
  defp fmt(ms), do: "#{Float.round(ms, 0)} ms"
end

InferenceLatency.run()
