# Benchmark Regression CI
#
# Measures inference latency for key architectures on BinaryBackend (CPU).
# Catches algorithmic regressions (extra ops, complexity changes).
#
# Usage:
#   mix run bench/regression_ci.exs                      # compare against baseline
#   BENCH_UPDATE_BASELINE=1 mix run bench/regression_ci.exs  # write new baseline
#
# Baseline stored in bench/results/ci_baseline.json (checked into repo).
# Threshold: 20% regression after normalizing for machine speed. The median
# current/baseline ratio across architectures is treated as the hardware scale
# (CI runners are far slower than the machine that wrote the baseline), so only
# architectures that slow down relative to the others are flagged.

Logger.configure(level: :warning)

defmodule RegressionCI do
  @threshold 1.20
  @baseline_path Path.join([__DIR__, "results", "ci_baseline.json"])

  def architectures do
    [
      {:mlp, [input_size: 32, hidden_sizes: [16, 16]], {2, 32}},
      {:lstm, [embed_dim: 32, hidden_size: 16, cell_type: :lstm, seq_len: 8], {2, 8, 32}},
      {:mamba, [embed_dim: 32, hidden_size: 16, state_size: 8, num_layers: 2, window_size: 8],
       {2, 8, 32}},
      {:gqa,
       [
         embed_dim: 32,
         hidden_size: 16,
         num_heads: 4,
         num_kv_heads: 2,
         head_dim: 8,
         num_layers: 2,
         window_size: 8
       ], {2, 8, 32}},
      {:min_gru, [embed_dim: 32, hidden_size: 16, num_layers: 2, window_size: 8], {2, 8, 32}},
      {:vit,
       [
         image_size: 16,
         patch_size: 8,
         in_channels: 3,
         embed_dim: 32,
         depth: 2,
         num_heads: 4
       ], {2, 3, 16, 16}},
      {:detr,
       [
         image_size: 16,
         in_channels: 3,
         hidden_dim: 16,
         num_heads: 4,
         num_encoder_layers: 1,
         num_decoder_layers: 1,
         ffn_dim: 32,
         num_queries: 4,
         num_classes: 10,
         backbone_channels: 8,
         backbone_stages: 1,
         dropout: 0.0
       ], {2, 16, 16, 3}}
    ]
  end

  defp rand(shape) do
    key = Nx.Random.key(42)
    {tensor, _key} = Nx.Random.normal(key, shape: shape)
    tensor
  end

  defp make_input(:detr, shape), do: %{"image" => rand(shape)}
  defp make_input(:vit, shape), do: %{"image" => rand(shape)}
  defp make_input(_name, shape), do: rand(shape)

  defp make_template(%{} = input) when not is_struct(input) do
    Map.new(input, fn {k, v} -> {k, Nx.template(Nx.shape(v), Nx.type(v))} end)
  end

  defp make_template(input), do: Nx.template(Nx.shape(input), Nx.type(input))

  def run do
    IO.puts("=" |> String.duplicate(60))
    IO.puts("Edifice Benchmark Regression CI — BinaryBackend")
    IO.puts("=" |> String.duplicate(60))
    IO.puts("")

    # Build and init all models
    scenarios =
      for {name, opts, shape} <- architectures() do
        IO.puts("  Building #{name}...")
        model = Edifice.build(name, opts)
        input = make_input(name, shape)
        {init_fn, predict_fn} = Axon.build(model)
        params = init_fn.(make_template(input), Axon.ModelState.empty())
        {name, predict_fn, params, input}
      end

    IO.puts("")

    # Run Benchee
    benchmarks =
      Map.new(scenarios, fn {name, predict_fn, params, input} ->
        {Atom.to_string(name), fn -> predict_fn.(params, input) end}
      end)

    IO.puts("Running benchmarks (warmup: 2s, time: 5s per arch)...")
    IO.puts("")

    suite =
      Benchee.run(benchmarks,
        warmup: 2,
        time: 5,
        print: [configuration: false],
        formatters: [Benchee.Formatters.Console]
      )

    # Extract median timings
    results =
      Map.new(suite.scenarios, fn scenario ->
        stats = scenario.run_time_data.statistics
        {scenario.name, %{median_ns: round(stats.median), ips: Float.round(stats.ips, 2)}}
      end)

    update_baseline? = System.get_env("BENCH_UPDATE_BASELINE") == "1"

    if update_baseline? do
      write_baseline(results)
    else
      compare_baseline(results)
    end
  end

  defp write_baseline(results) do
    baseline = %{
      generated_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      architectures: results
    }

    json = Jason.encode!(baseline, pretty: true)
    File.mkdir_p!(Path.dirname(@baseline_path))
    File.write!(@baseline_path, json <> "\n")

    IO.puts("")
    IO.puts("Baseline written to #{@baseline_path}")

    for {name, %{median_ns: ns}} <- Enum.sort(results) do
      IO.puts("  #{String.pad_trailing(name, 12)} #{format_time(ns)}")
    end
  end

  defp compare_baseline(results) do
    unless File.exists?(@baseline_path) do
      IO.puts("")
      IO.puts("No baseline found at #{@baseline_path}")
      IO.puts("Run with BENCH_UPDATE_BASELINE=1 to generate one.")
      IO.puts("Skipping comparison — exit 0.")
      System.halt(0)
    end

    baseline =
      @baseline_path
      |> File.read!()
      |> Jason.decode!()
      |> Map.get("architectures")

    IO.puts("")
    IO.puts("-" |> String.duplicate(60))
    IO.puts("Regression Check (threshold: #{round((@threshold - 1) * 100)}%)")
    IO.puts("-" |> String.duplicate(60))

    scale = hardware_scale(results, baseline)
    IO.puts("Hardware scale (median current/baseline): #{Float.round(scale, 2)}x")

    regressions =
      for {name, %{median_ns: current_ns}} <- Enum.sort(results), reduce: [] do
        acc ->
          case Map.get(baseline, name) do
            nil ->
              IO.puts("  #{String.pad_trailing(name, 12)} NEW (no baseline)")
              acc

            %{"median_ns" => baseline_ns} ->
              ratio = current_ns / baseline_ns / scale
              status = if ratio > @threshold, do: "REGRESSED", else: "ok"

              IO.puts(
                "  #{String.pad_trailing(name, 12)} " <>
                  "#{format_time(current_ns)} vs #{format_time(baseline_ns)} " <>
                  "(#{Float.round(ratio, 2)}x normalized) #{status}"
              )

              if ratio > @threshold, do: [name | acc], else: acc
          end
      end

    IO.puts("")

    if regressions != [] do
      IO.puts(
        "FAIL: #{length(regressions)} regression(s) detected: #{Enum.join(regressions, ", ")}"
      )

      System.halt(1)
    else
      IO.puts("OK: All architectures within #{round((@threshold - 1) * 100)}% of baseline.")
    end
  end

  # Median of current/baseline across architectures with a baseline entry.
  # A uniform slowdown is machine speed; a single outlier is a regression.
  defp hardware_scale(results, baseline) do
    ratios =
      for {name, %{median_ns: current_ns}} <- results,
          %{"median_ns" => baseline_ns} <- [Map.get(baseline, name)],
          do: current_ns / baseline_ns

    case Enum.sort(ratios) do
      [] -> 1.0
      sorted -> Enum.at(sorted, div(length(sorted), 2))
    end
  end

  defp format_time(ns) when ns >= 1_000_000_000 do
    "#{Float.round(ns / 1_000_000_000, 2)}s"
  end

  defp format_time(ns) when ns >= 1_000_000 do
    "#{Float.round(ns / 1_000_000, 2)}ms"
  end

  defp format_time(ns) when ns >= 1_000 do
    "#{Float.round(ns / 1_000, 2)}µs"
  end

  defp format_time(ns), do: "#{ns}ns"
end

RegressionCI.run()
