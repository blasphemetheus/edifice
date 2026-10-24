defmodule Edifice.Training.MemoryTracker do
  # EXLA is optional; retain undefined-function checks when it is installed.
  unless Code.ensure_loaded?(EXLA) do
    @compile {:no_warn_undefined, [EXLA.Client]}
  end

  @moduledoc """
  GPU memory tracking for training and inference.

  Wraps `EXLA.Client.get_memory_statistics/1` and `reset_peak_memory/1`
  with training-aware utilities: per-step tracking, peak memory reporting,
  OOM prediction, and Axon.Loop integration.

  ## Quick Start

      # Snapshot current memory
      Edifice.Training.MemoryTracker.snapshot()
      #=> %{allocated: 134217728, peak: 268435456, per_device: %{0 => 134217728}}

      # Measure peak memory of a computation
      {result, peak} = Edifice.Training.MemoryTracker.measure(fn ->
        predict_fn.(params, input)
      end)
      #=> peak: 256.0 MB

  ## Axon.Loop Integration

      loop
      |> Axon.Loop.trainer(loss, optimizer)
      |> Edifice.Training.MemoryTracker.attach(every: 10)

  This logs memory stats every 10 steps and warns if memory usage
  exceeds a threshold.

  ## Requirements

  Requires EXLA. Memory tracking is a property of the EXLA allocator —
  returns zeros on non-EXLA backends.
  """

  require Logger

  # ============================================================================
  # Snapshot API
  # ============================================================================

  @doc """
  Get current memory statistics.

  Returns a map with:
    * `:allocated` - Currently allocated bytes
    * `:peak` - Peak allocation since last reset
    * `:per_device` - Map of device_id => allocated bytes

  ## Options

    * `:client` - EXLA client name (default: auto-detect)
  """
  @spec snapshot(keyword()) :: map()
  def snapshot(opts \\ []) do
    client = resolve_client(opts)
    stats = EXLA.Client.get_memory_statistics(client)

    %{
      allocated: stats.allocated,
      peak: stats.peak,
      per_device: stats.per_device,
      timestamp: System.monotonic_time(:millisecond)
    }
  end

  @doc """
  Reset peak memory counter.

  ## Options

    * `:client` - EXLA client name (default: auto-detect)
  """
  @spec reset_peak(keyword()) :: :ok
  def reset_peak(opts \\ []) do
    client = resolve_client(opts)
    EXLA.Client.reset_peak_memory(client)
  end

  @doc """
  Measure peak memory used by a computation.

  Resets peak counter, runs the function, then reads peak.
  Returns `{result, memory_info}` where `memory_info` includes
  peak bytes and allocated bytes before/after.

  ## Options

    * `:client` - EXLA client name (default: auto-detect)
    * `:label` - Label for logging (default: `"computation"`)
    * `:log` - Whether to log results (default: `true`)

  ## Examples

      {output, mem} = Edifice.Training.MemoryTracker.measure(fn ->
        predict_fn.(params, input)
      end)
      #=> [MemoryTracker] computation: peak=256.0 MB, delta=+128.0 MB
  """
  @spec measure(function(), keyword()) :: {any(), map()}
  def measure(fun, opts \\ []) when is_function(fun, 0) do
    label = Keyword.get(opts, :label, "computation")
    log? = Keyword.get(opts, :log, true)
    client = resolve_client(opts)

    before = EXLA.Client.get_memory_statistics(client)
    EXLA.Client.reset_peak_memory(client)

    result = fun.()

    after_stats = EXLA.Client.get_memory_statistics(client)

    mem_info = %{
      peak: after_stats.peak,
      before_allocated: before.allocated,
      after_allocated: after_stats.allocated,
      delta: after_stats.allocated - before.allocated
    }

    if log? do
      Logger.info(
        "[MemoryTracker] #{label}: peak=#{readable_size(mem_info.peak)}, " <>
          "delta=#{format_delta(mem_info.delta)}"
      )
    end

    {result, mem_info}
  end

  @doc """
  Print a formatted memory report.

  ## Options

    * `:client` - EXLA client name (default: auto-detect)
  """
  @spec report(keyword()) :: map()
  def report(opts \\ []) do
    stats = snapshot(opts)

    IO.puts("\n[MemoryTracker] GPU Memory Report")
    IO.puts("  Allocated: #{readable_size(stats.allocated)}")
    IO.puts("  Peak:      #{readable_size(stats.peak)}")

    if map_size(stats.per_device) > 0 do
      IO.puts("  Per device:")

      Enum.sort(stats.per_device)
      |> Enum.each(fn {dev_id, bytes} ->
        IO.puts("    Device #{dev_id}: #{readable_size(bytes)}")
      end)
    end

    IO.puts("")
    stats
  end

  # ============================================================================
  # Axon.Loop integration
  # ============================================================================

  @doc """
  Attach memory tracking to an Axon.Loop.

  Logs memory statistics during training and optionally warns when
  memory exceeds a threshold.

  ## Options

    * `:every` - Log every N steps (default: 100)
    * `:warn_threshold` - Warn when allocated exceeds this many bytes (default: nil)
    * `:client` - EXLA client name (default: auto-detect)
  """
  @spec attach(Axon.Loop.t(), keyword()) :: Axon.Loop.t()
  def attach(loop, opts \\ []) do
    every = Keyword.get(opts, :every, 100)
    warn_threshold = Keyword.get(opts, :warn_threshold, nil)
    client_opts = Keyword.take(opts, [:client])

    loop
    |> Axon.Loop.handle_event(:started, fn state ->
      reset_peak(client_opts)

      Logger.info(
        "[MemoryTracker] Tracking started. " <>
          "Allocated: #{readable_size(snapshot(client_opts).allocated)}"
      )

      {:continue, state}
    end)
    |> Axon.Loop.handle_event(:iteration_completed, fn state ->
      step = state.iteration

      if rem(step, every) == 0 do
        stats = snapshot(client_opts)

        Logger.info(
          "[MemoryTracker] step=#{step} " <>
            "allocated=#{readable_size(stats.allocated)} " <>
            "peak=#{readable_size(stats.peak)}"
        )

        if warn_threshold && stats.allocated > warn_threshold do
          Logger.warning(
            "[MemoryTracker] Memory usage #{readable_size(stats.allocated)} " <>
              "exceeds threshold #{readable_size(warn_threshold)}"
          )
        end
      end

      {:continue, state}
    end)
    |> Axon.Loop.handle_event(:completed, fn state ->
      stats = snapshot(client_opts)

      Logger.info(
        "[MemoryTracker] Training complete. " <>
          "Final allocated: #{readable_size(stats.allocated)}, " <>
          "Peak: #{readable_size(stats.peak)}"
      )

      {:continue, state}
    end)
  end

  # ============================================================================
  # Comparison utilities
  # ============================================================================

  @doc """
  Compare memory usage between two configurations.

  Runs each function, measures peak memory, and reports the difference.
  Useful for comparing models with/without gradient checkpointing,
  different precisions, or quantization.

  ## Examples

      Edifice.Training.MemoryTracker.compare(
        {"f32", fn -> predict_fn.(params, input) end},
        {"bf16", fn -> predict_fn.(bf16_params, input) end}
      )
  """
  @spec compare({String.t(), function()}, {String.t(), function()}, keyword()) :: map()
  def compare({label_a, fun_a}, {label_b, fun_b}, opts \\ []) do
    {_, mem_a} = measure(fun_a, Keyword.merge(opts, label: label_a, log: false))
    {_, mem_b} = measure(fun_b, Keyword.merge(opts, label: label_b, log: false))

    ratio =
      if mem_b.peak > 0,
        do: Float.round(mem_a.peak / mem_b.peak, 2),
        else: 0.0

    IO.puts("\n[MemoryTracker] Comparison")
    IO.puts("  #{label_a}: peak=#{readable_size(mem_a.peak)}")
    IO.puts("  #{label_b}: peak=#{readable_size(mem_b.peak)}")
    IO.puts("  Ratio: #{label_a}/#{label_b} = #{ratio}x")
    IO.puts("")

    %{
      a: %{label: label_a, peak: mem_a.peak, delta: mem_a.delta},
      b: %{label: label_b, peak: mem_b.peak, delta: mem_b.delta},
      ratio: ratio
    }
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp resolve_client(opts) do
    client_name = Keyword.get(opts, :client, EXLA.Client.default_name())
    EXLA.Client.fetch!(client_name)
  end

  defp readable_size(n) when n < 1_024, do: "#{n} B"
  defp readable_size(n) when n < 1_048_576, do: "#{Float.round(n / 1_024, 1)} KB"
  defp readable_size(n) when n < 1_073_741_824, do: "#{Float.round(n / 1_048_576, 1)} MB"
  defp readable_size(n), do: "#{Float.round(n / 1_073_741_824, 2)} GB"

  defp format_delta(delta) when delta >= 0, do: "+#{readable_size(delta)}"
  defp format_delta(delta), do: "-#{readable_size(abs(delta))}"
end
