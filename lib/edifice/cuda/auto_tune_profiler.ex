defmodule Edifice.CUDA.AutoTuneProfiler do
  @moduledoc """
  Profiling and runtime feedback for AutoTune kernel dispatch.

  Tracks which kernels are dispatched, how often, via which path (fused
  custom call, NIF, or fallback), and with what input shapes. Reports
  coverage gaps where AutoTune benchmarks don't match actual training shapes.

  ## Quick Start

      # Before training
      Edifice.CUDA.AutoTuneProfiler.start()

      # Run training...
      Axon.Loop.run(loop, data, %{}, epochs: 5)

      # After training
      Edifice.CUDA.AutoTuneProfiler.report()
      Edifice.CUDA.AutoTuneProfiler.stop()

  ## Axon.Loop integration

      loop
      |> Axon.Loop.trainer(loss, optimizer)
      |> Edifice.CUDA.AutoTuneProfiler.attach()

  ## Defn-level output validation

  During kernel development, use `validate_output/3` inside defn to compare
  fused kernel output against the Elixir fallback via `runtime_call`:

      defn checked_scan(gates, candidates) do
        fused = FusedScan.mingru(gates, candidates)
        Edifice.CUDA.AutoTuneProfiler.validate_output(
          fused, {gates, candidates},
          kernel: :mingru, fallback: &MinGRU.min_gru_scan/2
        )
      end
  """

  require Logger
  import Nx.Defn

  @profiler_key :__edifice_autotune_profiler__

  # ============================================================================
  # Profiling lifecycle
  # ============================================================================

  @doc """
  Start profiling kernel dispatches.

  Initializes an ETS table to track dispatch counts, paths, and shapes.
  """
  def start do
    if :ets.whereis(@profiler_key) != :undefined do
      Logger.warning("[AutoTuneProfiler] Already running — call stop() first")
      :already_running
    else
      :ets.new(@profiler_key, [:named_table, :public, :set])
      Logger.info("[AutoTuneProfiler] Started")
      :ok
    end
  end

  @doc "Stop profiling and delete the ETS table."
  def stop do
    if :ets.whereis(@profiler_key) != :undefined do
      :ets.delete(@profiler_key)
      Logger.info("[AutoTuneProfiler] Stopped")
      :ok
    else
      :not_running
    end
  end

  @doc "Check if profiling is active."
  def active? do
    :ets.whereis(@profiler_key) != :undefined
  end

  @doc """
  Record a kernel dispatch. Called from FusedScan dispatch functions.

  ## Parameters

    * `kernel` - Kernel name atom (e.g., `:mingru`)
    * `path` - Dispatch path: `:custom_call`, `:nif`, or `:fallback`
    * `shape` - Input tensor shape tuple
    * `dtype` - Input tensor type
  """
  def record(kernel, path, shape, dtype) do
    if active?() do
      key = {kernel, path, shape, dtype}

      try do
        :ets.update_counter(@profiler_key, key, {2, 1})
      catch
        :error, :badarg ->
          :ets.insert(@profiler_key, {key, 1})
      end
    end
  end

  @doc """
  Print a summary of kernel dispatch statistics.

  Shows dispatch counts by kernel, path, shape, and flags coverage gaps
  where no AutoTune benchmark exists for an observed shape.
  """
  def report do
    unless active?() do
      IO.puts("[AutoTuneProfiler] Not running. Call start() first.")
      return_nothing()
    end

    entries =
      :ets.tab2list(@profiler_key)
      |> Enum.map(fn {{kernel, path, shape, dtype}, count} ->
        %{kernel: kernel, path: path, shape: shape, dtype: dtype, count: count}
      end)
      |> Enum.sort_by(& &1.count, :desc)

    if entries == [] do
      IO.puts("[AutoTuneProfiler] No dispatches recorded yet.")
      return_nothing()
    end

    total = Enum.sum(Enum.map(entries, & &1.count))

    IO.puts("\n[AutoTuneProfiler] Kernel Dispatch Report")
    IO.puts("  Total dispatches: #{total}\n")

    # Group by kernel
    by_kernel =
      entries
      |> Enum.group_by(& &1.kernel)
      |> Enum.sort_by(fn {_k, es} -> -Enum.sum(Enum.map(es, & &1.count)) end)

    IO.puts(
      "  #{pad("Kernel", 25)} #{pad("Path", 14)} #{pad("Shape", 25)} #{pad("DType", 8)} Count"
    )

    IO.puts("  #{String.duplicate("-", 85)}")

    for {_kernel, kernel_entries} <- by_kernel do
      kernel_total = Enum.sum(Enum.map(kernel_entries, & &1.count))

      for entry <- Enum.sort_by(kernel_entries, & &1.count, :desc) do
        IO.puts(
          "  #{pad(to_string(entry.kernel), 25)} #{pad(to_string(entry.path), 14)} " <>
            "#{pad(inspect(entry.shape), 25)} #{pad(format_dtype(entry.dtype), 8)} #{entry.count}"
        )
      end

      if length(kernel_entries) > 1 do
        IO.puts("  #{pad("", 25)} #{pad("", 14)} #{pad("subtotal", 25)} #{pad("", 8)} #{kernel_total}")
      end
    end

    # Coverage analysis
    IO.puts("\n  Coverage Analysis:")
    uncovered = find_uncovered(entries)

    if uncovered == [] do
      IO.puts("  All observed shapes have AutoTune benchmark coverage.")
    else
      IO.puts("  The following kernel/dim combinations lack benchmark coverage:")

      for {kernel, dim, dtype} <- uncovered do
        IO.puts("    - #{kernel} dim=#{dim} dtype=#{format_dtype(dtype)}")
      end

      IO.puts("\n  Run: AutoTune.warmup(hidden: <dim>, dtype: <type>, kernels: [...])")
    end

    IO.puts("")
  end

  # ============================================================================
  # Axon.Loop integration
  # ============================================================================

  @doc """
  Attach profiler to an Axon.Loop.

  Starts profiling at loop start and prints a report at loop end.
  """
  def attach(loop) do
    loop
    |> Axon.Loop.handle_event(:started, fn state ->
      start()
      {:continue, state}
    end)
    |> Axon.Loop.handle_event(:completed, fn state ->
      report()
      stop()
      {:continue, state}
    end)
  end

  # ============================================================================
  # Defn-level output validation (via runtime_call)
  # ============================================================================

  @doc """
  Validate fused kernel output against a fallback inside defn.

  Uses `runtime_call` to run the fallback function on the same inputs
  and compare outputs. Logs a warning if the outputs diverge beyond
  tolerance. Returns the fused output unchanged.

  This is a **debug tool** for kernel development — it doubles computation
  cost. Remove from production training.

  ## Options

    * `:kernel` - Kernel name for logging (default: `"unknown"`)
    * `:fallback` - Fallback function to compare against (required).
      Must be a named capture with arity matching the inputs.
    * `:atol` - Absolute tolerance (default: 1.0e-4)
    * `:rtol` - Relative tolerance (default: 1.0e-3)
  """
  defn validate_output(fused_output, inputs, opts \\ []) do
    # Pack fused_output and inputs together for the callback
    Nx.runtime_call(
      fused_output,
      {fused_output, inputs},
      opts,
      &__MODULE__.validate_callback/2
    )
  end

  @doc false
  def validate_callback({fused_output, inputs}, opts) do
    kernel = Keyword.get(opts, :kernel, "unknown")
    fallback = Keyword.get(opts, :fallback)
    atol = Keyword.get(opts, :atol, 1.0e-4)
    rtol = Keyword.get(opts, :rtol, 1.0e-3)

    if fallback do
      reference =
        case inputs do
          {a, b} -> fallback.(a, b)
          {a, b, c} -> fallback.(a, b, c)
          {a, b, c, d} -> fallback.(a, b, c, d)
          {a, b, c, d, e} -> fallback.(a, b, c, d, e)
          single -> fallback.(single)
        end

      close = Nx.all_close(fused_output, reference, atol: atol, rtol: rtol) |> Nx.to_number()

      if close != 1 do
        max_diff =
          Nx.subtract(fused_output, reference)
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        Logger.warning(
          "[AutoTuneProfiler] #{kernel} output mismatch: max_diff=#{Float.round(max_diff, 8)}" <>
            " (atol=#{atol}, rtol=#{rtol})"
        )
      end
    end

    fused_output
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp find_uncovered(entries) do
    entries
    |> Enum.map(fn %{kernel: k, shape: shape, dtype: dtype} ->
      dim = elem(shape, tuple_size(shape) - 1)
      {k, dim, dtype}
    end)
    |> Enum.uniq()
    |> Enum.filter(fn {kernel, dim, dtype} ->
      key = {:edifice_autotune, kernel, dim, dtype}
      :persistent_term.get(key, :not_cached) == :not_cached
    end)
  end

  defp return_nothing, do: :ok

  defp pad(str, width), do: String.pad_trailing(str, width)

  defp format_dtype({:f, 32}), do: "f32"
  defp format_dtype({:bf, 16}), do: "bf16"
  defp format_dtype(dtype), do: inspect(dtype)
end
