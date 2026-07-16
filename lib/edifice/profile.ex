defmodule Edifice.Profile do
  @moduledoc """
  Graph-level profiling for Edifice models compiled with EXLA.

  Wraps EXLA's telemetry events and memory statistics to provide
  compilation time, inference latency, and memory usage reporting.

  ## Usage

      # Profile a model's compilation + inference
      Edifice.Profile.run(:min_gru,
        embed_dim: 256,
        hidden_size: 256,
        seq_len: 32,
        batch_size: 1,
        warmup: 2,
        iterations: 10
      )

      # Profile with telemetry event logging
      Edifice.Profile.attach_telemetry()
      # ... run EXLA compilations ...
      Edifice.Profile.detach_telemetry()

      # Get EXLA memory stats
      Edifice.Profile.memory_stats()
  """

  require Logger

  @telemetry_handler_id "edifice-profile"

  @doc """
  Attach a telemetry handler that logs EXLA compilation events.

  Logs eval_time, compile_time, and total_time for each JIT compilation.
  Call `detach_telemetry/0` to remove the handler.
  """
  def attach_telemetry do
    :telemetry.attach(
      @telemetry_handler_id,
      [:exla, :compilation],
      &handle_telemetry_event/4,
      %{events: []}
    )
  end

  @doc """
  Detach the EXLA compilation telemetry handler.
  """
  def detach_telemetry do
    :telemetry.detach(@telemetry_handler_id)
  end

  defp handle_telemetry_event([:exla, :compilation], measurements, metadata, _config) do
    eval_ms = measurements[:eval_time] / 1_000
    compile_ms = measurements[:compile_time] / 1_000
    total_ms = measurements[:total_time] / 1_000

    Logger.info(
      "[EXLA compile] eval=#{Float.round(eval_ms, 1)}ms " <>
        "compile=#{Float.round(compile_ms, 1)}ms " <>
        "total=#{Float.round(total_ms, 1)}ms " <>
        "key=#{inspect(metadata[:key])}"
    )
  end

  @doc """
  Get EXLA memory statistics for the default client.

  Returns a map with `:allocated` and `:peak` in bytes, or an error
  if EXLA is not available.
  """
  def memory_stats(client_name \\ :default) do
    if Code.ensure_loaded?(EXLA.Client) do
      try do
        client = EXLA.Client.fetch!(client_name)
        EXLA.Client.get_memory_statistics(client)
      rescue
        _ -> {:error, :client_not_available}
      end
    else
      {:error, :exla_not_loaded}
    end
  end

  @doc """
  Profile compilation and inference for an Edifice architecture.

  ## Options

    - `:embed_dim` - Embedding dimension (default: 64)
    - `:hidden_size` - Hidden size (default: 64)
    - `:seq_len` - Sequence length (default: 32)
    - `:batch_size` - Batch size (default: 1)
    - `:warmup` - Number of warmup iterations (default: 2)
    - `:iterations` - Number of timed iterations (default: 10)
    - `:compiler` - Compiler module (default: nil for BinaryBackend)

  ## Modes

  With `mode: :step` (architectures implementing `Edifice.Stateful` only),
  profiles per-frame `step/3` latency instead of full-sequence forwards:
  batch defaults to 1, and the result reports `init_ms`, `p50_step_ms`,
  `p95_step_ms`, `mean_step_ms`, `max_step_ms`, and `state_bytes` (the
  recurrent state's memory footprint — what a netplay rollback snapshot
  costs). Any architecture build option (e.g. `:state_size`, `:num_layers`)
  passes through. This is the harness for picking a 60 FPS bot backbone:
  the budget is 16.67 ms/frame.

  ## Returns

    A map with profiling results:
    - `:compile_time_ms` - Time to compile the model (ms)
    - `:mean_inference_ms` - Mean inference latency (ms)
    - `:min_inference_ms` - Min inference latency (ms)
    - `:max_inference_ms` - Max inference latency (ms)
    - `:std_inference_ms` - Std dev of inference latency (ms)
    - `:param_count` - Total number of parameters
    - `:output_shape` - Shape of model output

    (or the step-mode map described above when `mode: :step`)
  """
  def run(arch, opts \\ []) do
    case Keyword.get(opts, :mode, :forward) do
      :step -> run_step(arch, opts)
      :forward -> run_forward(arch, opts)
      other -> raise ArgumentError, "unknown profile mode #{inspect(other)} (:forward | :step)"
    end
  end

  defp run_forward(arch, opts) do
    embed_dim = Keyword.get(opts, :embed_dim, 64)
    hidden_size = Keyword.get(opts, :hidden_size, 64)
    seq_len = Keyword.get(opts, :seq_len, 32)
    batch_size = Keyword.get(opts, :batch_size, 1)
    warmup = Keyword.get(opts, :warmup, 2)
    iterations = Keyword.get(opts, :iterations, 10)
    compiler = Keyword.get(opts, :compiler, nil)

    model =
      Edifice.build(arch,
        embed_dim: embed_dim,
        hidden_size: hidden_size,
        seq_len: seq_len
      )

    template = %{
      "state_sequence" => Nx.template({batch_size, seq_len, embed_dim}, :f32)
    }

    build_opts = if compiler, do: [compiler: compiler], else: []

    # Measure compilation
    {compile_us, {init_fn, predict_fn}} = :timer.tc(fn -> Axon.build(model, build_opts) end)
    compile_ms = compile_us / 1_000.0

    # Initialize params
    params = init_fn.(template, Axon.ModelState.empty())

    param_count = count_params(params)

    # Create input
    {input, _key} =
      Nx.Random.uniform(Nx.Random.key(42), shape: {batch_size, seq_len, embed_dim})

    input_map = %{"state_sequence" => input}

    # Warmup
    for _ <- 1..warmup do
      predict_fn.(params, input_map)
    end

    # Timed iterations
    times_ms =
      Enum.map(1..iterations, fn _ ->
        {usec, _} = :timer.tc(fn -> predict_fn.(params, input_map) end)
        usec / 1_000.0
      end)

    output = predict_fn.(params, input_map)

    mean = Enum.sum(times_ms) / length(times_ms)
    min_t = Enum.min(times_ms)
    max_t = Enum.max(times_ms)

    variance =
      times_ms
      |> Enum.map(fn t -> (t - mean) * (t - mean) end)
      |> Enum.sum()
      |> Kernel./(length(times_ms))

    std = :math.sqrt(variance)

    %{
      arch: arch,
      compile_time_ms: Float.round(compile_ms, 2),
      mean_inference_ms: Float.round(mean, 3),
      min_inference_ms: Float.round(min_t, 3),
      max_inference_ms: Float.round(max_t, 3),
      std_inference_ms: Float.round(std, 3),
      param_count: param_count,
      output_shape: Nx.shape(output),
      config: %{
        embed_dim: embed_dim,
        hidden_size: hidden_size,
        seq_len: seq_len,
        batch_size: batch_size
      }
    }
  end

  # Per-frame step latency profiling (Edifice.Stateful architectures)
  defp run_step(arch, opts) do
    unless Edifice.Stateful.stateful?(arch) do
      stateful =
        Edifice.list_architectures()
        |> Enum.filter(&Edifice.Stateful.stateful?/1)
        |> Enum.join(", ")

      raise ArgumentError,
            "#{inspect(arch)} does not implement Edifice.Stateful — step-mode " <>
              "profiling supports: #{stateful}"
    end

    batch_size = Keyword.get(opts, :batch_size, 1)
    warmup = Keyword.get(opts, :warmup, 10)
    iterations = Keyword.get(opts, :iterations, 200)
    compiler = Keyword.get(opts, :compiler)
    step_opts = if compiler, do: [compiler: compiler], else: []

    build_opts =
      opts
      |> Keyword.drop([:mode, :warmup, :iterations, :batch_size, :compiler, :archs, :seq_len])
      |> Keyword.put_new(:embed_dim, 64)
      |> Keyword.put_new(:hidden_size, 64)
      |> then(fn o ->
        # GatedSSM can only be stepped in its causal scan mode
        if arch == :gated_ssm, do: Keyword.put_new(o, :scan_mode, :causal), else: o
      end)

    embed_dim = Keyword.fetch!(build_opts, :embed_dim)

    # Build once (tiny seq_len — only needed to initialize params)
    model = Edifice.build(arch, Keyword.put(build_opts, :seq_len, 4))
    {init_fn, _predict_fn} = Axon.build(model)
    template = %{"state_sequence" => Nx.template({batch_size, 4, embed_dim}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())
    param_count = count_params(params)

    # init_state also warms the JIT when a compiler is given, so init_ms
    # includes compilation — that's the honest number for a frame loop
    {init_us, state} =
      :timer.tc(fn ->
        Edifice.init_state(arch, params, build_opts ++ [batch_size: batch_size] ++ step_opts)
      end)

    state_bytes = state_byte_size(state)

    # Pre-materialize every frame BEFORE the timed loop: a fresh frame per
    # step keeps caches honest, but eager per-index slicing must not be
    # timed — on EXLA each distinct slice start compiles its own executable
    # (~10ms), which poisoned the first-profiled arch's numbers until this
    # was hoisted. Slice on BinaryBackend (no compilation), then move each
    # frame to the active backend.
    {all_frames, _key} =
      Nx.Random.uniform(Nx.Random.key(42), -1.0, 1.0,
        shape: {warmup + iterations, batch_size, embed_dim}
      )

    all_frames_host = Nx.backend_copy(all_frames, Nx.BinaryBackend)

    frames =
      for i <- 0..(warmup + iterations - 1) do
        all_frames_host
        |> Nx.slice_along_axis(i, 1, axis: 0)
        |> Nx.squeeze(axes: [0])
        |> Nx.backend_copy(Nx.default_backend())
      end

    {warmup_frames, timed_frames} = Enum.split(frames, warmup)

    state =
      Enum.reduce(warmup_frames, state, fn frame, s ->
        {_out, s2} = Edifice.step(arch, params, s, frame, step_opts)
        s2
      end)

    {_state, times_rev} =
      Enum.reduce(timed_frames, {state, []}, fn frame, {s, acc} ->
        {us, {_out, s2}} =
          :timer.tc(fn -> Edifice.step(arch, params, s, frame, step_opts) end)

        {s2, [us / 1_000.0 | acc]}
      end)

    times = Enum.sort(times_rev)
    mean = Enum.sum(times) / length(times)

    %{
      arch: arch,
      mode: :step,
      init_ms: Float.round(init_us / 1_000.0, 3),
      p50_step_ms: Float.round(percentile(times, 0.5), 3),
      p95_step_ms: Float.round(percentile(times, 0.95), 3),
      mean_step_ms: Float.round(mean, 3),
      max_step_ms: Float.round(List.last(times), 3),
      state_bytes: state_bytes,
      param_count: param_count,
      backend: inspect(Nx.default_backend() |> elem(0)),
      config: %{
        embed_dim: embed_dim,
        hidden_size: Keyword.fetch!(build_opts, :hidden_size),
        batch_size: batch_size,
        iterations: iterations,
        compiler: compiler
      }
    }
  end

  defp percentile(sorted, q) do
    Enum.at(sorted, round(q * (length(sorted) - 1)))
  end

  defp state_byte_size(%Nx.Tensor{} = t), do: Nx.byte_size(t)

  defp state_byte_size(%{} = map) when not is_struct(map) do
    Enum.reduce(map, 0, fn {_k, v}, acc -> acc + state_byte_size(v) end)
  end

  @doc """
  Profile multiple architectures and print a comparison table.

  ## Options

  Same as `run/2`, plus:
    - `:archs` - List of architecture atoms to profile (required)
  """
  def compare(opts \\ []) do
    archs = Keyword.fetch!(opts, :archs)
    profile_opts = Keyword.drop(opts, [:archs])

    results =
      Enum.map(archs, fn arch ->
        IO.write("  Profiling #{arch}...")

        try do
          result = run(arch, profile_opts)

          if result[:mode] == :step do
            IO.puts(" done (p50 #{result.p50_step_ms}ms/step)")
          else
            IO.puts(" done (#{result.mean_inference_ms}ms)")
          end

          result
        rescue
          e ->
            IO.puts(" FAILED: #{Exception.message(e)}")
            %{arch: arch, error: Exception.message(e)}
        end
      end)

    IO.puts("")
    print_table(results)
    results
  end

  @doc """
  Print a formatted comparison table from profile results.

  Dispatches on the results' `:mode` — step-mode results get a per-step
  latency table (p50/p95/max, state bytes) with a backend footer.
  """
  def print_table(results) do
    if Enum.any?(results, &(Map.get(&1, :mode) == :step)) do
      print_step_table(results)
    else
      print_forward_table(results)
    end
  end

  defp print_step_table(results) do
    header =
      String.pad_trailing("Architecture", 25) <>
        String.pad_leading("Init(ms)", 10) <>
        String.pad_leading("p50(ms)", 9) <>
        String.pad_leading("p95(ms)", 9) <>
        String.pad_leading("Max(ms)", 9) <>
        String.pad_leading("State(KB)", 11) <>
        String.pad_leading("Params", 10)

    IO.puts(header)
    IO.puts(String.duplicate("-", String.length(header)))

    Enum.each(results, fn result ->
      if Map.has_key?(result, :error) do
        IO.puts(String.pad_trailing(to_string(result.arch), 25) <> "  ERROR: #{result.error}")
      else
        IO.puts(
          String.pad_trailing(to_string(result.arch), 25) <>
            String.pad_leading(to_string(result.init_ms), 10) <>
            String.pad_leading(to_string(result.p50_step_ms), 9) <>
            String.pad_leading(to_string(result.p95_step_ms), 9) <>
            String.pad_leading(to_string(result.max_step_ms), 9) <>
            String.pad_leading(to_string(Float.round(result.state_bytes / 1_024, 1)), 11) <>
            String.pad_leading(format_params(result.param_count), 10)
        )
      end
    end)

    backend =
      results
      |> Enum.find_value(&Map.get(&1, :backend))
      |> Kernel.||("unknown")

    IO.puts("")

    IO.puts(
      "Backend: #{backend} — CPU-only numbers; re-measure on idle GPU with: " <>
        "EXLA_TARGET=cuda mix run bench/step_latency.exs"
    )
  end

  defp print_forward_table(results) do
    header =
      String.pad_trailing("Architecture", 25) <>
        String.pad_leading("Compile(ms)", 12) <>
        String.pad_leading("Mean(ms)", 10) <>
        String.pad_leading("Min(ms)", 9) <>
        String.pad_leading("Std(ms)", 9) <>
        String.pad_leading("Params", 10)

    IO.puts(header)
    IO.puts(String.duplicate("-", String.length(header)))

    Enum.each(results, fn result ->
      if Map.has_key?(result, :error) do
        IO.puts(
          String.pad_trailing(to_string(result.arch), 25) <>
            "  ERROR: #{result.error}"
        )
      else
        IO.puts(
          String.pad_trailing(to_string(result.arch), 25) <>
            String.pad_leading(to_string(result.compile_time_ms), 12) <>
            String.pad_leading(to_string(result.mean_inference_ms), 10) <>
            String.pad_leading(to_string(result.min_inference_ms), 9) <>
            String.pad_leading(to_string(result.std_inference_ms), 9) <>
            String.pad_leading(format_params(result.param_count), 10)
        )
      end
    end)
  end

  defp count_params(model_state) do
    model_state.data
    |> flatten_params()
    |> Enum.reduce(0, fn {_path, tensor}, acc -> acc + Nx.size(tensor) end)
  end

  defp flatten_params(map) when is_map(map) do
    Enum.flat_map(map, fn
      {key, %Nx.Tensor{} = tensor} -> [{key, tensor}]
      {key, inner} when is_map(inner) ->
        flatten_params(inner) |> Enum.map(fn {k, v} -> {"#{key}.#{k}", v} end)
      _ -> []
    end)
  end

  defp format_params(count) when count >= 1_000_000, do: "#{Float.round(count / 1_000_000, 1)}M"
  defp format_params(count) when count >= 1_000, do: "#{Float.round(count / 1_000, 1)}K"
  defp format_params(count), do: "#{count}"
end
