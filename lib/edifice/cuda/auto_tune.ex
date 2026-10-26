defmodule Edifice.CUDA.AutoTune do
  @moduledoc """
  Runtime auto-tune: benchmarks fused vs fallback on first use, caches winner.

  Each fused CUDA kernel is benchmarked against the XLA fallback on first use.
  Results are cached in `:persistent_term` for zero-overhead subsequent lookups.

  ## Environment Variables

  - `EDIFICE_DISABLE_FUSED=1` — All fused disabled (highest priority)
  - `EDIFICE_AUTOTUNE=0` — Auto-tune disabled, always use fused when available (legacy)
  - `EDIFICE_AUTOTUNE=fallback` — Force all fallback (skip fused entirely)
  - Unset (default) — Auto-tune enabled, benchmark on first use
  """

  require Logger

  alias Edifice.CUDA.FusedScan

  @bench_batch 4
  @bench_seq 60
  @warmup_runs 3
  @timed_runs 10

  @all_kernels [
    :mingru,
    :minlstm,
    :elu_gru,
    :real_gru,
    :diag_linear,
    :liquid,
    :linear_scan,
    :delta_net_scan,
    :gated_delta_net_scan,
    :delta_product_scan,
    :slstm_scan,
    :lstm_scan,
    :gru_scan,
    :ttt_scan,
    :selective_scan,
    :kda_scan,
    :rla_scan,
    :flash_attention,
    :laser_attention,
    :fox_attention,
    :reservoir_scan,
    :titans_scan,
    :miras_scan,
    :gsa_scan,
    :mingru_block,
    :minlstm_block,
    :linear_block,
    :lstm_block,
    :gru_block
  ]

  # ============================================================================
  # Public API
  # ============================================================================

  @doc """
  Check whether to use the fused CUDA kernel for `kernel_name`.

  Called by dispatch functions in FusedScan. Returns `true` to use the fused
  custom call path, `false` to use the XLA fallback.
  """
  def use_fused?(kernel_name, tensor) do
    cond do
      System.get_env("EDIFICE_DISABLE_FUSED") == "1" ->
        false

      System.get_env("EDIFICE_AUTOTUNE") == "0" ->
        FusedScan.custom_call_available?(kernel_name)

      System.get_env("EDIFICE_AUTOTUNE") == "fallback" ->
        false

      # Per-op: ops without a native handler must fall through to the NIF
      # arm (or the fallback) instead of compiling the fallback in-graph.
      not FusedScan.custom_call_available?(kernel_name) ->
        false

      # During benchmarking, defer to custom_call_available?/1 (respects force_fallback)
      Process.get(:__edifice_benchmarking__, false) ->
        FusedScan.custom_call_available?(kernel_name)

      true ->
        dim = extract_key_dim(kernel_name, tensor)
        dtype = Nx.type(tensor)
        key = cache_key(kernel_name, dim, dtype)

        case :persistent_term.get(key, :not_cached) do
          :fused ->
            true

          :fallback ->
            false

          :not_cached ->
            result = run_benchmark(kernel_name, dim, dtype)
            result == :fused
        end
    end
  end

  @doc """
  Explicitly benchmark all kernels for given hidden dim and dtype.

  Call before training to pre-populate the cache and avoid benchmark
  latency during first compilation.

  ## Options

  - `:hidden` — Hidden dimension to benchmark (default: 64)
  - `:dtype` — Data type (default: `{:f, 32}`)
  - `:kernels` — List of kernel names to benchmark (default: all)
  """
  def warmup(opts \\ []) do
    hidden = Keyword.get(opts, :hidden, 64)
    dtype = Keyword.get(opts, :dtype, {:f, 32})
    kernels = Keyword.get(opts, :kernels, @all_kernels)

    Logger.info(
      "[AutoTune] Warming up #{length(kernels)} kernels (hidden=#{hidden}, dtype=#{inspect(dtype)})"
    )

    results =
      Enum.map(kernels, fn kernel ->
        key = cache_key(kernel, hidden, dtype)

        case :persistent_term.get(key, :not_cached) do
          :not_cached -> {kernel, run_benchmark(kernel, hidden, dtype)}
          cached -> {kernel, cached}
        end
      end)

    Logger.info("[AutoTune] Warmup complete")
    results
  end

  @doc "Print cached results as a formatted table."
  def report do
    results = cached_results()

    if results == [] do
      IO.puts("[AutoTune] No cached results. Run AutoTune.warmup() first.")
    else
      IO.puts("\n[AutoTune] Cached Results:")

      IO.puts(
        "  #{String.pad_trailing("Kernel", 25)} #{String.pad_trailing("Dim", 6)} #{String.pad_trailing("DType", 10)} Winner"
      )

      IO.puts("  #{String.duplicate("-", 55)}")

      Enum.each(results, fn {kernel, dim, dtype, winner} ->
        IO.puts(
          "  #{String.pad_trailing(to_string(kernel), 25)} #{String.pad_trailing(to_string(dim), 6)} #{String.pad_trailing(format_dtype(dtype), 10)} #{winner}"
        )
      end)

      IO.puts("")
    end
  end

  @doc "Load results from disk cache."
  def load_disk_cache(path \\ "cache/autotune.json") do
    case File.read(path) do
      {:ok, json} ->
        data = decode_json(json)
        gpu_name = get_gpu_name()

        if data["gpu"] != gpu_name do
          Logger.warning(
            "[AutoTune] Disk cache GPU mismatch: cached=#{data["gpu"]}, current=#{gpu_name}. Ignoring."
          )

          {:error, :gpu_mismatch}
        else
          count = install_cached_results(data["results"] || %{})
          Logger.info("[AutoTune] Loaded #{count} results from #{path}")
          {:ok, count}
        end

      {:error, :enoent} ->
        {:error, :not_found}

      {:error, reason} ->
        Logger.warning("[AutoTune] Failed to load disk cache: #{inspect(reason)}")
        {:error, reason}
    end
  end

  # Install each parseable cache entry into persistent_term; returns the count.
  defp install_cached_results(results) do
    Enum.reduce(results, 0, fn {key_str, winner_str}, acc ->
      case parse_cache_key(key_str) do
        {:ok, kernel, dim, dtype} ->
          winner = String.to_existing_atom(winner_str)
          :persistent_term.put(cache_key(kernel, dim, dtype), winner)
          acc + 1

        _ ->
          acc
      end
    end)
  end

  @doc "Save cached results to disk."
  def save_disk_cache(path \\ "cache/autotune.json") do
    results = cached_results()

    data = %{
      "gpu" => get_gpu_name(),
      "results" =>
        Map.new(results, fn {kernel, dim, dtype, winner} ->
          {"#{kernel}:#{dim}:#{format_dtype(dtype)}", to_string(winner)}
        end)
    }

    dir = Path.dirname(path)
    File.mkdir_p!(dir)
    File.write!(path, encode_json(data))
    Logger.info("[AutoTune] Saved #{length(results)} results to #{path}")
    :ok
  end

  @doc "Clear all cached results from persistent_term."
  def clear_cache do
    results = cached_results()

    Enum.each(results, fn {kernel, dim, dtype, _winner} ->
      key = cache_key(kernel, dim, dtype)
      :persistent_term.erase(key)
    end)

    Logger.info("[AutoTune] Cleared #{length(results)} cached results")
    :ok
  end

  @doc "Return the list of all known kernel names."
  def all_kernels, do: @all_kernels

  # ============================================================================
  # Cache key helpers
  # ============================================================================

  defp cache_key(kernel, dim, dtype), do: {:edifice_autotune, kernel, dim, dtype}

  defp extract_key_dim(kernel_name, tensor) do
    shape = Nx.shape(tensor)
    last_dim = elem(shape, tuple_size(shape) - 1)

    case kernel_name do
      k when k in [:slstm_scan, :lstm_scan] -> div(last_dim, 4)
      :gru_scan -> div(last_dim, 3)
      :titans_scan -> div(last_dim, 4)
      :miras_scan -> div(last_dim, 5)
      _ -> last_dim
    end
  end

  # ============================================================================
  # Benchmarking
  # ============================================================================

  defp run_benchmark(kernel_name, dim, dtype) do
    inputs = generate_inputs(kernel_name, dim, dtype)

    if inputs == :unsupported do
      Logger.info("[AutoTune] #{kernel_name} — no benchmark generator, defaulting to fused")
      key = cache_key(kernel_name, dim, dtype)
      :persistent_term.put(key, :fused)
      :fused
    else
      try do
        Process.put(:__edifice_benchmarking__, true)

        # Fused path
        Process.delete(:__edifice_force_fallback__)
        fused_fn = make_fused_bench_fn(kernel_name)
        fused_jit = Nx.Defn.jit(fused_fn, compiler: EXLA)
        fused_us = time_runs(fused_jit, inputs)

        # Fallback path
        Process.put(:__edifice_force_fallback__, true)
        fallback_fn = make_fallback_bench_fn(kernel_name)
        fallback_jit = Nx.Defn.jit(fallback_fn, compiler: EXLA)
        fallback_us = time_runs(fallback_jit, inputs)

        winner = if fused_us <= fallback_us, do: :fused, else: :fallback
        key = cache_key(kernel_name, dim, dtype)
        :persistent_term.put(key, winner)

        log_result(kernel_name, dim, dtype, fused_us, fallback_us, winner)
        winner
      rescue
        e ->
          msg = Exception.message(e)

          if String.contains?(msg, "JIT compilation") do
            # Only log once per kernel to avoid spam during tracing
            jit_warn_key = {:__autotune_jit_warned__, kernel_name}

            unless Process.get(jit_warn_key, false) do
              Process.put(jit_warn_key, true)

              Logger.info(
                "[AutoTune] #{kernel_name} — skipped (inside JIT). Call AutoTune.warmup() before training for optimal dispatch."
              )
            end
          else
            Logger.warning(
              "[AutoTune] Benchmark failed for #{kernel_name}: #{msg}, defaulting to fused"
            )
          end

          # Don't cache — allows warmup/1 to re-benchmark outside JIT
          :fused
      after
        Process.delete(:__edifice_benchmarking__)
        Process.delete(:__edifice_force_fallback__)
      end
    end
  end

  defp time_runs(jit_fn, inputs) do
    # Warmup (includes JIT compilation on first call)
    for _ <- 1..@warmup_runs do
      result = apply(jit_fn, inputs)
      Nx.backend_transfer(result, Nx.BinaryBackend)
    end

    # Timed runs
    times =
      for _ <- 1..@timed_runs do
        {us, _} =
          :timer.tc(fn ->
            result = apply(jit_fn, inputs)
            Nx.backend_transfer(result, Nx.BinaryBackend)
            :ok
          end)

        us
      end

    # Median
    sorted = Enum.sort(times)
    Enum.at(sorted, div(length(sorted), 2))
  end

  defp log_result(kernel_name, dim, dtype, fused_us, fallback_us, winner) do
    fused_ms = Float.round(fused_us / 1000, 2)
    fallback_ms = Float.round(fallback_us / 1000, 2)
    dtype_str = format_dtype(dtype)

    {ratio_str, detail} =
      if winner == :fused do
        ratio = Float.round(fallback_us / max(fused_us, 1), 2)
        {"#{ratio}x faster", "using fused"}
      else
        ratio = Float.round(fused_us / max(fallback_us, 1), 2)
        {"fused #{ratio}x slower", "using fallback"}
      end

    Logger.info(
      "[AutoTune] fused_#{kernel_name}_#{dtype_str} (dim=#{dim}) — fused: #{fused_ms}ms, fallback: #{fallback_ms}ms → #{detail} (#{ratio_str})"
    )
  end

  # ============================================================================
  # Bench function generators — fused variants
  #
  # Each function clause creates a unique fn literal (different source location)
  # ensuring separate JIT cache entries from the fallback variants below.
  # ============================================================================

  # 2-arg kernels
  for k <- [
        :mingru,
        :elu_gru,
        :real_gru,
        :diag_linear,
        :liquid,
        :linear_scan,
        :slstm_scan,
        :lstm_scan,
        :gru_scan,
        :reservoir_scan
      ] do
    defp make_fused_bench_fn(unquote(k)) do
      fn a, b -> apply(Edifice.CUDA.FusedScan, unquote(k), [a, b]) end
    end
  end

  # 3-arg kernels
  defp make_fused_bench_fn(:minlstm) do
    fn a, b, c -> Edifice.CUDA.FusedScan.minlstm(a, b, c) end
  end

  # 4-arg kernels
  for k <- [:delta_net_scan, :delta_product_scan, :fox_attention, :gsa_scan] do
    defp make_fused_bench_fn(unquote(k)) do
      fn a, b, c, d -> apply(Edifice.CUDA.FusedScan, unquote(k), [a, b, c, d]) end
    end
  end

  # 5-arg kernels
  for k <- [:gated_delta_net_scan, :selective_scan, :kda_scan] do
    defp make_fused_bench_fn(unquote(k)) do
      fn a, b, c, d, e -> apply(Edifice.CUDA.FusedScan, unquote(k), [a, b, c, d, e]) end
    end
  end

  # 6-arg: rla_scan (default opts)
  defp make_fused_bench_fn(:rla_scan) do
    fn q, k, v, a, b, g -> Edifice.CUDA.FusedScan.rla_scan(q, k, v, a, b, g) end
  end

  # 7-arg: ttt_scan
  defp make_fused_bench_fn(:ttt_scan) do
    fn a, b, c, d, e, f, g -> Edifice.CUDA.FusedScan.ttt_scan(a, b, c, d, e, f, g) end
  end

  # 3-tensor + opts: attention with causal
  for k <- [:flash_attention, :laser_attention] do
    defp make_fused_bench_fn(unquote(k)) do
      fn q, k_t, v -> apply(Edifice.CUDA.FusedScan, unquote(k), [q, k_t, v, [causal: true]]) end
    end
  end

  # 1-tensor + opts: titans
  defp make_fused_bench_fn(:titans_scan) do
    fn combined ->
      {_, _, dim} = Nx.shape(combined)
      Edifice.CUDA.FusedScan.titans_scan(combined, memory_size: div(dim, 4))
    end
  end

  # 1-tensor + opts: miras
  defp make_fused_bench_fn(:miras_scan) do
    fn combined ->
      {_, _, dim} = Nx.shape(combined)
      Edifice.CUDA.FusedScan.miras_scan(combined, memory_size: div(dim, 5))
    end
  end

  # 3-tensor + num_layers: block scans
  for k <- [:mingru_block, :minlstm_block, :linear_block, :gru_block] do
    defp make_fused_bench_fn(unquote(k)) do
      fn input, weights, h0 ->
        apply(Edifice.CUDA.FusedScan, unquote(k), [input, weights, h0, 2])
      end
    end
  end

  # 4-tensor + num_layers: lstm_block
  defp make_fused_bench_fn(:lstm_block) do
    fn input, weights, h0, c0 ->
      Edifice.CUDA.FusedScan.lstm_block(input, weights, h0, c0, 2)
    end
  end

  # ============================================================================
  # Bench function generators — fallback variants
  #
  # Identical bodies but different fn source locations → different JIT cache keys.
  # During fallback benchmarks, __edifice_force_fallback__ is set in process dict,
  # so custom_call_available?() returns false and dispatch uses XLA fallback.
  # ============================================================================

  # 2-arg kernels
  for k <- [
        :mingru,
        :elu_gru,
        :real_gru,
        :diag_linear,
        :liquid,
        :linear_scan,
        :slstm_scan,
        :lstm_scan,
        :gru_scan,
        :reservoir_scan
      ] do
    defp make_fallback_bench_fn(unquote(k)) do
      fn a, b -> apply(Edifice.CUDA.FusedScan, unquote(k), [a, b]) end
    end
  end

  # 3-arg kernels
  defp make_fallback_bench_fn(:minlstm) do
    fn a, b, c -> Edifice.CUDA.FusedScan.minlstm(a, b, c) end
  end

  # 4-arg kernels
  for k <- [:delta_net_scan, :delta_product_scan, :fox_attention, :gsa_scan] do
    defp make_fallback_bench_fn(unquote(k)) do
      fn a, b, c, d -> apply(Edifice.CUDA.FusedScan, unquote(k), [a, b, c, d]) end
    end
  end

  # 5-arg kernels
  for k <- [:gated_delta_net_scan, :selective_scan, :kda_scan] do
    defp make_fallback_bench_fn(unquote(k)) do
      fn a, b, c, d, e -> apply(Edifice.CUDA.FusedScan, unquote(k), [a, b, c, d, e]) end
    end
  end

  # 6-arg: rla_scan
  defp make_fallback_bench_fn(:rla_scan) do
    fn q, k, v, a, b, g -> Edifice.CUDA.FusedScan.rla_scan(q, k, v, a, b, g) end
  end

  # 7-arg: ttt_scan
  defp make_fallback_bench_fn(:ttt_scan) do
    fn a, b, c, d, e, f, g -> Edifice.CUDA.FusedScan.ttt_scan(a, b, c, d, e, f, g) end
  end

  # 3-tensor + opts: attention
  for k <- [:flash_attention, :laser_attention] do
    defp make_fallback_bench_fn(unquote(k)) do
      fn q, k_t, v -> apply(Edifice.CUDA.FusedScan, unquote(k), [q, k_t, v, [causal: true]]) end
    end
  end

  # 1-tensor + opts: titans
  defp make_fallback_bench_fn(:titans_scan) do
    fn combined ->
      {_, _, dim} = Nx.shape(combined)
      Edifice.CUDA.FusedScan.titans_scan(combined, memory_size: div(dim, 4))
    end
  end

  # 1-tensor + opts: miras
  defp make_fallback_bench_fn(:miras_scan) do
    fn combined ->
      {_, _, dim} = Nx.shape(combined)
      Edifice.CUDA.FusedScan.miras_scan(combined, memory_size: div(dim, 5))
    end
  end

  # 3-tensor + num_layers: block scans
  for k <- [:mingru_block, :minlstm_block, :linear_block, :gru_block] do
    defp make_fallback_bench_fn(unquote(k)) do
      fn input, weights, h0 ->
        apply(Edifice.CUDA.FusedScan, unquote(k), [input, weights, h0, 2])
      end
    end
  end

  # 4-tensor + num_layers: lstm_block
  defp make_fallback_bench_fn(:lstm_block) do
    fn input, weights, h0, c0 ->
      Edifice.CUDA.FusedScan.lstm_block(input, weights, h0, c0, 2)
    end
  end

  # ============================================================================
  # Input generators
  # ============================================================================

  @scan_2tensor_kernels [:mingru, :elu_gru, :real_gru, :diag_linear, :liquid, :linear_scan]
  @matmul_4h_kernels [:slstm_scan, :lstm_scan]
  @head_kernels [
    :delta_net_scan,
    :gated_delta_net_scan,
    :delta_product_scan,
    :kda_scan,
    :rla_scan,
    :gsa_scan
  ]
  @attention_kernels [:flash_attention, :laser_attention, :fox_attention]
  @block_kernels [:mingru_block, :minlstm_block, :linear_block, :gru_block, :lstm_block]

  defp generate_inputs(kernel_name, dim, dtype) do
    b = @bench_batch
    t = @bench_seq
    h = dim

    cond do
      kernel_name in @head_kernels -> head_inputs(kernel_name, b, t, h, dtype)
      kernel_name in @attention_kernels -> attention_inputs(kernel_name, b, t, h, dtype)
      kernel_name in @block_kernels -> block_inputs(kernel_name, b, t, h, dtype)
      true -> scan_inputs(kernel_name, b, t, h, dtype)
    end
  end

  # Element-wise / matmul scan kernels operating on {B,T,H}-shaped tensors.
  defp scan_inputs(k, b, t, h, dtype) when k in @scan_2tensor_kernels,
    do: [rand({b, t, h}, dtype), rand({b, t, h}, dtype)]

  # 3-tensor {B,T,H}
  defp scan_inputs(:minlstm, b, t, h, dtype),
    do: [rand({b, t, h}, dtype), rand({b, t, h}, dtype), rand({b, t, h}, dtype)]

  # Matmul scans: wx {B,T,4H} + R {H,4H}
  defp scan_inputs(k, b, t, h, dtype) when k in @matmul_4h_kernels,
    do: [rand({b, t, 4 * h}, dtype), rand({h, 4 * h}, dtype)]

  # Matmul scan: wx {B,T,3H} + R {H,3H}
  defp scan_inputs(:gru_scan, b, t, h, dtype),
    do: [rand({b, t, 3 * h}, dtype), rand({h, 3 * h}, dtype)]

  # TTT: q,k,v,eta {B,T,H} + W0 {H,H} + gamma,beta {H}
  defp scan_inputs(:ttt_scan, b, t, h, dtype) do
    [
      rand({b, t, h}, dtype),
      rand({b, t, h}, dtype),
      rand({b, t, h}, dtype),
      rand({b, t, h}, dtype),
      rand({h, h}, dtype),
      rand({h}, dtype),
      rand({h}, dtype)
    ]
  end

  # Selective scan: x,dt {B,T,H} + A {H,S} + B,C {B,T,S}
  defp scan_inputs(:selective_scan, b, t, h, dtype) do
    state_size = 16

    [
      rand({b, t, h}, dtype),
      rand({b, t, h}, dtype),
      rand({h, state_size}, dtype),
      rand({b, t, state_size}, dtype),
      rand({b, t, state_size}, dtype)
    ]
  end

  # Reservoir: wx {B,T,H} + w_res {H,H}
  defp scan_inputs(:reservoir_scan, b, t, h, dtype),
    do: [rand({b, t, h}, dtype), rand({h, h}, dtype)]

  # Titans: combined {B,T,4*M} where M = dim
  defp scan_inputs(:titans_scan, b, t, h, dtype), do: [rand({b, t, 4 * h}, dtype)]

  # MIRAS: combined {B,T,5*M} where M = dim
  defp scan_inputs(:miras_scan, b, t, h, dtype), do: [rand({b, t, 5 * h}, dtype)]

  defp scan_inputs(_kernel, _b, _t, _h, _dtype), do: :unsupported

  # Multi-head delta-family kernels operating on {B,T,heads,d}-shaped tensors.
  defp head_inputs(kernel, b, t, h, dtype) do
    {num_heads, head_dim} = head_split(h)
    qkv = rand({b, t, num_heads, head_dim}, dtype)
    head_inputs(kernel, qkv, b, t, num_heads, head_dim, dtype)
  end

  # DeltaNet: q,k,v,beta all {B,T,heads,d}
  defp head_inputs(:delta_net_scan, qkv, b, t, nh, d, dtype),
    do: [qkv, rand({b, t, nh, d}, dtype), rand({b, t, nh, d}, dtype), rand({b, t, nh, d}, dtype)]

  # GatedDeltaNet / KDA: q,k,v,beta {B,T,H,d} + alpha {B,T,H}
  defp head_inputs(k, qkv, b, t, nh, d, dtype) when k in [:gated_delta_net_scan, :kda_scan] do
    [
      qkv,
      rand({b, t, nh, d}, dtype),
      rand({b, t, nh, d}, dtype),
      rand({b, t, nh, d}, dtype),
      rand({b, t, nh}, dtype)
    ]
  end

  # DeltaProduct: q {B,T,H,d}, k/v {B,T,n_h,H,d}, beta {B,T,n_h,H}
  defp head_inputs(:delta_product_scan, qkv, b, t, nh, d, dtype) do
    n_h = 2

    [
      qkv,
      rand({b, t, n_h, nh, d}, dtype),
      rand({b, t, n_h, nh, d}, dtype),
      rand({b, t, n_h, nh}, dtype)
    ]
  end

  # RLA: q,k,v {B,T,H,d} + alpha,beta,gamma {B,T,H,1,1}
  defp head_inputs(:rla_scan, qkv, b, t, nh, d, dtype) do
    [
      qkv,
      rand({b, t, nh, d}, dtype),
      rand({b, t, nh, d}, dtype),
      rand({b, t, nh, 1, 1}, dtype),
      rand({b, t, nh, 1, 1}, dtype),
      rand({b, t, nh, 1, 1}, dtype)
    ]
  end

  # GSA: q {B,T,H,d}, k_slot {B,T,H,slots}, v {B,T,H,d}, alpha {B,T,H}
  defp head_inputs(:gsa_scan, qkv, b, t, nh, d, dtype) do
    num_slots = 4

    [
      qkv,
      rand({b, t, nh, num_slots}, dtype),
      rand({b, t, nh, d}, dtype),
      rand({b, t, nh}, dtype)
    ]
  end

  # Attention: q,k,v {B,heads,T,d} (+ cs {B,heads,T} for Fox)
  defp attention_inputs(kernel, b, t, h, dtype) do
    {num_heads, head_dim} = head_split(h)

    qkv = [
      rand({b, num_heads, t, head_dim}, dtype),
      rand({b, num_heads, t, head_dim}, dtype),
      rand({b, num_heads, t, head_dim}, dtype)
    ]

    case kernel do
      :fox_attention -> qkv ++ [rand({b, num_heads, t}, dtype)]
      _ -> qkv
    end
  end

  # Block scans: input {B,T,H} + flat weights {layers*stride} + h0 (and c0) {B,layers,H}
  defp block_inputs(kernel, b, t, h, dtype) do
    num_layers = 2
    layer_stride = block_layer_stride(kernel, h)

    base = [
      rand({b, t, h}, dtype),
      rand({num_layers * layer_stride}, dtype),
      rand({b, num_layers, h}, dtype)
    ]

    case kernel do
      :lstm_block -> base ++ [rand({b, num_layers, h}, dtype)]
      _ -> base
    end
  end

  defp block_layer_stride(k, h) when k in [:mingru_block, :linear_block], do: 2 * h * h + 4 * h
  defp block_layer_stride(:minlstm_block, h), do: 3 * h * h + 5 * h
  defp block_layer_stride(:gru_block, h), do: 6 * h * h + 5 * h
  defp block_layer_stride(:lstm_block, h), do: 8 * h * h + 6 * h

  defp rand(shape, dtype) do
    key = Nx.Random.key(System.os_time())
    {tensor, _key} = Nx.Random.uniform(key, shape: shape, type: dtype)
    tensor
  end

  defp head_split(h) do
    num_heads = min(4, h)
    head_dim = max(div(h, num_heads), 1)
    {num_heads, head_dim}
  end

  # ============================================================================
  # Disk cache helpers
  # ============================================================================

  defp get_gpu_name do
    case System.cmd("nvidia-smi", ["--query-gpu=name", "--format=csv,noheader,nounits"],
           stderr_to_stdout: true
         ) do
      {name, 0} -> String.trim(name)
      _ -> "unknown"
    end
  rescue
    _ -> "unknown"
  end

  defp cached_results do
    :persistent_term.get()
    |> Enum.filter(fn {key, _val} ->
      match?({:edifice_autotune, _, _, _}, key)
    end)
    |> Enum.map(fn {{:edifice_autotune, kernel, dim, dtype}, winner} ->
      {kernel, dim, dtype, winner}
    end)
    |> Enum.sort()
  end

  defp parse_cache_key(key_str) do
    case String.split(key_str, ":") do
      [kernel, dim_str, dtype_str] ->
        kernel_atom = String.to_existing_atom(kernel)
        dim = String.to_integer(dim_str)
        dtype = parse_dtype(dtype_str)
        if dtype != :error, do: {:ok, kernel_atom, dim, dtype}, else: :error

      _ ->
        :error
    end
  rescue
    _ -> :error
  end

  defp format_dtype({:f, 32}), do: "f32"
  defp format_dtype({:bf, 16}), do: "bf16"
  defp format_dtype(dtype), do: inspect(dtype)

  defp parse_dtype("f32"), do: {:f, 32}
  defp parse_dtype("bf16"), do: {:bf, 16}
  defp parse_dtype(_), do: :error

  defp encode_json(data) do
    data |> JSON.encode!() |> IO.iodata_to_binary()
  end

  defp decode_json(string) do
    JSON.decode!(string)
  end
end
