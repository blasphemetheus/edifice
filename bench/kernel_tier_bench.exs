# Kernel tier benchmark: XLA fallback vs EXLA FFI custom call vs NIF bridge.
#
# Times the two ops that have handlers in every tier (GRU scan, Mamba
# selective scan) and the two models built on them, so the three dispatch
# tiers in `Edifice.CUDA.FusedScan` can be compared on one GPU:
#
#   xla        the pure-Nx fallback compiled by EXLA inside `Nx.Defn.jit`
#              (what every jitted model gets without a native handler)
#   ffi        the CUDA kernel as an XLA custom call inside the same jit
#              (`EDIFICE_FUSED_CUSTOM_CALL=1`, `libedifice_xla.so`)
#   nif        the same kernel through the C NIF on eager EXLA tensors
#              (forward only; gradients run inside jit where the NIF cannot)
#   eager_xla  the fallback evaluated op by op on eager EXLA tensors
#              (the NIF's only competitor when nothing is jitted)
#
# Reports first-call time (trace + XLA compile), steady-state median and
# min, and the max abs difference of each tier from the xla tier.
#
# Usage (GPU host, libraries built by scripts/check_cuda_kernels.sh):
#
#   EDIFICE_FUSED_CUSTOM_CALL=1 EDIFICE_AUTOTUNE=0 \
#     mix run bench/kernel_tier_bench.exs
#
# Env:
#   BENCH_OUT     directory for README.md + cuda.json (default: print only)
#   BENCH_QUICK=1 small grid, fewer iterations (smoke run)
#   BENCH_ITERS   timed iterations per cell (default 30)

Logger.configure(level: :warning)

unless System.get_env("EDIFICE_FUSED_CUSTOM_CALL") == "1",
  do: raise("set EDIFICE_FUSED_CUSTOM_CALL=1 (the FFI library loads at VM start)")

unless System.get_env("EDIFICE_AUTOTUNE") == "0",
  do: raise("set EDIFICE_AUTOTUNE=0 so the ffi tier is taken deterministically")

# The NIF allocates with cudaMalloc outside XLA's pool; leave it room.
clients = Application.get_env(:exla, :clients, [])
cuda = Keyword.merge(Keyword.get(clients, :cuda, platform: :cuda), memory_fraction: 0.6)
Application.put_env(:exla, :clients, Keyword.put(clients, :cuda, cuda))
Application.put_env(:exla, :default_client, :cuda)

defmodule KernelTierBench do
  @moduledoc false
  alias Edifice.CUDA.FusedScan

  @cuda {EXLA.Backend, client: :cuda}
  @quick System.get_env("BENCH_QUICK") == "1"
  @iters String.to_integer(System.get_env("BENCH_ITERS", if(@quick, do: "10", else: "30")))
  @warmup 5
  @eager_iters if(@quick, do: 3, else: 8)

  # {batch, seq_len, hidden}. The fallbacks are unrolled over seq_len and
  # their gradient graphs take XLA tens of seconds to compile at T=64, so
  # the grid stops at T=128 (the backward kernels allow up to 256).
  @gru_configs if @quick,
                 do: [{1, 32, 256}, {8, 64, 512}],
                 else: [
                   {1, 32, 256},
                   {1, 128, 512},
                   {32, 128, 512},
                   {32, 128, 1024}
                 ]

  # {batch, seq_len, hidden, state}
  @ssm_configs if @quick,
                 do: [{1, 32, 256, 16}],
                 else: [
                   {1, 32, 256, 16},
                   {1, 128, 512, 16},
                   {32, 128, 512, 16},
                   {32, 128, 1024, 16}
                 ]

  # {batch, seq_len, hidden} for the 2-layer models
  @model_configs if @quick,
                   do: [{1, 32, 256}],
                   else: [{1, 60, 512}, {32, 60, 512}, {32, 128, 1024}]

  # ── timing ─────────────────────────────────────────────────────────

  # Block until every output buffer is defined (EXLA executes asynchronously).
  # Outputs may be tensors, tuples or any Nx.Container (Axon.ModelState).
  defp sync(out), do: Nx.Defn.Composite.reduce(out, :ok, fn t, :ok -> sync_tensor(t) end)

  defp sync_tensor(%Nx.Tensor{data: %EXLA.Backend{buffer: %EXLA.DeviceBuffer{} = buf}} = t) do
    _ = EXLA.DeviceBuffer.read(buf, elem_bytes(t))
    :ok
  end

  defp sync_tensor(%Nx.Tensor{} = t),
    do:
      (
        _ = Nx.to_binary(t)
        :ok
      )

  defp elem_bytes(t) do
    {_, bits} = Nx.type(t)
    div(bits, 8)
  end

  defp timed(fun) do
    {us, out} =
      :timer.tc(fn ->
        out = fun.()
        sync(out)
        out
      end)

    {us / 1000, out}
  end

  # Returns %{first: ms, median: ms, min: ms, p95: ms, out: last output}
  defp measure(fun, iters) do
    {first, out} = timed(fun)
    for _ <- 1..@warmup, do: timed(fun)
    samples = for _ <- 1..iters, do: elem(timed(fun), 0)
    sorted = Enum.sort(samples)
    n = length(sorted)

    %{
      first: first,
      median: Enum.at(sorted, div(n, 2)),
      min: hd(sorted),
      p95: Enum.at(sorted, min(n - 1, round(n * 0.95))),
      out: out
    }
  end

  defp forced_fallback(fun) do
    Process.put(:__edifice_force_fallback__, true)

    try do
      fun.()
    after
      Process.delete(:__edifice_force_fallback__)
    end
  end

  # Flatten any container to host tensors so tiers can be compared.
  defp to_host(out),
    do: Nx.Defn.Composite.flatten_list([out]) |> Enum.map(&Nx.backend_copy(&1, Nx.BinaryBackend))

  # {max |a - b|, that divided by max |b|} over every tensor in the output.
  defp max_diff(a, b) when is_list(a) and is_list(b) do
    Enum.zip(a, b)
    |> Enum.map(fn {x, y} ->
      if Nx.to_number(Nx.any(Nx.is_nan(x))) == 1 do
        {:nan, :nan}
      else
        abs = Nx.subtract(x, y) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
        scale = Nx.abs(y) |> Nx.reduce_max() |> Nx.to_number()
        {abs, abs / max(scale, 1.0e-30)}
      end
    end)
    |> Enum.max_by(fn
      {:nan, _} -> :infinity
      {v, _} -> v
    end)
  end

  # The eager fallback: trace the same entry point under the Evaluator, which
  # runs every op one by one on the (CUDA) EXLA backend of the arguments.
  # Calling the entry point eagerly would reach the NIF arm instead.
  defp eager(fun), do: Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator)

  defp hp(fun), do: Nx.Defn.jit(fun, compiler: EXLA, precision: :highest)

  # Prove a tier really took the dispatch arm it claims (the profiler counts
  # every `record_dispatch`); `shape` may be `:any`.
  defp assert_dispatch(kernel, path, shape \\ :any) do
    hits =
      :ets.tab2list(:__edifice_autotune_profiler__)
      |> Enum.filter(fn {{k, p, s, _}, n} ->
        k == kernel and p == path and n > 0 and (shape == :any or s == shape)
      end)

    if hits == [], do: raise("#{kernel} #{inspect(shape)} never dispatched via #{path}")
    :ok
  end

  defp assert_nif_dispatch(kernel, shape), do: assert_dispatch(kernel, :nif, shape)

  defp rand(key, shape, lo, hi) do
    {t, key} = Nx.Random.uniform(key, lo, hi, shape: shape, type: {:f, 32})
    {Nx.backend_transfer(t, @cuda), key}
  end

  # ── op-level cases ─────────────────────────────────────────────────

  # Each case returns the ordered list of tiers:
  #   {tier_name, measure_result} with tier_name in
  #   xla ffi nif eager_xla xla_grad ffi_grad
  defp gru_case({b, t, h}) do
    key = Nx.Random.key(42)
    {wx, key} = rand(key, {b, t, 3 * h}, -0.5, 0.5)
    {r, key} = rand(key, {h, 3 * h}, -0.2, 0.2)
    {h0, key} = rand(key, {b, h}, -0.5, 0.5)
    {bhn, _} = rand(key, {h}, -0.3, 0.3)

    fwd_xla =
      Nx.Defn.jit(fn wx, r, h0, bhn -> FusedScan.gru_scan(wx, r, h0, bhn) end, compiler: EXLA)

    fwd_hp = hp(fn wx, r, h0, bhn -> FusedScan.gru_scan(wx, r, h0, bhn) end)

    fwd_ffi =
      Nx.Defn.jit(fn wx, r, h0, bhn -> FusedScan.gru_scan(wx, r, h0, bhn) end, compiler: EXLA)

    fwd_eager = eager(fn wx, r, h0, bhn -> FusedScan.gru_scan(wx, r, h0, bhn) end)

    grad_xla =
      Nx.Defn.jit(
        fn wx, r, bhn, h0 ->
          Nx.Defn.grad({wx, r, bhn}, fn {wx, r, bhn} ->
            FusedScan.gru_scan(wx, r, h0, bhn) |> Nx.pow(2) |> Nx.sum()
          end)
        end,
        compiler: EXLA
      )

    grad_hp =
      hp(fn wx, r, bhn, h0 ->
        Nx.Defn.grad({wx, r, bhn}, fn {wx, r, bhn} ->
          FusedScan.gru_scan(wx, r, h0, bhn) |> Nx.pow(2) |> Nx.sum()
        end)
      end)

    grad_ffi =
      Nx.Defn.jit(
        fn wx, r, bhn, h0 ->
          Nx.Defn.grad({wx, r, bhn}, fn {wx, r, bhn} ->
            FusedScan.gru_scan(wx, r, h0, bhn) |> Nx.pow(2) |> Nx.sum()
          end)
        end,
        compiler: EXLA
      )

    run_tiers(
      xla: fn ->
        forced_fallback(fn -> measure(fn -> fwd_xla.(wx, r, h0, bhn) end, @iters) end)
      end,
      xla_hp: fn ->
        forced_fallback(fn -> measure(fn -> fwd_hp.(wx, r, h0, bhn) end, @iters) end)
      end,
      ffi: fn -> measure(fn -> fwd_ffi.(wx, r, h0, bhn) end, @iters) end,
      nif: fn ->
        res =
          forced_fallback(fn -> measure(fn -> FusedScan.gru_scan(wx, r, h0, bhn) end, @iters) end)

        assert_nif_dispatch(:gru_scan, {b, t, 3 * h})
        res
      end,
      eager_xla: fn ->
        forced_fallback(fn -> measure(fn -> fwd_eager.(wx, r, h0, bhn) end, @eager_iters) end)
      end,
      xla_grad: fn ->
        forced_fallback(fn -> measure(fn -> grad_xla.(wx, r, bhn, h0) end, @iters) end)
      end,
      xla_hp_grad: fn ->
        forced_fallback(fn -> measure(fn -> grad_hp.(wx, r, bhn, h0) end, @iters) end)
      end,
      ffi_grad: fn -> measure(fn -> grad_ffi.(wx, r, bhn, h0) end, @iters) end
    )
  end

  defp ssm_case({b, t, h, n}) do
    key = Nx.Random.key(7)
    {x, key} = rand(key, {b, t, h}, -0.5, 0.5)
    {dt, key} = rand(key, {b, t, h}, 0.001, 0.1)
    {bm, key} = rand(key, {b, t, n}, -0.5, 0.5)
    {cm, _} = rand(key, {b, t, n}, -0.5, 0.5)
    a_diag = Nx.negate(Nx.add(Nx.iota({n}, type: {:f, 32}), 1.0))
    a = Nx.broadcast(Nx.reshape(a_diag, {1, n}), {h, n}) |> Nx.backend_transfer(@cuda)

    fwd_xla =
      Nx.Defn.jit(fn x, dt, a, b, c -> FusedScan.selective_scan(x, dt, a, b, c) end,
        compiler: EXLA
      )

    fwd_hp = hp(fn x, dt, a, b, c -> FusedScan.selective_scan(x, dt, a, b, c) end)

    fwd_ffi =
      Nx.Defn.jit(fn x, dt, a, b, c -> FusedScan.selective_scan(x, dt, a, b, c) end,
        compiler: EXLA
      )

    fwd_eager = eager(fn x, dt, a, b, c -> FusedScan.selective_scan(x, dt, a, b, c) end)

    grad_xla =
      Nx.Defn.jit(
        fn x, dt, b, c, a ->
          Nx.Defn.grad({x, dt, b, c}, fn {x, dt, b, c} ->
            FusedScan.selective_scan(x, dt, a, b, c) |> Nx.pow(2) |> Nx.sum()
          end)
        end,
        compiler: EXLA
      )

    grad_hp =
      hp(fn x, dt, b, c, a ->
        Nx.Defn.grad({x, dt, b, c}, fn {x, dt, b, c} ->
          FusedScan.selective_scan(x, dt, a, b, c) |> Nx.pow(2) |> Nx.sum()
        end)
      end)

    grad_ffi =
      Nx.Defn.jit(
        fn x, dt, b, c, a ->
          Nx.Defn.grad({x, dt, b, c}, fn {x, dt, b, c} ->
            FusedScan.selective_scan(x, dt, a, b, c) |> Nx.pow(2) |> Nx.sum()
          end)
        end,
        compiler: EXLA
      )

    run_tiers(
      xla: fn ->
        forced_fallback(fn -> measure(fn -> fwd_xla.(x, dt, a, bm, cm) end, @iters) end)
      end,
      xla_hp: fn ->
        forced_fallback(fn -> measure(fn -> fwd_hp.(x, dt, a, bm, cm) end, @iters) end)
      end,
      ffi: fn -> measure(fn -> fwd_ffi.(x, dt, a, bm, cm) end, @iters) end,
      nif: fn ->
        res =
          forced_fallback(fn ->
            measure(fn -> FusedScan.selective_scan(x, dt, a, bm, cm) end, @iters)
          end)

        assert_nif_dispatch(:selective_scan, {b, t, h})
        res
      end,
      eager_xla: fn ->
        forced_fallback(fn -> measure(fn -> fwd_eager.(x, dt, a, bm, cm) end, @eager_iters) end)
      end,
      xla_grad: fn ->
        forced_fallback(fn -> measure(fn -> grad_xla.(x, dt, bm, cm, a) end, @iters) end)
      end,
      xla_hp_grad: fn ->
        forced_fallback(fn -> measure(fn -> grad_hp.(x, dt, bm, cm, a) end, @iters) end)
      end,
      ffi_grad: fn -> measure(fn -> grad_ffi.(x, dt, bm, cm, a) end, @iters) end
    )
  end

  # ── model-level cases (2 layers, Axon.build compiler: EXLA) ────────

  # `xla` here is whatever the model does without the custom-call tier:
  # GRU -> the unrolled gru_scan fallback in-graph (the NIF is loaded so the
  # fused parameter layout is still chosen); Mamba -> the Blelloch parallel
  # scan, which is its production path without the kernel.
  defp model_case(arch, {b, t, h}) do
    build_opts = [embed_dim: h, hidden_size: h, num_layers: 2, seq_len: t, dropout: 0.0]

    build_opts =
      if arch == :mamba, do: Keyword.put(build_opts, :state_size, 16), else: build_opts

    {x, _} = rand(Nx.Random.key(3), {b, t, h}, -1.0, 1.0)
    input = %{"state_sequence" => x}

    # Build once per tier so the layer graph reflects that tier's dispatch
    # decision (`custom_call_available?` is consulted at build/trace time).
    build = fn ->
      model = Edifice.build(arch, build_opts)
      {init_fn, predict_fn} = Axon.build(model, mode: :inference, seed: 0)
      params = init_fn.(Nx.template({b, t, h}, :f32), Axon.ModelState.empty())
      params = Nx.backend_transfer(params, @cuda)
      {predict_fn, params}
    end

    # EXLA keys its compilation cache on the closure (code + captured env),
    # so each tier needs its own literal `fn` here: two identical builds
    # wrapped by the same helper closure would share one executable.
    {pf_x, params_x} = forced_fallback(build)
    # Same layout in both tiers (the NIF is loaded, so the fused GRU layout
    # is chosen either way); share the parameters so outputs are comparable.
    {pf_f, _} = build.()
    params_f = params_x

    loss = fn pf, p, x -> pf.(p, x) |> Nx.pow(2) |> Nx.sum() end
    fwd_xla = Nx.Defn.jit(fn p, x -> pf_x.(p, x) end, compiler: EXLA)
    fwd_hp = hp(fn p, x -> pf_x.(p, x) end)
    fwd_ffi = Nx.Defn.jit(fn p, x -> pf_f.(p, x) end, compiler: EXLA)
    grad_xla = Nx.Defn.jit(fn p, x -> Nx.Defn.grad(p, &loss.(pf_x, &1, x)) end, compiler: EXLA)
    grad_hp = hp(fn p, x -> Nx.Defn.grad(p, &loss.(pf_x, &1, x)) end)
    grad_ffi = Nx.Defn.jit(fn p, x -> Nx.Defn.grad(p, &loss.(pf_f, &1, x)) end, compiler: EXLA)

    kernel = if arch == :mamba, do: :selective_scan, else: :gru_scan
    Edifice.CUDA.AutoTuneProfiler.stop()
    Edifice.CUDA.AutoTuneProfiler.start()

    # Mamba's no-kernel path is the Blelloch scan (never enters FusedScan),
    # so only the custom-call arm can be asserted for both models.
    run_tiers(
      xla: fn ->
        forced_fallback(fn -> measure(fn -> fwd_xla.(params_x, input) end, @iters) end)
      end,
      xla_hp: fn ->
        forced_fallback(fn -> measure(fn -> fwd_hp.(params_x, input) end, @iters) end)
      end,
      ffi: fn ->
        res = measure(fn -> fwd_ffi.(params_f, input) end, @iters)
        assert_dispatch(kernel, :custom_call)
        res
      end,
      xla_grad: fn ->
        forced_fallback(fn -> measure(fn -> grad_xla.(params_x, input) end, @iters) end)
      end,
      xla_hp_grad: fn ->
        forced_fallback(fn -> measure(fn -> grad_hp.(params_x, input) end, @iters) end)
      end,
      ffi_grad: fn -> measure(fn -> grad_ffi.(params_f, input) end, @iters) end
    )
  end

  # Run each tier, attach the max abs diff from the `xla` (or `xla_grad`)
  # tier, drop the outputs.
  defp run_tiers(tiers) do
    results =
      Enum.map(tiers, fn {name, fun} ->
        IO.write("    #{name} ")
        res = fun.()
        IO.puts("median #{fmt(res.median)} ms (first #{fmt(res.first)} ms)")
        {name, res}
      end)

    # Reference = the exact-f32 XLA tiers; the default-precision ones use
    # TF32 matmuls and drift visibly over a 128-step recurrence at batch 32.
    ref_fwd = results[:xla_hp] && to_host(results[:xla_hp].out)
    ref_grad = results[:xla_hp_grad] && to_host(results[:xla_hp_grad].out)

    Enum.map(results, fn {name, res} ->
      ref = if String.ends_with?(to_string(name), "grad"), do: ref_grad, else: ref_fwd
      diff = if ref, do: max_diff(to_host(res.out), ref), else: nil

      {name, res |> Map.delete(:out) |> Map.put(:max_abs_diff, diff && Tuple.to_list(diff))}
      |> tap(fn {_, r} ->
        if diff,
          do:
            IO.puts(
              "      max abs diff vs xla_hp #{sci(elem(diff, 0))} (rel #{sci(elem(diff, 1))})"
            ),
          else: r
      end)
    end)
  end

  # ── report ─────────────────────────────────────────────────────────

  defp fmt(nil), do: "-"
  defp fmt(:nan), do: "NaN"
  defp fmt(ms) when ms >= 100, do: :erlang.float_to_binary(ms / 1, decimals: 0)
  defp fmt(ms) when ms >= 10, do: :erlang.float_to_binary(ms / 1, decimals: 1)
  defp fmt(ms) when ms >= 1, do: :erlang.float_to_binary(ms / 1, decimals: 2)
  defp fmt(ms), do: :erlang.float_to_binary(ms / 1, decimals: 3)

  defp sci(nil), do: "-"
  defp sci(:nan), do: "NaN"
  defp sci(x), do: :erlang.float_to_binary(x / 1, scientific: 1)

  defp speedup(nil, _), do: "-"
  defp speedup(_, nil), do: "-"
  defp speedup(base, this), do: :erlang.float_to_binary(base / this, decimals: 1) <> "x"

  defp table(title, rows, tiers) do
    header = "| config | " <> Enum.map_join(tiers, " | ", &to_string/1) <> " |"
    sep = "|---|" <> String.duplicate("---:|", length(tiers))

    body =
      Enum.map_join(rows, "\n", fn {label, res} ->
        "| #{label} | " <>
          Enum.map_join(tiers, " | ", fn tier ->
            case res[tier] do
              nil -> "-"
              %{median: m} -> fmt(m)
            end
          end) <> " |"
      end)

    "### #{title}\n\n#{header}\n#{sep}\n#{body}\n"
  end

  defp detail_table(title, rows) do
    header =
      "| config | tier | first call | median | min | p95 | max abs diff vs xla_hp | / max abs xla_hp |"

    sep = "|---|---|---:|---:|---:|---:|---:|---:|"

    body =
      Enum.flat_map(rows, fn {label, res} ->
        Enum.map(res, fn {tier, r} ->
          [abs, rel] = r.max_abs_diff || [nil, nil]

          "| #{label} | #{tier} | #{fmt(r.first)} | #{fmt(r.median)} | #{fmt(r.min)} | #{fmt(r.p95)} | #{sci(abs)} | #{sci(rel)} |"
        end)
      end)
      |> Enum.join("\n")

    "### #{title}\n\n#{header}\n#{sep}\n#{body}\n"
  end

  defp speedup_table(title, rows) do
    header =
      "| config | ffi vs xla (fwd) | ffi vs xla_hp (fwd) | nif vs xla (fwd) | nif vs eager_xla | ffi vs xla (grad) | ffi vs xla_hp (grad) |"

    sep = "|---|---:|---:|---:|---:|---:|---:|"

    body =
      Enum.map_join(rows, "\n", fn {label, res} ->
        med = fn tier -> res[tier] && res[tier].median end

        "| #{label} | #{speedup(med.(:xla), med.(:ffi))} | #{speedup(med.(:xla_hp), med.(:ffi))} | " <>
          "#{speedup(med.(:xla), med.(:nif))} | #{speedup(med.(:eager_xla), med.(:nif))} | " <>
          "#{speedup(med.(:xla_grad), med.(:ffi_grad))} | #{speedup(med.(:xla_hp_grad), med.(:ffi_grad))} |"
      end)

    "### #{title}\n\n#{header}\n#{sep}\n#{body}\n"
  end

  defp gpu_name do
    case System.cmd("nvidia-smi", ["--query-gpu=name,driver_version", "--format=csv,noheader"],
           stderr_to_stdout: true
         ) do
      {out, 0} -> String.trim(out)
      _ -> "unknown GPU"
    end
  end

  defp version(app), do: Application.spec(app, :vsn) |> to_string()

  def run do
    date = NaiveDateTime.local_now() |> NaiveDateTime.to_date() |> Date.to_iso8601()
    IO.puts("Kernel tier benchmark — #{gpu_name()} — nx #{version(:nx)} exla #{version(:exla)}")
    IO.puts("iters=#{@iters} warmup=#{@warmup} quick=#{@quick}")
    unless Edifice.CUDA.NIF.loaded?(), do: raise("NIF not loaded; build with make -C c_src")
    Edifice.CUDA.Library.ensure_loaded!()
    Edifice.CUDA.AutoTuneProfiler.start()

    gru =
      for {b, t, h} = cfg <- @gru_configs do
        label = "B=#{b} T=#{t} H=#{h}"
        IO.puts("\n## gru_scan #{label}")
        {label, gru_case(cfg)}
      end

    ssm =
      for {b, t, h, n} = cfg <- @ssm_configs do
        label = "B=#{b} T=#{t} H=#{h} N=#{n}"
        IO.puts("\n## selective_scan #{label}")
        {label, ssm_case(cfg)}
      end

    models =
      for arch <- [:gru, :mamba], {b, t, h} = cfg <- @model_configs do
        label = "#{arch} 2L B=#{b} T=#{t} H=#{h}"
        IO.puts("\n## model #{label}")
        {label, model_case(arch, cfg)}
      end

    op_tiers = [:xla, :xla_hp, :ffi, :nif, :eager_xla, :xla_grad, :xla_hp_grad, :ffi_grad]
    model_tiers = [:xla, :xla_hp, :ffi, :xla_grad, :xla_hp_grad, :ffi_grad]

    md = """
    # Kernel dispatch tiers — #{date}

    #{gpu_name()}; nx #{version(:nx)}, exla #{version(:exla)}, axon #{version(:axon)}.
    Generated by `bench/kernel_tier_bench.exs` (#{@iters} timed iterations after
    #{@warmup} warmups; eager_xla uses #{@eager_iters}). Times are milliseconds and
    include a one-element device read so asynchronous EXLA execution is
    complete. "first call" is trace + XLA compile for the jitted tiers.

    Tiers: `xla` = pure-Nx fallback compiled in-graph (EXLA default precision,
    TF32 matmuls); `xla_hp` = the same at `precision: :highest` (exact f32, like
    the kernels); `ffi` = CUDA kernel as an XLA custom call in the same graph;
    `nif` = same kernel through the C NIF on eager tensors (forward only);
    `eager_xla` = fallback op by op on eager tensors under `Nx.Defn.Evaluator`.
    `*_grad` = forward + gradient of a sum-of-squares loss (default precision).

    ## Medians (ms)

    #{table("gru_scan", gru, op_tiers)}
    #{table("selective_scan", ssm, op_tiers)}
    #{table("models (2 layers, Axon.build compiler: EXLA)", models, model_tiers)}
    ## Speedups (median over median)

    #{speedup_table("gru_scan", gru)}
    #{speedup_table("selective_scan", ssm)}
    #{speedup_table("models", models)}
    ## Detail

    #{detail_table("gru_scan", gru)}
    #{detail_table("selective_scan", ssm)}
    #{detail_table("models", models)}
    """

    IO.puts("\n" <> md)

    if out = System.get_env("BENCH_OUT") do
      File.mkdir_p!(out)
      File.write!(Path.join(out, "README.md"), md)

      json = %{
        date: date,
        gpu: gpu_name(),
        versions: %{nx: version(:nx), exla: version(:exla), axon: version(:axon)},
        iters: @iters,
        gru_scan: Map.new(gru, fn {k, v} -> {k, Map.new(v)} end),
        selective_scan: Map.new(ssm, fn {k, v} -> {k, Map.new(v)} end),
        models: Map.new(models, fn {k, v} -> {k, Map.new(v)} end)
      }

      File.write!(Path.join(out, "cuda.json"), Jason.encode!(json, pretty: true))
      IO.puts("wrote #{out}/README.md and cuda.json")
    end
  end
end

KernelTierBench.run()
