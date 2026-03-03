# Broad Fused-Kernel Benchmark
#
# Comprehensive benchmark measuring actual speedups across all fused CUDA
# kernels and the architectures that use them.
#
# Phase 1: Raw kernel latency — eager FusedScan calls on EXLA tensors.
#           Some custom calls work eagerly (delta_product, slstm, etc.),
#           others require the full Axon compilation pipeline and will fail here.
#           Fallback comparison uses BinaryBackend (public fallbacks only).
#
# Phase 2: Full model inference — Edifice.build/2 + Axon.build(compiler: EXLA).
#           This is the proper pipeline that triggers custom_call dispatch
#           for ALL kernels. The primary benchmark.
#
# Usage:
#   EDIFICE_LOCAL_NX=1 nix-shell --run "mix run bench/fused_kernel_sweep.exs" shell.nix
#
# Environment variables:
#   BENCH_BATCH    - Batch size (default: 1)
#   BENCH_SEQ_LEN  - Sequence length (default: 32)
#   BENCH_HIDDEN   - Hidden dimension (default: 256)
#   BENCH_HEADS    - Number of heads (default: 8)
#   BENCH_HEAD_DIM - Head dimension (default: 32)
#   BENCH_STATE    - SSM state size (default: 16)

Nx.default_backend(EXLA.Backend)
Logger.configure(level: :warning)

defmodule FusedKernelSweep do
  @moduledoc false

  alias Edifice.CUDA.FusedScan

  # ── Configuration ──────────────────────────────────────────────

  @batch String.to_integer(System.get_env("BENCH_BATCH", "1"))
  @seq_len String.to_integer(System.get_env("BENCH_SEQ_LEN", "32"))
  @hidden String.to_integer(System.get_env("BENCH_HIDDEN", "256"))
  @heads String.to_integer(System.get_env("BENCH_HEADS", "8"))
  @head_dim String.to_integer(System.get_env("BENCH_HEAD_DIM", "32"))
  @state_size String.to_integer(System.get_env("BENCH_STATE", "16"))
  @memory_size 64
  @num_slots 8

  @fused_warmup 10
  @fused_iters 100
  @fallback_warmup 2
  @fallback_iters 10

  # Kernels that segfault when called eagerly (custom_call bugs in eager mode).
  # These still work correctly through the Axon compilation pipeline in Phase 2.
  @phase1_skip MapSet.new([
    "reservoir", "titans", "miras", "gsa",
    "fox_attention", "laser_attention", "flash_attention"
  ])

  @phase2_warmup 5
  @phase2_iters 50

  # Architectures that previously segfaulted (fixed via scalar packing / causal hardcoding).
  @phase2_skip MapSet.new([])

  @fps_target_ms 16.0

  # ── Helpers ────────────────────────────────────────────────────

  defp rand(shape) do
    key = Nx.Random.key(:rand.uniform(100_000))
    {t, _} = Nx.Random.uniform(key, -2.0, 2.0, shape: shape)
    t
  end

  defp rand_positive(shape) do
    key = Nx.Random.key(:rand.uniform(100_000))
    {t, _} = Nx.Random.uniform(key, 0.01, 2.0, shape: shape)
    t
  end

  defp to_cpu(t), do: Nx.backend_copy(t, Nx.BinaryBackend)

  defp measure(warmup, iters, fun) do
    for _ <- 1..warmup, do: fun.()

    times =
      for _ <- 1..iters do
        {us, _} = :timer.tc(fun)
        us
      end

    times = Enum.sort(times)
    Enum.at(times, div(length(times), 2))
  end

  defp fmt_ms(ms) when ms < 0.01, do: "#{Float.round(ms * 1000, 1)} us"
  defp fmt_ms(ms) when ms < 1, do: "#{Float.round(ms, 3)} ms"
  defp fmt_ms(ms) when ms < 100, do: "#{Float.round(ms, 2)} ms"
  defp fmt_ms(ms), do: "#{Float.round(ms, 0)} ms"

  defp sep, do: String.duplicate("-", 72)

  # ── Phase 1: Raw kernel latency (eager dispatch) ──────────────
  #
  # Calls FusedScan functions eagerly on EXLA tensors. Custom calls that
  # support eager execution will run on GPU; others will fail (caught).
  # For kernels that succeed AND have public fallbacks, also measures
  # the Elixir sequential scan on BinaryBackend for speedup comparison.

  defp kernel_specs do
    b = @batch
    t = @seq_len
    h = @hidden
    heads = @heads
    d = @head_dim
    n = @state_size

    # EXLA inputs
    g1 = rand({b, t, h})
    g2 = rand({b, t, h})
    g3 = rand({b, t, h})

    q4 = rand({b, t, heads, d})
    k4 = rand({b, t, heads, d})
    v4 = rand({b, t, heads, d})
    beta4d = Nx.sigmoid(rand({b, t, heads, d}))
    alpha4 = Nx.sigmoid(rand({b, t, heads}))
    alpha4d = rand({b, t, heads, d})

    rla_a = Nx.sigmoid(rand({b, t, heads, 1, 1}))
    rla_b = Nx.sigmoid(rand({b, t, heads, 1, 1}))
    rla_g = Nx.sigmoid(rand({b, t, heads, 1, 1}))

    ss_x = rand({b, t, h})
    ss_dt = rand_positive({b, t, h})
    ss_a = Nx.negate(rand_positive({h, n}))
    ss_b = rand({b, t, n})
    ss_c = rand({b, t, n})

    wx4h = rand({b, t, 4 * h})
    wx3h = rand({b, t, 3 * h})
    r4h = rand({h, 4 * h})
    r3h = rand({h, 3 * h})

    ttt_q = rand({b, t, d})
    ttt_k = rand({b, t, d})
    ttt_v = rand({b, t, d})
    ttt_eta = Nx.sigmoid(rand({b, t, d}))
    ttt_w0 = rand({d, d})
    ttt_lng = rand({d})
    ttt_lnb = rand({d})

    aq = rand({b, heads, t, d})
    ak = rand({b, heads, t, d})
    av = rand({b, heads, t, d})
    cs = rand({b, heads, t})

    n_h = 2
    dp_q = rand({b, t, heads, d})
    dp_k = rand({b, t, n_h, heads, d})
    dp_v = rand({b, t, n_h, heads, d})
    dp_beta = Nx.sigmoid(rand({b, t, n_h, heads}))

    res_wx = rand({b, t, h})
    res_w = rand({h, h})
    tc = rand({b, t, 4 * @memory_size})
    mc = rand({b, t, 5 * @memory_size})

    gsa_q = rand({b, t, heads, d})
    gsa_ks = Nx.exp(rand({b, t, heads, @num_slots}))
    gsa_ks = Nx.divide(gsa_ks, Nx.sum(gsa_ks, axes: [3], keep_axes: true))
    gsa_v = rand({b, t, heads, d})
    gsa_a = Nx.sigmoid(rand({b, t, heads}))

    # BinaryBackend copies for fallback (only for kernels with public fallbacks)
    g1_c = to_cpu(g1)
    g2_c = to_cpu(g2)
    g3_c = to_cpu(g3)
    q4_c = to_cpu(q4)
    k4_c = to_cpu(k4)
    v4_c = to_cpu(v4)
    beta4d_c = to_cpu(beta4d)
    alpha4_c = to_cpu(alpha4)
    ss_x_c = to_cpu(ss_x)
    ss_dt_c = to_cpu(ss_dt)
    ss_a_c = to_cpu(ss_a)
    ss_b_c = to_cpu(ss_b)
    ss_c_c = to_cpu(ss_c)
    wx4h_c = to_cpu(wx4h)
    r4h_c = to_cpu(r4h)
    dp_q_c = to_cpu(dp_q)
    dp_k_c = to_cpu(dp_k)
    dp_v_c = to_cpu(dp_v)
    dp_beta_c = to_cpu(dp_beta)
    ttt_q_c = to_cpu(ttt_q)
    ttt_k_c = to_cpu(ttt_k)
    ttt_v_c = to_cpu(ttt_v)
    ttt_eta_c = to_cpu(ttt_eta)
    ttt_w0_c = to_cpu(ttt_w0)
    ttt_lng_c = to_cpu(ttt_lng)
    ttt_lnb_c = to_cpu(ttt_lnb)

    # {name, category, fused_fn, fallback_fn | nil}
    [
      # Simple scans [B,T,H]
      {"mingru", :simple,
       fn -> FusedScan.mingru(g1, g2) end,
       fn -> Edifice.Recurrent.MinGRU.min_gru_scan(g1_c, g2_c) end},

      {"minlstm", :simple,
       fn -> FusedScan.minlstm(g1, g2, g3) end,
       fn -> Edifice.Recurrent.MinLSTM.min_lstm_scan(g1_c, g2_c, g3_c) end},

      {"elu_gru", :simple,
       fn -> FusedScan.elu_gru(g1, g2) end,
       fn -> Edifice.Recurrent.NativeRecurrence.elu_gru_scan(g1_c, g2_c) end},

      {"real_gru", :simple,
       fn -> FusedScan.real_gru(g1, g2) end,
       fn -> Edifice.Recurrent.NativeRecurrence.real_gru_scan(g1_c, g2_c) end},

      {"diag_linear", :simple,
       fn -> FusedScan.diag_linear(g1, g2) end,
       fn -> Edifice.Recurrent.NativeRecurrence.diag_linear_scan(g1_c, g2_c) end},

      {"liquid", :simple,
       fn -> FusedScan.liquid(g1, g2) end,
       fn -> Edifice.Liquid.liquid_exact_scan(g1_c, g2_c) end},

      {"linear_scan", :simple,
       fn -> FusedScan.linear_scan(g1, g2) end,
       fn -> FusedScan.linear_scan_fallback(g1_c, g2_c) end},

      {"selective_scan", :simple,
       fn -> FusedScan.selective_scan(ss_x, ss_dt, ss_a, ss_b, ss_c) end,
       fn -> Edifice.SSM.Common.selective_scan_fallback(ss_x_c, ss_dt_c, ss_a_c, ss_b_c, ss_c_c) end},

      {"lstm_scan", :simple,
       fn -> FusedScan.lstm_scan(wx4h, r4h) end,
       nil},

      {"gru_scan", :simple,
       fn -> FusedScan.gru_scan(wx3h, r3h) end,
       nil},

      # Matrix-state scans [B,T,H,d]
      {"delta_net", :matrix,
       fn -> FusedScan.delta_net_scan(q4, k4, v4, beta4d) end,
       fn -> Edifice.Recurrent.DeltaNet.delta_net_sequential_scan(q4_c, k4_c, v4_c, beta4d_c) end},

      {"gated_delta_net", :matrix,
       fn -> FusedScan.gated_delta_net_scan(q4, k4, v4, beta4d, alpha4) end,
       fn -> Edifice.Recurrent.GatedDeltaNet.gated_delta_net_sequential_scan(q4_c, k4_c, v4_c, beta4d_c, alpha4_c) end},

      {"delta_product", :matrix,
       fn -> FusedScan.delta_product_scan(dp_q, dp_k, dp_v, dp_beta) end,
       fn -> Edifice.Recurrent.DeltaProduct.delta_product_scan_fallback(dp_q_c, dp_k_c, dp_v_c, dp_beta_c) end},

      {"slstm", :matrix,
       fn -> FusedScan.slstm_scan(wx4h, r4h) end,
       fn -> Edifice.Recurrent.SLSTM.slstm_scan_fallback(wx4h_c, r4h_c) end},

      {"ttt", :matrix,
       fn -> FusedScan.ttt_scan(ttt_q, ttt_k, ttt_v, ttt_eta, ttt_w0, ttt_lng, ttt_lnb) end,
       fn -> Edifice.Recurrent.TTT.ttt_scan_fallback(ttt_q_c, ttt_k_c, ttt_v_c, ttt_eta_c, ttt_w0_c, ttt_lng_c, ttt_lnb_c) end},

      {"kda", :matrix,
       fn -> FusedScan.kda_scan(q4, k4, v4, alpha4d, alpha4) end,
       nil},

      {"rla", :matrix,
       fn -> FusedScan.rla_scan(q4, k4, v4, rla_a, rla_b, rla_g) end,
       nil},

      # Memory / slot kernels
      {"reservoir", :memory,
       fn -> FusedScan.reservoir_scan(res_wx, res_w) end,
       nil},

      {"titans", :memory,
       fn -> FusedScan.titans_scan(tc, memory_size: @memory_size) end,
       nil},

      {"miras", :memory,
       fn -> FusedScan.miras_scan(mc, memory_size: @memory_size) end,
       nil},

      {"gsa", :memory,
       fn -> FusedScan.gsa_scan(gsa_q, gsa_ks, gsa_v, gsa_a) end,
       nil},

      # Attention [B,H,T,d] — last (flash may crash in some configs)
      {"fox_attention", :attention,
       fn -> FusedScan.fox_attention(aq, ak, av, cs) end,
       nil},

      {"laser_attention", :attention,
       fn -> FusedScan.laser_attention(aq, ak, av, causal: true) end,
       nil},

      {"flash_attention", :attention,
       fn -> FusedScan.flash_attention(aq, ak, av, causal: true) end,
       nil}
    ]
  end

  def phase1 do
    IO.puts(sep())
    IO.puts("Phase 1: Raw Kernel Latency (eager custom_call dispatch)")
    IO.puts("  batch=#{@batch}, seq=#{@seq_len}, hidden=#{@hidden}, " <>
            "heads=#{@heads}, head_dim=#{@head_dim}, state=#{@state_size}")
    IO.puts("  Note: Not all custom calls support eager dispatch. Kernels that fail")
    IO.puts("  here may still work in Phase 2 (Axon compilation pipeline).")
    IO.puts(sep())
    IO.puts("")

    specs = kernel_specs()

    header =
      "  #{String.pad_trailing("Kernel", 22)}" <>
      "#{String.pad_trailing("Category", 12)}" <>
      "#{String.pad_leading("CUDA (ms)", 12)}" <>
      "#{String.pad_leading("Elixir (ms)", 14)}" <>
      "#{String.pad_leading("Speedup", 10)}"

    IO.puts(header)
    IO.puts("  " <> String.duplicate("-", 68))

    results =
      for {name, category, fused_fn, fallback_fn} <- specs do
        IO.write("  #{String.pad_trailing(name, 22)}#{String.pad_trailing(to_string(category), 12)}")

        fused_us =
          if MapSet.member?(@phase1_skip, name) do
            IO.puts(String.pad_leading("SKIP", 12) <> "  (segfaults eagerly, see Phase 2)")
            nil
          else
            try do
              measure(@fused_warmup, @fused_iters, fused_fn)
            rescue
              e ->
                msg = Exception.message(e) |> String.slice(0, 40)
                IO.puts(String.pad_leading("FAIL", 12) <> "  #{msg}")
                nil
            end
          end

        if fused_us do
          fused_ms = fused_us / 1000

          {fallback_us, speedup_str} =
            if fallback_fn do
              try do
                fb_us = measure(@fallback_warmup, @fallback_iters, fallback_fn)
                speedup = fb_us / max(fused_us, 1)
                {fb_us, "#{Float.round(speedup, 1)}x"}
              rescue
                _ -> {nil, "-"}
              end
            else
              {nil, "-"}
            end

          fallback_str = if fallback_us, do: fmt_ms(fallback_us / 1000), else: "-"

          IO.puts(
            "#{String.pad_leading(fmt_ms(fused_ms), 12)}" <>
            "#{String.pad_leading(fallback_str, 14)}" <>
            "#{String.pad_leading(speedup_str, 10)}"
          )

          {name, category, fused_us, fallback_us}
        else
          {name, category, nil, nil}
        end
      end

    IO.puts("")

    # Summary
    succeeded = Enum.filter(results, fn {_, _, f, _} -> f != nil end)
    with_speedup =
      results
      |> Enum.filter(fn {_, _, f, fb} -> f != nil and fb != nil end)
      |> Enum.map(fn {name, _, f, fb} -> {name, fb / max(f, 1)} end)
      |> Enum.sort_by(fn {_, s} -> -s end)

    failed = Enum.filter(results, fn {_, _, f, _} -> f == nil end)

    IO.puts("  #{length(succeeded)}/#{length(results)} kernels ran eagerly" <>
            if(failed != [], do: " (#{length(failed)} need Axon pipeline)", else: ""))

    if with_speedup != [] do
      speedups = Enum.map(with_speedup, fn {_, s} -> s end) |> Enum.sort()
      median = Enum.at(speedups, div(length(speedups), 2))
      {best_name, best_s} = hd(with_speedup)
      {worst_name, worst_s} = List.last(with_speedup)

      IO.puts("  #{length(with_speedup)} with speedup comparison:")
      IO.puts("    Median: #{Float.round(median, 1)}x | " <>
              "Best: #{best_name} (#{Float.round(best_s, 1)}x) | " <>
              "Worst: #{worst_name} (#{Float.round(worst_s, 1)}x)")
    end

    if failed != [] do
      IO.puts("  Eager-unsupported: #{Enum.map_join(failed, ", ", fn {n, _, _, _} -> n end)}")
    end

    IO.puts("")
    results
  end

  # ── Phase 2: Full model inference ─────────────────────────────

  @shared_opts [
    embed_dim: @hidden,
    hidden_size: @hidden,
    state_size: @state_size,
    num_layers: 2,
    seq_len: @seq_len,
    window_size: @seq_len,
    head_dim: @head_dim,
    num_heads: @heads,
    dropout: 0.0
  ]

  defp model_specs do
    [
      {:min_gru, :recurrent, @shared_opts},
      {:min_lstm, :recurrent, @shared_opts},
      {:delta_net, :recurrent, @shared_opts},
      {:gated_delta_net, :recurrent, @shared_opts},
      {:delta_product, :recurrent, @shared_opts},
      {:slstm, :recurrent, @shared_opts},
      {:ttt, :recurrent, @shared_opts},
      {:titans, :recurrent, @shared_opts},
      {:miras, :recurrent, @shared_opts},
      {:liquid, :recurrent, @shared_opts},
      {:reservoir, :recurrent,
       [input_size: @hidden, reservoir_size: @hidden, output_size: @hidden, seq_len: @seq_len]},
      {:mamba, :ssm, @shared_opts},
      {:griffin, :attention, @shared_opts},
      {:mega, :attention, @shared_opts},
      {:gqa, :attention, @shared_opts},
      {:laser, :attention, @shared_opts},
      {:fox, :attention, @shared_opts},
      {:kda, :attention, @shared_opts},
      {:rla, :attention, @shared_opts},
      {:gsa, :attention, @shared_opts},
      {:infini_attention, :attention, @shared_opts}
    ]
  end

  defp input_name(:reservoir), do: "input"
  defp input_name(_), do: "state_sequence"

  defp compile_model(arch, opts) do
    model = Edifice.build(arch, opts)
    input_key = input_name(arch)
    template = %{input_key => Nx.template({@batch, @seq_len, @hidden}, :f32)}

    {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
    params = init_fn.(template, Axon.ModelState.empty())

    input = rand({@batch, @seq_len, @hidden})
    input_map = %{input_key => input}

    for _ <- 1..@phase2_warmup, do: predict_fn.(params, input_map)

    {predict_fn, params, input_map}
  end

  def phase2 do
    IO.puts(sep())
    IO.puts("Phase 2: Full Model Inference (Axon + compiler: EXLA)")
    IO.puts("  batch=#{@batch}, seq=#{@seq_len}, hidden=#{@hidden}, layers=2")
    IO.puts("  #{@phase2_warmup} warmup + #{@phase2_iters} timed iterations, reporting median")
    IO.puts("  Custom calls are properly dispatched through the Axon pipeline.")
    IO.puts(sep())
    IO.puts("")

    header =
      "  #{String.pad_trailing("Architecture", 22)}" <>
      "#{String.pad_trailing("Family", 12)}" <>
      "#{String.pad_leading("Latency", 12)}" <>
      "#{String.pad_leading("FPS", 10)}" <>
      "  Status"

    IO.puts(header)
    IO.puts("  " <> String.duplicate("-", 68))

    results =
      for {arch, family, opts} <- model_specs() do
        IO.write("  #{String.pad_trailing(to_string(arch), 22)}#{String.pad_trailing(to_string(family), 12)}")

        if MapSet.member?(@phase2_skip, arch) do
          IO.puts("#{String.pad_leading("-", 12)}#{String.pad_leading("-", 10)}  SKIP (segfault)")
          {arch, family, nil}
        else
        try do
          {predict_fn, params, input_map} = compile_model(arch, opts)

          times =
            for _ <- 1..@phase2_iters do
              {us, _} = :timer.tc(fn -> predict_fn.(params, input_map) end)
              us
            end

          times = Enum.sort(times)
          median_us = Enum.at(times, div(length(times), 2))
          median_ms = median_us / 1000
          fps = if median_ms > 0, do: Float.round(1000.0 / median_ms, 1), else: 0.0

          marker =
            cond do
              median_ms < @fps_target_ms -> "<< 60 FPS"
              median_ms < @fps_target_ms * 2 -> "~  30 FPS"
              true -> ""
            end

          IO.puts(
            "#{String.pad_leading(fmt_ms(median_ms), 12)}" <>
            "#{String.pad_leading("#{fps}", 10)}" <>
            "  #{marker}"
          )

          {arch, family, median_us}
        rescue
          e ->
            msg = Exception.message(e) |> String.slice(0, 50)
            IO.puts("#{String.pad_leading("-", 12)}#{String.pad_leading("-", 10)}  FAIL: #{msg}")
            {arch, family, nil}
        end
        end
      end

    IO.puts("")

    successful =
      results
      |> Enum.filter(fn {_, _, us} -> us != nil end)
      |> Enum.sort_by(fn {_, _, us} -> us end)

    failed = Enum.filter(results, fn {_, _, us} -> us == nil end)

    IO.puts("  Latency Ranking:")
    IO.puts("  " <> String.duplicate("-", 60))

    successful
    |> Enum.with_index(1)
    |> Enum.each(fn {{arch, family, us}, rank} ->
      ms = us / 1000
      fps = if ms > 0, do: Float.round(1000.0 / ms, 1), else: 0.0
      marker = if ms < @fps_target_ms, do: " *", else: ""

      IO.puts(
        "  #{String.pad_trailing("#{rank}.", 5)}" <>
        "#{String.pad_trailing(to_string(arch), 22)}" <>
        "#{String.pad_trailing(to_string(family), 12)}" <>
        "#{String.pad_leading(fmt_ms(ms), 12)}" <>
        "#{String.pad_leading("#{fps}", 10)}" <>
        marker
      )
    end)

    IO.puts("")

    viable = Enum.count(successful, fn {_, _, us} -> us / 1000 < @fps_target_ms end)
    IO.puts("  #{viable}/#{length(successful)} architectures under #{@fps_target_ms}ms (60 FPS)")

    if failed != [] do
      IO.puts("  Failed: #{Enum.map_join(failed, ", ", fn {arch, _, _} -> to_string(arch) end)}")
    end

    IO.puts("")
    IO.puts("  Category Summary (median latency):")
    IO.puts("  " <> String.duplicate("-", 60))

    successful
    |> Enum.group_by(fn {_, family, _} -> family end)
    |> Enum.map(fn {family, entries} ->
      times = Enum.map(entries, fn {_, _, us} -> us / 1000 end) |> Enum.sort()
      median = Enum.at(times, div(length(times), 2))
      best = hd(times)
      worst = List.last(times)
      {family, median, best, worst, length(entries)}
    end)
    |> Enum.sort_by(fn {_, median, _, _, _} -> median end)
    |> Enum.each(fn {family, median, best, worst, count} ->
      IO.puts(
        "  #{String.pad_trailing(to_string(family), 15)}" <>
        "median=#{String.pad_trailing(fmt_ms(median), 10)}" <>
        "best=#{String.pad_trailing(fmt_ms(best), 10)}" <>
        "worst=#{String.pad_trailing(fmt_ms(worst), 10)}" <>
        "(#{count} archs)"
      )
    end)

    IO.puts("")
    results
  end

  # ── Phase 3: Training throughput (forward + backward) ────────

  @phase3_warmup 3
  @phase3_iters 20

  defp compile_train_step(arch, opts) do
    model = Edifice.build(arch, opts)
    input_key = input_name(arch)
    template = %{input_key => Nx.template({@batch, @seq_len, @hidden}, :f32)}

    {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
    params = init_fn.(template, Axon.ModelState.empty())

    input = rand({@batch, @seq_len, @hidden})
    input_map = %{input_key => input}

    # Compile the train step: forward + backward (MSE loss on output mean)
    train_fn =
      EXLA.jit(fn params, input_map ->
        Nx.Defn.grad(params, fn p ->
          out = predict_fn.(p, input_map)
          out |> Nx.mean()
        end)
      end)

    # Warmup
    for _ <- 1..@phase3_warmup, do: train_fn.(params, input_map)

    {train_fn, params, input_map}
  end

  def phase3 do
    IO.puts(sep())
    IO.puts("Phase 3: Training Throughput (forward + backward via value_and_grad)")
    IO.puts("  batch=#{@batch}, seq=#{@seq_len}, hidden=#{@hidden}, layers=2")
    IO.puts("  #{@phase3_warmup} warmup + #{@phase3_iters} timed iterations, reporting median")
    IO.puts("  Uses fused backward CUDA kernels where available.")
    IO.puts(sep())
    IO.puts("")

    header =
      "  #{String.pad_trailing("Architecture", 22)}" <>
      "#{String.pad_trailing("Family", 12)}" <>
      "#{String.pad_leading("Fwd+Bwd", 12)}" <>
      "#{String.pad_leading("FPS", 10)}" <>
      "  Status"

    IO.puts(header)
    IO.puts("  " <> String.duplicate("-", 68))

    results =
      for {arch, family, opts} <- model_specs() do
        IO.write("  #{String.pad_trailing(to_string(arch), 22)}#{String.pad_trailing(to_string(family), 12)}")

        if MapSet.member?(@phase2_skip, arch) do
          IO.puts("#{String.pad_leading("-", 12)}#{String.pad_leading("-", 10)}  SKIP (segfault)")
          {arch, family, nil}
        else
        try do
          {train_fn, params, input_map} = compile_train_step(arch, opts)

          times =
            for _ <- 1..@phase3_iters do
              {us, _} = :timer.tc(fn -> train_fn.(params, input_map) end)
              us
            end

          times = Enum.sort(times)
          median_us = Enum.at(times, div(length(times), 2))
          median_ms = median_us / 1000
          fps = if median_ms > 0, do: Float.round(1000.0 / median_ms, 1), else: 0.0

          IO.puts(
            "#{String.pad_leading(fmt_ms(median_ms), 12)}" <>
            "#{String.pad_leading("#{fps}", 10)}" <>
            "  "
          )

          {arch, family, median_us}
        rescue
          e ->
            msg = Exception.message(e) |> String.slice(0, 50)
            IO.puts("#{String.pad_leading("-", 12)}#{String.pad_leading("-", 10)}  FAIL: #{msg}")
            {arch, family, nil}
        end
        end
      end

    IO.puts("")

    successful =
      results
      |> Enum.filter(fn {_, _, us} -> us != nil end)
      |> Enum.sort_by(fn {_, _, us} -> us end)

    failed = Enum.filter(results, fn {_, _, us} -> us == nil end)

    IO.puts("  Training Latency Ranking:")
    IO.puts("  " <> String.duplicate("-", 60))

    successful
    |> Enum.with_index(1)
    |> Enum.each(fn {{arch, family, us}, rank} ->
      ms = us / 1000
      fps = if ms > 0, do: Float.round(1000.0 / ms, 1), else: 0.0

      IO.puts(
        "  #{String.pad_trailing("#{rank}.", 5)}" <>
        "#{String.pad_trailing(to_string(arch), 22)}" <>
        "#{String.pad_trailing(to_string(family), 12)}" <>
        "#{String.pad_leading(fmt_ms(ms), 12)}" <>
        "#{String.pad_leading("#{fps}", 10)}"
      )
    end)

    if failed != [] do
      IO.puts("")
      IO.puts("  Failed: #{Enum.map_join(failed, ", ", fn {arch, _, _} -> to_string(arch) end)}")
    end

    IO.puts("")
    results
  end

  # ── Main ───────────────────────────────────────────────────────

  def run do
    IO.puts("=" |> String.duplicate(72))
    IO.puts("Fused Kernel Sweep — Comprehensive CUDA Benchmark")
    IO.puts("=" |> String.duplicate(72))
    IO.puts("")

    cc = FusedScan.custom_call_available?()
    nif = Code.ensure_loaded?(Edifice.CUDA.NIF) and
          function_exported?(Edifice.CUDA.NIF, :fused_mingru_scan, 7)
    IO.puts("  Dispatch: custom_call=#{cc}, NIF=#{nif}")
    IO.puts("  Backend:  #{inspect(Nx.default_backend())}")
    IO.puts("")

    IO.puts("Phase 0: GPU warmup...")
    _w = Nx.add(rand({2, 2}), rand({2, 2}))
    IO.puts("  Done.")
    IO.puts("")

    phase1()
    phase2()
    phase3()

    IO.puts("=" |> String.duplicate(72))
    IO.puts("Done.")
    IO.puts("=" |> String.duplicate(72))
  end
end

FusedKernelSweep.run()
