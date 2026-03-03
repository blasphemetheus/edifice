# Melee Architecture Comparison Benchmark
#
# Sweeps ~24 Edifice architectures at Melee-relevant dimensions to help
# choose a backbone for a 60fps fighting-game agent. Reports inference
# latency, training throughput, memory footprint, and sequence-length
# scaling in five phases.
#
# Usage (tee to log for later review):
#   EDIFICE_LOCAL_NX=1 nix-shell --run "mix run bench/architecture_comparison.exs" shell.nix 2>&1 | tee bench/results/arch_comparison_$(date +%Y%m%d_%H%M).log
#
# Quick run:
#   EDIFICE_LOCAL_NX=1 BENCH_WARMUP=3 BENCH_ITERS=20 nix-shell --run "mix run bench/architecture_comparison.exs" shell.nix 2>&1 | tee bench/results/arch_comparison_quick.log
#
# Larger hidden (4090):
#   EDIFICE_LOCAL_NX=1 MELEE_HIDDEN=128 MELEE_SEQ=32 nix-shell --run "mix run bench/architecture_comparison.exs" shell.nix 2>&1 | tee bench/results/arch_comparison_h128.log
#
# Environment variables:
#   MELEE_INPUT        - Game state features (default: 128)
#   MELEE_OUTPUT       - Action space size (default: 40)
#   MELEE_HIDDEN       - Hidden dimension (default: 64)
#   MELEE_SEQ          - Frames of history (default: 16, ~267ms at 60fps)
#   MELEE_BATCH_INFER  - Inference batch size (default: 1)
#   MELEE_BATCH_TRAIN  - Training batch size (default: 64)
#   BENCH_WARMUP       - Warmup iterations (default: 10)
#   BENCH_ITERS        - Timed iterations (default: 100)
#   BENCH_TRAIN_ITERS  - Training phase iterations (default: 20)

Nx.default_backend(EXLA.Backend)
Logger.configure(level: :warning)

defmodule MeleeArchComparison do
  @moduledoc false

  # ── Configuration ──────────────────────────────────────────────

  @input_dim String.to_integer(System.get_env("MELEE_INPUT", "128"))
  @output_dim String.to_integer(System.get_env("MELEE_OUTPUT", "40"))
  @hidden String.to_integer(System.get_env("MELEE_HIDDEN", "64"))
  @seq_len String.to_integer(System.get_env("MELEE_SEQ", "16"))
  @batch_infer String.to_integer(System.get_env("MELEE_BATCH_INFER", "1"))
  @batch_train String.to_integer(System.get_env("MELEE_BATCH_TRAIN", "64"))
  @warmup String.to_integer(System.get_env("BENCH_WARMUP", "10"))
  @iters String.to_integer(System.get_env("BENCH_ITERS", "100"))
  @train_iters String.to_integer(System.get_env("BENCH_TRAIN_ITERS", "20"))

  # Derived dimensions
  @head_dim max(div(@hidden, 4), 8)
  @num_heads max(div(@hidden, @head_dim), 1)
  @state_size min(@hidden, 16)
  @num_layers 2

  @fps_target_ms 16.0

  # Architectures known to segfault through Axon pipeline
  @skip_archs MapSet.new([])

  # When BENCH_ONLY_NEW=1, run only architectures added after the initial 23
  @only_new System.get_env("BENCH_ONLY_NEW", "0") == "1"

  # ── Shared opts ────────────────────────────────────────────────

  @shared_opts [
    embed_dim: @input_dim,
    hidden_size: @hidden,
    state_size: @state_size,
    num_layers: @num_layers,
    seq_len: @seq_len,
    window_size: @seq_len,
    head_dim: @head_dim,
    num_heads: @num_heads,
    dropout: 0.0
  ]

  # ── Architecture tiers ─────────────────────────────────────────

  # {atom, tier, opts | :mlp_temporal | :reservoir, new?}
  # new? = true for architectures added after initial 23 (for BENCH_ONLY_NEW filtering)
  defp architecture_specs do
    all = tier1_fast() ++ tier2_classic() ++ tier3_heavy() ++ tier4_exotic() ++
          tier5_attention() ++ tier6_ssm() ++ tier7_recurrent() ++ tier8_feedforward()

    if @only_new do
      Enum.filter(all, fn {_, _, _, new?} -> new? end)
    else
      all
    end
  end

  # Original 23 (new? = false)
  defp tier1_fast do
    [
      {:mlp_temporal, 1, :mlp_temporal, false},
      {:min_gru, 1, @shared_opts, false},
      {:min_lstm, 1, @shared_opts, false},
      {:gated_ssm, 1, @shared_opts, false},
      {:fnet, 1, @shared_opts, false},
      {:native_recurrence, 1, @shared_opts, false}
    ]
  end

  defp tier2_classic do
    [
      {:lstm, 2, @shared_opts, false},
      {:gru, 2, @shared_opts, false},
      {:mamba, 2, @shared_opts, false},
      {:liquid, 2, @shared_opts, false},
      {:s4, 2, @shared_opts, false},
      {:griffin, 2, @shared_opts, false}
    ]
  end

  defp tier3_heavy do
    [
      {:gqa, 3, @shared_opts, false},
      {:retnet, 3, @shared_opts, false},
      {:samba, 3, @shared_opts, false},
      {:xlstm, 3, Keyword.merge(@shared_opts, hidden_size: @hidden), false},
      {:slstm, 3, @shared_opts, false},
      {:hawk, 3, @shared_opts, false}
    ]
  end

  defp tier4_exotic do
    [
      {:delta_net, 4, @shared_opts, false},
      {:ttt, 4, @shared_opts, false},
      {:titans, 4, @shared_opts, false},
      {:reservoir, 4, :reservoir, false},
      {:huginn, 4, @shared_opts, false}
    ]
  end

  # New attention architectures
  defp tier5_attention do
    rwkv_opts = Keyword.merge(@shared_opts, head_size: max(div(@hidden, 4), 4))
    [
      {:rwkv, 5, rwkv_opts, true},
      {:gla, 5, @shared_opts, true},
      {:hgrn, 5, @shared_opts, true},
      {:mega, 5, @shared_opts, true},
      {:based, 5, @shared_opts, true},
      {:linear_transformer, 5, @shared_opts, true},
      {:nystromformer, 5, @shared_opts, true},
      {:performer, 5, @shared_opts, true},
      {:mla, 5, @shared_opts, true},
      {:diff_transformer, 5, @shared_opts, true},
      {:retnet_v2, 5, @shared_opts, true},
      {:megalodon, 5, @shared_opts, true},
      {:lightning_attention, 5, @shared_opts, true},
      {:gla_v2, 5, @shared_opts, true},
      {:hgrn_v2, 5, @shared_opts, true},
      {:flash_linear_attention, 5, @shared_opts, true},
      {:kda, 5, @shared_opts, true},
      {:gated_attention, 5, @shared_opts, true},
      {:sigmoid_attention, 5, @shared_opts, true},
      {:fox, 5, @shared_opts, true},
      {:log_linear, 5, @shared_opts, true},
      {:nha, 5, @shared_opts, true},
      {:laser, 5, @shared_opts, true},
      {:rla, 5, @shared_opts, true},
      {:rdn, 5, @shared_opts, true},
      {:tnn, 5, @shared_opts, true},
      {:spla, 5, @shared_opts, true},
      {:gsa, 5, @shared_opts, true},
      {:infini_attention, 5, @shared_opts, true},
      {:conformer, 5, @shared_opts, true}
    ]
  end

  # New SSM architectures
  defp tier6_ssm do
    [
      {:mamba_ssd, 6, @shared_opts, true},
      {:mamba_cumsum, 6, @shared_opts, true},
      {:mamba_hillis_steele, 6, @shared_opts, true},
      {:s4d, 6, @shared_opts, true},
      {:s5, 6, @shared_opts, true},
      {:h3, 6, @shared_opts, true},
      {:hyena, 6, @shared_opts, true},
      {:bimamba, 6, @shared_opts, true},
      {:jamba, 6, @shared_opts, true},
      {:zamba, 6, @shared_opts, true},
      {:striped_hyena, 6, @shared_opts, true},
      {:mamba3, 6, @shared_opts, true},
      {:gss, 6, @shared_opts, true},
      {:hyena_v2, 6, @shared_opts, true},
      {:hymba, 6, @shared_opts, true},
      {:ss_transformer, 6, @shared_opts, true},
      {:longhorn, 6, @shared_opts, true}
    ]
  end

  # New recurrent architectures
  defp tier7_recurrent do
    [
      {:mlstm, 7, @shared_opts, true},
      {:gated_delta_net, 7, @shared_opts, true},
      {:ttt_e2e, 7, @shared_opts, true},
      {:miras, 7, @shared_opts, true},
      {:xlstm_v2, 7, @shared_opts, true},
      {:transformer_like, 7, @shared_opts, true},
      {:deep_res_lstm, 7, @shared_opts, true},
      {:delta_product, 7, @shared_opts, true}
    ]
  end

  # New feedforward architectures
  defp tier8_feedforward do
    [
      {:kan, 8, @shared_opts, true},
      {:kat, 8, @shared_opts, true},
      {:bitnet, 8, @shared_opts, true},
      {:decoder_only, 8, @shared_opts, true}
    ]
  end

  # ── Helpers ────────────────────────────────────────────────────

  defp rand(shape) do
    key = Nx.Random.key(:rand.uniform(100_000))
    {t, _} = Nx.Random.uniform(key, -1.0, 1.0, shape: shape)
    t
  end

  defp measure(warmup_n, iter_n, fun) do
    if warmup_n > 0, do: for(_ <- 1..warmup_n, do: fun.())

    times =
      for _ <- 1..iter_n do
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

  defp fmt_number(n) when n < 1_000, do: "#{n}"
  defp fmt_number(n) when n < 1_000_000, do: "#{Float.round(n / 1_000, 1)}K"
  defp fmt_number(n), do: "#{Float.round(n / 1_000_000, 2)}M"

  defp fmt_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  defp fmt_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 1)} KB"

  defp fmt_bytes(bytes) when bytes < 1024 * 1024 * 1024,
    do: "#{Float.round(bytes / (1024 * 1024), 2)} MB"

  defp fmt_bytes(bytes), do: "#{Float.round(bytes / (1024 * 1024 * 1024), 2)} GB"

  defp sep, do: String.duplicate("=", 80)
  defp thin_sep, do: String.duplicate("-", 80)

  defp gpu_memory_mib do
    case System.cmd("nvidia-smi", [
           "--query-gpu=memory.used",
           "--format=csv,noheader,nounits"
         ]) do
      {output, 0} ->
        output |> String.trim() |> String.split("\n") |> hd() |> String.trim() |> String.to_integer()

      _ ->
        nil
    end
  rescue
    _ -> nil
  end

  defp count_params(model_state) do
    flatten_params(model_state.data)
    |> Enum.reduce({0, 0}, fn {_path, tensor}, {count, bytes} ->
      {count + Nx.size(tensor), bytes + Nx.byte_size(tensor)}
    end)
  end

  defp flatten_params(map) when is_map(map) do
    Enum.flat_map(map, fn
      {key, %Nx.Tensor{} = tensor} -> [{key, tensor}]
      {key, inner} when is_map(inner) ->
        flatten_params(inner) |> Enum.map(fn {k, v} -> {"#{key}.#{k}", v} end)
      _ -> []
    end)
  end

  defp flatten_params(_), do: []

  # ── Model building ─────────────────────────────────────────────

  defp build_model(:mlp_temporal, _opts) do
    Edifice.Feedforward.MLP.build_temporal(
      embed_dim: @input_dim,
      seq_len: @seq_len,
      hidden_sizes: [@hidden, @hidden],
      dropout: 0.0
    )
  end

  defp build_model(:reservoir, _opts) do
    Edifice.build(:reservoir,
      input_size: @input_dim,
      reservoir_size: @hidden,
      output_size: @hidden,
      seq_len: @seq_len
    )
  end

  defp build_model(arch, opts) do
    Edifice.build(arch, opts)
  end

  defp input_name(:reservoir), do: "input"
  defp input_name(_), do: "state_sequence"

  defp compile_model(arch, opts, batch) do
    model = build_model(arch, opts)
    input_key = input_name(arch)
    template = %{input_key => Nx.template({batch, @seq_len, @input_dim}, :f32)}

    {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
    params = init_fn.(template, Axon.ModelState.empty())

    input = rand({batch, @seq_len, @input_dim})
    input_map = %{input_key => input}

    {predict_fn, params, input_map}
  end

  defp compile_model_for_seq(arch, opts, batch, seq) do
    effective_opts =
      case opts do
        :mlp_temporal -> :mlp_temporal
        :reservoir -> :reservoir
        kw when is_list(kw) -> Keyword.merge(kw, seq_len: seq, window_size: seq)
      end

    model =
      case effective_opts do
        :mlp_temporal ->
          Edifice.Feedforward.MLP.build_temporal(
            embed_dim: @input_dim,
            seq_len: seq,
            hidden_sizes: [@hidden, @hidden],
            dropout: 0.0
          )

        :reservoir ->
          Edifice.build(:reservoir,
            input_size: @input_dim,
            reservoir_size: @hidden,
            output_size: @hidden,
            seq_len: seq
          )

        kw ->
          Edifice.build(arch, kw)
      end

    input_key = input_name(arch)
    template = %{input_key => Nx.template({batch, seq, @input_dim}, :f32)}

    {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
    params = init_fn.(template, Axon.ModelState.empty())

    input = rand({batch, seq, @input_dim})
    input_map = %{input_key => input}

    {predict_fn, params, input_map}
  end

  # ── Phase 0: GPU warmup + env detection ────────────────────────

  defp phase0 do
    IO.puts(sep())
    IO.puts("Melee Architecture Comparison Benchmark")
    IO.puts(sep())
    IO.puts("")
    IO.puts("  Configuration:")
    IO.puts("    Input dim:   #{@input_dim}  (game state features)")
    IO.puts("    Output dim:  #{@output_dim}  (action space)")
    IO.puts("    Hidden dim:  #{@hidden}")
    IO.puts("    Seq len:     #{@seq_len}  (~#{round(@seq_len / 60 * 1000)}ms at 60fps)")
    IO.puts("    Batch infer: #{@batch_infer}")
    IO.puts("    Batch train: #{@batch_train}")
    IO.puts("    Head dim:    #{@head_dim}, Num heads: #{@num_heads}")
    IO.puts("    State size:  #{@state_size}, Num layers: #{@num_layers}")
    IO.puts("    Warmup:      #{@warmup}, Iters: #{@iters}, Train iters: #{@train_iters}")
    IO.puts("")

    has_gpu = gpu_memory_mib() != nil

    if has_gpu do
      IO.puts("  GPU detected (nvidia-smi)")
      IO.puts("  Baseline GPU memory: #{gpu_memory_mib()} MiB")
    else
      IO.puts("  No GPU detected (nvidia-smi unavailable)")
    end

    IO.puts("  Backend: #{inspect(Nx.default_backend())}")

    cc =
      Code.ensure_loaded?(Edifice.CUDA.FusedScan) and
        function_exported?(Edifice.CUDA.FusedScan, :custom_call_available?, 0) and
        Edifice.CUDA.FusedScan.custom_call_available?()

    nif =
      Code.ensure_loaded?(Edifice.CUDA.NIF) and
        function_exported?(Edifice.CUDA.NIF, :fused_mingru_scan, 7)

    IO.puts("  Dispatch: custom_call=#{cc}, NIF=#{nif}")
    IO.puts("")

    IO.puts("  Phase 0: GPU warmup...")
    _w = Nx.add(rand({2, 2}), rand({2, 2}))
    IO.puts("  Done.")
    IO.puts("")

    has_gpu
  end

  # ── Phase 1: Single-sample inference latency ───────────────────

  defp phase1 do
    IO.puts(sep())
    IO.puts("Phase 1: Inference Latency (batch=#{@batch_infer}, 60fps = <#{@fps_target_ms}ms)")
    IO.puts(thin_sep())
    IO.puts("")

    header =
      "  #{String.pad_trailing("Architecture", 22)}" <>
        "#{String.pad_trailing("Tier", 6)}" <>
        "#{String.pad_leading("Latency", 12)}" <>
        "#{String.pad_leading("FPS", 10)}" <>
        "  60fps?"

    IO.puts(header)
    IO.puts("  " <> String.duplicate("-", 60))

    results =
      for {arch, tier, opts, _new?} <- architecture_specs() do
        IO.write(
          "  #{String.pad_trailing(to_string(arch), 22)}" <>
            "#{String.pad_trailing("T#{tier}", 6)}"
        )

        if MapSet.member?(@skip_archs, arch) do
          IO.puts("#{String.pad_leading("SKIP", 12)}#{String.pad_leading("-", 10)}  (segfault)")
          {arch, tier, nil}
        else
          try do
            {predict_fn, params, input_map} = compile_model(arch, opts, @batch_infer)

            median_us = measure(@warmup, @iters, fn -> predict_fn.(params, input_map) end)
            median_ms = median_us / 1000
            fps = if median_ms > 0, do: Float.round(1000.0 / median_ms, 1), else: 0.0

            viable = if median_ms < @fps_target_ms, do: "YES", else: "no"

            IO.puts(
              "#{String.pad_leading(fmt_ms(median_ms), 12)}" <>
                "#{String.pad_leading("#{fps}", 10)}" <>
                "  #{viable}"
            )

            {arch, tier, median_us}
          rescue
            e ->
              msg = Exception.message(e) |> String.slice(0, 40)
              IO.puts("#{String.pad_leading("FAIL", 12)}#{String.pad_leading("-", 10)}  #{msg}")
              {arch, tier, nil}
          end
        end
      end

    IO.puts("")

    # Summary
    succeeded = Enum.filter(results, fn {_, _, us} -> us != nil end)
    viable = Enum.count(succeeded, fn {_, _, us} -> us / 1000 < @fps_target_ms end)
    failed = Enum.filter(results, fn {_, _, us} -> us == nil end)

    IO.puts("  #{length(succeeded)}/#{length(results)} architectures compiled")
    IO.puts("  #{viable}/#{length(succeeded)} under #{@fps_target_ms}ms (60 FPS viable)")

    if failed != [] do
      IO.puts("  Failed: #{Enum.map_join(failed, ", ", fn {a, _, _} -> to_string(a) end)}")
    end

    IO.puts("")

    # Ranking
    IO.puts("  Inference Ranking (fastest first):")
    IO.puts("  " <> String.duplicate("-", 55))

    succeeded
    |> Enum.sort_by(fn {_, _, us} -> us end)
    |> Enum.with_index(1)
    |> Enum.each(fn {{arch, tier, us}, rank} ->
      ms = us / 1000
      fps = if ms > 0, do: Float.round(1000.0 / ms, 1), else: 0.0
      marker = if ms < @fps_target_ms, do: " *", else: ""

      IO.puts(
        "  #{String.pad_trailing("#{rank}.", 5)}" <>
          "#{String.pad_trailing(to_string(arch), 22)}" <>
          "T#{tier}  " <>
          "#{String.pad_leading(fmt_ms(ms), 12)}" <>
          "#{String.pad_leading("#{fps}", 10)}" <>
          marker
      )
    end)

    IO.puts("")
    results
  end

  # ── Phase 2: Training throughput ───────────────────────────────

  defp phase2(_phase1_results) do
    IO.puts(sep())
    IO.puts("Phase 2: Training Throughput (batch=#{@batch_train}, fwd+bwd)")
    IO.puts(thin_sep())
    IO.puts("")

    header =
      "  #{String.pad_trailing("Architecture", 22)}" <>
        "#{String.pad_trailing("Tier", 6)}" <>
        "#{String.pad_leading("Fwd+Bwd", 12)}" <>
        "#{String.pad_leading("Samples/s", 12)}"

    IO.puts(header)
    IO.puts("  " <> String.duplicate("-", 54))

    results =
      for {arch, tier, opts, _new?} <- architecture_specs() do
        IO.write(
          "  #{String.pad_trailing(to_string(arch), 22)}" <>
            "#{String.pad_trailing("T#{tier}", 6)}"
        )

        if MapSet.member?(@skip_archs, arch) do
          IO.puts("#{String.pad_leading("SKIP", 12)}#{String.pad_leading("-", 12)}")
          {arch, tier, nil}
        else
          try do
            {predict_fn, params, input_map} = compile_model(arch, opts, @batch_train)

            train_fn =
              EXLA.jit(fn params, input_map ->
                Nx.Defn.grad(params, fn p ->
                  predict_fn.(p, input_map) |> Nx.mean()
                end)
              end)

            # Warmup training
            for _ <- 1..min(@warmup, 3), do: train_fn.(params, input_map)

            median_us =
              measure(0, @train_iters, fn -> train_fn.(params, input_map) end)

            median_ms = median_us / 1000
            samples_per_sec =
              if median_ms > 0, do: Float.round(@batch_train * 1000.0 / median_ms, 1), else: 0.0

            IO.puts(
              "#{String.pad_leading(fmt_ms(median_ms), 12)}" <>
                "#{String.pad_leading("#{samples_per_sec}", 12)}"
            )

            {arch, tier, median_us}
          rescue
            e ->
              msg = Exception.message(e) |> String.slice(0, 40)
              IO.puts("#{String.pad_leading("FAIL", 12)}#{String.pad_leading("-", 12)}  #{msg}")
              {arch, tier, nil}
          end
        end
      end

    IO.puts("")

    # Ranking
    succeeded = Enum.filter(results, fn {_, _, us} -> us != nil end)
    failed = Enum.filter(results, fn {_, _, us} -> us == nil end)

    IO.puts("  Training Ranking (fastest first):")
    IO.puts("  " <> String.duplicate("-", 55))

    succeeded
    |> Enum.sort_by(fn {_, _, us} -> us end)
    |> Enum.with_index(1)
    |> Enum.each(fn {{arch, tier, us}, rank} ->
      ms = us / 1000
      sps = if ms > 0, do: Float.round(@batch_train * 1000.0 / ms, 1), else: 0.0

      IO.puts(
        "  #{String.pad_trailing("#{rank}.", 5)}" <>
          "#{String.pad_trailing(to_string(arch), 22)}" <>
          "T#{tier}  " <>
          "#{String.pad_leading(fmt_ms(ms), 12)}" <>
          "#{String.pad_leading("#{sps} s/s", 12)}"
      )
    end)

    if failed != [] do
      IO.puts("")
      IO.puts("  Failed: #{Enum.map_join(failed, ", ", fn {a, _, _} -> to_string(a) end)}")
    end

    IO.puts("")
    results
  end

  # ── Phase 3: Memory profile ────────────────────────────────────

  defp phase3(has_gpu) do
    IO.puts(sep())
    IO.puts("Phase 3: Memory Profile (param count + GPU delta)")
    IO.puts(thin_sep())
    IO.puts("")

    baseline_gpu = if has_gpu, do: gpu_memory_mib(), else: nil

    gpu_col_header = if has_gpu, do: "#{String.pad_leading("GPU Delta", 12)}", else: ""

    header =
      "  #{String.pad_trailing("Architecture", 22)}" <>
        "#{String.pad_trailing("Tier", 6)}" <>
        "#{String.pad_leading("Params", 12)}" <>
        "#{String.pad_leading("Size", 12)}" <>
        gpu_col_header

    IO.puts(header)
    IO.puts("  " <> String.duplicate("-", 52 + if(has_gpu, do: 12, else: 0)))

    results =
      for {arch, tier, opts, _new?} <- architecture_specs() do
        IO.write(
          "  #{String.pad_trailing(to_string(arch), 22)}" <>
            "#{String.pad_trailing("T#{tier}", 6)}"
        )

        if MapSet.member?(@skip_archs, arch) do
          IO.puts("#{String.pad_leading("SKIP", 12)}#{String.pad_leading("-", 12)}")
          {arch, tier, nil, nil, nil}
        else
          try do
            model = build_model(arch, opts)
            input_key = input_name(arch)
            template = %{input_key => Nx.template({@batch_infer, @seq_len, @input_dim}, :f32)}

            {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
            model_state = init_fn.(template, Axon.ModelState.empty())
            {param_count, param_bytes} = count_params(model_state)

            # Run inference to load activations on GPU
            input = rand({@batch_infer, @seq_len, @input_dim})
            predict_fn.(model_state, %{input_key => input})

            gpu_delta =
              if has_gpu && baseline_gpu do
                current = gpu_memory_mib()
                if current, do: current - baseline_gpu, else: nil
              end

            gpu_col =
              if has_gpu do
                val = if gpu_delta, do: "#{gpu_delta} MiB", else: "?"
                String.pad_leading(val, 12)
              else
                ""
              end

            IO.puts(
              "#{String.pad_leading(fmt_number(param_count), 12)}" <>
                "#{String.pad_leading(fmt_bytes(param_bytes), 12)}" <>
                gpu_col
            )

            {arch, tier, param_count, param_bytes, gpu_delta}
          rescue
            e ->
              msg = Exception.message(e) |> String.slice(0, 40)
              IO.puts("#{String.pad_leading("FAIL", 12)}#{String.pad_leading("-", 12)}  #{msg}")
              {arch, tier, nil, nil, nil}
          end
        end
      end

    IO.puts("")

    # Size ranking
    succeeded = Enum.filter(results, fn {_, _, pc, _, _} -> pc != nil end)

    IO.puts("  Size Ranking (smallest first):")
    IO.puts("  " <> String.duplicate("-", 55))

    succeeded
    |> Enum.sort_by(fn {_, _, _, pb, _} -> pb end)
    |> Enum.with_index(1)
    |> Enum.each(fn {{arch, tier, pc, pb, _gpu}, rank} ->
      IO.puts(
        "  #{String.pad_trailing("#{rank}.", 5)}" <>
          "#{String.pad_trailing(to_string(arch), 22)}" <>
          "T#{tier}  " <>
          "#{String.pad_leading(fmt_number(pc), 12)}" <>
          "#{String.pad_leading(fmt_bytes(pb), 12)}"
      )
    end)

    IO.puts("")
    results
  end

  # ── Phase 4: Sequence length scaling ───────────────────────────

  @seq_lengths [4, 8, 16, 32]

  defp phase4(phase1_results) do
    IO.puts(sep())
    IO.puts("Phase 4: Sequence Length Scaling (inference, batch=#{@batch_infer})")
    IO.puts(thin_sep())
    IO.puts("")

    # Only test architectures that passed Phase 1 and are within 2x target
    viable =
      phase1_results
      |> Enum.filter(fn {_, _, us} -> us != nil and us / 1000 < @fps_target_ms * 2 end)
      |> Enum.map(fn {arch, _, _} -> arch end)
      |> MapSet.new()

    specs =
      architecture_specs()
      |> Enum.filter(fn {arch, _, _, _} -> MapSet.member?(viable, arch) end)

    if specs == [] do
      IO.puts("  No architectures within 2x target latency — skipping Phase 4.")
      IO.puts("")
      return = []
      return
    else
      IO.puts("  Testing #{length(specs)} architectures across seq_len = #{inspect(@seq_lengths)}")
      IO.puts("")

      # Header
      seq_cols = Enum.map_join(@seq_lengths, "", fn s -> String.pad_leading("seq=#{s}", 12) end)

      header =
        "  #{String.pad_trailing("Architecture", 22)}" <>
          "#{String.pad_trailing("Tier", 6)}" <>
          seq_cols

      IO.puts(header)
      IO.puts("  " <> String.duplicate("-", 28 + 12 * length(@seq_lengths)))

      results =
        for {arch, tier, opts, _new?} <- specs do
          IO.write(
            "  #{String.pad_trailing(to_string(arch), 22)}" <>
              "#{String.pad_trailing("T#{tier}", 6)}"
          )

          timings =
            for seq <- @seq_lengths do
              try do
                {predict_fn, params, input_map} =
                  compile_model_for_seq(arch, opts, @batch_infer, seq)

                median_us =
                  measure(min(@warmup, 5), min(@iters, 30), fn ->
                    predict_fn.(params, input_map)
                  end)

                median_us
              rescue
                _ -> nil
              end
            end

          cols =
            Enum.map_join(timings, "", fn
              nil -> String.pad_leading("FAIL", 12)
              us -> String.pad_leading(fmt_ms(us / 1000), 12)
            end)

          IO.puts(cols)
          {arch, tier, Enum.zip(@seq_lengths, timings)}
        end

      IO.puts("")

      # Scaling analysis
      IO.puts("  Scaling Factor (seq=32 / seq=4):")
      IO.puts("  " <> String.duplicate("-", 40))

      results
      |> Enum.each(fn {arch, _tier, timings} ->
        t4 = Enum.find_value(timings, fn {s, us} -> if s == 4 and us, do: us end)
        t32 = Enum.find_value(timings, fn {s, us} -> if s == 32 and us, do: us end)

        if t4 && t32 do
          ratio = Float.round(t32 / max(t4, 1), 2)
          IO.puts("  #{String.pad_trailing(to_string(arch), 22)} #{ratio}x")
        end
      end)

      IO.puts("")
      results
    end
  end

  # ── Phase 5: Final comparison table + recommendations ──────────

  defp phase5(phase1_results, phase2_results, phase3_results) do
    IO.puts(sep())
    IO.puts("Phase 5: Final Comparison")
    IO.puts(thin_sep())
    IO.puts("")

    # Build lookup maps
    infer_map =
      phase1_results
      |> Enum.filter(fn {_, _, us} -> us != nil end)
      |> Map.new(fn {arch, _, us} -> {arch, us} end)

    train_map =
      phase2_results
      |> Enum.filter(fn {_, _, us} -> us != nil end)
      |> Map.new(fn {arch, _, us} -> {arch, us} end)

    mem_map =
      phase3_results
      |> Enum.filter(fn {_, _, pc, _, _} -> pc != nil end)
      |> Map.new(fn {arch, _, pc, pb, gpu} -> {arch, {pc, pb, gpu}} end)

    header =
      "  #{String.pad_trailing("Architecture", 20)}" <>
        "#{String.pad_trailing("T", 3)}" <>
        "#{String.pad_leading("Infer(ms)", 11)}" <>
        "#{String.pad_leading("FPS", 8)}" <>
        "#{String.pad_leading("Train(ms)", 11)}" <>
        "#{String.pad_leading("S/s", 9)}" <>
        "#{String.pad_leading("Params", 10)}" <>
        "#{String.pad_leading("Size", 10)}" <>
        "  60fps?"

    IO.puts(header)
    IO.puts("  " <> String.duplicate("-", 82))

    specs = architecture_specs()

    for {arch, tier, _opts, _new?} <- specs do
      infer_us = Map.get(infer_map, arch)
      train_us = Map.get(train_map, arch)
      mem = Map.get(mem_map, arch)

      infer_str =
        if infer_us do
          ms = infer_us / 1000
          fps = if ms > 0, do: Float.round(1000.0 / ms, 1), else: 0.0
          {fmt_ms(ms), "#{fps}", ms < @fps_target_ms}
        else
          {"-", "-", false}
        end

      train_str =
        if train_us do
          ms = train_us / 1000
          sps = if ms > 0, do: Float.round(@batch_train * 1000.0 / ms, 1), else: 0.0
          {fmt_ms(ms), "#{sps}"}
        else
          {"-", "-"}
        end

      mem_str =
        if mem do
          {pc, pb, _gpu} = mem
          {fmt_number(pc), fmt_bytes(pb)}
        else
          {"-", "-"}
        end

      {infer_ms_str, fps_str, viable} = infer_str
      {train_ms_str, sps_str} = train_str
      {params_str, size_str} = mem_str

      viable_str = if viable, do: "YES", else: if(infer_us, do: "no", else: "-")

      IO.puts(
        "  #{String.pad_trailing(to_string(arch), 20)}" <>
          "#{String.pad_trailing("#{tier}", 3)}" <>
          "#{String.pad_leading(infer_ms_str, 11)}" <>
          "#{String.pad_leading(fps_str, 8)}" <>
          "#{String.pad_leading(train_ms_str, 11)}" <>
          "#{String.pad_leading(sps_str, 9)}" <>
          "#{String.pad_leading(params_str, 10)}" <>
          "#{String.pad_leading(size_str, 10)}" <>
          "  #{viable_str}"
      )
    end

    IO.puts("")

    # Recommendations
    IO.puts("  Recommendations:")
    IO.puts("  " <> String.duplicate("-", 60))

    # T400 recommendation: strict <16ms + smallest memory
    t400_candidates =
      specs
      |> Enum.filter(fn {arch, _, _, _} ->
        infer_us = Map.get(infer_map, arch)
        infer_us != nil and infer_us / 1000 < @fps_target_ms
      end)
      |> Enum.sort_by(fn {arch, _, _, _} ->
        case Map.get(mem_map, arch) do
          {_pc, pb, _} -> pb
          nil -> :infinity
        end
      end)

    if t400_candidates != [] do
      {best, tier, _, _} = hd(t400_candidates)
      infer_ms = Map.get(infer_map, best) / 1000

      IO.puts(
        "  T400 (2GB, deployment): #{best} (T#{tier}) — " <>
          "#{fmt_ms(infer_ms)} inference, 60fps viable"
      )

      if length(t400_candidates) > 1 do
        runners_up =
          t400_candidates
          |> tl()
          |> Enum.take(3)
          |> Enum.map_join(", ", fn {a, t, _, _} -> "#{a}(T#{t})" end)

        IO.puts("    Runners-up: #{runners_up}")
      end
    else
      IO.puts("  T400 (2GB, deployment): No architecture under #{@fps_target_ms}ms!")
    end

    IO.puts("")

    # 4090 recommendation: best training throughput among viable archs
    train_candidates =
      specs
      |> Enum.filter(fn {arch, _, _, _} ->
        Map.get(train_map, arch) != nil
      end)
      |> Enum.sort_by(fn {arch, _, _, _} -> Map.get(train_map, arch) end)

    if train_candidates != [] do
      {best, tier, _, _} = hd(train_candidates)
      train_ms = Map.get(train_map, best) / 1000
      sps = Float.round(@batch_train * 1000.0 / train_ms, 1)

      IO.puts(
        "  4090 (training): #{best} (T#{tier}) — " <>
          "#{fmt_ms(train_ms)} fwd+bwd, #{sps} samples/s"
      )

      if length(train_candidates) > 1 do
        runners_up =
          train_candidates
          |> tl()
          |> Enum.take(3)
          |> Enum.map_join(", ", fn {a, t, _, _} -> "#{a}(T#{t})" end)

        IO.puts("    Runners-up: #{runners_up}")
      end
    else
      IO.puts("  4090 (training): No architecture completed training!")
    end

    IO.puts("")

    # Machine-readable summary
    IO.puts("  SUMMARY_LINE: archs=#{length(specs)}" <>
      " viable_60fps=#{length(t400_candidates)}" <>
      " fastest_infer=#{if t400_candidates != [], do: elem(hd(t400_candidates), 0), else: "none"}" <>
      " fastest_train=#{if train_candidates != [], do: elem(hd(train_candidates), 0), else: "none"}")

    IO.puts("")
  end

  # ── Decision Transformer (separate benchmark) ─────────────────

  defp benchmark_decision_transformer do
    IO.puts(thin_sep())
    IO.puts("  Bonus: Decision Transformer (separate — 4 inputs)")
    IO.puts(thin_sep())
    IO.puts("")

    context_len = @seq_len

    try do
      model =
        Edifice.build(:decision_transformer,
          state_dim: @input_dim,
          action_dim: @output_dim,
          hidden_size: @hidden,
          num_heads: max(@num_heads, 2),
          num_layers: @num_layers,
          context_len: context_len,
          max_timestep: 10_000,
          dropout: 0.0
        )

      template = %{
        "returns" => Nx.template({@batch_infer, context_len}, :f32),
        "states" => Nx.template({@batch_infer, context_len, @input_dim}, :f32),
        "actions" => Nx.template({@batch_infer, context_len, @output_dim}, :f32),
        "timesteps" => Nx.template({@batch_infer, context_len}, :s64)
      }

      {init_fn, predict_fn} = Axon.build(model, compiler: EXLA)
      params = init_fn.(template, Axon.ModelState.empty())

      key = Nx.Random.key(:rand.uniform(100_000))
      {returns, key} = Nx.Random.uniform(key, shape: {_batch = @batch_infer, context_len})
      {states, key} = Nx.Random.uniform(key, shape: {@batch_infer, context_len, @input_dim})
      {actions, key} = Nx.Random.uniform(key, shape: {@batch_infer, context_len, @output_dim})
      {ts_f, _key} = Nx.Random.uniform(key, 0, 1000, shape: {@batch_infer, context_len})
      timesteps = Nx.as_type(ts_f, :s64)

      input_map = %{
        "returns" => returns,
        "states" => states,
        "actions" => actions,
        "timesteps" => timesteps
      }

      median_us = measure(@warmup, @iters, fn -> predict_fn.(params, input_map) end)
      median_ms = median_us / 1000
      fps = if median_ms > 0, do: Float.round(1000.0 / median_ms, 1), else: 0.0
      viable = if median_ms < @fps_target_ms, do: "YES", else: "no"

      {param_count, param_bytes} = count_params(params)

      IO.puts("  Decision Transformer:")
      IO.puts("    Inference: #{fmt_ms(median_ms)} (#{fps} FPS, 60fps: #{viable})")
      IO.puts("    Params:    #{fmt_number(param_count)} (#{fmt_bytes(param_bytes)})")
      IO.puts("")
    rescue
      e ->
        msg = Exception.message(e) |> String.slice(0, 60)
        IO.puts("  Decision Transformer: FAIL — #{msg}")
        IO.puts("")
    end
  end

  # ── Main ───────────────────────────────────────────────────────

  def run do
    has_gpu = phase0()
    phase1_results = phase1()
    phase2_results = phase2(phase1_results)
    phase3_results = phase3(has_gpu)
    _phase4_results = phase4(phase1_results)
    phase5(phase1_results, phase2_results, phase3_results)
    benchmark_decision_transformer()

    IO.puts(sep())
    IO.puts("Done.")
    IO.puts(sep())
  end
end

MeleeArchComparison.run()
