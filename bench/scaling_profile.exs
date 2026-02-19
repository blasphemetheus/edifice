# Scaling Profile: How do architectures scale with sequence length and embed dim?
#
# Tests the top ExPhil backbone candidates from inference_latency results.
# Sweeps seq_len and embed_dim independently to characterize scaling behavior.
# SSMs should scale linearly in seq_len vs quadratic for attention.
#
# Usage:
#   EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/scaling_profile.exs
#
# Environment variables:
#   BENCH_BATCH  - Batch size (default: 1)
#   BENCH_ITERS  - Timing iterations per data point (default: 20)

Nx.default_backend(EXLA.Backend)

defmodule ScalingProfile do
  @batch String.to_integer(System.get_env("BENCH_BATCH", "1"))
  @iters String.to_integer(System.get_env("BENCH_ITERS", "20"))

  # Top backbone candidates from inference latency benchmark
  @architectures [
    {:gated_ssm, :ssm},
    {:fnet, :attention},
    {:gqa, :attention},
    {:mamba_hillis_steele, :ssm},
    {:jamba, :ssm},
    {:s4, :ssm},
    {:hyena, :ssm},
    {:retnet, :attention},
    {:mamba, :ssm},
    {:griffin, :attention},
    {:min_lstm, :recurrent},
    {:lstm, :recurrent}
  ]

  defp rand(shape) do
    key = Nx.Random.key(42)
    {tensor, _key} = Nx.Random.normal(key, shape: shape)
    tensor
  end

  defp fmt(ms) when ms < 0.01, do: "#{Float.round(ms * 1000, 1)} us"
  defp fmt(ms) when ms < 1, do: "#{Float.round(ms, 3)} ms"
  defp fmt(ms) when ms < 100, do: "#{Float.round(ms, 2)} ms"
  defp fmt(ms), do: "#{Float.round(ms, 0)} ms"

  defp build_opts(arch, embed, seq_len) do
    head_dim = max(div(embed, 8), 8)
    num_heads = max(div(embed, head_dim), 1)

    base = [
      embed_dim: embed,
      hidden_size: embed,
      state_size: min(embed, 16),
      num_layers: 2,
      seq_len: seq_len,
      window_size: seq_len,
      head_dim: head_dim,
      num_heads: num_heads,
      dropout: 0.0
    ]

    case arch do
      :rwkv -> Keyword.merge(base, head_size: max(div(embed, 4), 4))
      _ -> base
    end
  end

  defp time_inference(arch, embed, seq_len) do
    opts = build_opts(arch, embed, seq_len)
    model = Edifice.build(arch, opts)

    input_key = "state_sequence"
    input = rand({@batch, seq_len, embed})
    template = %{input_key => Nx.template({@batch, seq_len, embed}, :f32)}

    {init_fn, predict_fn} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())
    input_map = %{input_key => input}

    # Warmup
    for _ <- 1..3, do: predict_fn.(params, input_map)

    # Time
    {total_us, _} =
      :timer.tc(fn ->
        for _ <- 1..@iters, do: predict_fn.(params, input_map)
      end)

    total_us / @iters / 1_000
  rescue
    e ->
      IO.puts(
        "    (#{arch} failed at embed=#{embed}, seq=#{seq_len}: #{Exception.message(e) |> String.slice(0, 60)})"
      )

      nil
  end

  def run do
    IO.puts("=" |> String.duplicate(80))
    IO.puts("Edifice Scaling Profile — EXLA Backend")
    IO.puts("batch=#{@batch}, iters=#{@iters}")
    IO.puts("=" |> String.duplicate(80))
    IO.puts("")

    # Warmup GPU with first architecture
    IO.puts("Warming up GPU...")
    time_inference(:gated_ssm, 64, 8)
    IO.puts("")

    seq_len_scaling()
    IO.puts("")
    embed_dim_scaling()
    IO.puts("")
    gcn_scaling()
  end

  # ── Sequence length scaling (fixed embed=256) ──

  defp seq_len_scaling do
    embed = 256
    seq_lengths = [8, 16, 32, 64, 128, 256]

    IO.puts("## Sequence Length Scaling (embed=#{embed})")
    IO.puts("-" |> String.duplicate(70))

    # Header
    arch_names = Enum.map(@architectures, fn {arch, _} -> to_string(arch) end)

    IO.puts(
      "  #{String.pad_trailing("seq_len", 10)}" <>
        Enum.map_join(arch_names, "", fn name ->
          String.pad_trailing(String.slice(name, 0, 12), 13)
        end)
    )

    IO.puts("  " <> String.duplicate("-", 10 + 13 * length(@architectures)))

    for seq_len <- seq_lengths do
      results =
        Enum.map(@architectures, fn {arch, _cat} ->
          ms = time_inference(arch, embed, seq_len)
          if ms, do: fmt(ms), else: "FAIL"
        end)

      IO.puts(
        "  #{String.pad_trailing(Integer.to_string(seq_len), 10)}" <>
          Enum.map_join(results, "", fn r -> String.pad_trailing(r, 13) end)
      )
    end

    # Scaling ratio table (256 / 16)
    IO.puts("")
    IO.puts("  Scaling ratio (seq=256 / seq=16):")

    Enum.each(@architectures, fn {arch, cat} ->
      ms_16 = time_inference(arch, embed, 16)
      ms_256 = time_inference(arch, embed, 256)

      if ms_16 && ms_256 && ms_16 > 0 do
        ratio = Float.round(ms_256 / ms_16, 1)
        theoretical = if cat == :attention, do: "O(n^2) expect ~256x", else: "O(n) expect ~16x"
        IO.puts("    #{String.pad_trailing(to_string(arch), 25)} #{ratio}x  (#{theoretical})")
      end
    end)
  end

  # ── Embed dim scaling (fixed seq_len=32) ──

  defp embed_dim_scaling do
    seq_len = 32
    embed_dims = [64, 128, 256, 512]

    IO.puts("## Embed Dim Scaling (seq_len=#{seq_len})")
    IO.puts("-" |> String.duplicate(70))

    arch_names = Enum.map(@architectures, fn {arch, _} -> to_string(arch) end)

    IO.puts(
      "  #{String.pad_trailing("embed", 10)}" <>
        Enum.map_join(arch_names, "", fn name ->
          String.pad_trailing(String.slice(name, 0, 12), 13)
        end)
    )

    IO.puts("  " <> String.duplicate("-", 10 + 13 * length(@architectures)))

    for embed <- embed_dims do
      results =
        Enum.map(@architectures, fn {arch, _cat} ->
          ms = time_inference(arch, embed, seq_len)
          if ms, do: fmt(ms), else: "FAIL"
        end)

      IO.puts(
        "  #{String.pad_trailing(Integer.to_string(embed), 10)}" <>
          Enum.map_join(results, "", fn r -> String.pad_trailing(r, 13) end)
      )
    end
  end

  # ── GCN scaling with number of nodes ──

  defp gcn_scaling do
    IO.puts("## GCN — Graph Size Scaling")
    IO.puts("batch=#{@batch}, input_dim=8, hidden=[32,32]")
    IO.puts("-" |> String.duplicate(50))

    IO.puts("  #{String.pad_trailing("nodes", 10)}#{String.pad_trailing("inference", 15)}compile")

    IO.puts("  " <> String.duplicate("-", 40))

    node_counts = [16, 32, 64, 128, 256]

    for num_nodes <- node_counts do
      model =
        Edifice.Graph.GCN.build(
          input_dim: 8,
          hidden_dims: [32, 32],
          num_classes: 10
        )

      nodes = rand({@batch, num_nodes, 8})
      adj = Nx.eye(num_nodes) |> Nx.broadcast({@batch, num_nodes, num_nodes})
      input = %{"nodes" => nodes, "adjacency" => adj}

      template = %{
        "nodes" => Nx.template({@batch, num_nodes, 8}, :f32),
        "adjacency" => Nx.template({@batch, num_nodes, num_nodes}, :f32)
      }

      {compile_us, {init_fn, predict_fn}} =
        :timer.tc(fn -> Axon.build(model) end)

      params = init_fn.(template, Axon.ModelState.empty())

      # Warm up
      for _ <- 1..3, do: predict_fn.(params, input)

      {total_us, _} =
        :timer.tc(fn ->
          for _ <- 1..@iters, do: predict_fn.(params, input)
        end)

      avg_ms = total_us / @iters / 1_000
      compile_ms = compile_us / 1_000

      IO.puts(
        "  #{String.pad_trailing(Integer.to_string(num_nodes), 10)}" <>
          "#{String.pad_trailing(fmt(avg_ms), 15)}" <>
          "#{fmt(compile_ms)}"
      )
    end
  end
end

ScalingProfile.run()
