# Training Throughput Benchmark
#
# Measures forward + backward pass throughput (samples/sec) for each
# sequence architecture. Uses Nx.Defn.value_and_grad for the backward pass.
#
# Note: Conv-based models are excluded (Axon predict_fn not defn-traceable).
# See gradient_smoke_test.exs for details on the Axon limitation.
#
# Usage:
#   EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/training_throughput.exs
#
# Environment variables:
#   BENCH_EMBED   - Embedding dimension (default: 256)
#   BENCH_SEQ_LEN - Sequence length (default: 32)
#   BENCH_BATCH   - Batch size (default: 4)
#   BENCH_LAYERS  - Number of layers (default: 2)
#   BENCH_ITERS   - Timing iterations (default: 10)

Nx.default_backend(EXLA.Backend)

defmodule TrainingThroughput do
  @embed String.to_integer(System.get_env("BENCH_EMBED", "256"))
  @seq_len String.to_integer(System.get_env("BENCH_SEQ_LEN", "32"))
  @batch String.to_integer(System.get_env("BENCH_BATCH", "4"))
  @num_layers String.to_integer(System.get_env("BENCH_LAYERS", "2"))
  @iters String.to_integer(System.get_env("BENCH_ITERS", "10"))

  @head_dim max(div(@embed, 8), 8)
  @num_heads max(div(@embed, @head_dim), 1)

  @shared_opts [
    embed_dim: @embed,
    hidden_size: @embed,
    state_size: min(@embed, 16),
    num_layers: @num_layers,
    seq_len: @seq_len,
    window_size: @seq_len,
    head_dim: @head_dim,
    num_heads: @num_heads,
    dropout: 0.0
  ]

  # Architectures that support value_and_grad (no conv layers)
  @architectures [
    {:gated_ssm, :ssm},
    {:fnet, :attention},
    {:gqa, :attention},
    {:mamba_hillis_steele, :ssm},
    {:jamba, :ssm},
    {:s4, :ssm},
    {:hyena, :ssm},
    {:mamba, :ssm},
    {:mamba_ssd, :ssm},
    {:retnet, :attention},
    {:griffin, :attention},
    {:nystromformer, :attention},
    {:linear_transformer, :attention},
    {:performer, :attention},
    {:hgrn, :attention},
    {:gla, :attention},
    {:rwkv, :attention},
    {:min_lstm, :recurrent},
    {:min_gru, :recurrent},
    {:xlstm, :recurrent},
    {:delta_net, :recurrent},
    {:ttt, :recurrent},
    {:liquid, :recurrent},
    {:lstm, :recurrent},
    {:gru, :recurrent},
    {:kan, :other}
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

  defp build_opts(arch) do
    case arch do
      :rwkv -> Keyword.merge(@shared_opts, head_size: max(div(@embed, 4), 4))
      _ -> @shared_opts
    end
  end

  def run do
    IO.puts("=" |> String.duplicate(80))
    IO.puts("Edifice Training Throughput — EXLA Backend")

    IO.puts(
      "embed=#{@embed}, seq_len=#{@seq_len}, batch=#{@batch}, " <>
        "layers=#{@num_layers}, iters=#{@iters}"
    )

    IO.puts("=" |> String.duplicate(80))
    IO.puts("")

    # GPU warmup
    IO.puts("Warming up GPU...")
    warmup_model = Edifice.build(:gated_ssm, @shared_opts)
    {init_fn, predict_fn} = Axon.build(warmup_model, mode: :inference)
    template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed}, :f32)}
    params = init_fn.(template, Axon.ModelState.empty())

    for _ <- 1..5,
        do: predict_fn.(params, %{"state_sequence" => rand({@batch, @seq_len, @embed})})

    IO.puts("")

    # Header
    IO.puts(
      "  #{String.pad_trailing("Architecture", 25)}" <>
        "#{String.pad_trailing("Category", 12)}" <>
        "#{String.pad_trailing("Fwd (ms)", 12)}" <>
        "#{String.pad_trailing("Fwd+Bwd (ms)", 14)}" <>
        "#{String.pad_trailing("Bwd/Fwd", 10)}" <>
        "samples/s"
    )

    IO.puts("  " <> String.duplicate("-", 80))

    results =
      for {arch, category} <- @architectures do
        profile_arch(arch, category)
      end

    IO.puts("")
    IO.puts("  " <> String.duplicate("-", 80))

    {ok, failed} = Enum.split_with(results, fn r -> r.status == :ok end)
    IO.puts("  #{length(ok)}/#{length(@architectures)} succeeded")

    if failed != [] do
      IO.puts("")
      IO.puts("  FAILURES:")

      for r <- failed do
        IO.puts("    #{r.arch}: #{r.error}")
      end
    end

    IO.puts("")

    # Ranked by throughput
    IO.puts("## Throughput Ranking (samples/sec)")
    IO.puts("-" |> String.duplicate(60))

    ok
    |> Enum.sort_by(fn r -> -r.throughput end)
    |> Enum.with_index(1)
    |> Enum.each(fn {r, rank} ->
      IO.puts(
        "  #{String.pad_trailing(Integer.to_string(rank), 6)}" <>
          "#{String.pad_trailing(to_string(r.arch), 25)}" <>
          "#{String.pad_trailing(to_string(r.category), 12)}" <>
          "#{Float.round(r.throughput, 1)} samples/s"
      )
    end)

    IO.puts("")

    # Category summary
    IO.puts("## Category Summary")
    IO.puts("-" |> String.duplicate(60))

    ok
    |> Enum.group_by(fn r -> r.category end)
    |> Enum.map(fn {cat, entries} ->
      throughputs = Enum.map(entries, fn r -> r.throughput end) |> Enum.sort(:desc)
      median = Enum.at(throughputs, div(length(throughputs), 2))
      best = hd(throughputs)
      {cat, median, best, length(entries)}
    end)
    |> Enum.sort_by(fn {_, median, _, _} -> -median end)
    |> Enum.each(fn {cat, median, best, count} ->
      IO.puts(
        "  #{String.pad_trailing(to_string(cat), 15)}" <>
          "median=#{String.pad_trailing("#{Float.round(median, 1)}/s", 12)}" <>
          "best=#{String.pad_trailing("#{Float.round(best, 1)}/s", 12)}" <>
          "(#{count} archs)"
      )
    end)

    IO.puts("")
    IO.puts("Done.")
  end

  defp profile_arch(arch, category) do
    try do
      opts = build_opts(arch)
      model = Edifice.build(arch, opts)
      input = rand({@batch, @seq_len, @embed})
      input_map = %{"state_sequence" => input}
      template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed}, :f32)}

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)
      model_state = init_fn.(template, Axon.ModelState.empty())

      # Forward-only timing
      for _ <- 1..3, do: predict_fn.(model_state, input_map)

      {fwd_us, _} =
        :timer.tc(fn ->
          for _ <- 1..@iters, do: predict_fn.(model_state, input_map)
        end)

      fwd_ms = fwd_us / @iters / 1_000

      # Forward + backward timing
      params_data = model_state.data

      loss_fn = fn params ->
        state = %{model_state | data: params}
        output = predict_fn.(state, input_map)
        Nx.mean(output)
      end

      # Warmup grad
      for _ <- 1..3, do: Nx.Defn.value_and_grad(params_data, loss_fn)

      {grad_us, _} =
        :timer.tc(fn ->
          for _ <- 1..@iters, do: Nx.Defn.value_and_grad(params_data, loss_fn)
        end)

      grad_ms = grad_us / @iters / 1_000

      bwd_ratio = if fwd_ms > 0, do: Float.round(grad_ms / fwd_ms, 1), else: 0.0
      throughput = if grad_ms > 0, do: @batch * 1_000 / grad_ms, else: 0.0

      IO.puts(
        "  #{String.pad_trailing(to_string(arch), 25)}" <>
          "#{String.pad_trailing(to_string(category), 12)}" <>
          "#{String.pad_trailing(fmt(fwd_ms), 12)}" <>
          "#{String.pad_trailing(fmt(grad_ms), 14)}" <>
          "#{String.pad_trailing("#{bwd_ratio}x", 10)}" <>
          "#{Float.round(throughput, 1)}"
      )

      %{
        arch: arch,
        category: category,
        fwd_ms: fwd_ms,
        grad_ms: grad_ms,
        bwd_ratio: bwd_ratio,
        throughput: throughput,
        status: :ok
      }
    rescue
      e ->
        msg = Exception.message(e) |> String.slice(0, 80)

        IO.puts(
          "  #{String.pad_trailing(to_string(arch), 25)}" <>
            "#{String.pad_trailing(to_string(category), 12)}" <>
            "FAIL: #{msg}"
        )

        %{arch: arch, category: category, status: :fail, error: msg}
    end
  end
end

TrainingThroughput.run()
