# GPU Memory Profile
#
# Measures parameter count and peak memory usage per architecture.
# Uses nvidia-smi polling for GPU memory and Nx tensor byte counting
# for parameter sizes.
#
# Usage:
#   EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/memory_profile.exs
#
# Environment variables:
#   BENCH_EMBED   - Embedding dimension (default: 256)
#   BENCH_SEQ_LEN - Sequence length (default: 32)
#   BENCH_BATCH   - Batch size (default: 1)
#   BENCH_LAYERS  - Number of layers (default: 2)

Nx.default_backend(EXLA.Backend)

defmodule MemoryProfile do
  @embed String.to_integer(System.get_env("BENCH_EMBED", "256"))
  @seq_len String.to_integer(System.get_env("BENCH_SEQ_LEN", "32"))
  @batch String.to_integer(System.get_env("BENCH_BATCH", "1"))
  @num_layers String.to_integer(System.get_env("BENCH_LAYERS", "2"))

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

  @architectures [
    {:gated_ssm, :ssm},
    {:fnet, :attention},
    {:gqa, :attention},
    {:mamba_hillis_steele, :ssm},
    {:jamba, :ssm},
    {:s4, :ssm},
    {:s4d, :ssm},
    {:s5, :ssm},
    {:hyena, :ssm},
    {:mamba, :ssm},
    {:mamba_ssd, :ssm},
    {:mamba_cumsum, :ssm},
    {:bimamba, :ssm},
    {:striped_hyena, :ssm},
    {:zamba, :ssm},
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
    {:titans, :recurrent},
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

  defp build_opts(arch) do
    case arch do
      :rwkv -> Keyword.merge(@shared_opts, head_size: max(div(@embed, 4), 4))
      _ -> @shared_opts
    end
  end

  defp fmt_bytes(bytes) when bytes < 1024, do: "#{bytes} B"
  defp fmt_bytes(bytes) when bytes < 1024 * 1024, do: "#{Float.round(bytes / 1024, 1)} KB"

  defp fmt_bytes(bytes) when bytes < 1024 * 1024 * 1024,
    do: "#{Float.round(bytes / (1024 * 1024), 2)} MB"

  defp fmt_bytes(bytes), do: "#{Float.round(bytes / (1024 * 1024 * 1024), 2)} GB"

  defp count_params(model_state) do
    flatten_params(model_state.data)
    |> Enum.reduce({0, 0}, fn {_path, tensor}, {count, bytes} ->
      n = Nx.size(tensor)
      b = Nx.byte_size(tensor)
      {count + n, bytes + b}
    end)
  end

  defp flatten_params(map) when is_map(map) do
    Enum.flat_map(map, fn
      {key, %Nx.Tensor{} = tensor} ->
        [{key, tensor}]

      {key, inner} when is_map(inner) ->
        flatten_params(inner) |> Enum.map(fn {k, v} -> {"#{key}.#{k}", v} end)

      _ ->
        []
    end)
  end

  defp flatten_params(_), do: []

  defp gpu_memory_mib do
    case System.cmd("nvidia-smi", [
           "--query-gpu=memory.used",
           "--format=csv,noheader,nounits"
         ]) do
      {output, 0} ->
        output
        |> String.trim()
        |> String.split("\n")
        |> hd()
        |> String.trim()
        |> String.to_integer()

      _ ->
        nil
    end
  rescue
    _ -> nil
  end

  def run do
    IO.puts("=" |> String.duplicate(80))
    IO.puts("Edifice Memory Profile — EXLA Backend")

    IO.puts("embed=#{@embed}, seq_len=#{@seq_len}, batch=#{@batch}, layers=#{@num_layers}")

    IO.puts("=" |> String.duplicate(80))
    IO.puts("")

    has_nvidia_smi = gpu_memory_mib() != nil

    if has_nvidia_smi do
      IO.puts("  nvidia-smi detected — GPU memory tracking enabled")
    else
      IO.puts("  nvidia-smi not found — GPU memory tracking disabled")
    end

    IO.puts("")

    # GPU warmup
    IO.puts("Warming up GPU...")
    warmup_model = Edifice.build(:gated_ssm, @shared_opts)
    {init_fn, predict_fn} = Axon.build(warmup_model)
    template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed}, :f32)}
    _params = init_fn.(template, Axon.ModelState.empty())
    IO.puts("")

    baseline_gpu_mib = if has_nvidia_smi, do: gpu_memory_mib(), else: nil

    if baseline_gpu_mib do
      IO.puts("  Baseline GPU memory: #{baseline_gpu_mib} MiB")
      IO.puts("")
    end

    # Header
    gpu_col = if has_nvidia_smi, do: "#{String.pad_trailing("GPU Delta", 12)}", else: ""

    IO.puts(
      "  #{String.pad_trailing("Architecture", 25)}" <>
        "#{String.pad_trailing("Category", 12)}" <>
        "#{String.pad_trailing("Params", 12)}" <>
        "#{String.pad_trailing("Param Size", 12)}" <>
        gpu_col <>
        "Layers"
    )

    IO.puts("  " <> String.duplicate("-", 75 + if(has_nvidia_smi, do: 12, else: 0)))

    results =
      for {arch, category} <- @architectures do
        profile_arch(arch, category, baseline_gpu_mib, has_nvidia_smi)
      end

    IO.puts("")

    {ok, _failed} = Enum.split_with(results, fn r -> r.status == :ok end)

    # Size ranking
    IO.puts("## Parameter Size Ranking (smallest first)")
    IO.puts("-" |> String.duplicate(60))

    ok
    |> Enum.sort_by(fn r -> r.param_bytes end)
    |> Enum.with_index(1)
    |> Enum.each(fn {r, rank} ->
      IO.puts(
        "  #{String.pad_trailing(Integer.to_string(rank), 6)}" <>
          "#{String.pad_trailing(to_string(r.arch), 25)}" <>
          "#{String.pad_trailing(fmt_bytes(r.param_bytes), 12)}" <>
          "(#{r.param_count} params)"
      )
    end)

    IO.puts("")

    # Category summary
    IO.puts("## Category Summary (median param size)")
    IO.puts("-" |> String.duplicate(60))

    ok
    |> Enum.group_by(fn r -> r.category end)
    |> Enum.map(fn {cat, entries} ->
      sizes = Enum.map(entries, fn r -> r.param_bytes end) |> Enum.sort()
      median = Enum.at(sizes, div(length(sizes), 2))
      smallest = hd(sizes)
      largest = List.last(sizes)
      {cat, median, smallest, largest, length(entries)}
    end)
    |> Enum.sort_by(fn {_, median, _, _, _} -> median end)
    |> Enum.each(fn {cat, median, smallest, largest, count} ->
      IO.puts(
        "  #{String.pad_trailing(to_string(cat), 15)}" <>
          "median=#{String.pad_trailing(fmt_bytes(median), 12)}" <>
          "min=#{String.pad_trailing(fmt_bytes(smallest), 12)}" <>
          "max=#{String.pad_trailing(fmt_bytes(largest), 12)}" <>
          "(#{count} archs)"
      )
    end)

    IO.puts("")
    IO.puts("Done.")
  end

  defp profile_arch(arch, category, baseline_gpu_mib, has_nvidia_smi) do
    try do
      opts = build_opts(arch)
      model = Edifice.build(arch, opts)

      input = rand({@batch, @seq_len, @embed})
      input_map = %{"state_sequence" => input}
      template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed}, :f32)}

      {init_fn, predict_fn} = Axon.build(model)
      model_state = init_fn.(template, Axon.ModelState.empty())

      {param_count, param_bytes} = count_params(model_state)

      # Run inference to measure GPU memory with activations loaded
      predict_fn.(model_state, input_map)

      gpu_delta =
        if has_nvidia_smi && baseline_gpu_mib do
          current = gpu_memory_mib()
          if current, do: current - baseline_gpu_mib, else: nil
        end

      # Count layers from flat params (unique layer prefixes)
      layer_count =
        flatten_params(model_state.data)
        |> Enum.map(fn {path, _} -> path |> String.split(".") |> hd() end)
        |> Enum.uniq()
        |> length()

      gpu_col =
        if has_nvidia_smi do
          val = if gpu_delta, do: "#{gpu_delta} MiB", else: "?"
          String.pad_trailing(val, 12)
        else
          ""
        end

      IO.puts(
        "  #{String.pad_trailing(to_string(arch), 25)}" <>
          "#{String.pad_trailing(to_string(category), 12)}" <>
          "#{String.pad_trailing(Integer.to_string(param_count), 12)}" <>
          "#{String.pad_trailing(fmt_bytes(param_bytes), 12)}" <>
          gpu_col <>
          "#{layer_count}"
      )

      %{
        arch: arch,
        category: category,
        param_count: param_count,
        param_bytes: param_bytes,
        gpu_delta_mib: gpu_delta,
        layer_count: layer_count,
        status: :ok
      }
    rescue
      e ->
        msg = Exception.message(e) |> String.slice(0, 60)

        IO.puts(
          "  #{String.pad_trailing(to_string(arch), 25)}" <>
            "#{String.pad_trailing(to_string(category), 12)}" <>
            "FAIL: #{msg}"
        )

        %{arch: arch, category: category, status: :fail, error: msg}
    end
  end
end

MemoryProfile.run()
