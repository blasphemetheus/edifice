# Mixed Precision Throughput Benchmark
#
# Compares f32 vs bf16 forward pass throughput on representative architectures.
#
# Usage:
#   mix run bench/mixed_precision_bench.exs
#   EXLA_TARGET=cuda mix run bench/mixed_precision_bench.exs   # GPU

alias Edifice.MixedPrecision

IO.puts("=" |> String.duplicate(70))
IO.puts("Mixed Precision Throughput Benchmark")
IO.puts("=" |> String.duplicate(70))
IO.puts("")

# Representative architectures with small dims for CPU benchmarking
configs = [
  {:decoder_only, [embed_dim: 64, hidden_size: 64, seq_len: 32, num_layers: 2, num_heads: 4, num_kv_heads: 4]},
  {:mlp, [layers: [64, 64, 32], dropout: 0.0]},
  {:min_gru, [hidden_size: 64, seq_len: 32]}
]

batch_size = 4
warmup = 3
iterations = 10

results = Enum.map(configs, fn {arch, opts} ->
  IO.puts("--- #{arch} ---")

  # Build f32 and bf16 models
  model_f32 = Edifice.build(arch, opts)
  model_bf16 = MixedPrecision.apply(model_f32, :bf16)

  summary = MixedPrecision.summary(model_bf16)
  IO.puts("  Layers: #{summary.total} total, #{summary.with_policy} bf16, #{summary.without_policy} f32")

  # Determine input shape
  seq_len = Keyword.get(opts, :seq_len, 32)
  embed_dim = opts[:embed_dim] || opts[:hidden_size] || hd(opts[:layers] || [64])

  input_shape = {batch_size, seq_len, embed_dim}
  template = %{"state_sequence" => Nx.template(input_shape, :f32)}
  {input, _key} = Nx.Random.uniform(Nx.Random.key(42), shape: input_shape)
  input_map = %{"state_sequence" => input}

  # Build both
  {init_f32, pred_f32} = Axon.build(model_f32)
  {init_bf16, pred_bf16} = Axon.build(model_bf16)

  params_f32 = init_f32.(template, Axon.ModelState.empty())
  params_bf16 = init_bf16.(template, Axon.ModelState.empty())

  # Warmup
  for _ <- 1..warmup do
    pred_f32.(params_f32, input_map)
    pred_bf16.(params_bf16, input_map)
  end

  # Benchmark f32
  {f32_us, _} = :timer.tc(fn ->
    for _ <- 1..iterations, do: pred_f32.(params_f32, input_map)
  end)

  # Benchmark bf16
  {bf16_us, _} = :timer.tc(fn ->
    for _ <- 1..iterations, do: pred_bf16.(params_bf16, input_map)
  end)

  f32_ms = f32_us / (iterations * 1000)
  bf16_ms = bf16_us / (iterations * 1000)
  speedup = f32_ms / max(bf16_ms, 0.001)

  IO.puts("  f32:  #{Float.round(f32_ms, 2)} ms/iter")
  IO.puts("  bf16: #{Float.round(bf16_ms, 2)} ms/iter")
  IO.puts("  Speedup: #{Float.round(speedup, 2)}x")
  IO.puts("")

  {arch, f32_ms, bf16_ms, speedup}
end)

# Summary table
IO.puts("=" |> String.duplicate(70))
IO.puts("Summary")
IO.puts("=" |> String.duplicate(70))
IO.puts(String.pad_trailing("Architecture", 20) <>
  String.pad_trailing("f32 (ms)", 12) <>
  String.pad_trailing("bf16 (ms)", 12) <>
  "Speedup")
IO.puts("-" |> String.duplicate(56))

Enum.each(results, fn {arch, f32_ms, bf16_ms, speedup} ->
  IO.puts(
    String.pad_trailing("#{arch}", 20) <>
    String.pad_trailing("#{Float.round(f32_ms, 2)}", 12) <>
    String.pad_trailing("#{Float.round(bf16_ms, 2)}", 12) <>
    "#{Float.round(speedup, 2)}x"
  )
end)

IO.puts("")
IO.puts("Note: Real speedup requires GPU (EXLA_TARGET=cuda). On CPU/BinaryBackend,")
IO.puts("bf16 may not be faster due to lack of hardware bf16 units.")
