# Architecture Performance Profile
#
# Profiles 12 representative architectures across all major computational patterns.
# Measures EXLA compilation time, warm inference throughput, and memory usage.
#
# Usage:
#   mix run bench/architecture_profile.exs
#
# Requires EXLA compiled (EXLA_TARGET=host for CPU, or CUDA).

Nx.default_backend(EXLA.Backend)

defmodule ArchProfile do
  @batch 4
  @seq_len 32
  @hidden 64
  @num_classes 10

  def architectures do
    [
      {"MLP (baseline)", &build_mlp/0, &input_mlp/0},
      {"MultiHead Attention", &build_multihead/0, &input_seq/0},
      {"Linear Transformer", &build_linear_transformer/0, &input_seq/0},
      {"Mamba", &build_mamba/0, &input_seq/0},
      {"S4", &build_s4/0, &input_seq/0},
      {"GCN", &build_gcn/0, &input_gcn/0},
      {"ResNet-18", &build_resnet/0, &input_image/0},
      {"ViT", &build_vit/0, &input_image_chw/0},
      {"DeepSets", &build_deep_sets/0, &input_set/0},
      {"NTM", &build_ntm/0, &input_ntm/0},
      {"VAE (encoder)", &build_vae_encoder/0, &input_flat/0},
      {"VAE (decoder)", &build_vae_decoder/0, &input_latent/0}
    ]
  end

  # -- Builders --

  def build_mlp do
    Edifice.Feedforward.MLP.build(
      input_size: @hidden,
      hidden_sizes: [128, @num_classes]
    )
  end

  def build_multihead do
    Edifice.Attention.MultiHead.build(
      embed_dim: @hidden,
      hidden_size: @hidden,
      num_heads: 4,
      head_dim: 16,
      num_layers: 2,
      window_size: @seq_len
    )
  end

  def build_linear_transformer do
    Edifice.Attention.LinearTransformer.build(
      embed_dim: @hidden,
      hidden_size: @hidden,
      num_heads: 4,
      num_layers: 2,
      window_size: @seq_len
    )
  end

  def build_mamba do
    Edifice.SSM.Mamba.build(
      embed_dim: @hidden,
      hidden_size: @hidden,
      state_size: 16,
      num_layers: 2,
      window_size: @seq_len
    )
  end

  def build_s4 do
    Edifice.SSM.S4.build(
      embed_dim: @hidden,
      hidden_size: @hidden,
      state_size: 16,
      num_layers: 2,
      window_size: @seq_len
    )
  end

  def build_gcn do
    Edifice.Graph.GCN.build(
      input_dim: 8,
      hidden_dims: [32, 32],
      num_classes: @num_classes
    )
  end

  def build_resnet do
    Edifice.Convolutional.ResNet.build(
      input_shape: {nil, 32, 32, 3},
      num_classes: @num_classes,
      block_sizes: [2, 2, 2, 2],
      initial_channels: 16
    )
  end

  def build_vit do
    Edifice.Vision.ViT.build(
      image_size: 32,
      patch_size: 8,
      in_channels: 3,
      embed_dim: @hidden,
      depth: 2,
      num_heads: 4,
      num_classes: @num_classes
    )
  end

  def build_deep_sets do
    Edifice.Sets.DeepSets.build(
      input_dim: 8,
      hidden_dim: 32,
      output_dim: @num_classes
    )
  end

  def build_ntm do
    Edifice.Memory.NTM.build(
      input_size: @hidden,
      memory_size: 32,
      memory_dim: 16,
      controller_size: @hidden,
      output_size: @num_classes
    )
  end

  def build_vae_encoder do
    {encoder, _decoder} =
      Edifice.Generative.VAE.build(
        input_size: @hidden,
        latent_size: 16,
        encoder_sizes: [64, 32],
        decoder_sizes: [32, 64]
      )

    encoder
  end

  def build_vae_decoder do
    {_encoder, decoder} =
      Edifice.Generative.VAE.build(
        input_size: @hidden,
        latent_size: 16,
        encoder_sizes: [64, 32],
        decoder_sizes: [32, 64]
      )

    decoder
  end

  # -- Inputs --

  defp rand(shape) do
    key = Nx.Random.key(42)
    {tensor, _key} = Nx.Random.normal(key, shape: shape)
    tensor
  end

  def input_mlp, do: rand({@batch, @hidden})
  def input_seq, do: rand({@batch, @seq_len, @hidden})
  def input_flat, do: rand({@batch, @hidden})
  def input_latent, do: rand({@batch, 16})
  def input_image, do: rand({@batch, 32, 32, 3})
  def input_image_chw, do: rand({@batch, 3, 32, 32})
  def input_set, do: rand({@batch, 16, 8})

  def input_gcn do
    nodes = rand({@batch, 16, 8})
    adj = Nx.eye(16) |> Nx.broadcast({@batch, 16, 16})
    %{"nodes" => nodes, "adjacency" => adj}
  end

  def input_ntm do
    %{
      "input" => rand({@batch, @hidden}),
      "memory" => rand({@batch, 32, 16})
    }
  end

  # -- Profiling --

  def compile_and_init(model, %{} = input) when not is_struct(input) do
    template =
      Map.new(input, fn {k, v} -> {k, Nx.template(Nx.shape(v), Nx.type(v))} end)

    {init_fn, predict_fn} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())
    {predict_fn, params}
  end

  def compile_and_init(model, input) do
    template = Nx.template(Nx.shape(input), Nx.type(input))
    {init_fn, predict_fn} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())
    {predict_fn, params}
  end

  def run do
    IO.puts("=" |> String.duplicate(70))
    IO.puts("Edifice Architecture Profile — EXLA Backend")
    IO.puts("batch=#{@batch}, seq_len=#{@seq_len}, hidden=#{@hidden}")
    IO.puts("=" |> String.duplicate(70))
    IO.puts("")

    # Phase 1: Compilation timing
    IO.puts("## Phase 1: EXLA Compilation Time")
    IO.puts("-" |> String.duplicate(50))

    compiled =
      for {name, build_fn, input_fn} <- architectures() do
        model = build_fn.()
        input = input_fn.()

        {compile_us, {predict_fn, params}} =
          :timer.tc(fn -> compile_and_init(model, input) end)

        compile_ms = compile_us / 1_000
        IO.puts("  #{String.pad_trailing(name, 25)} #{Float.round(compile_ms, 1)} ms")
        {name, predict_fn, params, input}
      end

    IO.puts("")

    # Phase 2: Warm inference throughput via Benchee.
    # NTM uses while loops that have EXLA cross-process issues with Benchee's
    # parallel execution, so we benchmark it separately with :timer.tc.
    IO.puts("## Phase 2: Warm Inference Throughput")
    IO.puts("-" |> String.duplicate(50))
    IO.puts("")

    {ntm_entries, benchmarkable} =
      Enum.split_with(compiled, fn {name, _, _, _} -> name == "NTM" end)

    # Benchmark NTM separately (in-process, no Benchee parallelism)
    for {name, predict_fn, params, input} <- ntm_entries do
      # Warm up
      predict_fn.(params, input)

      {total_us, _} =
        :timer.tc(fn ->
          for _ <- 1..20, do: predict_fn.(params, input)
        end)

      avg_ms = total_us / 20 / 1_000
      IO.puts("  #{name}: ~#{Float.round(avg_ms, 2)} ms/iter (manual, 20 iters)")
    end

    if ntm_entries != [], do: IO.puts("")

    benchmarks =
      Map.new(benchmarkable, fn {name, predict_fn, params, input} ->
        {name, fn -> predict_fn.(params, input) end}
      end)

    Benchee.run(benchmarks,
      warmup: 1,
      time: 3,
      memory_time: 1,
      print: [configuration: false],
      formatters: [Benchee.Formatters.Console]
    )
  end
end

ArchProfile.run()
