# Architecture Performance Profile
#
# Profiles representative architectures across all major computational patterns
# and families. Measures EXLA compilation time, warm inference throughput, and
# memory usage.
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
      # ── Core families (original 12) ──────────────────────────────
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
      {"VAE (decoder)", &build_vae_decoder/0, &input_latent/0},
      # ── New families ─────────────────────────────────────────────
      {"DiffTransformer", &build_diff_transformer/0, &input_seq/0},
      {"GLA v2", &build_gla_v2/0, &input_seq/0},
      {"DINOv2 (student)", &build_dino_v2/0, &input_image_chw/0},
      {"EfficientViT", &build_efficient_vit/0, &input_image_chw/0},
      {"DiT", &build_dit/0, &input_dit/0},
      {"SiT", &build_sit/0, &input_dit/0},
      {"SparseAutoencoder", &build_sae/0, &input_flat/0},
      {"MoE v2", &build_moe_v2/0, &input_moe/0},
      {"PolicyValue", &build_policy_value/0, &input_flat/0},
      {"FNO", &build_fno/0, &input_fno/0},
      {"SoundStorm", &build_soundstorm/0, &input_tokens/0},
      {"Medusa", &build_medusa/0, &input_flat/0},
      {"Multimodal Fusion", &build_fusion/0, &input_fusion/0},
      {"EGNN", &build_egnn/0, &input_egnn/0}
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

  # -- New family builders --

  def build_diff_transformer do
    Edifice.Attention.DiffTransformer.build(
      embed_dim: @hidden,
      hidden_size: @hidden,
      num_heads: 4,
      num_layers: 2,
      window_size: @seq_len,
      dropout: 0.0
    )
  end

  def build_gla_v2 do
    Edifice.Attention.GLAv2.build(
      embed_dim: @hidden,
      hidden_size: @hidden,
      num_heads: 4,
      head_dim: 16,
      num_layers: 2,
      window_size: @seq_len,
      dropout: 0.0
    )
  end

  def build_dino_v2 do
    {student, _teacher} =
      Edifice.Vision.DINOv2.build(
        image_size: 32,
        patch_size: 8,
        in_channels: 3,
        embed_dim: @hidden,
        depth: 2,
        num_heads: 4,
        dropout: 0.0
      )

    student
  end

  def build_efficient_vit do
    Edifice.Vision.EfficientViT.build(
      image_size: 32,
      patch_size: 8,
      in_channels: 3,
      embed_dim: 32,
      depths: [1, 1],
      num_heads: [2, 2]
    )
  end

  def build_dit do
    Edifice.Generative.DiT.build(
      input_dim: @hidden,
      hidden_size: @hidden,
      depth: 2,
      num_heads: 4,
      dropout: 0.0
    )
  end

  def build_sit do
    Edifice.Generative.SiT.build(
      input_dim: @hidden,
      hidden_size: @hidden,
      depth: 2,
      num_heads: 4,
      dropout: 0.0
    )
  end

  def build_sae do
    Edifice.Interpretability.SparseAutoencoder.build(
      input_size: @hidden,
      dict_size: @hidden * 4
    )
  end

  def build_moe_v2 do
    Edifice.Meta.MoEv2.build(
      input_size: @hidden,
      hidden_size: @hidden * 4,
      output_size: @hidden,
      num_shared_experts: 1,
      num_routed_experts: 2,
      tokens_per_expert: 2,
      dropout: 0.0
    )
  end

  def build_policy_value do
    Edifice.RL.PolicyValue.build(
      input_size: @hidden,
      action_size: @num_classes,
      hidden_size: @hidden
    )
  end

  def build_fno do
    Edifice.Scientific.FNO.build(
      in_channels: 3,
      out_channels: 1,
      modes: 8,
      hidden_channels: @hidden,
      num_layers: 2
    )
  end

  def build_soundstorm do
    Edifice.Audio.SoundStorm.build(
      num_codebooks: 2,
      codebook_size: 64,
      hidden_dim: @hidden,
      num_layers: 2,
      num_heads: 4,
      conv_kernel_size: 3,
      dropout: 0.0
    )
  end

  def build_medusa do
    Edifice.Inference.Medusa.build(
      base_hidden_dim: @hidden,
      vocab_size: 64,
      num_medusa_heads: 2,
      medusa_num_layers: 1
    )
  end

  def build_fusion do
    Edifice.Multimodal.Fusion.build(
      vision_dim: @hidden,
      llm_dim: @hidden,
      num_visual_tokens: 4,
      text_seq_len: @seq_len
    )
  end

  def build_egnn do
    Edifice.Graph.EGNN.build(
      in_node_features: 8,
      hidden_dim: 32,
      num_layers: 2,
      coord_dim: 3
    )
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

  def input_dit do
    %{
      "noisy_input" => rand({@batch, @hidden}),
      "timestep" => rand({@batch})
    }
  end

  def input_moe, do: rand({@batch, @seq_len, @hidden})

  def input_tokens do
    Nx.iota({@batch, @seq_len}, axis: 1) |> Nx.remainder(64)
  end

  def input_fno, do: rand({@batch, @seq_len, 3})

  def input_fusion do
    %{
      "visual_tokens" => rand({@batch, 4, @hidden}),
      "text_embeddings" => rand({@batch, @seq_len, @hidden})
    }
  end

  def input_egnn do
    %{
      "nodes" => rand({@batch, 16, 8}),
      "coords" => rand({@batch, 16, 3}),
      "edge_index" => Nx.tensor([[[0, 1], [1, 2], [2, 0], [0, 0], [1, 1], [2, 2], [3, 4], [4, 5], [5, 3], [6, 7], [7, 8], [8, 6], [9, 10], [10, 11], [11, 9], [12, 13]]]) |> Nx.broadcast({@batch, 16, 2})
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
