# Full Architecture Sweep Benchmark
#
# Builds and profiles EVERY architecture in the Edifice registry on EXLA.
# Measures build time, EXLA compilation time, and warm inference time.
# Flags outliers that are suspiciously slow relative to their family.
#
# Usage:
#   mix run bench/full_sweep.exs
#
# Requires EXLA compiled (EXLA_TARGET=host for CPU, or CUDA).

Nx.default_backend(EXLA.Backend)

defmodule FullSweep do
  # Shared small dims (same as registry_sweep_test.exs)
  @batch 4
  @embed 32
  @hidden 16
  @seq_len 8
  @state_size 8
  @num_layers 2
  @image_size 16
  @in_channels 3
  @num_nodes 6
  @node_dim 16
  @num_classes 4
  @num_points 12
  @point_dim 3
  @num_memories 4
  @memory_dim 8
  @latent_size 8
  @action_dim 4
  @action_horizon 4
  @warmup_iters 3
  @timing_iters 10

  defp rand(shape) do
    key = Nx.Random.key(42)
    {tensor, _key} = Nx.Random.normal(key, shape: shape)
    tensor
  end

  # ── Architecture Specs ─────────────────────────────────────────────
  # Each: {name, family, build_fn, input_fn}

  def specs do
    sequence_specs() ++
      special_sequence_specs() ++
      feedforward_specs() ++
      vision_specs() ++
      graph_specs() ++
      set_specs() ++
      energy_specs() ++
      probabilistic_specs() ++
      memory_specs() ++
      meta_specs() ++
      neuromorphic_specs() ++
      convolutional_specs() ++
      generative_specs() ++
      contrastive_specs()
  end

  @sequence_opts [
    embed_dim: @embed,
    hidden_size: @hidden,
    state_size: @state_size,
    num_layers: @num_layers,
    seq_len: @seq_len,
    window_size: @seq_len,
    head_dim: 8,
    num_heads: 2,
    dropout: 0.0
  ]

  @sequence_archs ~w(
    mamba mamba_ssd mamba_cumsum mamba_hillis_steele
    s4 s4d s5 h3 hyena bimamba gated_ssm jamba zamba
    lstm gru xlstm min_gru min_lstm delta_net ttt titans
    retnet gla hgrn griffin gqa fnet linear_transformer nystromformer performer
    kan liquid
  )a

  defp sequence_specs do
    for arch <- @sequence_archs do
      family =
        cond do
          arch in ~w(mamba mamba_ssd mamba_cumsum mamba_hillis_steele s4 s4d s5 h3 hyena bimamba gated_ssm jamba zamba)a ->
            "ssm"

          arch in ~w(lstm gru xlstm min_gru min_lstm delta_net ttt titans)a ->
            "recurrent"

          arch in ~w(retnet gla hgrn griffin gqa fnet linear_transformer nystromformer performer)a ->
            "attention"

          arch == :kan ->
            "feedforward"

          arch == :liquid ->
            "liquid"
        end

      {arch, family, fn -> Edifice.build(arch, @sequence_opts) end,
       fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end}
    end
  end

  defp special_sequence_specs do
    [
      {:reservoir, "recurrent",
       fn ->
         Edifice.build(:reservoir,
           input_size: @embed,
           reservoir_size: @hidden,
           output_size: @hidden,
           seq_len: @seq_len
         )
       end, fn -> %{"input" => rand({@batch, @seq_len, @embed})} end},
      {:rwkv, "attention",
       fn ->
         Edifice.build(:rwkv,
           embed_dim: @embed,
           hidden_size: @hidden,
           head_size: 8,
           num_layers: @num_layers,
           seq_len: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      {:switch_moe, "meta",
       fn ->
         Edifice.build(:switch_moe,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_layers: @num_layers,
           seq_len: @seq_len,
           num_experts: 2,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      {:soft_moe, "meta",
       fn ->
         Edifice.build(:soft_moe,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_layers: @num_layers,
           seq_len: @seq_len,
           num_experts: 2,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end}
    ]
  end

  defp feedforward_specs do
    [
      {:mlp, "feedforward",
       fn -> Edifice.build(:mlp, input_size: @embed, hidden_sizes: [@hidden]) end,
       fn -> %{"input" => rand({@batch, @embed})} end},
      {:tabnet, "feedforward",
       fn -> Edifice.build(:tabnet, input_size: @embed, output_size: @num_classes) end,
       fn -> %{"input" => rand({@batch, @embed})} end}
    ]
  end

  defp vision_specs do
    vision_opts = [
      image_size: @image_size,
      in_channels: @in_channels,
      patch_size: 4,
      embed_dim: @hidden,
      hidden_dim: @hidden,
      depth: 1,
      num_heads: 2,
      dropout: 0.0
    ]

    simple =
      for arch <- [:vit, :deit, :mlp_mixer] do
        {arch, "vision", fn -> Edifice.build(arch, vision_opts) end,
         fn -> %{"image" => rand({@batch, @in_channels, @image_size, @image_size})} end}
      end

    simple ++
      [
        {:swin, "vision",
         fn ->
           Edifice.build(:swin,
             image_size: 32,
             in_channels: @in_channels,
             patch_size: 4,
             embed_dim: @hidden,
             depths: [1, 1],
             num_heads: [2, 2],
             window_size: 4,
             dropout: 0.0
           )
         end, fn -> %{"image" => rand({@batch, @in_channels, 32, 32})} end},
        {:convnext, "vision",
         fn ->
           Edifice.build(:convnext,
             image_size: 32,
             in_channels: @in_channels,
             patch_size: 4,
             dims: [@hidden, @hidden * 2],
             depths: [1, 1],
             dropout: 0.0
           )
         end, fn -> %{"image" => rand({@batch, @in_channels, 32, 32})} end},
        {:unet, "vision",
         fn ->
           Edifice.build(:unet,
             in_channels: @in_channels,
             out_channels: 1,
             image_size: @image_size,
             base_features: 8,
             depth: 2,
             dropout: 0.0
           )
         end, fn -> %{"image" => rand({@batch, @in_channels, @image_size, @image_size})} end}
      ]
  end

  defp graph_specs do
    graph_opts = [
      input_dim: @node_dim,
      hidden_dim: @hidden,
      num_classes: @num_classes,
      num_layers: @num_layers,
      num_heads: 2,
      dropout: 0.0
    ]

    graph_input = fn ->
      nodes = rand({@batch, @num_nodes, @node_dim})
      adj = Nx.eye(@num_nodes) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})
      %{"nodes" => nodes, "adjacency" => adj}
    end

    standard =
      for arch <- [:gcn, :gat, :graph_sage, :gin, :pna, :graph_transformer] do
        {arch, "graph", fn -> Edifice.build(arch, graph_opts) end, graph_input}
      end

    standard ++
      [
        {:schnet, "graph",
         fn ->
           Edifice.build(:schnet,
             input_dim: @node_dim,
             hidden_dim: @hidden,
             num_interactions: 2,
             num_filters: @hidden,
             num_rbf: 10
           )
         end, graph_input}
      ]
  end

  defp set_specs do
    [
      {:deep_sets, "sets",
       fn -> Edifice.build(:deep_sets, input_dim: @point_dim, output_dim: @num_classes) end,
       fn -> %{"input" => rand({@batch, @num_points, @point_dim})} end},
      {:pointnet, "sets",
       fn -> Edifice.build(:pointnet, num_classes: @num_classes, input_dim: @point_dim) end,
       fn -> %{"input" => rand({@batch, @num_points, @point_dim})} end}
    ]
  end

  defp energy_specs do
    [
      {:ebm, "energy", fn -> Edifice.build(:ebm, input_size: @embed) end,
       fn -> %{"input" => rand({@batch, @embed})} end},
      {:hopfield, "energy", fn -> Edifice.build(:hopfield, input_dim: @embed) end,
       fn -> %{"input" => rand({@batch, @embed})} end},
      {:neural_ode, "energy",
       fn -> Edifice.build(:neural_ode, input_size: @embed, hidden_size: @hidden) end,
       fn -> %{"input" => rand({@batch, @embed})} end}
    ]
  end

  defp probabilistic_specs do
    [
      {:bayesian, "probabilistic",
       fn -> Edifice.build(:bayesian, input_size: @embed, output_size: @num_classes) end,
       fn -> %{"input" => rand({@batch, @embed})} end},
      {:mc_dropout, "probabilistic",
       fn -> Edifice.build(:mc_dropout, input_size: @embed, output_size: @num_classes) end,
       fn -> %{"input" => rand({@batch, @embed})} end},
      {:evidential, "probabilistic",
       fn -> Edifice.build(:evidential, input_size: @embed, num_classes: @num_classes) end,
       fn -> %{"input" => rand({@batch, @embed})} end}
    ]
  end

  defp memory_specs do
    [
      {:ntm, "memory",
       fn ->
         Edifice.build(:ntm,
           input_size: @embed,
           output_size: @hidden,
           memory_size: @num_memories,
           memory_dim: @memory_dim,
           num_heads: 1
         )
       end,
       fn ->
         %{
           "input" => rand({@batch, @embed}),
           "memory" => rand({@batch, @num_memories, @memory_dim})
         }
       end},
      {:memory_network, "memory",
       fn ->
         Edifice.build(:memory_network,
           input_dim: @embed,
           output_dim: @hidden,
           num_memories: @num_memories
         )
       end,
       fn ->
         %{
           "query" => rand({@batch, @embed}),
           "memories" => rand({@batch, @num_memories, @embed})
         }
       end}
    ]
  end

  defp meta_specs do
    [
      {:moe, "meta",
       fn ->
         Edifice.build(:moe,
           input_size: @embed,
           hidden_size: @hidden * 4,
           output_size: @hidden,
           num_experts: 2,
           top_k: 1
         )
       end, fn -> %{"moe_input" => rand({@batch, @seq_len, @embed})} end},
      {:lora, "meta",
       fn -> Edifice.build(:lora, input_size: @embed, output_size: @hidden, rank: 4) end,
       fn -> %{"input" => rand({@batch, @embed})} end},
      {:adapter, "meta", fn -> Edifice.build(:adapter, hidden_size: @hidden) end,
       fn -> %{"input" => rand({@batch, @hidden})} end},
      {:hypernetwork, "meta",
       fn ->
         Edifice.build(:hypernetwork,
           conditioning_size: @embed,
           target_layer_sizes: [{@embed, @hidden}],
           input_size: @embed
         )
       end,
       fn ->
         %{
           "conditioning" => rand({@batch, @embed}),
           "data_input" => rand({@batch, @embed})
         }
       end},
      {:capsule, "meta",
       fn ->
         Edifice.build(:capsule,
           input_shape: {nil, 28, 28, 1},
           conv_channels: 32,
           conv_kernel: 9,
           num_primary_caps: 8,
           primary_cap_dim: 4,
           num_digit_caps: @num_classes,
           digit_cap_dim: 4
         )
       end, fn -> %{"input" => rand({@batch, 28, 28, 1})} end}
    ]
  end

  defp neuromorphic_specs do
    [
      {:snn, "neuromorphic",
       fn ->
         Edifice.build(:snn,
           input_size: @embed,
           output_size: @num_classes,
           hidden_sizes: [@hidden]
         )
       end, fn -> %{"input" => rand({@batch, @embed})} end},
      {:ann2snn, "neuromorphic",
       fn -> Edifice.build(:ann2snn, input_size: @embed, output_size: @num_classes) end,
       fn -> %{"input" => rand({@batch, @embed})} end}
    ]
  end

  defp convolutional_specs do
    [
      {:resnet, "convolutional",
       fn ->
         Edifice.build(:resnet,
           input_shape: {nil, @image_size, @image_size, @in_channels},
           num_classes: @num_classes,
           block_sizes: [1, 1],
           initial_channels: 8
         )
       end, fn -> %{"input" => rand({@batch, @image_size, @image_size, @in_channels})} end},
      {:densenet, "convolutional",
       fn ->
         Edifice.build(:densenet,
           input_shape: {nil, 32, 32, @in_channels},
           num_classes: @num_classes,
           growth_rate: 8,
           block_config: [2, 2],
           initial_channels: 16
         )
       end, fn -> %{"input" => rand({@batch, 32, 32, @in_channels})} end},
      {:tcn, "convolutional",
       fn -> Edifice.build(:tcn, input_size: @embed, hidden_size: @hidden, num_layers: 2) end,
       fn -> %{"input" => rand({@batch, @seq_len, @embed})} end},
      {:mobilenet, "convolutional",
       fn ->
         Edifice.build(:mobilenet,
           input_dim: @embed,
           hidden_dim: @hidden,
           num_classes: @num_classes
         )
       end, fn -> %{"input" => rand({@batch, @embed})} end},
      {:efficientnet, "convolutional",
       fn ->
         Edifice.build(:efficientnet,
           input_dim: 64,
           base_dim: 16,
           width_multiplier: 1.0,
           depth_multiplier: 1.0,
           num_classes: @num_classes
         )
       end, fn -> %{"input" => rand({@batch, 64})} end}
    ]
  end

  defp generative_specs do
    [
      # VAE: tuple return (encoder, decoder)
      {:vae_encoder, "generative",
       fn ->
         {enc, _dec} = Edifice.build(:vae, input_size: @embed, latent_size: @latent_size)
         enc
       end, fn -> %{"input" => rand({@batch, @embed})} end},
      {:vae_decoder, "generative",
       fn ->
         {_enc, dec} = Edifice.build(:vae, input_size: @embed, latent_size: @latent_size)
         dec
       end, fn -> %{"latent" => rand({@batch, @latent_size})} end},
      # GAN
      {:gan_generator, "generative",
       fn ->
         {gen, _disc} = Edifice.build(:gan, output_size: @embed, latent_size: @latent_size)
         gen
       end, fn -> %{"noise" => rand({@batch, @latent_size})} end},
      {:gan_discriminator, "generative",
       fn ->
         {_gen, disc} = Edifice.build(:gan, output_size: @embed, latent_size: @latent_size)
         disc
       end, fn -> %{"data" => rand({@batch, @embed})} end},
      # VQ-VAE encoder only
      {:vq_vae_encoder, "generative",
       fn ->
         {enc, _dec} = Edifice.build(:vq_vae, input_size: @embed, embedding_dim: @latent_size)
         enc
       end, fn -> %{"input" => rand({@batch, @embed})} end},
      # Normalizing Flow
      {:normalizing_flow, "generative",
       fn -> Edifice.build(:normalizing_flow, input_size: @embed, num_flows: 2) end,
       fn -> %{"input" => rand({@batch, @embed})} end},
      # Diffusion-family
      {:diffusion, "generative",
       fn ->
         Edifice.build(:diffusion,
           obs_size: @embed,
           action_dim: @action_dim,
           action_horizon: @action_horizon,
           hidden_size: @hidden,
           num_layers: @num_layers,
           dropout: 0.0
         )
       end,
       fn ->
         %{
           "noisy_actions" => rand({@batch, @action_horizon, @action_dim}),
           "timestep" => rand({@batch}),
           "observations" => rand({@batch, @embed})
         }
       end},
      {:ddim, "generative",
       fn ->
         Edifice.build(:ddim,
           obs_size: @embed,
           action_dim: @action_dim,
           action_horizon: @action_horizon,
           hidden_size: @hidden,
           num_layers: @num_layers,
           dropout: 0.0
         )
       end,
       fn ->
         %{
           "noisy_actions" => rand({@batch, @action_horizon, @action_dim}),
           "timestep" => rand({@batch}),
           "observations" => rand({@batch, @embed})
         }
       end},
      {:flow_matching, "generative",
       fn ->
         Edifice.build(:flow_matching,
           obs_size: @embed,
           action_dim: @action_dim,
           action_horizon: @action_horizon,
           hidden_size: @hidden,
           num_layers: @num_layers,
           dropout: 0.0
         )
       end,
       fn ->
         %{
           "x_t" => rand({@batch, @action_horizon, @action_dim}),
           "timestep" => rand({@batch}),
           "observations" => rand({@batch, @embed})
         }
       end},
      {:dit, "generative",
       fn ->
         Edifice.build(:dit,
           input_dim: @embed,
           hidden_size: @hidden,
           depth: 1,
           num_heads: 2,
           dropout: 0.0
         )
       end,
       fn ->
         %{
           "noisy_input" => rand({@batch, @embed}),
           "timestep" => rand({@batch})
         }
       end},
      {:score_sde, "generative",
       fn ->
         Edifice.build(:score_sde,
           input_dim: @embed,
           hidden_size: @hidden,
           num_layers: @num_layers
         )
       end,
       fn ->
         %{
           "noisy_input" => rand({@batch, @embed}),
           "timestep" => rand({@batch})
         }
       end},
      {:consistency_model, "generative",
       fn ->
         Edifice.build(:consistency_model,
           input_dim: @embed,
           hidden_size: @hidden,
           num_layers: @num_layers
         )
       end,
       fn ->
         %{
           "noisy_input" => rand({@batch, @embed}),
           "sigma" => rand({@batch})
         }
       end},
      {:latent_diffusion_denoiser, "generative",
       fn ->
         {_enc, _dec, denoiser} =
           Edifice.build(:latent_diffusion,
             input_size: @embed,
             latent_size: @latent_size,
             hidden_size: @hidden,
             num_layers: @num_layers
           )

         denoiser
       end,
       fn ->
         %{
           "noisy_z" => rand({@batch, @latent_size}),
           "timestep" => rand({@batch})
         }
       end}
    ]
  end

  defp contrastive_specs do
    simple =
      for {arch, opts} <- [
            {:simclr, [encoder_dim: @embed, projection_dim: @hidden]},
            {:barlow_twins, [encoder_dim: @embed, projection_dim: @hidden]},
            {:vicreg, [encoder_dim: @embed, projection_dim: @hidden]}
          ] do
        {arch, "contrastive", fn -> Edifice.build(arch, opts) end,
         fn -> %{"features" => rand({@batch, @embed})} end}
      end

    simple ++
      [
        {:byol_online, "contrastive",
         fn ->
           {online, _target} =
             Edifice.build(:byol, encoder_dim: @embed, projection_dim: @hidden)

           online
         end, fn -> %{"features" => rand({@batch, @embed})} end},
        {:mae_encoder, "contrastive",
         fn ->
           {enc, _dec} =
             Edifice.build(:mae,
               input_dim: @embed,
               embed_dim: @hidden,
               num_patches: 4,
               depth: 1,
               num_heads: 2,
               decoder_depth: 1,
               decoder_num_heads: 2
             )

           enc
         end, fn -> %{"visible_patches" => rand({@batch, 4, @embed})} end}
      ]
  end

  # ── Runner ─────────────────────────────────────────────────────────

  def run do
    all_specs = specs()
    total = length(all_specs)

    IO.puts("=" |> String.duplicate(80))
    IO.puts("Edifice Full Sweep — #{total} architectures on EXLA")
    IO.puts("batch=#{@batch}, embed=#{@embed}, hidden=#{@hidden}, seq_len=#{@seq_len}")
    IO.puts("=" |> String.duplicate(80))
    IO.puts("")

    header =
      "  #{String.pad_trailing("Architecture", 28)}" <>
        "#{String.pad_trailing("Family", 15)}" <>
        "#{String.pad_trailing("Build", 10)}" <>
        "#{String.pad_trailing("Compile", 10)}" <>
        "#{String.pad_trailing("Inference", 12)}" <>
        "Status"

    IO.puts(header)
    IO.puts("  " <> String.duplicate("-", 78))

    results =
      for {name, family, build_fn, input_fn} <- all_specs do
        profile_one(name, family, build_fn, input_fn)
      end

    IO.puts("")
    IO.puts("  " <> String.duplicate("-", 78))

    # Summary
    {ok, failed} = Enum.split_with(results, fn r -> r.status == :ok end)

    IO.puts("  #{length(ok)}/#{total} succeeded, #{length(failed)} failed")
    IO.puts("")

    if failed != [] do
      IO.puts("  FAILURES:")

      for r <- failed do
        IO.puts("    #{r.name}: #{r.error}")
      end

      IO.puts("")
    end

    # Flag outliers (>5x family median)
    family_groups =
      ok
      |> Enum.group_by(& &1.family)
      |> Map.new(fn {fam, entries} ->
        times = Enum.map(entries, & &1.inference_ms) |> Enum.sort()
        median = Enum.at(times, div(length(times), 2))
        {fam, median}
      end)

    outliers =
      Enum.filter(ok, fn r ->
        median = Map.get(family_groups, r.family, r.inference_ms)
        median > 0 and r.inference_ms > median * 5
      end)

    if outliers != [] do
      IO.puts("  OUTLIERS (>5x family median):")

      for r <- outliers do
        median = Map.get(family_groups, r.family)

        IO.puts(
          "    #{r.name} (#{r.family}): #{fmt(r.inference_ms)} " <>
            "(#{Float.round(r.inference_ms / median, 1)}x family median of #{fmt(median)})"
        )
      end

      IO.puts("")
    end

    # Family summary
    IO.puts("  FAMILY MEDIANS:")

    for {fam, median} <- Enum.sort_by(family_groups, fn {_f, m} -> m end, :desc) do
      IO.puts("    #{String.pad_trailing(fam, 18)} #{fmt(median)}")
    end
  end

  defp profile_one(name, family, build_fn, input_fn) do
    try do
      # Build
      {build_us, model} = :timer.tc(fn -> build_fn.() end)
      build_ms = build_us / 1_000

      # Generate input
      input = input_fn.()

      template =
        case input do
          %{} = map when not is_struct(map) ->
            Map.new(map, fn {k, v} -> {k, Nx.template(Nx.shape(v), Nx.type(v))} end)

          tensor ->
            Nx.template(Nx.shape(tensor), Nx.type(tensor))
        end

      # Compile (includes init)
      {compile_us, {predict_fn, params}} =
        :timer.tc(fn ->
          {init_fn, predict_fn} = Axon.build(model)
          params = init_fn.(template, Axon.ModelState.empty())
          {predict_fn, params}
        end)

      compile_ms = compile_us / 1_000

      # Warm up
      for _ <- 1..@warmup_iters, do: predict_fn.(params, input)

      # Time inference
      {total_us, _} =
        :timer.tc(fn ->
          for _ <- 1..@timing_iters, do: predict_fn.(params, input)
        end)

      inference_ms = total_us / @timing_iters / 1_000

      IO.puts(
        "  #{String.pad_trailing(to_string(name), 28)}" <>
          "#{String.pad_trailing(family, 15)}" <>
          "#{String.pad_trailing(fmt(build_ms), 10)}" <>
          "#{String.pad_trailing(fmt(compile_ms), 10)}" <>
          "#{String.pad_trailing(fmt(inference_ms), 12)}" <>
          "ok"
      )

      %{
        name: name,
        family: family,
        build_ms: build_ms,
        compile_ms: compile_ms,
        inference_ms: inference_ms,
        status: :ok
      }
    rescue
      e ->
        msg = Exception.message(e) |> String.slice(0, 60)

        IO.puts(
          "  #{String.pad_trailing(to_string(name), 28)}" <>
            "#{String.pad_trailing(family, 15)}" <>
            "#{String.pad_trailing("-", 10)}" <>
            "#{String.pad_trailing("-", 10)}" <>
            "#{String.pad_trailing("-", 12)}" <>
            "FAIL"
        )

        %{name: name, family: family, status: :fail, error: msg}
    end
  end

  defp fmt(ms) when ms < 1, do: "#{Float.round(ms * 1000, 0)} us"
  defp fmt(ms) when ms < 100, do: "#{Float.round(ms, 1)} ms"
  defp fmt(ms), do: "#{Float.round(ms, 0)} ms"
end

FullSweep.run()
