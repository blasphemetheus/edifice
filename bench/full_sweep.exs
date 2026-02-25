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
  @vocab_size 64
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
      contrastive_specs() ++
      interpretability_specs() ++
      multimodal_specs() ++
      world_model_specs() ++
      rl_specs() ++
      scientific_specs() ++
      inference_specs() ++
      robotics_specs() ++
      audio_specs()
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
    s4 s4d s5 h3 hyena bimamba gated_ssm jamba zamba striped_hyena mamba3
    gss hyena_v2 hymba ss_transformer
    lstm gru xlstm mlstm min_gru min_lstm delta_net gated_delta_net ttt ttt_e2e titans
    slstm xlstm_v2 native_recurrence
    retnet gla hgrn griffin gqa fnet linear_transformer nystromformer performer
    based mega mla diff_transformer hawk retnet_v2 megalodon
    gla_v2 hgrn_v2 kda gated_attention
    rnope_swa ssmax softpick
    kan liquid
  )a

  defp sequence_specs do
    for arch <- @sequence_archs do
      family =
        cond do
          arch in ~w(mamba mamba_ssd mamba_cumsum mamba_hillis_steele s4 s4d s5 h3 hyena bimamba gated_ssm jamba zamba striped_hyena mamba3 gss hyena_v2 hymba ss_transformer)a ->
            "ssm"

          arch in ~w(lstm gru xlstm mlstm min_gru min_lstm delta_net gated_delta_net ttt ttt_e2e titans slstm xlstm_v2 native_recurrence)a ->
            "recurrent"

          arch in ~w(retnet gla hgrn griffin gqa fnet linear_transformer nystromformer performer based mega mla diff_transformer hawk retnet_v2 megalodon gla_v2 hgrn_v2 kda gated_attention rnope_swa ssmax softpick)a ->
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
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # v0.2.0 architectures with non-standard options
      {:decoder_only, "transformer",
       fn ->
         Edifice.build(:decoder_only,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_kv_heads: 1,
           num_layers: @num_layers,
           seq_len: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      {:conformer, "attention",
       fn ->
         Edifice.build(:conformer,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           conv_kernel_size: 3,
           seq_len: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      {:infini_attention, "attention",
       fn ->
         Edifice.build(:infini_attention,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           segment_size: 4,
           seq_len: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      {:ring_attention, "attention",
       fn ->
         Edifice.build(:ring_attention,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_chunks: 2,
           num_layers: @num_layers,
           window_size: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # lightning_attention: needs num_blocks > 1, so seq_len > block_size
      {:lightning_attention, "attention",
       fn ->
         Edifice.build(:lightning_attention,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           block_size: div(@seq_len, 2),
           seq_len: @seq_len,
           window_size: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # flash_linear_attention: needs num_chunks > 1, so seq_len > chunk_size
      {:flash_linear_attention, "attention",
       fn ->
         Edifice.build(:flash_linear_attention,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           chunk_size: div(@seq_len, 2),
           seq_len: @seq_len,
           window_size: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # nsa: needs block_size and num_selected_blocks
      {:nsa, "attention",
       fn ->
         Edifice.build(:nsa,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           head_dim: 8,
           num_layers: @num_layers,
           block_size: 4,
           num_selected_blocks: 2,
           compression_ratio: 2,
           seq_len: @seq_len,
           window_size: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # dual_chunk_attention: seq_len must be divisible by chunk_size
      {:dual_chunk_attention, "attention",
       fn ->
         Edifice.build(:dual_chunk_attention,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           chunk_size: @seq_len,
           seq_len: @seq_len,
           window_size: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # perceiver: needs input_dim
      {:perceiver, "attention",
       fn ->
         Edifice.build(:perceiver,
           input_dim: @embed,
           latent_dim: @hidden,
           num_latents: 4,
           num_layers: @num_layers,
           num_heads: 2,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      {:kat, "feedforward",
       fn ->
         Edifice.build(:kat,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           seq_len: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      {:bitnet, "feedforward",
       fn ->
         Edifice.build(:bitnet,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           seq_len: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      {:mixture_of_depths, "meta",
       fn ->
         Edifice.build(:mixture_of_depths,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           capacity_ratio: 0.5,
           seq_len: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      {:mixture_of_agents, "meta",
       fn ->
         Edifice.build(:mixture_of_agents,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_proposers: 2,
           num_layers: @num_layers,
           seq_len: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # yarn: position encoding model
      {:yarn, "attention",
       fn ->
         Edifice.build(:yarn,
           embed_dim: @embed,
           scale: 2,
           original_max_position: 16
         )
       end, fn -> %{"yarn_input" => rand({@batch, @seq_len, @embed})} end},
      # tmrope: multimodal position encoding
      {:tmrope, "attention",
       fn ->
         Edifice.build(:tmrope,
           embed_dim: @embed,
           modalities: [:text, :image],
           max_position: 32
         )
       end,
       fn ->
         %{
           "tmrope_query" => rand({@batch, @seq_len, @embed}),
           "tmrope_key" => rand({@batch, @seq_len, @embed}),
           "tmrope_positions" => rand({@batch, @seq_len})
         }
       end},
      # hybrid_builder: composite architecture
      {:hybrid_builder, "meta",
       fn ->
         Edifice.build(:hybrid_builder,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_layers: @num_layers,
           state_size: @state_size,
           num_heads: 2,
           head_dim: 8,
           seq_len: @seq_len,
           window_size: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # mixture_of_tokenizers: sequence model with multiple tokenizer embeddings
      {:mixture_of_tokenizers, "meta",
       fn ->
         Edifice.build(:mixture_of_tokenizers,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_layers: @num_layers,
           num_heads: 2,
           num_tokenizers: 2,
           tokenizer_vocab_sizes: [32, 32],
           tokenizer_embed_dims: [@embed, @embed],
           seq_len: @seq_len,
           window_size: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # qat: quantization-aware training model
      {:qat, "meta",
       fn ->
         Edifice.build(:qat,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           seq_len: @seq_len,
           window_size: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # distillation_head: maps student dim to teacher dim
      {:distillation_head, "meta",
       fn ->
         Edifice.build(:distillation_head,
           embed_dim: @embed,
           teacher_dim: @hidden,
           hidden_size: @hidden,
           num_layers: @num_layers,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # test_time_compute: returns container {backbone, scores}
      {:test_time_compute, "meta",
       fn ->
         Edifice.build(:test_time_compute,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_layers: @num_layers,
           num_heads: 2,
           scorer_hidden: @hidden,
           seq_len: @seq_len,
           window_size: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # nemotron_h: hybrid transformer (option is hidden_dim, not hidden_size)
      {:nemotron_h, "transformer",
       fn ->
         Edifice.build(:nemotron_h,
           embed_dim: @embed,
           hidden_dim: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           seq_len: @seq_len,
           window_size: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # multi_token_prediction: returns container {pred_1, ..., pred_N}
      {:multi_token_prediction, "transformer",
       fn ->
         Edifice.build(:multi_token_prediction,
           embed_dim: @embed,
           vocab_size: @vocab_size,
           hidden_size: @hidden,
           num_layers: @num_layers,
           num_heads: 2,
           num_predictions: 2,
           seq_len: @seq_len,
           window_size: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # byte_latent_transformer: returns {encoder, latent_transformer, decoder}
      {:byte_latent_encoder, "transformer",
       fn ->
         {enc, _lat, _dec} =
           Edifice.build(:byte_latent_transformer,
             embed_dim: @embed,
             hidden_size: @hidden,
             num_layers: @num_layers,
             num_heads: 2,
             max_byte_len: @seq_len,
             patch_size: 4,
             dropout: 0.0
           )

         enc
       end, fn -> %{"byte_ids" => Nx.iota({@batch, @seq_len}, axis: 1) |> Nx.remainder(256)} end},
      # speculative_head: returns container {pred_1, ..., pred_N}
      {:speculative_head, "meta",
       fn ->
         Edifice.build(:speculative_head,
           embed_dim: @embed,
           vocab_size: @vocab_size,
           hidden_size: @hidden,
           num_layers: @num_layers,
           num_heads: 2,
           num_predictions: 2,
           head_hidden: @hidden,
           seq_len: @seq_len,
           window_size: @seq_len,
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
         end, fn -> %{"image" => rand({@batch, @in_channels, @image_size, @image_size})} end},
        {:focalnet, "vision",
         fn ->
           Edifice.build(:focalnet,
             image_size: @image_size,
             in_channels: @in_channels,
             patch_size: 4,
             hidden_size: @hidden,
             num_layers: 2,
             focal_levels: 2,
             focal_kernel: 3
           )
         end, fn -> %{"image" => rand({@batch, @in_channels, @image_size, @image_size})} end},
        {:poolformer, "vision",
         fn ->
           Edifice.build(:poolformer,
             image_size: @image_size,
             in_channels: @in_channels,
             patch_size: 4,
             embed_dim: @hidden,
             depths: [1, 1],
             dropout: 0.0
           )
         end, fn -> %{"image" => rand({@batch, @in_channels, @image_size, @image_size})} end},
        {:nerf, "vision",
         fn -> Edifice.build(:nerf, coord_dim: 3, hidden_size: @hidden, use_viewdir: false) end,
         fn -> %{"coordinates" => rand({@batch, 3})} end},
        # gaussian_splat: render pipeline uses vectorized defn while loop for compositing
        {:gaussian_splat, "vision",
         fn ->
           Edifice.build(:gaussian_splat,
             num_gaussians: 20,
             sh_degree: 1,
             image_size: 16
           )
         end,
         fn ->
           %{
             "camera_position" => rand({@batch, 3}),
             "view_matrix" => Nx.eye(4) |> Nx.broadcast({@batch, 4, 4}),
             "proj_matrix" => Nx.eye(4) |> Nx.broadcast({@batch, 4, 4}),
             "image_height" => Nx.broadcast(16.0, {@batch}),
             "image_width" => Nx.broadcast(16.0, {@batch})
           }
         end},
        # Mamba Vision: hybrid ViT + Mamba (4 stages: dim, 2*dim, 4*dim, 8*dim)
        # mamba_vision needs image_size >= 64 so stage 4 has >=2x2 spatial tokens
        # (stem 4x + 3 downsamples 8x = 32x total, so 64/32 = 2x2 at stage 4)
        {:mamba_vision, "vision",
         fn ->
           Edifice.build(:mamba_vision,
             image_size: 64,
             in_channels: @in_channels,
             dim: 16,
             depths: [1, 1, 1, 1],
             num_heads: [1, 2, 4, 8],
             d_state: 4,
             d_conv: 3,
             dropout: 0.0
           )
         end, fn -> %{"image" => rand({@batch, @in_channels, 64, 64})} end},
        # DINOv2: self-distillation (returns tuple)
        # dino_v2 builds both student+teacher; use small embed_dim to avoid OOM on 4GB GPU
        {:dino_v2_student, "vision",
         fn ->
           {student, _teacher} =
             Edifice.build(:dino_v2,
               image_size: @image_size,
               in_channels: @in_channels,
               patch_size: 4,
               embed_dim: 8,
               depth: 1,
               num_heads: 2,
               dropout: 0.0
             )

           student
         end, fn -> %{"image" => rand({@batch, @in_channels, @image_size, @image_size})} end},
        # MetaFormer: pluggable token mixer
        {:metaformer, "vision",
         fn ->
           Edifice.build(:metaformer,
             image_size: @image_size,
             in_channels: @in_channels,
             patch_size: 4,
             embed_dim: @hidden,
             depths: [1, 1],
             num_heads: [2, 2],
             dropout: 0.0
           )
         end, fn -> %{"image" => rand({@batch, @in_channels, @image_size, @image_size})} end},
        # CAFormer: conv + attention stages
        {:caformer, "vision",
         fn ->
           Edifice.build(:caformer,
             image_size: @image_size,
             in_channels: @in_channels,
             patch_size: 4,
             embed_dim: @hidden,
             depths: [1, 1],
             num_heads: [2, 2],
             dropout: 0.0
           )
         end, fn -> %{"image" => rand({@batch, @in_channels, @image_size, @image_size})} end},
        # EfficientViT: linear attention with cascaded groups
        {:efficient_vit, "vision",
         fn ->
           Edifice.build(:efficient_vit,
             image_size: 32,
             in_channels: @in_channels,
             patch_size: 8,
             embed_dim: @hidden,
             depths: [1, 1],
             num_heads: [2, 2]
           )
         end, fn -> %{"image" => rand({@batch, @in_channels, 32, 32})} end}
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
         end, graph_input},
        {:gin_v2, "graph",
         fn ->
           Edifice.build(:gin_v2,
             input_dim: @node_dim,
             edge_dim: 4,
             hidden_dims: [@hidden],
             num_classes: @num_classes
           )
         end,
         fn ->
           nodes = rand({@batch, @num_nodes, @node_dim})
           adj = Nx.eye(@num_nodes) |> Nx.broadcast({@batch, @num_nodes, @num_nodes})
           edges = rand({@batch, @num_nodes, @num_nodes, 4})
           %{"nodes" => nodes, "adjacency" => adj, "edge_features" => edges}
         end},
        # EGNN: equivariant graph neural network (returns container)
        {:egnn, "graph",
         fn ->
           Edifice.build(:egnn,
             in_node_features: @node_dim,
             hidden_dim: @hidden,
             num_layers: @num_layers,
             coord_dim: 3
           )
         end,
         fn ->
           %{
             "nodes" => rand({@batch, @num_nodes, @node_dim}),
             "coords" => rand({@batch, @num_nodes, 3}),
             "edge_index" => Nx.tensor([[[0, 1], [1, 2], [2, 0], [0, 0], [1, 1], [2, 2]]]) |> Nx.broadcast({@batch, @num_nodes, 2})
           }
         end}
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
       end},
      # Engram: hash-based memory lookup
      {:engram, "memory",
       fn ->
         Edifice.build(:engram,
           key_dim: @embed,
           value_dim: @hidden,
           num_buckets: @num_memories,
           num_tables: 2
         )
       end,
       fn ->
         %{
           "query" => rand({@batch, @embed}),
           "memory_slots" => rand({2, @num_memories, @hidden})
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
      {:moe_v2, "meta",
       fn ->
         Edifice.build(:moe_v2,
           input_size: @embed,
           hidden_size: @hidden * 4,
           output_size: @hidden,
           num_shared_experts: 1,
           num_routed_experts: 2,
           tokens_per_expert: 2,
           dropout: 0.0
         )
       end, fn -> %{"moe_input" => rand({@batch, @seq_len, @embed})} end},
      # lora: standalone build only outputs the B*A*x delta; B is initialized to zeros
      # (by design — meant to be used with wrap/3). Use wrap to test base + delta.
      {:lora, "meta",
       fn ->
         input = Axon.input("input", shape: {nil, @embed})
         base = Axon.dense(input, @hidden, name: "lora_base")
         Edifice.Meta.LoRA.wrap(input, base, rank: 4, alpha: 1.0, name: "lora")
       end,
       fn -> %{"input" => rand({@batch, @embed})} end},
      {:dora, "meta",
       fn -> Edifice.build(:dora, input_size: @embed, output_size: @hidden, rank: 4) end,
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
      # capsule: larger cap dims improve squash function numerical stability
      {:capsule, "meta",
       fn ->
         Edifice.build(:capsule,
           input_shape: {nil, 28, 28, 1},
           conv_channels: 32,
           conv_kernel: 9,
           num_primary_caps: 8,
           primary_cap_dim: 8,
           num_digit_caps: @num_classes,
           digit_cap_dim: 8
         )
       end, fn -> %{"input" => rand({@batch, 28, 28, 1})} end},
      {:rlhf_head, "meta",
       fn ->
         Edifice.build(:rlhf_head,
           input_size: @embed,
           hidden_size: @hidden,
           variant: :reward
         )
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
      # DPO/KTO/GRPO: RL-alignment models (language model backbone)
      {:dpo, "meta",
       fn ->
         Edifice.build(:dpo,
           hidden_size: @hidden,
           num_layers: @num_layers,
           num_heads: 2,
           vocab_size: @vocab_size,
           max_seq_len: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"tokens" => Nx.iota({@batch, @seq_len}, axis: 1) |> Nx.remainder(@vocab_size)} end},
      {:kto, "meta",
       fn ->
         Edifice.build(:kto,
           hidden_size: @hidden,
           num_layers: @num_layers,
           num_heads: 2,
           vocab_size: @vocab_size,
           max_seq_len: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"tokens" => Nx.iota({@batch, @seq_len}, axis: 1) |> Nx.remainder(@vocab_size)} end},
      {:grpo, "meta",
       fn ->
         Edifice.build(:grpo,
           hidden_size: @hidden,
           num_layers: @num_layers,
           num_heads: 2,
           vocab_size: @vocab_size,
           max_seq_len: @seq_len,
           dropout: 0.0
         )
       end, fn -> %{"tokens" => Nx.iota({@batch, @seq_len}, axis: 1) |> Nx.remainder(@vocab_size)} end},
      # Speculative decoding: returns {draft, verifier} tuple
      {:speculative_decoding_draft, "meta",
       fn ->
         {draft, _verifier} =
           Edifice.build(:speculative_decoding,
             embed_dim: @embed,
             hidden_size: @hidden,
             num_layers: @num_layers,
             num_heads: 2,
             seq_len: @seq_len,
             window_size: @seq_len,
             dropout: 0.0
           )

         draft
       end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end}
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
      {:dit_v2, "generative",
       fn ->
         Edifice.build(:dit_v2,
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
      # consistency_model: sigma must be in [sigma_min, sigma_max] range (not random normal)
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
           "sigma" => Nx.add(Nx.multiply(Nx.abs(rand({@batch})), 10.0), 0.01)
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
       end},
      # MMDiT: multi-modal diffusion transformer
      {:mmdit, "generative",
       fn ->
         Edifice.build(:mmdit,
           img_dim: @embed,
           txt_dim: @embed,
           hidden_size: @hidden,
           depth: 1,
           num_heads: 2,
           img_tokens: 4,
           txt_tokens: 4,
           cond_dim: @hidden
         )
       end,
       fn ->
         %{
           "img_latent" => rand({@batch, 4, @embed}),
           "txt_embed" => rand({@batch, 4, @embed}),
           "timestep" => rand({@batch}),
           "pooled_text" => rand({@batch, @hidden})
         }
       end},
      # SoFlow: second-order flow matching
      {:soflow, "generative",
       fn ->
         Edifice.build(:soflow,
           obs_size: @embed,
           action_dim: @action_dim,
           action_horizon: @action_horizon,
           hidden_size: @hidden,
           num_layers: @num_layers
         )
       end,
       fn ->
         %{
           "x_t" => rand({@batch, @action_horizon, @action_dim}),
           "current_time" => rand({@batch}),
           "target_time" => rand({@batch}),
           "observations" => rand({@batch, @embed})
         }
       end},
      # VAR: visual autoregressive model (returns container per scale)
      # scales [1, 2] -> total_tokens = 1*1 + 2*2 = 5
      {:var, "generative",
       fn ->
         Edifice.build(:var,
           hidden_size: @hidden,
           num_layers: @num_layers,
           num_heads: 2,
           scales: [1, 2],
           codebook_size: @vocab_size,
           dropout: 0.0
         )
       end,
       fn ->
         %{"scale_embeddings" => rand({@batch, 5, @hidden})}
       end},
      # Linear DiT: efficient linear-attention DiT
      {:linear_dit, "generative",
       fn ->
         Edifice.build(:linear_dit,
           input_dim: @embed,
           hidden_size: @hidden,
           num_layers: @num_layers,
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
      # SiT: Scalable Interpolant Transformer
      {:sit, "generative",
       fn ->
         Edifice.build(:sit,
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
      # Transfusion: joint text + image generation (returns container)
      {:transfusion, "generative",
       fn ->
         Edifice.build(:transfusion,
           embed_dim: @embed,
           hidden_size: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           vocab_size: @vocab_size,
           patch_dim: @embed,
           dropout: 0.0
         )
       end,
       fn ->
         %{
           "sequence" => rand({@batch, @seq_len, @embed}),
           "modality_mask" => Nx.broadcast(1.0, {@batch, @seq_len}),
           "timestep" => rand({@batch})
         }
       end},
      # MAR: masked autoregressive model
      {:mar, "generative",
       fn ->
         Edifice.build(:mar,
           vocab_size: @vocab_size,
           embed_dim: @hidden,
           num_layers: @num_layers,
           num_heads: 2,
           seq_len: @seq_len,
           dropout: 0.0
         )
       end,
       fn ->
         %{"tokens" => Nx.iota({@batch, @seq_len}, axis: 1) |> Nx.remainder(@vocab_size)}
       end},
      # CogVideoX: video generation transformer (patch_size must be [t, h, w] list)
      {:cogvideox, "generative",
       fn ->
         Edifice.build(:cogvideox,
           patch_size: [1, 2, 2],
           hidden_size: @hidden,
           num_heads: 2,
           num_layers: @num_layers,
           num_frames: 2,
           text_hidden_size: @hidden
         )
       end,
       fn ->
         %{
           "video_latent" => rand({@batch, 2, 4, 4, @hidden}),
           "text_embed" => rand({@batch, @seq_len, @hidden}),
           "timestep" => rand({@batch})
         }
       end},
      # TRELLIS: 3D structured latent generation (conditioning must be 3D)
      {:trellis, "generative",
       fn ->
         Edifice.build(:trellis,
           voxel_resolution: 4,
           feature_dim: @hidden,
           hidden_size: @hidden,
           num_layers: @num_layers,
           num_heads: 2,
           window_size: 4,
           condition_dim: @hidden,
           max_voxels: 16
         )
       end,
       fn ->
         %{
           "sparse_features" => rand({@batch, 16, @hidden}),
           "voxel_positions" => rand({@batch, 16, 3}),
           "occupancy_mask" => Nx.broadcast(1.0, {@batch, 16}),
           "conditioning" => rand({@batch, 1, @hidden}),
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
         end, fn -> %{"visible_patches" => rand({@batch, 4, @embed})} end},
        {:jepa_encoder, "contrastive",
         fn ->
           {enc, _pred} =
             Edifice.build(:jepa,
               input_dim: @embed,
               embed_dim: @hidden
             )

           enc
         end, fn -> %{"features" => rand({@batch, @embed})} end},
        # Temporal JEPA: returns {context_encoder, predictor}
        # Context encoder uses ModelBuilder (expects "state_sequence" input)
        {:temporal_jepa_encoder, "contrastive",
         fn ->
           {enc, _pred} =
             Edifice.build(:temporal_jepa,
               input_dim: @embed,
               embed_dim: @hidden,
               predictor_embed_dim: @hidden,
               encoder_depth: 1,
               predictor_depth: 1,
               num_heads: 2,
               seq_len: @seq_len,
               dropout: 0.0
             )

           enc
         end, fn -> %{"state_sequence" => rand({@batch, @seq_len, @embed})} end},
        # SigLIP: returns {encoder, temperature_param}
        {:siglip, "contrastive",
         fn ->
           {enc, _temp} =
             Edifice.build(:siglip,
               input_dim: @embed,
               projection_dim: @hidden
             )

           enc
         end, fn -> %{"features" => rand({@batch, @embed})} end}
      ]
  end

  defp interpretability_specs do
    [
      {:sparse_autoencoder, "interpretability",
       fn ->
         Edifice.build(:sparse_autoencoder,
           input_size: @embed,
           dict_size: @embed * 4
         )
       end, fn -> %{"sae_input" => rand({@batch, @embed})} end},
      {:transcoder, "interpretability",
       fn ->
         Edifice.build(:transcoder,
           input_size: @embed,
           output_size: @hidden,
           dict_size: @embed * 4
         )
       end, fn -> %{"transcoder_input" => rand({@batch, @embed})} end}
    ]
  end

  defp multimodal_specs do
    [
      {:multimodal_mlp_fusion, "multimodal",
       fn ->
         Edifice.build(:multimodal_mlp_fusion,
           vision_dim: @embed,
           llm_dim: @hidden,
           num_visual_tokens: 4,
           text_seq_len: @seq_len
         )
       end,
       fn ->
         %{
           "visual_tokens" => rand({@batch, 4, @embed}),
           "text_embeddings" => rand({@batch, @seq_len, @hidden})
         }
       end}
    ]
  end

  defp world_model_specs do
    [
      # World model: returns {encoder, dynamics, reward_head}
      {:world_model_encoder, "world_model",
       fn ->
         {enc, _dyn, _rew} =
           Edifice.build(:world_model,
             obs_size: @embed,
             action_size: @action_dim,
             latent_size: @latent_size,
             hidden_size: @hidden
           )

         enc
       end, fn -> %{"observation" => rand({@batch, @embed})} end}
    ]
  end

  defp rl_specs do
    [
      # PolicyValue: returns container {policy, value}
      {:policy_value, "rl",
       fn ->
         Edifice.build(:policy_value,
           input_size: @embed,
           action_size: @action_dim,
           hidden_size: @hidden
         )
       end, fn -> %{"observation" => rand({@batch, @embed})} end}
    ]
  end

  defp scientific_specs do
    [
      {:fno, "scientific",
       fn ->
         Edifice.build(:fno,
           in_channels: @in_channels,
           out_channels: 1,
           modes: 4,
           hidden_channels: @hidden,
           num_layers: @num_layers
         )
       end, fn -> %{"input" => rand({@batch, @seq_len, @in_channels})} end}
    ]
  end

  defp inference_specs do
    [
      # Medusa: returns container {head_1, ..., head_K}
      {:medusa, "inference",
       fn ->
         Edifice.build(:medusa,
           base_hidden_dim: @embed,
           vocab_size: @vocab_size,
           num_medusa_heads: 2,
           medusa_num_layers: 1
         )
       end, fn -> %{"hidden_states" => rand({@batch, @embed})} end}
    ]
  end

  defp robotics_specs do
    [
      # OpenVLA: vision-language-action model (patch_size must divide image_size,
      # num_kv_heads must be <= num_heads for GQA)
      {:openvla, "robotics",
       fn ->
         Edifice.build(:openvla,
           image_size: 28,
           in_channels: @in_channels,
           patch_size: 14,
           hidden_dim: @hidden,
           num_heads: 2,
           num_kv_heads: 1,
           num_layers: @num_layers,
           max_text_len: @seq_len,
           action_dim: @action_dim,
           dropout: 0.0
         )
       end,
       fn ->
         %{
           "image" => rand({@batch, @in_channels, 28, 28}),
           "text_tokens" => rand({@batch, @seq_len, @hidden})
         }
       end},
      # ACT: returns {encoder, decoder}
      {:act_decoder, "robotics",
       fn ->
         {_enc, dec} =
           Edifice.build(:act,
             obs_dim: @embed,
             action_dim: @action_dim,
             chunk_size: @action_horizon,
             hidden_dim: @hidden,
             num_heads: 2,
             num_layers: @num_layers,
             latent_dim: @latent_size,
             dropout: 0.0
           )

         dec
       end,
       fn ->
         %{
           "obs" => rand({@batch, @embed}),
           "z" => rand({@batch, @latent_size})
         }
       end}
    ]
  end

  defp audio_specs do
    [
      # EnCodec: encoder uses channels:first 1D conv (320x downsample)
      {:encodec_encoder, "audio",
       fn ->
         {enc, _dec} = Edifice.build(:encodec, hidden_dim: @hidden)
         enc
       end,
       fn ->
         # samples must be divisible by 320 (total downsampling factor)
         %{"waveform" => rand({@batch, 1, 320})}
       end},
      # VALLE: returns {ar_model, nar_model}
      {:valle_ar, "audio",
       fn ->
         {ar, _nar} =
           Edifice.build(:valle,
             vocab_size: @vocab_size,
             num_codebooks: 2,
             hidden_dim: @hidden,
             num_heads: 2,
             num_layers: @num_layers,
             dropout: 0.0
           )

         ar
       end,
       fn ->
         %{
           "text_tokens" => Nx.iota({@batch, @seq_len}, axis: 1) |> Nx.remainder(@vocab_size),
           "prompt_tokens" => Nx.iota({@batch, 2, 4}, axis: 2) |> Nx.remainder(@vocab_size),
           "audio_tokens" => Nx.iota({@batch, @seq_len}, axis: 1) |> Nx.remainder(@vocab_size)
         }
       end},
      {:soundstorm, "audio",
       fn ->
         Edifice.build(:soundstorm,
           num_codebooks: 2,
           codebook_size: @vocab_size,
           hidden_dim: @hidden,
           num_layers: @num_layers,
           num_heads: 2,
           conv_kernel_size: 3,
           dropout: 0.0
         )
       end, fn -> %{"tokens" => Nx.iota({@batch, @seq_len}, axis: 1) |> Nx.remainder(@vocab_size)} end}
    ]
  end

  # ── Runner ─────────────────────────────────────────────────────────

  def run do
    all_specs = specs()

    # Filter by SWEEP_ONLY env var: comma-separated architecture names
    # Usage: SWEEP_ONLY=var,cogvideox,trellis mix run bench/full_sweep.exs
    all_specs =
      case System.get_env("SWEEP_ONLY") do
        nil ->
          all_specs

        "" ->
          all_specs

        filter ->
          names =
            filter
            |> String.split(",")
            |> Enum.map(&String.trim/1)
            |> Enum.map(&String.to_atom/1)
            |> MapSet.new()

          Enum.filter(all_specs, fn {name, _family, _build, _input} -> name in names end)
      end

    total = length(all_specs)

    IO.puts("=" |> String.duplicate(80))
    IO.puts("Edifice Full Sweep — #{total} architectures on EXLA")
    IO.puts("batch=#{@batch}, embed=#{@embed}, hidden=#{@hidden}, seq_len=#{@seq_len}")
    IO.puts("=" |> String.duplicate(80))
    IO.puts("")

    # Phase 0: GPU/EXLA runtime warmup — build, compile, and run a small model
    # to absorb one-time costs (cuDNN init, BFC allocator, EXLA JIT cache warmup).
    # This ensures the first real architecture gets a fair measurement.
    IO.puts("## Phase 0: GPU Runtime Warmup")
    IO.puts("-" |> String.duplicate(60))

    {warmup_us, _} =
      :timer.tc(fn ->
        model = Edifice.build(:gated_ssm, @sequence_opts)
        {init_fn, predict_fn} = Axon.build(model)
        template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed}, :f32)}
        params = init_fn.(template, Axon.ModelState.empty())
        input = %{"state_sequence" => rand({@batch, @seq_len, @embed})}
        for _ <- 1..@warmup_iters, do: predict_fn.(params, input)
      end)

    IO.puts("  Warmed up EXLA runtime in #{Float.round(warmup_us / 1_000, 0)} ms (discarded)")
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
        result = profile_one(name, family, build_fn, input_fn)
        # Free GPU memory between architectures to avoid OOM on small GPUs.
        # GC all processes so EXLA buffer finalizers run and release GPU memory.
        for pid <- Process.list(), do: :erlang.garbage_collect(pid)
        result
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

    # Health warnings
    unhealthy = Enum.filter(ok, fn r -> health_status_tag(r.health) != "ok" end)

    if unhealthy != [] do
      IO.puts("  HEALTH WARNINGS:")

      for r <- unhealthy do
        IO.puts("    #{r.name}: #{health_status_tag(r.health)}")
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
      {total_us, output} =
        :timer.tc(fn ->
          for _ <- 1..(@timing_iters - 1), do: predict_fn.(params, input)
          predict_fn.(params, input)
        end)

      inference_ms = total_us / @timing_iters / 1_000

      # Health checks on output (negligible cost — a few reductions)
      health = check_output_health(output)
      health_tag = health_status_tag(health)

      IO.puts(
        "  #{String.pad_trailing(to_string(name), 28)}" <>
          "#{String.pad_trailing(family, 15)}" <>
          "#{String.pad_trailing(fmt(build_ms), 10)}" <>
          "#{String.pad_trailing(fmt(compile_ms), 10)}" <>
          "#{String.pad_trailing(fmt(inference_ms), 12)}" <>
          health_tag
      )

      %{
        name: name,
        family: family,
        build_ms: build_ms,
        compile_ms: compile_ms,
        inference_ms: inference_ms,
        health: health,
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

  # Flatten any output structure (container tuples/maps, plain tensors) to a list of tensors
  defp collect_tensors(%Nx.Tensor{} = t), do: [t]

  defp collect_tensors(tuple) when is_tuple(tuple) do
    tuple |> Tuple.to_list() |> Enum.flat_map(&collect_tensors/1)
  end

  defp collect_tensors(%{} = map) when not is_struct(map) do
    map |> Map.values() |> Enum.flat_map(&collect_tensors/1)
  end

  defp collect_tensors(_other), do: []

  defp check_output_health(output) do
    tensors = collect_tensors(output)

    if tensors == [] do
      %{has_nan: false, has_inf: false, all_zero: true, low_variance: true}
    else
      has_nan = Enum.any?(tensors, fn t -> Nx.any(Nx.is_nan(t)) |> Nx.to_number() == 1 end)
      has_inf = Enum.any?(tensors, fn t -> Nx.any(Nx.is_infinity(t)) |> Nx.to_number() == 1 end)

      all_zero =
        Enum.all?(tensors, fn t ->
          Nx.all(Nx.equal(t, 0)) |> Nx.to_number() == 1
        end)

      low_variance =
        not has_nan and not has_inf and not all_zero and
          Enum.all?(tensors, fn t ->
            flat = Nx.reshape(t, {Nx.size(t)})
            variance = Nx.variance(flat) |> Nx.to_number()
            variance < 1.0e-8
          end)

      %{has_nan: has_nan, has_inf: has_inf, all_zero: all_zero, low_variance: low_variance}
    end
  end

  defp health_status_tag(%{has_nan: true}), do: "ok (NaN!)"
  defp health_status_tag(%{has_inf: true}), do: "ok (Inf!)"
  defp health_status_tag(%{all_zero: true}), do: "ok (ZERO)"
  defp health_status_tag(%{low_variance: true}), do: "ok (FLAT)"
  defp health_status_tag(_), do: "ok"

  defp fmt(ms) when ms < 1, do: "#{Float.round(ms * 1000, 0)} us"
  defp fmt(ms) when ms < 100, do: "#{Float.round(ms, 1)} ms"
  defp fmt(ms), do: "#{Float.round(ms, 0)} ms"
end

# Guard: don't auto-run when loaded by another script
unless System.get_env("EDIFICE_NO_AUTORUN") do
  FullSweep.run()
end
