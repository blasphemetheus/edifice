defmodule Edifice.RegistryIntegrityTest do
  @moduledoc """
  Verifies that every registered architecture in Edifice.list_architectures/0
  can be built via Edifice.build/2. No forward passes — just Axon graph
  construction. Catches missing required options, broken module references,
  and build-time crashes for all 200+ architectures.
  """
  use ExUnit.Case, async: true

  # ── Broad build opts ────────────────────────────────────────────────
  # Edifice.build/2 normalizes embed_dim → input_size/input_dim/obs_size.
  # Most modules use Keyword.get with defaults, so extra keys are ignored.
  @broad_opts [
    embed_dim: 16,
    hidden_size: 8,
    num_layers: 1,
    num_heads: 2,
    num_kv_heads: 2,
    head_dim: 4,
    seq_len: 4,
    window_size: 4,
    state_size: 4,
    dropout: 0.0,
    # Vision
    image_size: 16,
    in_channels: 1,
    patch_size: 4,
    depth: 1,
    # Graph / classification
    num_classes: 4,
    output_size: 4,
    # Generative
    latent_size: 4,
    action_dim: 4,
    action_horizon: 4,
    # Contrastive
    encoder_dim: 16,
    projection_dim: 8,
    # MoE
    num_experts: 2,
    top_k: 1,
    # Vocab
    vocab_size: 32,
    # Feedforward
    hidden_sizes: [8],
    # PEFT
    rank: 4,
    # Audio
    num_codebooks: 2
  ]

  # ── Architecture-specific overrides ─────────────────────────────────
  # These are merged INTO @broad_opts (override wins on conflict).
  @overrides %{
    # Vision: list-valued opts
    swin: [depths: [1], num_heads: [2], window_size: 4],
    convnext: [dims: [8], depths: [1]],
    focalnet: [depths: [1], dims: [8]],
    poolformer: [depths: [1], dims: [8]],
    metaformer: [depths: [1], dims: [8], num_heads: [2]],
    caformer: [depths: [1], dims: [8], num_heads: [2]],
    efficient_vit: [depths: [1], dims: [8], num_heads: [2]],
    mamba_vision: [depths: [1, 1, 1, 1], dim: 8, num_heads: [1, 1, 2, 2]],
    dino_v2: [include_head: false],
    dino_v3: [include_head: false],
    # Vision: NeRF / 3D
    nerf: [coord_dim: 3, dir_dim: 3],
    gaussian_splat: [num_gaussians: 4, image_size: 8],
    # Convolutional: input_shape required
    resnet: [
      input_shape: {nil, 16, 16, 1},
      block_sizes: [1, 1],
      initial_channels: 4
    ],
    densenet: [
      input_shape: {nil, 32, 32, 1},
      block_config: [2],
      growth_rate: 4,
      initial_channels: 8,
      image_size: 32
    ],
    capsule: [
      input_shape: {nil, 28, 28, 1},
      conv_channels: 16,
      conv_kernel: 9,
      num_primary_caps: 4,
      primary_cap_dim: 4,
      num_digit_caps: 4,
      digit_cap_dim: 4
    ],
    # Graph
    schnet: [num_interactions: 1, num_filters: 8, num_rbf: 8],
    gin_v2: [edge_dim: 4],
    egnn: [in_node_features: 8],
    dimenet: [],
    se3_transformer: [],
    gps: [pe_dim: 4, rwse_walk_length: 4],
    # Detection
    detr: [
      hidden_dim: 8,
      num_encoder_layers: 1,
      num_decoder_layers: 1,
      ffn_dim: 8,
      num_queries: 4
    ],
    rt_detr: [
      hidden_dim: 8,
      num_encoder_layers: 1,
      num_decoder_layers: 1,
      ffn_dim: 8,
      num_queries: 4
    ],
    sam2: [hidden_dim: 8],
    # Audio
    whisper: [
      n_mels: 16,
      audio_len: 4,
      hidden_dim: 8,
      num_encoder_layers: 1,
      num_decoder_layers: 1
    ],
    encodec: [hidden_dim: 8],
    soundstorm: [],
    valle: [],
    # Transformer
    multi_token_prediction: [num_predictions: 2],
    byte_latent_transformer: [
      max_byte_len: 4,
      num_patches: 2,
      latent_dim: 8
    ],
    # Memory
    ntm: [memory_size: 4, memory_dim: 4, num_heads: 1],
    memory_network: [num_memories: 4],
    engram: [key_dim: 16, value_dim: 8, num_tables: 2, num_buckets: 4],
    # Meta: MoE variants need input_size/output_size explicitly
    moe: [input_size: 16, hidden_size: 32, output_size: 8],
    moe_v2: [
      input_size: 16,
      hidden_size: 32,
      output_size: 8,
      num_shared_experts: 1,
      num_routed_experts: 2,
      tokens_per_expert: 2
    ],
    remoe: [input_size: 16, hidden_size: 32, output_size: 8],
    hypernetwork: [
      conditioning_size: 16,
      target_layer_sizes: [{16, 8}],
      input_size: 16
    ],
    speculative_decoding: [
      draft_type: :gqa,
      verifier_type: :gqa,
      draft_model_opts: [
        hidden_size: 8,
        num_layers: 1,
        num_heads: 2,
        seq_len: 4,
        dropout: 0.0
      ],
      verifier_model_opts: [
        hidden_size: 8,
        num_layers: 1,
        num_heads: 2,
        seq_len: 4,
        dropout: 0.0
      ]
    ],
    test_time_compute: [scorer_hidden: 8],
    mixture_of_tokenizers: [
      num_tokenizers: 2,
      tokenizer_vocab_sizes: [16, 32],
      tokenizer_embed_dims: [8, 8]
    ],
    speculative_head: [head_hidden: 8, num_predictions: 2],
    eagle3: [hidden_size: 8, vocab_size: 32, num_heads: 2, num_kv_heads: 2],
    distillation_head: [teacher_dim: 8],
    # RL
    policy_value: [action_size: 4],
    decision_transformer: [state_dim: 16, action_dim: 4, context_len: 4],
    # Robotics
    act: [obs_dim: 16, latent_dim: 4, chunk_size: 4],
    openvla: [hidden_dim: 8],
    # Generative
    mmdit: [
      img_dim: 16,
      txt_dim: 16,
      hidden_size: 16,
      img_tokens: 4,
      txt_tokens: 4
    ],
    var: [scales: [1, 2]],
    cogvideox: [hidden_size: 8, text_hidden_size: 8, patch_size: [1, 2, 2]],
    trellis: [feature_dim: 8],
    # Contrastive
    jepa: [predictor_embed_dim: 8, encoder_depth: 1, predictor_depth: 1],
    temporal_jepa: [
      predictor_embed_dim: 8,
      encoder_depth: 1,
      predictor_depth: 1
    ],
    mae: [num_patches: 4, decoder_depth: 1, decoder_num_heads: 2],
    # Interpretability
    sparse_autoencoder: [input_size: 16, hidden_size: 32],
    transcoder: [input_size: 16, output_size: 8, hidden_size: 32],
    concept_bottleneck: [input_size: 16, num_concepts: 8],
    # World Model
    world_model: [action_size: 4],
    # Multimodal
    multimodal_mlp_fusion: [
      vision_dim: 16,
      llm_dim: 8,
      num_visual_tokens: 4,
      text_seq_len: 4
    ],
    # Scientific
    fno: [in_channels: 1, out_channels: 1, hidden_channels: 8, modes: 4],
    deep_onet: [num_sensors: 8, coord_dim: 3],
    # RWKV needs head_size <= hidden_size
    rwkv: [head_size: 4],
    # Reservoir uses input_size
    reservoir: [input_size: 16, reservoir_size: 8, output_size: 8],
    # Chunked attention
    lightning_attention: [block_size: 2, chunk_size: 2],
    flash_linear_attention: [block_size: 2, chunk_size: 2],
    dual_chunk_attention: [block_size: 2, chunk_size: 2],
    # Perceiver
    perceiver: [input_dim: 16, hidden_dim: 8, output_dim: 8, num_latents: 4],
    # Sets
    deep_sets: [input_dim: 16, output_dim: 4],
    pointnet_pp: [input_dim: 3],
    # Inference
    medusa: [base_hidden_dim: 8, vocab_size: 32],
    # Robotics
    diffusion_policy: [obs_dim: 16, action_dim: 4, action_horizon: 4],
    # Contrastive
    vjepa2: [patch_dim: 16],
    # Wave 5 generative
    deep_flow: [input_size: 8, patch_size: 2, in_channels: 1, num_branches: 2],
    meissonic: [codebook_size: 32, num_image_tokens: 16, text_dim: 8, cond_dim: 8]
  }

  for arch <- Edifice.list_architectures() do
    test "#{arch} builds via Edifice.build/2" do
      arch = unquote(arch)
      overrides = Map.get(@overrides, arch, [])
      opts = Keyword.merge(@broad_opts, overrides)

      result = Edifice.build(arch, opts)

      assert is_struct(result, Axon) or is_tuple(result),
             "#{arch}: expected %Axon{} or tuple, got #{inspect(result, limit: 3)}"
    end
  end
end
