defmodule Edifice do
  @moduledoc """
  Edifice - A comprehensive ML architecture library for Elixir.

  Provides implementations of all major neural network architecture families,
  built on Nx and Axon. From simple MLPs to state space models, attention
  mechanisms, generative models, and graph neural networks.

  ## Quick Start

      # Build any architecture by name
      model = Edifice.build(:mamba, embed_dim: 256, hidden_size: 512)

      # Or use the module directly
      model = Edifice.SSM.Mamba.build(embed_dim: 256, hidden_size: 512)

      # List all available architectures
      Edifice.list_architectures()

  ## Architecture Families

  | Family | Architectures |
  |--------|--------------|
  | Transformer | Decoder-Only (GPT-style), Multi-Token Prediction, Byte Latent Transformer, Nemotron-H, FreeTransformer |
  | Feedforward | MLP, KAN, KAT, TabNet, BitNet |
  | Convolutional | Conv1D/2D, ResNet, DenseNet, TCN, MobileNet, EfficientNet |
  | Recurrent | LSTM, GRU, xLSTM, xLSTM v2, mLSTM, sLSTM, MinGRU, MinLSTM, DeltaNet, Gated DeltaNet, DeltaProduct, TTT, TTT-E2E, Titans, MIRAS, Reservoir (ESN), Native Recurrence, TransformerLike, DeepResLSTM |
  | State Space | Mamba, Mamba-2 (SSD), Mamba-3, S4, S4D, S5, H3, Hyena, Hyena v2, BiMamba, GatedSSM, GSS, StripedHyena, Hymba, State Space Transformer, Longhorn |
  | Attention | Multi-Head, GQA, MLA, KDA (Kimi Delta Attention), DiffTransformer, Sigmoid Attention, FoX (Forgetting Transformer), Log-Linear, NHA (Native Hybrid Attention), LASER, MoBA, MTA (Multi-Token Attention), GSA (Gated Slot Attention), RLA/RDN (Residual Linear Attention), TNN (Toeplitz), Perceiver, FNet, Linear Transformer, Nystromformer, Performer, RetNet, RetNet v2, RWKV, GLA, GLA v2, HGRN, HGRN v2, Griffin, Hawk, Based, InfiniAttention, Conformer, Mega, MEGALODON, RingAttention, Lightning Attention, Flash Linear Attention, YaRN, NSA, SPLA, InfLLM-V2, Dual Chunk Attention |
  | Vision | ViT, DeiT, Swin, U-Net, ConvNeXt, MLP-Mixer, FocalNet, PoolFormer, NeRF, MambaVision, DINOv3, Janus, Vision KAN |
  | Generative | VAE, VQ-VAE, GAN, Diffusion, DDIM, DiT, DiT v2, MMDiT, Latent Diffusion, Consistency, Score SDE, Flow Matching, Rectified Flow, SoFlow, Normalizing Flow, TarFlow, STARFlow, Transfusion, CogVideoX, TRELLIS, MDLM, LLaDA, CaDDi, DeepFlow, Meissonic, MAGVIT-v2, Show-o, JanusFlow |
  | Graph | GCN, GAT, GraphSAGE, GIN, GINv2, PNA, GraphTransformer, SchNet, DimeNet, SE(3)-Transformer, GPS, KA-GNN, Message Passing |
  | Sets | DeepSets, PointNet, PointNet++ |
  | Energy | EBM, Hopfield, Neural ODE |
  | Probabilistic | Bayesian, MC Dropout, Evidential |
  | Memory | NTM, Memory Networks, Memory Layers |
  | Meta | MoE, MoE v2, Switch MoE, Soft MoE, ReMoE, MoR, MoED, LoRA, DoRA, VeRA, Kron-LoRA, Adapter, Hypernetworks, Capsules, MixtureOfDepths, MixtureOfAgents, RLHFHead, Speculative Decoding, Test-Time Compute, Mixture of Tokenizers, Speculative Head, EAGLE-3, Manifold HC, Distillation Head, QAT, Coconut, Hybrid Builder |
  | Liquid | Liquid Neural Networks |
  | Contrastive | SimCLR, BYOL, Barlow Twins, MAE, VICReg, JEPA, Temporal JEPA, V-JEPA 2 |
  | Interpretability | Sparse Autoencoder, Transcoder, Gated SAE, JumpReLU SAE, BatchTopK SAE, Linear Probe, Crosscoder, Concept Bottleneck, DAS Probe, LEACE, Matryoshka SAE, Cross-Layer Transcoder |
  | World Model | World Model |
  | Multimodal | MLP Projection Fusion, Cross-Attention Fusion, Perceiver Resampler |
  | RL | PolicyValue, Decision Transformer |
  | Neuromorphic | SNN, ANN2SNN |
  | Inference | Medusa |
  | Robotics | ACT, Diffusion Policy, OpenVLA |
  | Audio | SoundStorm, EnCodec, F5-TTS, VALL-E, Whisper, Wav2Vec 2.0 |
  | Detection | DETR, RT-DETR, SAM 2 |
  | Scientific | FNO, DeepONet |
  """

  @architecture_registry %{
    # Transformer
    decoder_only: Edifice.Transformer.DecoderOnly,
    multi_token_prediction: Edifice.Transformer.MultiTokenPrediction,
    byte_latent_transformer: Edifice.Transformer.ByteLatentTransformer,
    nemotron_h: Edifice.Transformer.NemotronH,
    free_transformer: Edifice.Transformer.FreeTransformer,
    # Feedforward
    mlp: Edifice.Feedforward.MLP,
    kan: Edifice.Feedforward.KAN,
    kat: Edifice.Feedforward.KAT,
    tabnet: Edifice.Feedforward.TabNet,
    bitnet: Edifice.Feedforward.BitNet,
    # Convolutional
    conv1d: Edifice.Convolutional.Conv,
    resnet: Edifice.Convolutional.ResNet,
    densenet: Edifice.Convolutional.DenseNet,
    tcn: Edifice.Convolutional.TCN,
    mobilenet: Edifice.Convolutional.MobileNet,
    efficientnet: Edifice.Convolutional.EfficientNet,
    # Recurrent
    lstm: {Edifice.Recurrent, [cell_type: :lstm]},
    gru: {Edifice.Recurrent, [cell_type: :gru]},
    xlstm: Edifice.Recurrent.XLSTM,
    mlstm: {Edifice.Recurrent.XLSTM, [variant: :mlstm]},
    min_gru: Edifice.Recurrent.MinGRU,
    min_lstm: Edifice.Recurrent.MinLSTM,
    delta_net: Edifice.Recurrent.DeltaNet,
    gated_delta_net: Edifice.Recurrent.GatedDeltaNet,
    ttt: Edifice.Recurrent.TTT,
    ttt_e2e: Edifice.Recurrent.TTTE2E,
    titans: Edifice.Recurrent.Titans,
    miras: Edifice.Recurrent.MIRAS,
    reservoir: Edifice.Recurrent.Reservoir,
    slstm: Edifice.Recurrent.SLSTM,
    xlstm_v2: Edifice.Recurrent.XLSTMv2,
    native_recurrence: Edifice.Recurrent.NativeRecurrence,
    transformer_like: Edifice.Recurrent.TransformerLike,
    deep_res_lstm: Edifice.Recurrent.DeepResLSTM,
    huginn: Edifice.Recurrent.Huginn,
    delta_product: Edifice.Recurrent.DeltaProduct,
    # SSM
    mamba: Edifice.SSM.Mamba,
    mamba_ssd: Edifice.SSM.MambaSSD,
    mamba_cumsum: Edifice.SSM.MambaCumsum,
    mamba_hillis_steele: Edifice.SSM.MambaHillisSteele,
    s4: Edifice.SSM.S4,
    s4d: Edifice.SSM.S4D,
    s5: Edifice.SSM.S5,
    h3: Edifice.SSM.H3,
    hyena: Edifice.SSM.Hyena,
    bimamba: Edifice.SSM.BiMamba,
    gated_ssm: Edifice.SSM.GatedSSM,
    jamba: Edifice.SSM.Hybrid,
    zamba: Edifice.SSM.Zamba,
    striped_hyena: Edifice.SSM.StripedHyena,
    mamba3: Edifice.SSM.Mamba3,
    gss: Edifice.SSM.GSS,
    hyena_v2: Edifice.SSM.HyenaV2,
    hymba: Edifice.SSM.Hymba,
    ss_transformer: Edifice.SSM.SSTransformer,
    longhorn: Edifice.SSM.Longhorn,
    samba: Edifice.SSM.Samba,
    mixture_of_mamba: Edifice.SSM.MixtureOfMamba,
    hybrid_builder: Edifice.Meta.HybridBuilder,
    # Attention
    attention: Edifice.Attention.MultiHead,
    retnet: Edifice.Attention.RetNet,
    rwkv: Edifice.Attention.RWKV,
    gla: Edifice.Attention.GLA,
    hgrn: Edifice.Attention.HGRN,
    griffin: Edifice.Attention.Griffin,
    gqa: Edifice.Attention.GQA,
    perceiver: Edifice.Attention.Perceiver,
    fnet: Edifice.Attention.FNet,
    linear_transformer: Edifice.Attention.LinearTransformer,
    nystromformer: Edifice.Attention.Nystromformer,
    performer: Edifice.Attention.Performer,
    mega: Edifice.Attention.Mega,
    based: Edifice.Attention.Based,
    infini_attention: Edifice.Attention.InfiniAttention,
    conformer: Edifice.Attention.Conformer,
    ring_attention: Edifice.Attention.RingAttention,
    mla: Edifice.Attention.MLA,
    diff_transformer: Edifice.Attention.DiffTransformer,
    hawk: Edifice.Attention.Hawk,
    retnet_v2: Edifice.Attention.RetNetV2,
    megalodon: Edifice.Attention.Megalodon,
    gla_v2: Edifice.Attention.GLAv2,
    hgrn_v2: Edifice.Attention.HGRNv2,
    flash_linear_attention: Edifice.Attention.FlashLinearAttention,
    kda: Edifice.Attention.KDA,
    gated_attention: Edifice.Attention.GatedAttention,
    sigmoid_attention: Edifice.Attention.SigmoidAttention,
    ssmax: Edifice.Blocks.SSMax,
    softpick: Edifice.Blocks.Softpick,
    rnope_swa: Edifice.Attention.RNoPESWA,
    yarn: Edifice.Attention.YARN,
    nsa: Edifice.Attention.NSA,
    spla: Edifice.Attention.SPLA,
    infllm_v2: Edifice.Attention.InfLLMV2,
    tmrope: Edifice.Attention.TMRoPE,
    fox: Edifice.Attention.FoX,
    log_linear: Edifice.Attention.LogLinear,
    nha: Edifice.Attention.NHA,
    laser: Edifice.Attention.LASER,
    moba: Edifice.Attention.MoBA,
    mta: Edifice.Attention.MTA,
    gsa: Edifice.Attention.GSA,
    rla: Edifice.Attention.RLA,
    rdn: {Edifice.Attention.RLA, [variant: :rdn]},
    tnn: Edifice.Attention.TNN,
    # Vision
    vit: Edifice.Vision.ViT,
    deit: Edifice.Vision.DeiT,
    swin: Edifice.Vision.SwinTransformer,
    unet: Edifice.Vision.UNet,
    convnext: Edifice.Vision.ConvNeXt,
    mlp_mixer: Edifice.Vision.MLPMixer,
    focalnet: Edifice.Vision.FocalNet,
    poolformer: Edifice.Vision.PoolFormer,
    nerf: Edifice.Vision.NeRF,
    gaussian_splat: Edifice.Vision.GaussianSplat,
    mamba_vision: Edifice.Vision.MambaVision,
    dino_v2: Edifice.Vision.DINOv2,
    dino_v3: Edifice.Vision.DINOv3,
    metaformer: Edifice.Vision.MetaFormer,
    caformer: {Edifice.Vision.MetaFormer, [variant: :caformer]},
    efficient_vit: Edifice.Vision.EfficientViT,
    janus: Edifice.Vision.Janus,
    vision_kan: Edifice.Vision.VisionKAN,
    # Detection
    detr: Edifice.Detection.DETR,
    rt_detr: Edifice.Detection.RTDETR,
    sam2: Edifice.Detection.SAM2,
    # Multimodal
    multimodal_mlp_fusion: Edifice.Multimodal.Fusion,
    # Generative
    diffusion: Edifice.Generative.Diffusion,
    ddim: Edifice.Generative.DDIM,
    dit: Edifice.Generative.DiT,
    dit_v2: Edifice.Generative.DiTv2,
    latent_diffusion: Edifice.Generative.LatentDiffusion,
    consistency_model: Edifice.Generative.ConsistencyModel,
    score_sde: Edifice.Generative.ScoreSDE,
    flow_matching: Edifice.Generative.FlowMatching,
    vae: Edifice.Generative.VAE,
    vq_vae: Edifice.Generative.VQVAE,
    gan: Edifice.Generative.GAN,
    normalizing_flow: Edifice.Generative.NormalizingFlow,
    mmdit: Edifice.Generative.MMDiT,
    soflow: Edifice.Generative.SoFlow,
    var: Edifice.Generative.VAR,
    linear_dit: Edifice.Generative.LinearDiT,
    sana: Edifice.Generative.LinearDiT,
    sit: Edifice.Generative.SiT,
    transfusion: Edifice.Generative.Transfusion,
    mar: Edifice.Generative.MAR,
    cogvideox: Edifice.Generative.CogVideoX,
    trellis: Edifice.Generative.TRELLIS,
    mdlm: Edifice.Generative.MDLM,
    rectified_flow: Edifice.Generative.RectifiedFlow,
    magvit2: Edifice.Generative.MAGVIT2,
    show_o: Edifice.Generative.ShowO,
    janus_flow: Edifice.Generative.JanusFlow,
    tar_flow: Edifice.Generative.TarFlow,
    star_flow: Edifice.Generative.STARFlow,
    llada: Edifice.Generative.LLaDA,
    deep_flow: Edifice.Generative.DeepFlow,
    caddi: Edifice.Generative.CaDDi,
    meissonic: Edifice.Generative.Meissonic,
    # Graph
    gcn: Edifice.Graph.GCN,
    gat: Edifice.Graph.GAT,
    graph_sage: Edifice.Graph.GraphSAGE,
    gin: Edifice.Graph.GIN,
    pna: Edifice.Graph.PNA,
    graph_transformer: Edifice.Graph.GraphTransformer,
    schnet: Edifice.Graph.SchNet,
    gin_v2: Edifice.Graph.GINv2,
    egnn: Edifice.Graph.EGNN,
    dimenet: Edifice.Graph.DimeNet,
    se3_transformer: Edifice.Graph.SE3Transformer,
    gps: Edifice.Graph.GPS,
    ka_gnn: Edifice.Graph.KAGNN,
    # Sets
    deep_sets: Edifice.Sets.DeepSets,
    pointnet: Edifice.Sets.PointNet,
    pointnet_pp: Edifice.Sets.PointNetPP,
    # Energy
    ebm: Edifice.Energy.EBM,
    hopfield: Edifice.Energy.Hopfield,
    neural_ode: Edifice.Energy.NeuralODE,
    # Probabilistic
    bayesian: Edifice.Probabilistic.Bayesian,
    mc_dropout: Edifice.Probabilistic.MCDropout,
    evidential: Edifice.Probabilistic.EvidentialNN,
    # Memory
    ntm: Edifice.Memory.NTM,
    memory_network: Edifice.Memory.MemoryNetwork,
    engram: Edifice.Memory.Engram,
    memory_layer: Edifice.Memory.MemoryLayer,
    # Meta
    moe: Edifice.Meta.MoE,
    switch_moe: Edifice.Meta.SwitchMoE,
    soft_moe: Edifice.Meta.SoftMoE,
    lora: Edifice.Meta.LoRA,
    adapter: Edifice.Meta.Adapter,
    hypernetwork: Edifice.Meta.Hypernetwork,
    capsule: Edifice.Meta.Capsule,
    mixture_of_depths: Edifice.Meta.MixtureOfDepths,
    mixture_of_agents: Edifice.Meta.MixtureOfAgents,
    rlhf_head: Edifice.Meta.RLHFHead,
    dpo: Edifice.Meta.DPO,
    kto: Edifice.Meta.KTO,
    grpo: Edifice.Meta.GRPO,
    moe_v2: Edifice.Meta.MoEv2,
    dora: Edifice.Meta.DoRA,
    speculative_decoding: Edifice.Meta.SpeculativeDecoding,
    test_time_compute: Edifice.Meta.TestTimeCompute,
    mixture_of_tokenizers: Edifice.Meta.MixtureOfTokenizers,
    speculative_head: Edifice.Meta.SpeculativeHead,
    eagle3: Edifice.Meta.Eagle3,
    manifold_hc: Edifice.Meta.ManifoldHC,
    distillation_head: Edifice.Meta.DistillationHead,
    qat: Edifice.Meta.QAT,
    remoe: Edifice.Meta.ReMoE,
    mixture_of_recursions: Edifice.Meta.MixtureOfRecursions,
    mixture_of_expert_depths: Edifice.Meta.MixtureOfExpertDepths,
    coconut: Edifice.Meta.Coconut,
    vera: Edifice.Meta.VeRA,
    kron_lora: Edifice.Meta.KronLoRA,
    # Contrastive / Self-Supervised
    simclr: Edifice.Contrastive.SimCLR,
    byol: Edifice.Contrastive.BYOL,
    barlow_twins: Edifice.Contrastive.BarlowTwins,
    mae: Edifice.Contrastive.MAE,
    vicreg: Edifice.Contrastive.VICReg,
    jepa: Edifice.Contrastive.JEPA,
    temporal_jepa: Edifice.Contrastive.TemporalJEPA,
    vjepa2: Edifice.Contrastive.VJEPA2,
    siglip: Edifice.Contrastive.SigLIP,
    # Interpretability
    sparse_autoencoder: Edifice.Interpretability.SparseAutoencoder,
    transcoder: Edifice.Interpretability.Transcoder,
    gated_sae: Edifice.Interpretability.GatedSAE,
    linear_probe: Edifice.Interpretability.LinearProbe,
    jump_relu_sae: Edifice.Interpretability.JumpReluSAE,
    batch_top_k_sae: Edifice.Interpretability.BatchTopKSAE,
    crosscoder: Edifice.Interpretability.Crosscoder,
    concept_bottleneck: Edifice.Interpretability.ConceptBottleneck,
    das_probe: Edifice.Interpretability.DASProbe,
    leace: Edifice.Interpretability.LEACE,
    matryoshka_sae: Edifice.Interpretability.MatryoshkaSAE,
    cross_layer_transcoder: Edifice.Interpretability.CrossLayerTranscoder,
    # World Model
    world_model: Edifice.WorldModel.WorldModel,
    # RL
    policy_value: Edifice.RL.PolicyValue,
    decision_transformer: Edifice.RL.DecisionTransformer,
    # Lightning Attention
    lightning_attention: Edifice.Attention.LightningAttention,
    # Dual Chunk Attention
    dual_chunk_attention: Edifice.Attention.DualChunk,
    # Liquid
    liquid: Edifice.Liquid,
    # Scientific
    fno: Edifice.Scientific.FNO,
    deep_onet: Edifice.Scientific.DeepONet,
    # Neuromorphic
    snn: Edifice.Neuromorphic.SNN,
    ann2snn: Edifice.Neuromorphic.ANN2SNN,
    # Inference
    medusa: Edifice.Inference.Medusa,
    # Robotics
    act: Edifice.Robotics.ACT,
    diffusion_policy: Edifice.Robotics.DiffusionPolicy,
    openvla: Edifice.Robotics.OpenVLA,
    # Audio
    soundstorm: Edifice.Audio.SoundStorm,
    encodec: Edifice.Audio.EnCodec,
    f5_tts: Edifice.Audio.F5TTS,
    valle: Edifice.Audio.VALLE,
    whisper: Edifice.Audio.Whisper,
    wav2vec2: Edifice.Audio.Wav2Vec2
  }

  @doc """
  List all available architecture names.

  ## Examples

      iex> :mamba in Edifice.list_architectures()
      true
  """
  @spec list_architectures() :: [atom()]
  def list_architectures do
    @architecture_registry |> Map.keys() |> Enum.sort()
  end

  @doc """
  List architectures grouped by family.

  ## Examples

      iex> families = Edifice.list_families()
      iex> is_map(families)
      true

      iex> families = Edifice.list_families()
      iex> :mamba in families.ssm
      true
  """
  @spec list_families() :: %{atom() => [atom()]}
  def list_families do
    %{
      transformer: [
        :decoder_only,
        :multi_token_prediction,
        :byte_latent_transformer,
        :nemotron_h,
        :free_transformer
      ],
      feedforward: [:mlp, :kan, :kat, :tabnet, :bitnet],
      convolutional: [:conv1d, :resnet, :densenet, :tcn, :mobilenet, :efficientnet],
      recurrent: [
        :lstm,
        :gru,
        :xlstm,
        :mlstm,
        :min_gru,
        :min_lstm,
        :delta_net,
        :gated_delta_net,
        :ttt,
        :ttt_e2e,
        :titans,
        :miras,
        :reservoir,
        :slstm,
        :xlstm_v2,
        :native_recurrence,
        :transformer_like,
        :deep_res_lstm,
        :huginn,
        :delta_product
      ],
      ssm: [
        :mamba,
        :mamba_ssd,
        :mamba_cumsum,
        :mamba_hillis_steele,
        :s4,
        :s4d,
        :s5,
        :h3,
        :hyena,
        :bimamba,
        :gated_ssm,
        :jamba,
        :zamba,
        :striped_hyena,
        :mamba3,
        :gss,
        :hyena_v2,
        :hymba,
        :ss_transformer,
        :longhorn,
        :samba,
        :mixture_of_mamba
      ],
      attention: [
        :attention,
        :retnet,
        :rwkv,
        :gla,
        :hgrn,
        :griffin,
        :gqa,
        :perceiver,
        :fnet,
        :linear_transformer,
        :nystromformer,
        :performer,
        :mega,
        :based,
        :infini_attention,
        :conformer,
        :ring_attention,
        :mla,
        :diff_transformer,
        :hawk,
        :retnet_v2,
        :megalodon,
        :lightning_attention,
        :gla_v2,
        :hgrn_v2,
        :flash_linear_attention,
        :kda,
        :gated_attention,
        :sigmoid_attention,
        :ssmax,
        :softpick,
        :rnope_swa,
        :yarn,
        :nsa,
        :spla,
        :infllm_v2,
        :tmrope,
        :dual_chunk_attention,
        :fox,
        :log_linear,
        :nha,
        :laser,
        :moba,
        :mta,
        :gsa,
        :rla,
        :rdn,
        :tnn
      ],
      vision: [
        :vit,
        :deit,
        :swin,
        :unet,
        :convnext,
        :mlp_mixer,
        :focalnet,
        :poolformer,
        :nerf,
        :gaussian_splat,
        :mamba_vision,
        :dino_v2,
        :dino_v3,
        :metaformer,
        :caformer,
        :efficient_vit,
        :janus,
        :vision_kan
      ],
      generative: [
        :diffusion,
        :ddim,
        :dit,
        :dit_v2,
        :latent_diffusion,
        :consistency_model,
        :score_sde,
        :flow_matching,
        :vae,
        :vq_vae,
        :gan,
        :normalizing_flow,
        :mmdit,
        :soflow,
        :var,
        :linear_dit,
        :sana,
        :sit,
        :transfusion,
        :mar,
        :cogvideox,
        :trellis,
        :mdlm,
        :rectified_flow,
        :magvit2,
        :show_o,
        :janus_flow,
        :tar_flow,
        :star_flow,
        :llada,
        :deep_flow,
        :caddi,
        :meissonic
      ],
      graph: [
        :gcn,
        :gat,
        :graph_sage,
        :gin,
        :gin_v2,
        :pna,
        :graph_transformer,
        :schnet,
        :egnn,
        :dimenet,
        :se3_transformer,
        :gps,
        :ka_gnn
      ],
      sets: [:deep_sets, :pointnet, :pointnet_pp],
      energy: [:ebm, :hopfield, :neural_ode],
      probabilistic: [:bayesian, :mc_dropout, :evidential],
      memory: [:ntm, :memory_network, :engram, :memory_layer],
      meta: [
        :moe,
        :switch_moe,
        :soft_moe,
        :lora,
        :adapter,
        :hypernetwork,
        :capsule,
        :mixture_of_depths,
        :mixture_of_agents,
        :rlhf_head,
        :dpo,
        :kto,
        :grpo,
        :moe_v2,
        :dora,
        :speculative_decoding,
        :test_time_compute,
        :mixture_of_tokenizers,
        :speculative_head,
        :eagle3,
        :manifold_hc,
        :distillation_head,
        :qat,
        :remoe,
        :mixture_of_recursions,
        :mixture_of_expert_depths,
        :coconut,
        :hybrid_builder,
        :vera,
        :kron_lora
      ],
      contrastive: [
        :simclr,
        :byol,
        :barlow_twins,
        :mae,
        :vicreg,
        :jepa,
        :temporal_jepa,
        :siglip,
        :vjepa2
      ],
      interpretability: [
        :sparse_autoencoder,
        :transcoder,
        :gated_sae,
        :linear_probe,
        :jump_relu_sae,
        :batch_top_k_sae,
        :crosscoder,
        :concept_bottleneck,
        :das_probe,
        :leace,
        :matryoshka_sae,
        :cross_layer_transcoder
      ],
      world_model: [:world_model],
      multimodal: [:multimodal_mlp_fusion],
      rl: [:policy_value, :decision_transformer],
      liquid: [:liquid],
      scientific: [:fno, :deep_onet],
      neuromorphic: [:snn, :ann2snn],
      inference: [:medusa],
      robotics: [:act, :diffusion_policy, :openvla],
      audio: [:soundstorm, :encodec, :f5_tts, :valle, :whisper, :wav2vec2],
      detection: [:detr, :rt_detr, :sam2]
    }
  end

  @doc """
  Build an architecture by name.

  ## Parameters
    - `name` - Architecture name (see `list_architectures/0`)
    - `opts` - Architecture-specific options. The primary input dimension option
      varies by family:
      - `:embed_dim` — sequence models (SSM, attention, recurrent)
      - `:input_size` — flat-vector models (MLP, probabilistic, energy)
      - `:input_dim` — graph/spatial models (GCN, DeepSets, vision)
      - `:obs_size` — diffusion models (Diffusion, DDIM, FlowMatching)

      You can pass any of the first three interchangeably — `build/2` will
      normalize to the name each module expects. `:obs_size` is also
      normalized from the above. Image models requiring `:input_shape`
      (a tuple like `{nil, 32, 32, 3}`) must be passed explicitly.

  ## Examples

      iex> model = Edifice.build(:mlp, input_size: 16, hidden_sizes: [32])
      iex> %Axon{} = model
      iex> true
      true

  ## Returns

  An `Axon.t()` model for most architectures. These architectures return tuples:

    - `:vae` — `{encoder, decoder}`
    - `:vq_vae` — `{encoder, decoder}`
    - `:gan` — `{generator, discriminator}`
    - `:normalizing_flow` — `{flow_model, log_det_fn}`
    - `:simclr` — `{backbone, projection_head}`
    - `:byol` — `{online_network, target_network}`
    - `:barlow_twins` — `{backbone, projection_head}`
    - `:vicreg` — `{backbone, projection_head}`
    - `:jepa` — `{context_encoder, predictor}`
    - `:temporal_jepa` — `{context_encoder, predictor}`
    - `:vjepa2` — `{context_encoder, predictor}`
    - `:mae` — `{encoder, decoder}`
    - `:world_model` — `{encoder, dynamics, reward_head}` (or 4-tuple with decoder)
    - `:byte_latent_transformer` — `{encoder, latent_transformer, decoder}`
    - `:speculative_decoding` — `{draft_model, verifier_model}`
    - `:multi_token_prediction` — `Axon.container(%{pred_1: ..., pred_N: ...})`
    - `:test_time_compute` — `Axon.container(%{backbone: ..., scores: ...})`
    - `:speculative_head` — `Axon.container(%{pred_1: ..., pred_N: ...})`
    - `:magvit2` — `{encoder, decoder}`
    - `:tar_flow` — `{encoder, decoder}`
    - `:star_flow` — `{encoder, decoder}`
    - `:act` — `{encoder, decoder}`
    - `:medusa` — `Axon.container(%{head_1: ..., head_K: ...})`
    - `:detr` — `Axon.container(%{class_logits: ..., bbox_pred: ...})`
    - `:rt_detr` — `Axon.container(%{class_logits: ..., bbox_pred: ...})`
    - `:sam2` — `Axon.container(%{masks: ..., iou_scores: ...})`
  """
  @spec build(atom(), keyword()) :: Axon.t() | tuple()
  def build(name, opts \\ []) do
    opts = normalize_input_dim(opts)

    case Map.fetch(@architecture_registry, name) do
      {:ok, {module, default_opts}} ->
        merged_opts = Keyword.merge(default_opts, opts)
        module.build(merged_opts)

      {:ok, module} ->
        module.build(opts)

      :error ->
        available = list_architectures() |> Enum.join(", ")
        raise ArgumentError, "Unknown architecture #{inspect(name)}. Available: #{available}"
    end
  end

  @doc """
  Get the module for a named architecture.

  ## Examples

      iex> Edifice.module_for(:mamba)
      Edifice.SSM.Mamba

      iex> Edifice.module_for(:mlp)
      Edifice.Feedforward.MLP
  """
  @spec module_for(atom()) :: module()
  def module_for(name) do
    case Map.fetch(@architecture_registry, name) do
      {:ok, {module, _}} ->
        module

      {:ok, module} ->
        module

      :error ->
        available = list_architectures() |> Enum.join(", ")
        raise ArgumentError, "Unknown architecture #{inspect(name)}. Available: #{available}"
    end
  end

  # Normalize input dimension options so users can pass any of the scalar
  # names and the right one reaches each module.
  defp normalize_input_dim(opts) do
    value =
      Keyword.get(opts, :embed_dim) ||
        Keyword.get(opts, :input_size) ||
        Keyword.get(opts, :input_dim) ||
        Keyword.get(opts, :obs_size)

    if value do
      opts
      |> Keyword.put_new(:embed_dim, value)
      |> Keyword.put_new(:input_size, value)
      |> Keyword.put_new(:input_dim, value)
      |> Keyword.put_new(:obs_size, value)
    else
      opts
    end
  end
end
