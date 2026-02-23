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
  | Transformer | Decoder-Only (GPT-style), Multi-Token Prediction, Byte Latent Transformer |
  | Feedforward | MLP, KAN, KAT, TabNet, BitNet |
  | Convolutional | Conv1D/2D, ResNet, DenseNet, TCN, MobileNet, EfficientNet |
  | Recurrent | LSTM, GRU, xLSTM, xLSTM v2, mLSTM, sLSTM, MinGRU, MinLSTM, DeltaNet, Gated DeltaNet, TTT, Titans, Reservoir (ESN), Native Recurrence |
  | State Space | Mamba, Mamba-2 (SSD), Mamba-3, S4, S4D, S5, H3, Hyena, Hyena v2, BiMamba, GatedSSM, GSS, StripedHyena, Hymba, State Space Transformer |
  | Attention | Multi-Head, GQA, MLA, DiffTransformer, Perceiver, FNet, Linear Transformer, Nystromformer, Performer, RetNet, RetNet v2, RWKV, GLA, GLA v2, HGRN, HGRN v2, Griffin, Hawk, Based, InfiniAttention, Conformer, Mega, MEGALODON, RingAttention, Lightning Attention, Flash Linear Attention |
  | Vision | ViT, DeiT, Swin, U-Net, ConvNeXt, MLP-Mixer, FocalNet, PoolFormer, NeRF |
  | Generative | VAE, VQ-VAE, GAN, Diffusion, DDIM, DiT, DiT v2, Latent Diffusion, Consistency, Score SDE, Flow Matching, Normalizing Flow |
  | Graph | GCN, GAT, GraphSAGE, GIN, GINv2, PNA, GraphTransformer, SchNet, Message Passing |
  | Sets | DeepSets, PointNet |
  | Energy | EBM, Hopfield, Neural ODE |
  | Probabilistic | Bayesian, MC Dropout, Evidential |
  | Memory | NTM, Memory Networks |
  | Meta | MoE, MoE v2, Switch MoE, Soft MoE, LoRA, DoRA, Adapter, Hypernetworks, Capsules, MixtureOfDepths, MixtureOfAgents, RLHFHead, Speculative Decoding, Test-Time Compute, Mixture of Tokenizers, Speculative Head, Distillation Head, QAT |
  | Liquid | Liquid Neural Networks |
  | Contrastive | SimCLR, BYOL, Barlow Twins, MAE, VICReg, JEPA, Temporal JEPA |
  | Interpretability | Sparse Autoencoder, Transcoder |
  | World Model | World Model |
  | RL | PolicyValue |
  | Neuromorphic | SNN, ANN2SNN |
  """

  @architecture_registry %{
    # Transformer
    decoder_only: Edifice.Transformer.DecoderOnly,
    multi_token_prediction: Edifice.Transformer.MultiTokenPrediction,
    byte_latent_transformer: Edifice.Transformer.ByteLatentTransformer,
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
    titans: Edifice.Recurrent.Titans,
    reservoir: Edifice.Recurrent.Reservoir,
    slstm: Edifice.Recurrent.SLSTM,
    xlstm_v2: Edifice.Recurrent.XLSTMv2,
    native_recurrence: Edifice.Recurrent.NativeRecurrence,
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
    # Graph
    gcn: Edifice.Graph.GCN,
    gat: Edifice.Graph.GAT,
    graph_sage: Edifice.Graph.GraphSAGE,
    gin: Edifice.Graph.GIN,
    pna: Edifice.Graph.PNA,
    graph_transformer: Edifice.Graph.GraphTransformer,
    schnet: Edifice.Graph.SchNet,
    gin_v2: Edifice.Graph.GINv2,
    # Sets
    deep_sets: Edifice.Sets.DeepSets,
    pointnet: Edifice.Sets.PointNet,
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
    moe_v2: Edifice.Meta.MoEv2,
    dora: Edifice.Meta.DoRA,
    speculative_decoding: Edifice.Meta.SpeculativeDecoding,
    test_time_compute: Edifice.Meta.TestTimeCompute,
    mixture_of_tokenizers: Edifice.Meta.MixtureOfTokenizers,
    speculative_head: Edifice.Meta.SpeculativeHead,
    distillation_head: Edifice.Meta.DistillationHead,
    qat: Edifice.Meta.QAT,
    # Contrastive / Self-Supervised
    simclr: Edifice.Contrastive.SimCLR,
    byol: Edifice.Contrastive.BYOL,
    barlow_twins: Edifice.Contrastive.BarlowTwins,
    mae: Edifice.Contrastive.MAE,
    vicreg: Edifice.Contrastive.VICReg,
    jepa: Edifice.Contrastive.JEPA,
    temporal_jepa: Edifice.Contrastive.TemporalJEPA,
    # Interpretability
    sparse_autoencoder: Edifice.Interpretability.SparseAutoencoder,
    transcoder: Edifice.Interpretability.Transcoder,
    # World Model
    world_model: Edifice.WorldModel.WorldModel,
    # RL
    policy_value: Edifice.RL.PolicyValue,
    # Lightning Attention
    lightning_attention: Edifice.Attention.LightningAttention,
    # Liquid
    liquid: Edifice.Liquid,
    # Neuromorphic
    snn: Edifice.Neuromorphic.SNN,
    ann2snn: Edifice.Neuromorphic.ANN2SNN
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

      Edifice.list_families()
      # => %{
      #   feedforward: [:mlp, :kan],
      #   ssm: [:mamba, :mamba_ssd, :s5, ...],
      #   ...
      # }
  """
  @spec list_families() :: %{atom() => [atom()]}
  def list_families do
    %{
      transformer: [:decoder_only, :multi_token_prediction, :byte_latent_transformer],
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
        :titans,
        :reservoir,
        :slstm,
        :xlstm_v2,
        :native_recurrence
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
        :ss_transformer
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
        :flash_linear_attention
      ],
      vision: [:vit, :deit, :swin, :unet, :convnext, :mlp_mixer, :focalnet, :poolformer, :nerf],
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
        :normalizing_flow
      ],
      graph: [:gcn, :gat, :graph_sage, :gin, :gin_v2, :pna, :graph_transformer, :schnet],
      sets: [:deep_sets, :pointnet],
      energy: [:ebm, :hopfield, :neural_ode],
      probabilistic: [:bayesian, :mc_dropout, :evidential],
      memory: [:ntm, :memory_network],
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
        :moe_v2,
        :dora,
        :speculative_decoding,
        :test_time_compute,
        :mixture_of_tokenizers,
        :speculative_head,
        :distillation_head,
        :qat
      ],
      contrastive: [:simclr, :byol, :barlow_twins, :mae, :vicreg, :jepa, :temporal_jepa],
      interpretability: [:sparse_autoencoder, :transcoder],
      world_model: [:world_model],
      rl: [:policy_value],
      liquid: [:liquid],
      neuromorphic: [:snn, :ann2snn]
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

      model = Edifice.build(:mamba, embed_dim: 256, hidden_size: 512, num_layers: 4)
      model = Edifice.build(:mlp, input_size: 256, hidden_sizes: [512, 256])
      model = Edifice.build(:gcn, input_dim: 8, hidden_dims: [32, 32], num_classes: 10)

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
    - `:mae` — `{encoder, decoder}`
    - `:world_model` — `{encoder, dynamics, reward_head}` (or 4-tuple with decoder)
    - `:byte_latent_transformer` — `{encoder, latent_transformer, decoder}`
    - `:speculative_decoding` — `{draft_model, verifier_model}`
    - `:multi_token_prediction` — `Axon.container(%{pred_1: ..., pred_N: ...})`
    - `:test_time_compute` — `Axon.container(%{backbone: ..., scores: ...})`
    - `:speculative_head` — `Axon.container(%{pred_1: ..., pred_N: ...})`
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

      Edifice.module_for(:mamba)
      # => Edifice.SSM.Mamba
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
