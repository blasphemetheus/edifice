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
  | Transformer | Decoder-Only (GPT-style) |
  | Feedforward | MLP, KAN, KAT, TabNet, BitNet |
  | Convolutional | Conv1D/2D, ResNet, DenseNet, TCN, MobileNet, EfficientNet |
  | Recurrent | LSTM, GRU, xLSTM, mLSTM, MinGRU, MinLSTM, DeltaNet, TTT, Titans, Reservoir (ESN) |
  | State Space | Mamba, Mamba-2 (SSD), S4, S4D, S5, H3, Hyena, BiMamba, GatedSSM, StripedHyena |
  | Attention | Multi-Head, GQA, Perceiver, FNet, Linear Transformer, Nystromformer, Performer, RetNet, RWKV, GLA, HGRN, Griffin, Based, InfiniAttention, Conformer, Mega, RingAttention |
  | Vision | ViT, DeiT, Swin, U-Net, ConvNeXt, MLP-Mixer, FocalNet, PoolFormer, NeRF |
  | Generative | VAE, VQ-VAE, GAN, Diffusion, DDIM, DiT, Latent Diffusion, Consistency, Score SDE, Flow Matching, Normalizing Flow |
  | Graph | GCN, GAT, GraphSAGE, GIN, GINv2, PNA, GraphTransformer, SchNet, Message Passing |
  | Sets | DeepSets, PointNet |
  | Energy | EBM, Hopfield, Neural ODE |
  | Probabilistic | Bayesian, MC Dropout, Evidential |
  | Memory | NTM, Memory Networks |
  | Meta | MoE, Switch MoE, Soft MoE, LoRA, Adapter, Hypernetworks, Capsules, MixtureOfDepths, MixtureOfAgents, RLHFHead |
  | Liquid | Liquid Neural Networks |
  | Contrastive | SimCLR, BYOL, Barlow Twins, MAE, VICReg |
  | Neuromorphic | SNN, ANN2SNN |
  """

  @architecture_registry %{
    # Transformer
    decoder_only: Edifice.Transformer.DecoderOnly,
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
    ttt: Edifice.Recurrent.TTT,
    titans: Edifice.Recurrent.Titans,
    reservoir: Edifice.Recurrent.Reservoir,
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
    # Contrastive / Self-Supervised
    simclr: Edifice.Contrastive.SimCLR,
    byol: Edifice.Contrastive.BYOL,
    barlow_twins: Edifice.Contrastive.BarlowTwins,
    mae: Edifice.Contrastive.MAE,
    vicreg: Edifice.Contrastive.VICReg,
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
      transformer: [:decoder_only],
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
        :ttt,
        :titans,
        :reservoir
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
        :striped_hyena
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
        :ring_attention
      ],
      vision: [:vit, :deit, :swin, :unet, :convnext, :mlp_mixer, :focalnet, :poolformer, :nerf],
      generative: [
        :diffusion,
        :ddim,
        :dit,
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
        :rlhf_head
      ],
      contrastive: [:simclr, :byol, :barlow_twins, :mae, :vicreg],
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
    - `:mae` — `{encoder, decoder}`
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
