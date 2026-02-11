defmodule Edifice do
  @moduledoc """
  Edifice - A comprehensive ML architecture library for Elixir.

  Provides implementations of all major neural network architecture families,
  built on Nx and Axon. From simple MLPs to state space models, attention
  mechanisms, generative models, and graph neural networks.

  ## Quick Start

      # Build any architecture by name
      model = Edifice.build(:mamba, embed_size: 256, hidden_size: 512)

      # Or use the module directly
      model = Edifice.SSM.Mamba.build(embed_size: 256, hidden_size: 512)

      # List all available architectures
      Edifice.list_architectures()

  ## Architecture Families

  | Family | Architectures |
  |--------|--------------|
  | Feedforward | MLP, KAN |
  | Convolutional | Conv1D/2D, ResNet, DenseNet, TCN |
  | Recurrent | LSTM, GRU, xLSTM, Reservoir (ESN) |
  | State Space | Mamba, Mamba-2 (SSD), S5, GatedSSM |
  | Attention | Multi-Head, RetNet, RWKV, GLA, HGRN, Griffin |
  | Generative | VAE, VQ-VAE, GAN, Diffusion, Flow Matching, Normalizing Flow |
  | Graph | GCN, GAT, Message Passing |
  | Sets | DeepSets, PointNet |
  | Energy | EBM, Hopfield |
  | Probabilistic | Bayesian, MC Dropout |
  | Memory | NTM, Memory Networks |
  | Meta | MoE, Hypernetworks, Capsules |
  | Liquid | Liquid Neural Networks |
  | Neuromorphic | SNN |
  """

  @architecture_registry %{
    # Feedforward
    mlp: Edifice.Feedforward.MLP,
    kan: Edifice.Feedforward.KAN,
    # Convolutional
    resnet: Edifice.Convolutional.ResNet,
    densenet: Edifice.Convolutional.DenseNet,
    tcn: Edifice.Convolutional.TCN,
    # Recurrent
    lstm: {Edifice.Recurrent, [cell_type: :lstm]},
    gru: {Edifice.Recurrent, [cell_type: :gru]},
    xlstm: Edifice.Recurrent.XLSTM,
    reservoir: Edifice.Recurrent.Reservoir,
    # SSM
    mamba: Edifice.SSM.Mamba,
    mamba_ssd: Edifice.SSM.MambaSSD,
    mamba_cumsum: Edifice.SSM.MambaCumsum,
    mamba_hillis_steele: Edifice.SSM.MambaHillisSteele,
    s5: Edifice.SSM.S5,
    gated_ssm: Edifice.SSM.GatedSSM,
    jamba: Edifice.SSM.Hybrid,
    zamba: Edifice.SSM.Zamba,
    # Attention
    attention: Edifice.Attention.MultiHead,
    sliding_window: Edifice.Attention.MultiHead,
    retnet: Edifice.Attention.RetNet,
    rwkv: Edifice.Attention.RWKV,
    gla: Edifice.Attention.GLA,
    hgrn: Edifice.Attention.HGRN,
    griffin: Edifice.Attention.Griffin,
    # Generative
    diffusion: Edifice.Generative.Diffusion,
    flow_matching: Edifice.Generative.FlowMatching,
    vae: Edifice.Generative.VAE,
    vq_vae: Edifice.Generative.VQVAE,
    gan: Edifice.Generative.GAN,
    normalizing_flow: Edifice.Generative.NormalizingFlow,
    # Graph
    gcn: Edifice.Graph.GCN,
    gat: Edifice.Graph.GAT,
    # Sets
    deep_sets: Edifice.Sets.DeepSets,
    pointnet: Edifice.Sets.PointNet,
    # Energy
    ebm: Edifice.Energy.EBM,
    hopfield: Edifice.Energy.Hopfield,
    # Probabilistic
    bayesian: Edifice.Probabilistic.Bayesian,
    mc_dropout: Edifice.Probabilistic.MCDropout,
    # Memory
    ntm: Edifice.Memory.NTM,
    # Meta
    moe: Edifice.Meta.MoE,
    hypernetwork: Edifice.Meta.Hypernetwork,
    capsule: Edifice.Meta.Capsule,
    # Liquid
    liquid: Edifice.Liquid,
    # Neuromorphic
    snn: Edifice.Neuromorphic.SNN
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
      feedforward: [:mlp, :kan],
      convolutional: [:resnet, :densenet, :tcn],
      recurrent: [:lstm, :gru, :xlstm, :reservoir],
      ssm: [:mamba, :mamba_ssd, :mamba_cumsum, :mamba_hillis_steele, :s5, :gated_ssm, :jamba, :zamba],
      attention: [:attention, :sliding_window, :retnet, :rwkv, :gla, :hgrn, :griffin],
      generative: [:diffusion, :flow_matching, :vae, :vq_vae, :gan, :normalizing_flow],
      graph: [:gcn, :gat],
      sets: [:deep_sets, :pointnet],
      energy: [:ebm, :hopfield],
      probabilistic: [:bayesian, :mc_dropout],
      memory: [:ntm],
      meta: [:moe, :hypernetwork, :capsule],
      liquid: [:liquid],
      neuromorphic: [:snn]
    }
  end

  @doc """
  Build an architecture by name.

  ## Parameters
    - `name` - Architecture name (see `list_architectures/0`)
    - `opts` - Architecture-specific options (at minimum `:embed_size` or `:input_size`)

  ## Examples

      model = Edifice.build(:mamba, embed_size: 256, hidden_size: 512, num_layers: 4)
      model = Edifice.build(:mlp, input_size: 256, hidden_sizes: [512, 256])
      model = Edifice.build(:lstm, embed_size: 256, hidden_size: 512)

  ## Returns
    An Axon model.
  """
  @spec build(atom(), keyword()) :: Axon.t()
  def build(name, opts \\ []) do
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
      {:ok, {module, _}} -> module
      {:ok, module} -> module
      :error ->
        available = list_architectures() |> Enum.join(", ")
        raise ArgumentError, "Unknown architecture #{inspect(name)}. Available: #{available}"
    end
  end
end
