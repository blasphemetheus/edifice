defmodule Edifice.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/blasphemetheus/edifice"

  def project do
    [
      app: :edifice,
      version: @version,
      elixir: "~> 1.18",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      aliases: aliases(),

      # Docs
      name: "Edifice",
      description: "A comprehensive ML architecture library for Elixir/Nx/Axon",
      source_url: @source_url,
      homepage_url: @source_url,
      docs: docs(),
      package: package()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      # ML Core
      {:nx, "~> 0.9"},
      {:axon, "~> 0.7"},
      {:polaris, "~> 0.1"},

      # GPU Backend (optional - users bring their own)
      {:exla, "~> 0.9", optional: true},

      # Dev & Test
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:stream_data, "~> 1.1", only: [:dev, :test]},
      {:benchee, "~> 1.0", only: :dev},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  defp docs do
    [
      main: "Edifice",
      extras: [
        "README.md",
        "CHANGELOG.md",
        "LICENSE",
        # Getting Started
        "guides/ml_foundations.md",
        "guides/core_vocabulary.md",
        "guides/problem_landscape.md",
        "guides/reading_edifice.md",
        "guides/learning_path.md",
        # Architecture Guides
        "guides/state_space_models.md",
        "guides/attention_mechanisms.md",
        "guides/recurrent_networks.md",
        "guides/vision_architectures.md",
        "guides/convolutional_networks.md",
        "guides/contrastive_learning.md",
        "guides/graph_and_set_networks.md",
        "guides/generative_models.md",
        "guides/dynamic_and_continuous.md",
        "guides/building_blocks.md",
        "guides/meta_learning.md",
        "guides/uncertainty_and_memory.md"
      ],
      groups_for_extras: [
        "Getting Started": [
          "guides/ml_foundations.md",
          "guides/core_vocabulary.md",
          "guides/problem_landscape.md",
          "guides/reading_edifice.md",
          "guides/learning_path.md"
        ],
        "Guides: Sequence Processing": [
          "guides/state_space_models.md",
          "guides/attention_mechanisms.md",
          "guides/recurrent_networks.md"
        ],
        "Guides: Representation Learning": [
          "guides/vision_architectures.md",
          "guides/convolutional_networks.md",
          "guides/contrastive_learning.md",
          "guides/graph_and_set_networks.md"
        ],
        "Guides: Generative & Dynamic": [
          "guides/generative_models.md",
          "guides/dynamic_and_continuous.md"
        ],
        "Guides: Composition & Enhancement": [
          "guides/building_blocks.md",
          "guides/meta_learning.md",
          "guides/uncertainty_and_memory.md"
        ]
      ],
      groups_for_modules: [
        Feedforward: [
          Edifice.Feedforward.MLP,
          Edifice.Feedforward.KAN,
          Edifice.Feedforward.TabNet
        ],
        Convolutional: [
          Edifice.Convolutional.Conv,
          Edifice.Convolutional.ResNet,
          Edifice.Convolutional.DenseNet,
          Edifice.Convolutional.TCN,
          Edifice.Convolutional.MobileNet,
          Edifice.Convolutional.EfficientNet
        ],
        Recurrent: [
          Edifice.Recurrent,
          Edifice.Recurrent.XLSTM,
          Edifice.Recurrent.MinGRU,
          Edifice.Recurrent.MinLSTM,
          Edifice.Recurrent.DeltaNet,
          Edifice.Recurrent.TTT,
          Edifice.Recurrent.Titans,
          Edifice.Recurrent.Reservoir
        ],
        "State Space Models": [
          Edifice.SSM.Mamba,
          Edifice.SSM.MambaSSD,
          Edifice.SSM.MambaCumsum,
          Edifice.SSM.MambaHillisSteele,
          Edifice.SSM.S4,
          Edifice.SSM.S4D,
          Edifice.SSM.S5,
          Edifice.SSM.H3,
          Edifice.SSM.Hyena,
          Edifice.SSM.BiMamba,
          Edifice.SSM.GatedSSM,
          Edifice.SSM.Hybrid,
          Edifice.SSM.Zamba,
          Edifice.SSM.HybridBuilder,
          Edifice.SSM.Common
        ],
        Attention: [
          Edifice.Attention.MultiHead,
          Edifice.Attention.GQA,
          Edifice.Attention.Perceiver,
          Edifice.Attention.FNet,
          Edifice.Attention.LinearTransformer,
          Edifice.Attention.Nystromformer,
          Edifice.Attention.Performer,
          Edifice.Attention.RetNet,
          Edifice.Attention.RWKV,
          Edifice.Attention.GLA,
          Edifice.Attention.HGRN,
          Edifice.Attention.Griffin
        ],
        Vision: [
          Edifice.Vision.ViT,
          Edifice.Vision.DeiT,
          Edifice.Vision.SwinTransformer,
          Edifice.Vision.UNet,
          Edifice.Vision.ConvNeXt,
          Edifice.Vision.MLPMixer
        ],
        Generative: [
          Edifice.Generative.VAE,
          Edifice.Generative.VQVAE,
          Edifice.Generative.GAN,
          Edifice.Generative.Diffusion,
          Edifice.Generative.DDIM,
          Edifice.Generative.DiT,
          Edifice.Generative.LatentDiffusion,
          Edifice.Generative.ConsistencyModel,
          Edifice.Generative.ScoreSDE,
          Edifice.Generative.FlowMatching,
          Edifice.Generative.NormalizingFlow
        ],
        Contrastive: [
          Edifice.Contrastive.SimCLR,
          Edifice.Contrastive.BYOL,
          Edifice.Contrastive.BarlowTwins,
          Edifice.Contrastive.MAE,
          Edifice.Contrastive.VICReg
        ],
        Graph: [
          Edifice.Graph.GCN,
          Edifice.Graph.GAT,
          Edifice.Graph.GIN,
          Edifice.Graph.GraphSAGE,
          Edifice.Graph.GraphTransformer,
          Edifice.Graph.PNA,
          Edifice.Graph.SchNet,
          Edifice.Graph.MessagePassing
        ],
        Sets: [
          Edifice.Sets.DeepSets,
          Edifice.Sets.PointNet
        ],
        Energy: [
          Edifice.Energy.EBM,
          Edifice.Energy.Hopfield,
          Edifice.Energy.NeuralODE
        ],
        Probabilistic: [
          Edifice.Probabilistic.Bayesian,
          Edifice.Probabilistic.MCDropout,
          Edifice.Probabilistic.EvidentialNN
        ],
        Memory: [
          Edifice.Memory.NTM,
          Edifice.Memory.MemoryNetwork
        ],
        Meta: [
          Edifice.Meta.MoE,
          Edifice.Meta.SwitchMoE,
          Edifice.Meta.SoftMoE,
          Edifice.Meta.LoRA,
          Edifice.Meta.Adapter,
          Edifice.Meta.Hypernetwork,
          Edifice.Meta.Capsule
        ],
        Liquid: [
          Edifice.Liquid
        ],
        Neuromorphic: [
          Edifice.Neuromorphic.SNN,
          Edifice.Neuromorphic.ANN2SNN
        ],
        "Building Blocks": [
          Edifice.Blocks.RMSNorm,
          Edifice.Blocks.SwiGLU,
          Edifice.Blocks.RoPE,
          Edifice.Blocks.ALiBi,
          Edifice.Blocks.PatchEmbed,
          Edifice.Blocks.SinusoidalPE,
          Edifice.Blocks.AdaptiveNorm,
          Edifice.Blocks.CrossAttention
        ],
        Utilities: [
          Edifice.Utils.FusedOps,
          Edifice.Utils.ODESolver,
          Edifice.Utils.Common
        ]
      ]
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ["lib", "mix.exs", "README.md", "CHANGELOG.md", "LICENSE", "guides"]
    ]
  end

  defp aliases do
    [
      setup: ["deps.get"]
    ]
  end
end
