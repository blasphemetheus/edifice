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
      {:stream_data, "~> 1.1", only: [:dev, :test]}
    ]
  end

  defp docs do
    [
      main: "Edifice",
      extras: ["README.md", "CHANGELOG.md", "LICENSE"],
      groups_for_modules: [
        "Feedforward": [
          Edifice.Feedforward.MLP,
          Edifice.Feedforward.KAN
        ],
        "Convolutional": [
          Edifice.Convolutional.Conv,
          Edifice.Convolutional.ResNet,
          Edifice.Convolutional.DenseNet,
          Edifice.Convolutional.TCN
        ],
        "Recurrent": [
          Edifice.Recurrent,
          Edifice.Recurrent.XLSTM,
          Edifice.Recurrent.Reservoir
        ],
        "State Space Models": [
          Edifice.SSM.Mamba,
          Edifice.SSM.MambaSSD,
          Edifice.SSM.MambaCumsum,
          Edifice.SSM.MambaHillisSteele,
          Edifice.SSM.S5,
          Edifice.SSM.GatedSSM,
          Edifice.SSM.Hybrid,
          Edifice.SSM.Zamba,
          Edifice.SSM.HybridBuilder,
          Edifice.SSM.Common
        ],
        "Attention": [
          Edifice.Attention.MultiHead,
          Edifice.Attention.RetNet,
          Edifice.Attention.RWKV,
          Edifice.Attention.GLA,
          Edifice.Attention.HGRN,
          Edifice.Attention.Griffin
        ],
        "Generative": [
          Edifice.Generative.VAE,
          Edifice.Generative.VQVAE,
          Edifice.Generative.GAN,
          Edifice.Generative.Diffusion,
          Edifice.Generative.FlowMatching,
          Edifice.Generative.NormalizingFlow
        ],
        "Graph": [
          Edifice.Graph.GCN,
          Edifice.Graph.GAT,
          Edifice.Graph.MessagePassing
        ],
        "Sets": [
          Edifice.Sets.DeepSets,
          Edifice.Sets.PointNet
        ],
        "Energy": [
          Edifice.Energy.EBM,
          Edifice.Energy.Hopfield
        ],
        "Probabilistic": [
          Edifice.Probabilistic.Bayesian,
          Edifice.Probabilistic.MCDropout
        ],
        "Memory": [
          Edifice.Memory.NTM,
          Edifice.Memory.MemoryNetwork
        ],
        "Meta": [
          Edifice.Meta.MoE,
          Edifice.Meta.Hypernetwork,
          Edifice.Meta.Capsule
        ],
        "Liquid": [
          Edifice.Liquid
        ],
        "Neuromorphic": [
          Edifice.Neuromorphic.SNN
        ],
        "Utilities": [
          Edifice.Utils.FusedOps,
          Edifice.Utils.ODESolver
        ]
      ]
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp aliases do
    [
      setup: ["deps.get"]
    ]
  end
end
