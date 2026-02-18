defmodule Edifice.Blocks.RMSNorm do
  @moduledoc """
  Root Mean Square Layer Normalization.

  Simpler and faster than standard LayerNorm -- normalizes by the RMS of the
  activations without centering (no mean subtraction). Used by LLaMA, Mamba-2,
  Mistral, and most modern transformer variants.

  ## Formula

      RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

  Compared to LayerNorm which computes both mean and variance, RMSNorm only
  computes the RMS, saving ~50% of the normalization compute.

  ## Usage

      # As an Axon layer
      normalized = RMSNorm.layer(input, hidden_size: 256)

  ## References
  - "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
  - https://arxiv.org/abs/1910.07467
  """
  import Nx.Defn

  @doc """
  Build an RMSNorm Axon layer.

  ## Options
    - `:hidden_size` - Feature dimension for the learnable scale (required)
    - `:epsilon` - Numerical stability constant (default: 1.0e-6)
    - `:name` - Layer name prefix (default: "rms_norm")
  """
  @spec layer(Axon.t(), keyword()) :: Axon.t()
  def layer(input, opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    name = Keyword.get(opts, :name, "rms_norm")
    epsilon = Keyword.get(opts, :epsilon, 1.0e-6)

    gamma = Axon.param("#{name}_gamma", {hidden_size}, initializer: :ones)

    Axon.layer(
      &rms_norm_impl/3,
      [input, gamma],
      name: name,
      epsilon: epsilon,
      op_name: :rms_norm
    )
  end

  @doc """
  Compute RMSNorm on a raw tensor.

  ## Parameters
    - `x` - Input tensor [..., hidden_size]
    - `gamma` - Learnable scale [hidden_size]
  """
  defn apply(x, gamma, opts \\ [epsilon: 1.0e-6]) do
    epsilon = opts[:epsilon]
    rms = Nx.sqrt(Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true) |> Nx.add(epsilon))
    Nx.multiply(Nx.divide(x, rms), gamma)
  end

  defp rms_norm_impl(x, gamma, opts) do
    epsilon = opts[:epsilon] || 1.0e-6
    rms = Nx.sqrt(Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true) |> Nx.add(epsilon))
    Nx.multiply(Nx.divide(x, rms), gamma)
  end
end
