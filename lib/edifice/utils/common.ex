defmodule Edifice.Utils.Common do
  @moduledoc """
  Common utility functions shared across architecture implementations.

  Provides building blocks for layer construction, shape manipulation,
  and initialization helpers.
  """

  import Nx.Defn

  @doc """
  Align a dimension to a multiple of N (for tensor core efficiency).

  GPU tensor cores work most efficiently with dimensions that are multiples
  of 8 (for FP16/BF16) or 16 (for INT8).

  ## Examples

      iex> Edifice.Utils.Common.align_dim(287, 8)
      288
      iex> Edifice.Utils.Common.align_dim(256, 8)
      256
  """
  @spec align_dim(non_neg_integer(), pos_integer()) :: non_neg_integer()
  def align_dim(dim, alignment \\ 8) do
    remainder = rem(dim, alignment)

    if remainder == 0 do
      dim
    else
      dim + (alignment - remainder)
    end
  end

  @doc """
  Extract the last timestep from a sequence tensor.

  Takes `[batch, seq_len, features]` and returns `[batch, features]`.
  """
  @spec last_timestep(Nx.Tensor.t()) :: Nx.Tensor.t()
  def last_timestep(tensor) do
    seq_len = Nx.axis_size(tensor, 1)

    Nx.slice_along_axis(tensor, seq_len - 1, 1, axis: 1)
    |> Nx.squeeze(axes: [1])
  end

  @doc """
  Build an Axon layer that extracts the last timestep.
  """
  @spec last_timestep_layer(Axon.t(), keyword()) :: Axon.t()
  def last_timestep_layer(input, opts \\ []) do
    name = Keyword.get(opts, :name, "last_timestep")

    Axon.nx(input, &last_timestep/1, name: name)
  end

  @doc """
  Build a feed-forward network block (dense -> activation -> dropout).

  Common pattern used across many architectures.
  """
  @spec ffn_block(Axon.t(), non_neg_integer(), keyword()) :: Axon.t()
  def ffn_block(input, output_size, opts \\ []) do
    activation = Keyword.get(opts, :activation, :relu)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.get(opts, :name, "ffn")
    use_layer_norm = Keyword.get(opts, :layer_norm, false)

    layer = Axon.dense(input, output_size, name: "#{name}_dense")

    layer =
      if use_layer_norm do
        Axon.layer_norm(layer, name: "#{name}_ln")
      else
        layer
      end

    layer = Axon.activation(layer, activation)

    if dropout > 0 do
      Axon.dropout(layer, rate: dropout, name: "#{name}_dropout")
    else
      layer
    end
  end

  @doc """
  Root Mean Square Layer Normalization (RMSNorm).

  Simpler and faster than standard LayerNorm - no mean subtraction.
  Used in LLaMA, Mamba, and other modern architectures.

  rms_norm(x) = x / sqrt(mean(x^2) + eps) * gamma
  """
  defn rms_norm(x, gamma, opts \\ [epsilon: 1.0e-6]) do
    epsilon = opts[:epsilon]

    rms = Nx.sqrt(Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true) |> Nx.add(epsilon))
    Nx.multiply(Nx.divide(x, rms), gamma)
  end

  @doc """
  SiLU (Swish) activation: x * sigmoid(x).
  """
  defn silu(x) do
    Nx.multiply(x, Nx.sigmoid(x))
  end

  @doc """
  GELU activation: x * Phi(x) where Phi is standard normal CDF.
  """
  defn gelu(x) do
    Nx.multiply(x, Nx.multiply(0.5, Nx.add(1.0, Nx.erf(Nx.multiply(x, 0.7071067811865476)))))
  end
end
