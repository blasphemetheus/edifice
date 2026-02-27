defmodule Edifice.Blocks.FFN do
  @moduledoc """
  Feed-Forward Network building blocks for transformer architectures.

  Provides standard and gated FFN variants used in the feed-forward sublayer
  of transformer blocks. This module unifies the duplicated `build_ffn/3`
  pattern found across attention architectures.

  ## Variants

  - **Standard**: `dense(hidden * expansion) -> activation -> dropout -> dense(hidden)`
  - **Gated**: Delegates to `SwiGLU.layer/2` for gated linear unit variants

  ## Usage

      # Standard FFN (default in most transformers)
      ffn = FFN.layer(input, hidden_size: 256)

      # With custom expansion factor and activation
      ffn = FFN.layer(input, hidden_size: 256, expansion_factor: 8, activation: :relu)

      # Gated variant (SwiGLU/GeGLU/ReGLU)
      ffn = FFN.gated_layer(input, hidden_size: 256, activation: :silu)

  ## References
  - "Attention Is All You Need" (Vaswani et al., 2017) - original FFN
  - "GLU Variants Improve Transformer" (Shazeer, 2020) - gated variants
  """

  alias Edifice.Blocks.SwiGLU

  @default_expansion_factor 4
  @default_activation :gelu

  @doc """
  Build a standard feed-forward network as an Axon layer.

  ## Options
    - `:hidden_size` - Input/output dimension (required)
    - `:expansion_factor` - Inner dimension multiplier (default: 4)
    - `:inner_size` - Explicit inner dimension (overrides expansion_factor)
    - `:activation` - Activation function (default: :gelu)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:name` - Layer name prefix (default: "ffn")

  ## Examples

      iex> input = Axon.input("x", shape: {nil, 32})
      iex> output = Edifice.Blocks.FFN.layer(input, hidden_size: 32)
      iex> %Axon{} = output
  """
  @spec layer(Axon.t(), keyword()) :: Axon.t()
  def layer(input, opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)

    inner_size =
      Keyword.get_lazy(opts, :inner_size, fn ->
        hidden_size * Keyword.get(opts, :expansion_factor, @default_expansion_factor)
      end)

    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.get(opts, :name, "ffn")

    output =
      input
      |> Axon.dense(inner_size, name: "#{name}_up")
      |> Axon.activation(activation, name: "#{name}_act")
      |> Axon.dense(hidden_size, name: "#{name}_down")

    if dropout > 0 do
      Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
    else
      output
    end
  end

  @doc """
  Build a gated feed-forward network (SwiGLU/GeGLU/ReGLU).

  Delegates to `Edifice.Blocks.SwiGLU.layer/2` with a unified API.

  ## Options
    - `:hidden_size` - Input/output dimension (required)
    - `:inner_size` - Intermediate dimension (default: hidden_size * 2.667)
    - `:activation` - Gate activation: :silu, :gelu, :relu (default: :silu)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:name` - Layer name prefix (default: "gated_ffn")
  """
  @spec gated_layer(Axon.t(), keyword()) :: Axon.t()
  def gated_layer(input, opts \\ []) do
    name = Keyword.get(opts, :name, "gated_ffn")
    SwiGLU.layer(input, Keyword.put(opts, :name, name))
  end
end
