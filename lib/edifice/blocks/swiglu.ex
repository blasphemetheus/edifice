defmodule Edifice.Blocks.SwiGLU do
  @moduledoc """
  SwiGLU / GeGLU / ReGLU gated feed-forward networks.

  Gated Linear Units with various activation functions, as used in the
  feed-forward blocks of modern transformers (LLaMA, PaLM, Mistral).
  The gating mechanism provides better gradient flow and expressiveness
  compared to standard dense + activation.

  ## Formula

      SwiGLU(x) = (xW1 * SiLU(xV)) W2
      GeGLU(x)  = (xW1 * GELU(xV)) W2
      ReGLU(x)  = (xW1 * ReLU(xV)) W2

  ## Architecture

  ```
  Input [batch, ..., dim]
        |
        +-------+-------+
        |               |
     Dense W1       Dense V (gate)
        |               |
        |          Activation (SiLU/GELU/ReLU)
        |               |
        +----> Multiply <+
                  |
               Dense W2
                  |
  Output [batch, ..., dim]
  ```

  ## Usage

      ffn = SwiGLU.layer(input, hidden_size: 256, inner_size: 1024)

  ## References
  - "GLU Variants Improve Transformer" (Shazeer, 2020)
  - https://arxiv.org/abs/2002.05202
  """

  require Axon

  @default_expansion_factor 2.667

  @doc """
  Build a SwiGLU feed-forward block as an Axon layer.

  ## Options
    - `:hidden_size` - Input/output dimension (required)
    - `:inner_size` - Intermediate dimension (default: hidden_size * 2.667, rounded to multiple of 8)
    - `:activation` - Gate activation: :silu, :gelu, :relu (default: :silu)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:name` - Layer name prefix (default: "swiglu")
  """
  @spec layer(Axon.t(), keyword()) :: Axon.t()
  def layer(input, opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)

    inner_size =
      Keyword.get_lazy(opts, :inner_size, fn ->
        raw = round(hidden_size * @default_expansion_factor)
        # Round up to multiple of 8 for tensor core alignment
        padding = rem(8 - rem(raw, 8), 8)
        raw + padding
      end)

    activation = Keyword.get(opts, :activation, :silu)
    dropout = Keyword.get(opts, :dropout, 0.0)
    name = Keyword.get(opts, :name, "swiglu")

    # Gate projection
    gate_proj = Axon.dense(input, inner_size, name: "#{name}_gate")
    gate = Axon.activation(gate_proj, activation, name: "#{name}_gate_act")

    # Up projection
    up_proj = Axon.dense(input, inner_size, name: "#{name}_up")

    # Gated multiplication
    gated = Axon.multiply(gate, up_proj, name: "#{name}_mul")

    # Down projection
    output = Axon.dense(gated, hidden_size, name: "#{name}_down")

    if dropout > 0 do
      Axon.dropout(output, rate: dropout, name: "#{name}_dropout")
    else
      output
    end
  end
end
