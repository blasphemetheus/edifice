defmodule Edifice.Attention.FNet do
  @moduledoc """
  FNet: Replacing Attention with Fourier Transform.

  FNet replaces the self-attention sublayer in Transformers with an
  unparameterized Fourier Transform, achieving O(N log N) token mixing
  with no learnable attention parameters.

  ## Key Innovation: FFT Mixing

  Instead of computing attention weights, FNet applies FFT along the
  sequence axis to mix token information. This is parameter-free and
  achieves surprisingly competitive performance:

  ```
  Standard Transformer:  LayerNorm -> Self-Attention -> Residual
  FNet:                  LayerNorm -> FFT Mixing     -> Residual
  ```

  The FFT provides global token mixing (every token interacts with
  every other token through frequency-domain multiplication).

  ## Architecture

  ```
  Input [batch, seq_len, embed_size]
        |
        v
  +-------------------------------------+
  |       FNet Block                     |
  |                                      |
  |  LayerNorm                           |
  |    -> FFT along seq axis             |
  |    -> Take real part                 |
  |  -> Residual                         |
  |                                      |
  |  LayerNorm                           |
  |    -> Dense(hidden * 4)              |
  |    -> GeLU                           |
  |    -> Dense(hidden)                  |
  |  -> Residual                         |
  +-------------------------------------+
        | (repeat for num_layers)
        v
  Last timestep -> [batch, hidden_size]
  ```

  ## Complexity

  | Component | Transformer | FNet |
  |-----------|------------|------|
  | Token mixing | O(N^2) | O(N log N) |
  | Parameters | Q,K,V weights | None (FFT) |
  | Training speed | Baseline | ~7x faster |
  | Quality | Baseline | 92-97% of BERT |

  ## Usage

      model = FNet.build(
        embed_size: 287,
        hidden_size: 256,
        num_layers: 4,
        dropout: 0.1
      )

  ## References
  - Paper: "FNet: Mixing Tokens with Fourier Transforms" (Lee-Thorp et al., Google 2021)
  """

  require Axon

  alias Edifice.Blocks.{TransformerBlock, ModelBuilder}

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 4
  @default_dropout 0.1

  @doc """
  Build an FNet model for sequence processing.

  ## Options

    - `:embed_size` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of FNet blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          name = "fnet_block_#{block_opts[:layer_idx]}"

          attn_fn = fn x, attn_name ->
            Axon.nx(x, &fourier_mixing/1, name: "#{attn_name}_fft_mix")
          end

          TransformerBlock.layer(input,
            attention_fn: attn_fn,
            hidden_size: hidden_size,
            dropout: dropout,
            name: name
          )
        end
      )
    )
  end

  @doc """
  Apply Fourier mixing to a sequence tensor.

  Applies 2D FFT (along sequence and feature axes) and takes the real part.
  This provides global token mixing without any learnable parameters.

  ## Parameters
    - `tensor` - Input tensor [batch, seq_len, hidden_dim]

  ## Returns
    Real part of FFT [batch, seq_len, hidden_dim]
  """
  @spec fourier_mixing(Nx.Tensor.t()) :: Nx.Tensor.t()
  def fourier_mixing(tensor) do
    # Paper: y = Real(FFT2(x)) — 2D FFT along both sequence and hidden axes.
    # Nx.fft operates on the last axis, so we apply it twice with transposition.
    {batch, seq_len, hidden_dim} = Nx.shape(tensor)

    # FFT along sequence axis (axis 1):
    # Transpose to [batch, hidden_dim, seq_len] so last axis = seq
    x = Nx.transpose(tensor, axes: [0, 2, 1])
    x = Nx.reshape(x, {batch * hidden_dim, seq_len})
    x = Nx.fft(x)
    x = Nx.reshape(x, {batch, hidden_dim, seq_len})
    x = Nx.transpose(x, axes: [0, 2, 1])

    # FFT along hidden axis (axis 2):
    # Already in [batch, seq_len, hidden_dim] so last axis = hidden
    x = Nx.reshape(x, {batch * seq_len, hidden_dim})
    x = Nx.fft(x)
    x = Nx.reshape(x, {batch, seq_len, hidden_dim})

    # Take real part of the 2D FFT result
    Nx.real(x)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an FNet model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  @doc """
  Calculate approximate parameter count for an FNet model.
  """
  @spec param_count(keyword()) :: non_neg_integer()
  def param_count(opts) do
    embed_size = Keyword.get(opts, :embed_size, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    inner_size = hidden_size * 4

    # Per layer (no attention params - FFT is parameter-free):
    # FFN: up + down
    ffn_params =
      hidden_size * inner_size +
        inner_size * hidden_size

    per_layer = ffn_params

    input_proj = if embed_size != hidden_size, do: embed_size * hidden_size, else: 0

    input_proj + per_layer * num_layers
  end

  @doc """
  Recommended default configuration for sequence processing.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      hidden_size: 256,
      num_layers: 4,
      window_size: 60,
      dropout: 0.1
    ]
  end
end
