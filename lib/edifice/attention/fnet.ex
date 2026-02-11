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

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 4
  @default_dropout 0.1
  @default_window_size 60

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
    embed_size = Keyword.fetch!(opts, :embed_size)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Use concrete seq_len for efficient JIT compilation
    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_size]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_size})

    # Project input to hidden dimension if different
    x =
      if embed_size != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack FNet blocks
    x =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_fnet_block(
          acc,
          hidden_size: hidden_size,
          dropout: dropout,
          name: "fnet_block_#{layer_idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      x,
      fn tensor ->
        seq_len_actual = Nx.axis_size(tensor, 1)

        Nx.slice_along_axis(tensor, seq_len_actual - 1, 1, axis: 1)
        |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  @doc """
  Build a single FNet block.

  Each block has:
  1. LayerNorm -> FFT Mixing -> Residual
  2. LayerNorm -> FFN -> Residual
  """
  @spec build_fnet_block(Axon.t(), keyword()) :: Axon.t()
  def build_fnet_block(input, opts) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    name = Keyword.get(opts, :name, "fnet_block")

    # 1. Fourier mixing branch
    fft_normed = Axon.layer_norm(input, name: "#{name}_fft_norm")

    fft_out = Axon.nx(
      fft_normed,
      fn tensor ->
        fourier_mixing(tensor)
      end,
      name: "#{name}_fft_mix"
    )

    # Residual
    after_fft = Axon.add(input, fft_out, name: "#{name}_fft_residual")

    # 2. FFN branch
    ffn_normed = Axon.layer_norm(after_fft, name: "#{name}_ffn_norm")
    ffn_out = build_ffn(ffn_normed, hidden_size, "#{name}_ffn")

    ffn_out =
      if dropout > 0 do
        Axon.dropout(ffn_out, rate: dropout, name: "#{name}_ffn_dropout")
      else
        ffn_out
      end

    Axon.add(after_fft, ffn_out, name: "#{name}_ffn_residual")
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
    # Apply FFT along the sequence axis (axis 1) for each feature independently
    # Nx.fft operates on the last axis, so we need to handle this carefully
    {batch, seq_len, hidden_dim} = Nx.shape(tensor)

    # Transpose to [batch, hidden_dim, seq_len] so FFT runs along seq axis (last axis)
    transposed = Nx.transpose(tensor, axes: [0, 2, 1])

    # Reshape to 2D for FFT: [batch * hidden_dim, seq_len]
    flat = Nx.reshape(transposed, {batch * hidden_dim, seq_len})

    # Apply FFT along sequence dimension and take real part
    fft_result = Nx.fft(flat)
    real_part = Nx.real(fft_result)

    # Reshape back: [batch, hidden_dim, seq_len] -> [batch, seq_len, hidden_dim]
    real_part
    |> Nx.reshape({batch, hidden_dim, seq_len})
    |> Nx.transpose(axes: [0, 2, 1])
  end

  # Feed-forward network
  defp build_ffn(input, hidden_size, name) do
    inner_size = hidden_size * 4

    input
    |> Axon.dense(inner_size, name: "#{name}_up")
    |> Axon.activation(:gelu, name: "#{name}_gelu")
    |> Axon.dense(hidden_size, name: "#{name}_down")
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
