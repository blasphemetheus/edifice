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
  Input [batch, seq_len, embed_dim]
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
  | Token mixing | O(N^2) | O(N^2)* |
  | Parameters | Q,K,V weights | None (DFT) |
  | Training speed | Baseline | ~7x faster |
  | Quality | Baseline | 92-97% of BERT |

  *Note: We use real-valued DFT matrix multiply instead of Nx.fft because
  EXLA's autodiff through complex FFT outputs triggers Nx.less/2 errors
  in LayerNorm's backward pass. For typical seq_len (30-128) and hidden_size
  (256-512), the O(N^2) matrix multiply is negligible vs the FFN layers.

  ## Usage

      model = FNet.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        dropout: 0.1
      )

  ## References
  - Paper: "FNet: Mixing Tokens with Fourier Transforms" (Lee-Thorp et al., Google 2021)
  """

  alias Edifice.Blocks.{ModelBuilder, TransformerBlock}

  # Default hyperparameters
  @default_hidden_size 256
  @default_num_layers 4
  @default_dropout 0.1

  @doc """
  Build an FNet model for sequence processing.

  ## Options

    - `:embed_dim` - Size of input embedding per timestep (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of FNet blocks (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:window_size` - Expected sequence length for JIT optimization (default: 60)

  ## Returns

    An Axon model that outputs [batch, hidden_size] from the last position.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    # Precompute real DFT matrices at build time (avoids complex numbers
    # that break EXLA's backward pass through Nx.fft)
    dft_seq_tensor = dft_real_matrix(seq_len)
    dft_hidden_tensor = dft_real_matrix(hidden_size)

    ModelBuilder.build_sequence_model(
      Keyword.merge(opts,
        hidden_size: hidden_size,
        num_layers: num_layers,
        block_builder: fn input, block_opts ->
          name = "fnet_block_#{block_opts[:layer_idx]}"

          # Create Axon constants for the DFT matrices (shared tensors, unique names)
          dft_seq = Axon.constant(dft_seq_tensor, name: "#{name}_dft_seq")
          dft_hidden = Axon.constant(dft_hidden_tensor, name: "#{name}_dft_hidden")

          attn_fn = fn x, attn_name ->
            Axon.layer(
              fn input_t, dft_s, dft_h, _opts ->
                fourier_mixing_real(input_t, dft_s, dft_h)
              end,
              [x, dft_seq, dft_hidden],
              name: "#{attn_name}_fft_mix"
            )
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
  Apply Fourier mixing using real-valued DFT matrix multiply.

  Computes Real(FFT2(x)) along both sequence and feature axes using
  precomputed cosine DFT matrices. This avoids Nx.fft entirely, preventing
  complex number issues in EXLA's backward pass.

  ## Parameters
    - `tensor` - Input tensor [batch, seq_len, hidden_dim]
    - `dft_seq` - Precomputed DFT matrix [seq_len, seq_len]
    - `dft_hidden` - Precomputed DFT matrix [hidden_dim, hidden_dim]

  ## Returns
    DFT-mixed tensor [batch, seq_len, hidden_dim]
  """
  @spec fourier_mixing_real(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def fourier_mixing_real(tensor, dft_seq, dft_hidden) do
    # Paper: y = Real(FFT2(x)) — 2D DFT along both sequence and hidden axes.
    # For real input x: Real(FFT(x))_k = sum_n x_n * cos(2π*n*k/N)
    # This is equivalent to multiplying by the real DFT matrix (cosine basis).
    # The DFT matrix is symmetric: cos(2π*n*k/N) = cos(2π*k*n/N).
    {batch, seq_len, hidden_dim} = Nx.shape(tensor)

    # Mix along sequence axis (axis 1):
    # Reshape to [batch*hidden, seq] for matmul with [seq, seq] DFT matrix
    x = Nx.transpose(tensor, axes: [0, 2, 1])
    x = Nx.reshape(x, {batch * hidden_dim, seq_len})
    x = Nx.dot(x, dft_seq)
    x = Nx.reshape(x, {batch, hidden_dim, seq_len})
    x = Nx.transpose(x, axes: [0, 2, 1])

    # Mix along hidden axis (axis 2):
    # Reshape to [batch*seq, hidden] for matmul with [hidden, hidden] DFT matrix
    x = Nx.reshape(x, {batch * seq_len, hidden_dim})
    x = Nx.dot(x, dft_hidden)
    Nx.reshape(x, {batch, seq_len, hidden_dim})
  end

  @doc """
  Build a real-valued DFT matrix: DFT[k, n] = cos(2π * k * n / N).

  For real inputs, Real(FFT(x)) = x @ DFT_real, avoiding complex arithmetic.
  """
  @spec dft_real_matrix(pos_integer()) :: Nx.Tensor.t()
  def dft_real_matrix(n) do
    row = Nx.iota({n, 1}, type: :f32)
    col = Nx.iota({1, n}, type: :f32)
    Nx.cos(Nx.multiply(Nx.multiply(row, col), 2.0 * :math.pi() / n))
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
    embed_dim = Keyword.get(opts, :embed_dim, 287)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)

    inner_size = hidden_size * 4

    # Per layer (no attention params - FFT is parameter-free):
    # FFN: up + down
    ffn_params =
      hidden_size * inner_size +
        inner_size * hidden_size

    per_layer = ffn_params

    input_proj = if embed_dim != hidden_size, do: embed_dim * hidden_size, else: 0

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
