defmodule Edifice.Generative.MAGVIT2 do
  @moduledoc """
  MAGVIT-v2 — Lookup-Free Quantization for visual token generation.

  <!-- verified: true, date: 2026-02-28 -->

  Replaces traditional VQ codebook lookup with binary quantization: each
  latent dimension is independently thresholded to {-1, +1}, yielding 2^D
  codewords for D dimensions. This eliminates codebook collapse and scales
  to very large codebook sizes (e.g. 2^18 = 262,144).

  ## Architecture

  ```
  Input [batch, input_size]
        |
  Encoder: MLP stack -> z_e [batch, latent_dim]
        |
  LFQ: sign(z_e) -> z_q in {-1, +1}^D   (straight-through gradient)
        |
  Decoder: MLP stack -> reconstruction [batch, input_size]
  ```

  The effective codebook size is `2^latent_dim`. For `latent_dim: 18`, this
  gives 262,144 codes matching the original MAGVIT-v2 paper.

  ## Returns

  `{encoder, decoder}` tuple:
  - Encoder: `[batch, input_size]` -> `[batch, latent_dim]`
  - Decoder: `[batch, latent_dim]` -> `[batch, input_size]`

  ## References

  - Yu et al., "Language Model Beats Diffusion — Tokenizer is Key to
    Visual Generation" (ICLR 2024)
  - https://arxiv.org/abs/2310.05737
  """

  import Nx.Defn

  @default_latent_dim 18
  @default_encoder_sizes [256, 128]
  @default_decoder_sizes [128, 256]
  @default_activation :relu

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:decoder_sizes, [pos_integer()]}
          | {:encoder_sizes, [pos_integer()]}
          | {:input_size, pos_integer()}
          | {:latent_dim, pos_integer()}

  @doc """
  Build MAGVIT-v2 encoder and decoder with lookup-free quantization.

  ## Options

    - `:input_size` - Input feature dimension (required)
    - `:latent_dim` - Number of binary dimensions (default: 18, giving 2^18 codes)
    - `:encoder_sizes` - Encoder hidden layer sizes (default: [256, 128])
    - `:decoder_sizes` - Decoder hidden layer sizes (default: [128, 256])
    - `:activation` - Activation function (default: :relu)

  ## Returns

    `{encoder, decoder}` — two Axon models.
  """
  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    latent_dim = Keyword.get(opts, :latent_dim, @default_latent_dim)
    encoder_sizes = Keyword.get(opts, :encoder_sizes, @default_encoder_sizes)
    decoder_sizes = Keyword.get(opts, :decoder_sizes, @default_decoder_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)

    encoder = build_encoder(input_size, latent_dim, encoder_sizes, activation)
    decoder = build_decoder(latent_dim, input_size, decoder_sizes, activation)

    {encoder, decoder}
  end

  @doc "Get the output size (decoder reconstruction dimension)."
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :input_size)
  end

  @doc """
  Get the effective codebook size for a given latent dimension.

  ## Examples

      iex> Edifice.Generative.MAGVIT2.codebook_size(18)
      262144

      iex> Edifice.Generative.MAGVIT2.codebook_size(14)
      16384
  """
  @spec codebook_size(pos_integer()) :: pos_integer()
  def codebook_size(latent_dim) do
    Integer.pow(2, latent_dim)
  end

  # ===========================================================================
  # Encoder
  # ===========================================================================

  defp build_encoder(input_size, latent_dim, hidden_sizes, activation) do
    input = Axon.input("input", shape: {nil, input_size})

    trunk =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "encoder_dense_#{idx}")
        |> Axon.activation(activation, name: "encoder_act_#{idx}")
      end)

    # Project to latent dim (no activation — continuous pre-quantization output)
    Axon.dense(trunk, latent_dim, name: "encoder_to_latent")
  end

  # ===========================================================================
  # Decoder
  # ===========================================================================

  defp build_decoder(latent_dim, output_size, hidden_sizes, activation) do
    input = Axon.input("quantized", shape: {nil, latent_dim})

    trunk =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(input, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "decoder_dense_#{idx}")
        |> Axon.activation(activation, name: "decoder_act_#{idx}")
      end)

    Axon.dense(trunk, output_size, name: "decoder_output")
  end

  # ===========================================================================
  # Lookup-Free Quantization
  # ===========================================================================

  @doc """
  Lookup-free quantization: binarize each dimension to {-1, +1}.

  Uses straight-through estimator for gradient flow. The quantized vector
  is `sign(z_e)`, and gradients pass through unchanged.

  ## Parameters

    - `z_e` - Encoder output `[batch, latent_dim]`

  ## Returns

    `{z_q, indices}` where:
    - `z_q` - Quantized vectors in {-1, +1}^D `[batch, latent_dim]`
    - `indices` - Integer code indices `[batch]` (each in 0..2^D-1)
  """
  @spec quantize(Nx.Tensor.t()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  defn quantize(z_e) do
    # Binary quantization: sign function
    z_q = Nx.select(Nx.greater_equal(z_e, 0), 1.0, -1.0)

    # Straight-through estimator
    z_q_st = z_e + stop_grad(z_q - z_e)

    # Convert binary codes to integer indices
    # bits: 1 where z_e >= 0, 0 otherwise
    bits = Nx.select(Nx.greater_equal(z_e, 0), 1, 0)
    {_batch, d} = Nx.shape(bits)
    # powers of 2: [2^(D-1), 2^(D-2), ..., 2^0]
    powers = Nx.pow(2, Nx.iota({d}) |> Nx.reverse())
    indices = Nx.dot(bits, powers)

    {z_q_st, indices}
  end

  @doc """
  Entropy loss for LFQ: encourages uniform codebook usage.

  Maximizes the entropy of the marginal distribution over codes in a batch.
  This prevents mode collapse where only a few codes are used.

  ## Parameters

    - `z_e` - Pre-quantization encoder outputs `[batch, latent_dim]`

  ## Returns

    Negative entropy (to be minimized).
  """
  @spec entropy_loss(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn entropy_loss(z_e) do
    # Per-dimension probability of being positive
    probs = Nx.mean(Nx.sigmoid(z_e), axes: [0])

    # Binary entropy per dimension: -p*log(p) - (1-p)*log(1-p)
    eps = 1.0e-7
    p = Nx.clip(probs, eps, 1.0 - eps)
    per_dim_entropy = Nx.negate(p * Nx.log(p) + (1.0 - p) * Nx.log(1.0 - p))

    # Total entropy (sum across dimensions)
    # We want to maximize entropy, so return negative
    Nx.negate(Nx.sum(per_dim_entropy))
  end

  @doc """
  Commitment loss: encourages encoder outputs to be close to {-1, +1}.

  Penalizes the distance from each latent dimension to its nearest
  quantization level.

  ## Parameters

    - `z_e` - Pre-quantization encoder outputs `[batch, latent_dim]`

  ## Returns

    Commitment loss scalar.
  """
  @spec commitment_loss(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn commitment_loss(z_e) do
    # Distance from each dimension to nearest {-1, +1}
    z_q = Nx.select(Nx.greater_equal(z_e, 0), 1.0, -1.0)
    Nx.mean(Nx.pow(z_e - stop_grad(z_q), 2))
  end

  @doc """
  Combined MAGVIT-v2 loss.

  ## Parameters

    - `reconstruction` - Decoder output `[batch, input_size]`
    - `target` - Original input `[batch, input_size]`
    - `z_e` - Pre-quantization encoder outputs `[batch, latent_dim]`
    - `opts` - Options:
      - `:entropy_weight` - Weight for entropy loss (default: 0.1)
      - `:commitment_weight` - Weight for commitment loss (default: 0.25)

  ## Returns

    Combined loss scalar.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def loss(reconstruction, target, z_e, opts \\ []) do
    ew = Keyword.get(opts, :entropy_weight, 0.1)
    cw = Keyword.get(opts, :commitment_weight, 0.25)
    do_loss(reconstruction, target, z_e, ew, cw)
  end

  defnp do_loss(reconstruction, target, z_e, entropy_weight, commitment_weight) do
    recon = Nx.mean(Nx.pow(reconstruction - target, 2))
    ent = entropy_loss(z_e)
    commit = commitment_loss(z_e)

    recon + entropy_weight * ent + commitment_weight * commit
  end
end
