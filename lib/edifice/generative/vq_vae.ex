defmodule Edifice.Generative.VQVAE do
  @moduledoc """
  Vector-Quantized Variational Autoencoder (VQ-VAE).

  Instead of learning a continuous latent space like a standard VAE,
  VQ-VAE learns a discrete codebook of embedding vectors. The encoder
  output is quantized to the nearest codebook entry, producing a
  discrete latent representation.

  This avoids posterior collapse (a common VAE failure mode) and
  produces sharper reconstructions since the decoder receives
  high-fidelity codebook vectors rather than noisy samples.

  ## Architecture

  ```
  Input [batch, input_size]
        |
        v
  +-------------------+
  | Encoder           |
  | Dense layers      |
  +-------------------+
        |
        v
  z_e [batch, embedding_dim]    (continuous encoder output)
        |
        v
  +-------------------+
  | Quantize          |
  | nearest codebook  |
  | vector lookup     |
  +-------------------+
        |
        v
  z_q [batch, embedding_dim]    (discrete quantized vector)
        |
        v
  +-------------------+
  | Decoder           |
  | Dense layers      |
  +-------------------+
        |
        v
  Reconstruction [batch, input_size]
  ```

  ## Training Losses

  VQ-VAE training uses three loss components:
  1. **Reconstruction loss**: MSE between input and reconstruction
  2. **Codebook loss**: `||sg(z_e) - e||^2` - moves codebook vectors toward encoder outputs
  3. **Commitment loss**: `||z_e - sg(e)||^2` - prevents encoder from fluctuating too far from codebook

  The straight-through estimator passes gradients from decoder directly to encoder,
  bypassing the non-differentiable quantization step.

  ## Usage

      # Build full VQ-VAE
      {encoder, decoder} = VQVAE.build(input_size: 784, embedding_dim: 64, num_embeddings: 512)

      # Quantize encoder output against codebook
      {z_q, indices} = VQVAE.quantize(z_e, codebook)

      # Training losses
      commit = VQVAE.commitment_loss(z_e, z_q)
      cb = VQVAE.codebook_loss(z_e, z_q)
  """

  require Axon
  import Nx.Defn

  @default_embedding_dim 64
  @default_encoder_sizes [256, 128]
  @default_decoder_sizes [128, 256]
  @default_activation :relu

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a complete VQ-VAE (encoder + decoder).

  The encoder maps inputs to continuous vectors of size `embedding_dim`.
  The decoder reconstructs from quantized codebook vectors of the same size.
  Quantization is performed externally via `quantize/2` during training.

  ## Options
    - `:input_size` - Input feature dimension (required)
    - `:embedding_dim` - Codebook vector dimension (default: 64)
    - `:num_embeddings` - Number of codebook entries (default: 512)
    - `:encoder_sizes` - Hidden layer sizes for encoder (default: [256, 128])
    - `:decoder_sizes` - Hidden layer sizes for decoder (default: [128, 256])
    - `:activation` - Activation function (default: :relu)

  ## Returns
    `{encoder, decoder}` - Tuple of Axon models.
    - Encoder: `[batch, input_size]` -> `[batch, embedding_dim]`
    - Decoder: `[batch, embedding_dim]` -> `[batch, input_size]`
  """
  @spec build(keyword()) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)

    encoder = build_encoder(opts)
    decoder = build_decoder(Keyword.put(opts, :output_size, input_size))

    {encoder, decoder}
  end

  @doc """
  Build the encoder network.

  Maps input to a continuous vector in the embedding space. This output
  is then quantized to the nearest codebook vector during training.

  ## Options
    - `:input_size` - Input feature dimension (required)
    - `:embedding_dim` - Output dimension, must match codebook (default: 64)
    - `:encoder_sizes` - Hidden layer sizes (default: [256, 128])
    - `:activation` - Activation function (default: :relu)

  ## Returns
    An Axon model: `[batch, input_size]` -> `[batch, embedding_dim]`.
  """
  @spec build_encoder(keyword()) :: Axon.t()
  def build_encoder(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    embedding_dim = Keyword.get(opts, :embedding_dim, @default_embedding_dim)
    encoder_sizes = Keyword.get(opts, :encoder_sizes, @default_encoder_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)

    input = Axon.input("input", shape: {nil, input_size})

    # Encoder trunk with activation
    trunk =
      Enum.with_index(encoder_sizes)
      |> Enum.reduce(input, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "encoder_dense_#{idx}")
        |> Axon.activation(activation, name: "encoder_act_#{idx}")
      end)

    # Project to embedding dimension (no activation - continuous output)
    Axon.dense(trunk, embedding_dim, name: "encoder_to_embedding")
  end

  @doc """
  Build the decoder network.

  Maps quantized codebook vectors back to the input space.

  ## Options
    - `:input_size` or `:output_size` - Reconstruction dimension (required)
    - `:embedding_dim` - Input dimension from codebook (default: 64)
    - `:decoder_sizes` - Hidden layer sizes (default: [128, 256])
    - `:activation` - Activation function (default: :relu)

  ## Returns
    An Axon model: `[batch, embedding_dim]` -> `[batch, output_size]`.
  """
  @spec build_decoder(keyword()) :: Axon.t()
  def build_decoder(opts \\ []) do
    output_size = Keyword.get(opts, :output_size) || Keyword.fetch!(opts, :input_size)
    embedding_dim = Keyword.get(opts, :embedding_dim, @default_embedding_dim)
    decoder_sizes = Keyword.get(opts, :decoder_sizes, @default_decoder_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)

    latent_input = Axon.input("quantized", shape: {nil, embedding_dim})

    # Decoder trunk
    trunk =
      Enum.with_index(decoder_sizes)
      |> Enum.reduce(latent_input, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "decoder_dense_#{idx}")
        |> Axon.activation(activation, name: "decoder_act_#{idx}")
      end)

    # Output reconstruction
    Axon.dense(trunk, output_size, name: "decoder_output")
  end

  # ============================================================================
  # Quantization
  # ============================================================================

  @doc """
  Quantize continuous encoder outputs to nearest codebook vectors.

  For each encoder output vector z_e, finds the nearest codebook
  entry by L2 distance and returns the quantized vector along with
  the codebook indices.

  Uses the straight-through estimator: gradients flow from z_q to z_e
  directly, bypassing the non-differentiable argmin.

  ## Parameters
    - `z_e` - Encoder output `[batch, embedding_dim]`
    - `codebook` - Codebook embeddings `[num_embeddings, embedding_dim]`

  ## Returns
    `{z_q, indices}` where:
    - `z_q` - Quantized vectors `[batch, embedding_dim]` (with straight-through gradient)
    - `indices` - Codebook indices `[batch]`
  """
  @spec quantize(Nx.Tensor.t(), Nx.Tensor.t()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  defn quantize(z_e, codebook) do
    # Compute L2 distances: ||z_e - e_j||^2
    # Expand: ||z_e||^2 - 2*z_e*e^T + ||e||^2
    # z_e: [batch, D], codebook: [K, D]

    # [batch, 1]
    z_e_sq = Nx.sum(Nx.pow(z_e, 2), axes: [-1], keep_axes: true)
    # [K, 1]
    cb_sq = Nx.sum(Nx.pow(codebook, 2), axes: [-1], keep_axes: true)
    # [batch, K]
    cross = Nx.dot(z_e, Nx.transpose(codebook))

    # distances: [batch, K]
    distances = z_e_sq - 2.0 * cross + Nx.transpose(cb_sq)

    # Find nearest codebook entry
    # [batch]
    indices = Nx.argmin(distances, axis: -1)

    # Gather quantized vectors
    # [batch, D]
    z_q = Nx.take(codebook, indices, axis: 0)

    # Straight-through estimator: gradient of z_q flows to z_e
    # z_q_st = z_e + stop_grad(z_q - z_e)
    z_q_st = z_e + stop_grad(z_q - z_e)

    {z_q_st, indices}
  end

  # ============================================================================
  # Loss Functions
  # ============================================================================

  @doc """
  Commitment loss: encourages encoder outputs to stay close to codebook vectors.

  Computed as `mean(||z_e - sg(z_q)||^2)` where `sg` is stop_gradient.
  This prevents the encoder output from growing unboundedly away from
  the codebook entries.

  ## Parameters
    - `z_e` - Encoder output `[batch, embedding_dim]`
    - `z_q` - Quantized vectors `[batch, embedding_dim]` (will be detached)

  ## Returns
    Commitment loss scalar.
  """
  @spec commitment_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn commitment_loss(z_e, z_q) do
    # ||z_e - sg(z_q)||^2 - gradients only flow to encoder
    diff = z_e - stop_grad(z_q)
    Nx.mean(Nx.pow(diff, 2))
  end

  @doc """
  Codebook loss: moves codebook vectors toward encoder outputs.

  Computed as `mean(||sg(z_e) - z_q||^2)` where `sg` is stop_gradient.
  This is equivalent to an EMA update of the codebook embeddings.

  ## Parameters
    - `z_e` - Encoder output `[batch, embedding_dim]` (will be detached)
    - `z_q` - Quantized vectors `[batch, embedding_dim]`

  ## Returns
    Codebook loss scalar.
  """
  @spec codebook_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn codebook_loss(z_e, z_q) do
    # ||sg(z_e) - z_q||^2 - gradients only flow to codebook
    diff = stop_grad(z_e) - z_q
    Nx.mean(Nx.pow(diff, 2))
  end

  @doc """
  Initialize a codebook with random normal vectors.

  ## Parameters
    - `num_embeddings` - Number of codebook entries
    - `embedding_dim` - Dimension of each codebook vector

  ## Returns
    Codebook tensor `[num_embeddings, embedding_dim]` initialized from N(0, 1).
  """
  @spec init_codebook(pos_integer(), pos_integer()) :: Nx.Tensor.t()
  defn init_codebook(num_embeddings, embedding_dim) do
    key = Nx.Random.key(42)
    {codebook, _key} = Nx.Random.normal(key, shape: {num_embeddings, embedding_dim})
    codebook
  end

  @doc """
  Combined VQ-VAE loss: reconstruction + codebook + commitment.

  ## Parameters
    - `reconstruction` - Decoder output `[batch, input_size]`
    - `target` - Original input `[batch, input_size]`
    - `z_e` - Encoder output `[batch, embedding_dim]`
    - `z_q` - Quantized vectors `[batch, embedding_dim]`
    - `commitment_weight` - Weight for commitment loss (default: 0.25)

  ## Returns
    Combined loss scalar.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def loss(reconstruction, target, z_e, z_q, opts \\ []) do
    commitment_weight = Keyword.get(opts, :commitment_weight, 0.25)
    do_loss(reconstruction, target, z_e, z_q, commitment_weight)
  end

  defnp do_loss(reconstruction, target, z_e, z_q, commitment_weight) do
    recon = Nx.mean(Nx.pow(reconstruction - target, 2))
    cb = codebook_loss(z_e, z_q)
    commit = commitment_loss(z_e, z_q)

    recon + cb + commitment_weight * commit
  end
end
