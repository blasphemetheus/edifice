defmodule Edifice.Contrastive.MAE do
  @moduledoc """
  MAE - Masked Autoencoder.

  Implements the Masked Autoencoder from "Masked Autoencoders Are Scalable
  Vision Learners" (He et al., CVPR 2022), adapted for 1D sequence data.
  MAE masks a large portion of input patches (tokens) and trains an
  autoencoder to reconstruct the missing patches.

  ## Key Innovations

  - **High masking ratio**: Masking 75% of patches creates a challenging
    pretext task that forces learning of strong representations
  - **Asymmetric encoder-decoder**: Encoder only processes unmasked patches
    (efficient), decoder is lightweight and processes all patches
  - **Mask tokens**: Learned placeholder tokens are added for masked positions
    before the decoder

  ## Architecture

  ```
  Input Patches [batch, num_patches, input_dim]
        |
        v
  [Random Masking] (keep 25%)
        |
        v
  +------------------+
  |     Encoder      |  (processes only unmasked patches)
  | (deeper, wider)  |
  +------------------+
        |
        v
  [Add Mask Tokens]  (insert learnable tokens at masked positions)
        |
        v
  +------------------+
  |     Decoder      |  (processes all patches)
  | (shallow, narrow)|
  +------------------+
        |
        v
  [Reconstruction]   MSE loss on masked patches only
  ```

  ## Usage

      # Build encoder and decoder
      {encoder, decoder} = MAE.build(
        input_dim: 64,
        embed_dim: 256,
        decoder_dim: 128,
        mask_ratio: 0.75,
        num_encoder_layers: 4,
        num_decoder_layers: 2
      )

      # For pretraining: use both encoder + decoder with masking
      # For downstream: use encoder only (discard decoder)

  ## References
  - Paper: https://arxiv.org/abs/2111.06377
  """

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default encoder embedding dimension"
  @spec default_embed_dim() :: pos_integer()
  def default_embed_dim, do: 256

  @doc "Default decoder dimension"
  @spec default_decoder_dim() :: pos_integer()
  def default_decoder_dim, do: 128

  @doc "Default masking ratio"
  @spec default_mask_ratio() :: float()
  def default_mask_ratio, do: 0.75

  @doc "Default number of encoder layers"
  @spec default_num_encoder_layers() :: pos_integer()
  def default_num_encoder_layers, do: 4

  @doc "Default number of decoder layers"
  @spec default_num_decoder_layers() :: pos_integer()
  def default_num_decoder_layers, do: 2

  @doc "Default feedforward expansion factor"
  @spec default_expand_factor() :: pos_integer()
  def default_expand_factor, do: 2

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build the MAE encoder and decoder.

  ## Options
    - `:input_dim` - Dimension of each input patch/token (required)
    - `:embed_dim` - Encoder embedding dimension (default: 256)
    - `:decoder_dim` - Decoder hidden dimension (default: 128)
    - `:mask_ratio` - Fraction of patches to mask (default: 0.75)
    - `:num_encoder_layers` - Number of encoder layers (default: 4)
    - `:num_decoder_layers` - Number of decoder layers (default: 2)
    - `:num_patches` - Number of input patches/tokens (default: nil for dynamic)

  ## Returns
    `{encoder, decoder}` tuple of Axon models.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:input_dim, pos_integer()}
          | {:embed_dim, pos_integer()}
          | {:num_encoder_layers, pos_integer()}
          | {:num_decoder_layers, pos_integer()}
          | {:decoder_dim, pos_integer()}
          | {:expand_factor, pos_integer()}
          | {:num_patches, pos_integer() | nil}
          | {:mask_ratio, float()}

  @spec build([build_opt()]) :: {Axon.t(), Axon.t()}
  def build(opts \\ []) do
    encoder = build_encoder(opts)
    decoder = build_decoder(opts)
    {encoder, decoder}
  end

  @doc """
  Build the MAE encoder.

  The encoder processes only unmasked patches. In the full MAE pipeline,
  masking is applied externally before feeding to the encoder.

  ## Options
    - `:input_dim` - Dimension of each input patch (required)
    - `:embed_dim` - Encoder embedding dimension (default: 256)
    - `:num_encoder_layers` - Number of encoder layers (default: 4)
    - `:num_patches` - Sequence length (default: nil)

  ## Returns
    An Axon model: [batch, num_visible, input_dim] -> [batch, num_visible, embed_dim]
  """
  @spec build_encoder(keyword()) :: Axon.t()
  def build_encoder(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    embed_dim = Keyword.get(opts, :embed_dim, default_embed_dim())
    num_layers = Keyword.get(opts, :num_encoder_layers, default_num_encoder_layers())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())
    num_patches = Keyword.get(opts, :num_patches, nil)

    # Input: [batch, num_visible_patches, input_dim]
    input = Axon.input("visible_patches", shape: {nil, num_patches, input_dim})

    # Patch embedding: project to embed_dim
    x = Axon.dense(input, embed_dim, name: "encoder_patch_embed")

    # Encoder layers (simplified transformer blocks without self-attention
    # for Axon compatibility - uses MLP mixer style)
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_encoder_block(acc, embed_dim, expand_factor, "encoder_block_#{layer_idx}")
      end)

    Axon.layer_norm(output, name: "encoder_final_norm")
  end

  @doc """
  Build the MAE decoder.

  The decoder takes encoder output (with mask tokens inserted at masked
  positions) and reconstructs all patches.

  ## Options
    - `:input_dim` - Original patch dimension for reconstruction target (required)
    - `:embed_dim` - Encoder output dimension / decoder input (default: 256)
    - `:decoder_dim` - Decoder hidden dimension (default: 128)
    - `:num_decoder_layers` - Number of decoder layers (default: 2)
    - `:num_patches` - Total number of patches (default: nil)

  ## Returns
    An Axon model: [batch, num_patches, embed_dim] -> [batch, num_patches, input_dim]
  """
  @spec build_decoder(keyword()) :: Axon.t()
  def build_decoder(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    embed_dim = Keyword.get(opts, :embed_dim, default_embed_dim())
    decoder_dim = Keyword.get(opts, :decoder_dim, default_decoder_dim())
    num_layers = Keyword.get(opts, :num_decoder_layers, default_num_decoder_layers())
    expand_factor = Keyword.get(opts, :expand_factor, default_expand_factor())
    num_patches = Keyword.get(opts, :num_patches, nil)

    # Input: [batch, num_patches, embed_dim] (includes mask tokens)
    input = Axon.input("decoder_input", shape: {nil, num_patches, embed_dim})

    # Project from encoder dim to decoder dim
    x = Axon.dense(input, decoder_dim, name: "decoder_embed")

    # Decoder layers
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        build_encoder_block(acc, decoder_dim, expand_factor, "decoder_block_#{layer_idx}")
      end)

    output = Axon.layer_norm(output, name: "decoder_final_norm")

    # Reconstruction head: project back to input_dim
    Axon.dense(output, input_dim, name: "decoder_reconstruction")
  end

  # ============================================================================
  # Encoder/Decoder Blocks
  # ============================================================================

  defp build_encoder_block(input, dim, expand_factor, name) do
    inner_dim = dim * expand_factor

    # Token mixing (cross-token interaction via shared dense)
    normed = Axon.layer_norm(input, name: "#{name}_norm1")

    mixed =
      normed
      |> Axon.dense(dim, name: "#{name}_token_mix")
      |> Axon.activation(:gelu, name: "#{name}_token_gelu")

    after_mix = Axon.add(input, mixed, name: "#{name}_residual1")

    # Channel mixing (per-token feedforward)
    normed2 = Axon.layer_norm(after_mix, name: "#{name}_norm2")

    ff =
      normed2
      |> Axon.dense(inner_dim, name: "#{name}_ff_up")
      |> Axon.activation(:gelu, name: "#{name}_ff_gelu")
      |> Axon.dense(dim, name: "#{name}_ff_down")

    Axon.add(after_mix, ff, name: "#{name}_residual2")
  end

  # ============================================================================
  # Masking Utilities
  # ============================================================================

  @doc """
  Generate a random mask for input patches.

  Returns a tuple of `{visible_indices, masked_indices}` for a given
  number of patches and mask ratio.

  ## Parameters
    - `num_patches` - Total number of patches
    - `mask_ratio` - Fraction to mask (default: 0.75)

  ## Returns
    `{visible_indices, masked_indices}` tensors.
  """
  @spec generate_mask(non_neg_integer(), float()) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def generate_mask(num_patches, mask_ratio \\ default_mask_ratio()) do
    num_masked = round(num_patches * mask_ratio)
    num_visible = num_patches - num_masked

    # Random permutation via argsort of random values
    key = Nx.Random.key(System.system_time(:microsecond))
    {rand_vals, _key} = Nx.Random.uniform(key, shape: {num_patches})
    indices = Nx.argsort(rand_vals)

    visible = Nx.slice(indices, [0], [num_visible])
    masked = Nx.slice(indices, [num_visible], [num_masked])

    {visible, masked}
  end

  @doc """
  Compute reconstruction loss on masked patches only.

  ## Parameters
    - `reconstructed` - Decoder output: [batch, num_patches, input_dim]
    - `original` - Original input: [batch, num_patches, input_dim]
    - `masked_indices` - Indices of masked patches: [num_masked]

  ## Returns
    Scalar MSE loss over masked patches.
  """
  @spec reconstruction_loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def reconstruction_loss(reconstructed, original, masked_indices) do
    # Gather masked patches from both
    recon_masked = Nx.take(reconstructed, masked_indices, axis: 1)
    orig_masked = Nx.take(original, masked_indices, axis: 1)

    # MSE loss
    Nx.mean(Nx.pow(Nx.subtract(recon_masked, orig_masked), 2))
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of the MAE encoder.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :embed_dim, default_embed_dim())
  end
end
