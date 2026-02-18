defmodule Edifice.Blocks.PatchEmbed do
  @moduledoc """
  Patch Embedding for Vision Transformers.

  Splits images into fixed-size patches and linearly projects each patch
  into an embedding vector. This is the standard input processing for ViT,
  DeiT, MAE, and other vision transformer architectures.

  ## How It Works

  1. Split image into non-overlapping patches of size P x P
  2. Flatten each patch into a vector of size P*P*C
  3. Linear project to embedding dimension

  For a 224x224 image with 16x16 patches: 196 patches, each 768-dim (16*16*3).

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
        v
  +----------------------------------+
  | Split into P x P patches         |
  | (H/P * W/P = num_patches total) |
  +----------------------------------+
        |
        v
  [batch, num_patches, P*P*C]
        |
        v
  +----------------------------------+
  | Linear projection to embed_dim   |
  +----------------------------------+
        |
        v
  [batch, num_patches, embed_dim]
  ```

  ## Usage

      patches = PatchEmbed.layer(image,
        image_size: 224,
        patch_size: 16,
        in_channels: 3,
        embed_dim: 768
      )

  ## References
  - "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
  """

  @doc """
  Build a patch embedding Axon layer.

  ## Options
    - `:image_size` - Input image size (square, default: 224)
    - `:patch_size` - Patch size (square, default: 16)
    - `:in_channels` - Number of input channels (default: 3)
    - `:embed_dim` - Output embedding dimension (required)
    - `:name` - Layer name prefix (default: "patch_embed")
  """
  @spec layer(Axon.t(), keyword()) :: Axon.t()
  def layer(input, opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    patch_size = Keyword.get(opts, :patch_size, 16)
    in_channels = Keyword.get(opts, :in_channels, 3)
    name = Keyword.get(opts, :name, "patch_embed")

    patch_dim = patch_size * patch_size * in_channels

    # Reshape image into patches and flatten each
    patched =
      Axon.nx(
        input,
        fn tensor ->
          # tensor: [batch, channels, height, width]
          batch = Nx.axis_size(tensor, 0)
          h = Nx.axis_size(tensor, 2)
          w = Nx.axis_size(tensor, 3)
          num_patches_h = div(h, patch_size)
          num_patches_w = div(w, patch_size)
          num_patches = num_patches_h * num_patches_w

          # Reshape to [batch, C, nH, P, nW, P]
          tensor
          |> Nx.reshape(
            {batch, in_channels, num_patches_h, patch_size, num_patches_w, patch_size}
          )
          # Transpose to [batch, nH, nW, C, P, P]
          |> Nx.transpose(axes: [0, 2, 4, 1, 3, 5])
          # Flatten patches: [batch, num_patches, patch_dim]
          |> Nx.reshape({batch, num_patches, patch_dim})
        end,
        name: "#{name}_extract"
      )

    # Linear projection to embed_dim
    Axon.dense(patched, embed_dim, name: "#{name}_proj")
  end

  @doc """
  Calculate the number of patches for given image and patch sizes.
  """
  @spec num_patches(pos_integer(), pos_integer()) :: pos_integer()
  def num_patches(image_size, patch_size) do
    n = div(image_size, patch_size)
    n * n
  end
end
