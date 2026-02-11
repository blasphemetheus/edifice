defmodule Edifice.Vision.ConvNeXt do
  @moduledoc """
  ConvNeXt - A Modernized ResNet implementation.

  Modernizes the classic ResNet design with techniques borrowed from
  transformers: depthwise-separable convolutions, inverted bottleneck
  blocks, GELU activation, LayerNorm, and fewer activation functions.

  Since Axon works with dense layers on flattened patch representations,
  this implementation uses the ConvNeXt block design pattern (inverted
  bottleneck with LayerNorm and GELU) applied to patch-based features.

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
  +-----v--------------------+
  | Stem (Patchify)           |  Non-overlapping patches -> embed
  +---------------------------+
        |
        v
  [batch, num_patches, dims[0]]
        |
  +-----v--------------------+
  | Stage 1                   |  depths[0] ConvNeXt blocks at dims[0]
  |   LN -> Dense -> GELU    |
  |   -> Dense -> Residual    |
  +---------------------------+
        |
  +-----v--------------------+
  | Downsample                |  LN + Dense to dims[1], merge 2x2
  +---------------------------+
        |
  +-----v--------------------+
  | Stage 2                   |  depths[1] ConvNeXt blocks at dims[1]
  +---------------------------+
        |
        ... (repeat for each stage)
        |
  +-----v--------------------+
  | Stage 4                   |  depths[3] ConvNeXt blocks at dims[3]
  +---------------------------+
        |
  +-----v--------------------+
  | Global Average Pool       |
  +---------------------------+
        |
  +-----v--------------------+
  | LayerNorm                 |
  +---------------------------+
        |
  +-----v--------------------+
  | Optional Classifier       |
  +---------------------------+
  ```

  ## ConvNeXt Block

  ```
  Input
    |
    +----------- Residual ----------+
    |                               |
  LayerNorm                         |
    |                               |
  Dense (dim -> dim * 4)            |
    |                               |
  GELU                              |
    |                               |
  Dense (dim * 4 -> dim)            |
    |                               |
  Scale (learnable layer_scale)     |
    |                               |
    +---------- Add ---------------+
    |
  Output
  ```

  ## Usage

      # ConvNeXt-Tiny
      model = ConvNeXt.build(
        image_size: 224,
        in_channels: 3,
        depths: [3, 3, 9, 3],
        dims: [96, 192, 384, 768],
        num_classes: 1000
      )

      # ConvNeXt-Small
      model = ConvNeXt.build(
        image_size: 224,
        depths: [3, 3, 27, 3],
        dims: [96, 192, 384, 768],
        num_classes: 1000
      )

      # ConvNeXt-Base
      model = ConvNeXt.build(
        image_size: 224,
        depths: [3, 3, 27, 3],
        dims: [128, 256, 512, 1024],
        num_classes: 1000
      )

  ## References

  - "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
  """

  require Axon

  alias Edifice.Blocks.PatchEmbed

  @default_image_size 224
  @default_patch_size 4
  @default_in_channels 3
  @default_depths [3, 3, 9, 3]
  @default_dims [96, 192, 384, 768]
  @default_dropout 0.0

  # ============================================================================
  # Model Builder
  # ============================================================================

  @doc """
  Build a ConvNeXt model.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Stem patchify size (default: 4)
    - `:in_channels` - Number of input channels (default: 3)
    - `:depths` - Number of blocks per stage (default: [3, 3, 9, 3])
    - `:dims` - Channel dimensions per stage (default: [96, 192, 384, 768])
    - `:dropout` - Dropout rate (default: 0.0)
    - `:num_classes` - Number of classes for classification head (optional)

  ## Returns

    An Axon model. Without `:num_classes`, outputs `[batch, dims[-1]]`.
    With `:num_classes`, outputs `[batch, num_classes]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    depths = Keyword.get(opts, :depths, @default_depths)
    dims = Keyword.get(opts, :dims, @default_dims)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    num_classes = Keyword.get(opts, :num_classes, nil)

    num_stages = length(depths)

    # Input: [batch, channels, height, width]
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    first_dim = List.first(dims)

    # Stem: patchify with patch_size x patch_size non-overlapping patches
    x =
      PatchEmbed.layer(input,
        image_size: image_size,
        patch_size: patch_size,
        in_channels: in_channels,
        embed_dim: first_dim,
        name: "stem"
      )

    x = Axon.layer_norm(x, name: "stem_norm")

    # Process through stages
    num_patches = PatchEmbed.num_patches(image_size, patch_size)

    {x, _current_patches} =
      Enum.reduce(
        Enum.zip([depths, dims, 0..(num_stages - 1)]),
        {x, num_patches},
        fn {stage_depth, dim, stage_idx}, {acc, patches} ->
          # ConvNeXt blocks for this stage
          acc =
            Enum.reduce(0..(stage_depth - 1), acc, fn block_idx, block_acc ->
              convnext_block(block_acc, dim, dropout, name: "stage#{stage_idx}_block#{block_idx}")
            end)

          # Downsampling between stages (except last)
          if stage_idx < num_stages - 1 do
            next_dim = Enum.at(dims, stage_idx + 1)

            {downsampled, new_patches} =
              downsample(acc, next_dim, patches, name: "downsample_#{stage_idx}")

            {downsampled, new_patches}
          else
            {acc, patches}
          end
        end
      )

    # Global average pool: [batch, num_patches, dim] -> [batch, dim]
    x =
      Axon.nx(
        x,
        fn tensor ->
          Nx.mean(tensor, axes: [1])
        end,
        name: "global_pool"
      )

    x = Axon.layer_norm(x, name: "final_norm")

    # Optional classification head
    if num_classes do
      x =
        if dropout > 0.0 do
          Axon.dropout(x, rate: dropout, name: "head_dropout")
        else
          x
        end

      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  # ============================================================================
  # ConvNeXt Block
  # ============================================================================

  defp convnext_block(input, dim, dropout, opts) do
    name = Keyword.get(opts, :name, "cnx_block")
    expansion = 4

    # LayerNorm -> Dense (expand) -> GELU -> Dense (project back)
    x =
      input
      |> Axon.layer_norm(name: "#{name}_norm")
      |> Axon.dense(dim * expansion, name: "#{name}_expand")
      |> Axon.activation(:gelu, name: "#{name}_gelu")
      |> Axon.dense(dim, name: "#{name}_project")

    x =
      if dropout > 0.0 do
        Axon.dropout(x, rate: dropout, name: "#{name}_drop")
      else
        x
      end

    # Residual connection
    Axon.add(input, x, name: "#{name}_residual")
  end

  # ============================================================================
  # Downsampling
  # ============================================================================

  defp downsample(input, next_dim, num_patches, opts) do
    name = Keyword.get(opts, :name, "downsample")

    # Merge 2x2 neighboring patches and project to next dimension
    # [batch, N, C] -> [batch, N/4, 4C] -> [batch, N/4, next_dim]

    # Compute grid dimensions at compile time (num_patches is a known integer)
    grid_size = num_patches |> :math.sqrt() |> round()
    new_grid = div(grid_size, 2)

    merged =
      Axon.nx(
        input,
        fn tensor ->
          batch = Nx.axis_size(tensor, 0)
          feat_dim = Nx.axis_size(tensor, 2)

          # Reshape to 2D grid
          grid = Nx.reshape(tensor, {batch, grid_size, grid_size, feat_dim})

          # Take quadrants
          tl =
            Nx.slice_along_axis(grid, 0, new_grid, axis: 1)
            |> Nx.slice_along_axis(0, new_grid, axis: 2)

          tr =
            Nx.slice_along_axis(grid, 0, new_grid, axis: 1)
            |> Nx.slice_along_axis(new_grid, new_grid, axis: 2)

          bl =
            Nx.slice_along_axis(grid, new_grid, new_grid, axis: 1)
            |> Nx.slice_along_axis(0, new_grid, axis: 2)

          br =
            Nx.slice_along_axis(grid, new_grid, new_grid, axis: 1)
            |> Nx.slice_along_axis(new_grid, new_grid, axis: 2)

          merged = Nx.concatenate([tl, tr, bl, br], axis: 3)
          Nx.reshape(merged, {batch, new_grid * new_grid, feat_dim * 4})
        end,
        name: "#{name}_merge"
      )

    x = Axon.layer_norm(merged, name: "#{name}_norm")
    x = Axon.dense(x, next_dim, name: "#{name}_proj")

    new_patches = div(num_patches, 4)
    {x, new_patches}
  end

  # ============================================================================
  # Output Size
  # ============================================================================

  @doc """
  Get the output size of a ConvNeXt model.

  Returns `:num_classes` if set, otherwise the last stage dimension.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    case Keyword.get(opts, :num_classes) do
      nil ->
        dims = Keyword.get(opts, :dims, @default_dims)
        List.last(dims)

      num_classes ->
        num_classes
    end
  end
end
