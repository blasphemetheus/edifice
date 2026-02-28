defmodule Edifice.Vision.ConvNeXt do
  @moduledoc """
  ConvNeXt - A Modernized ResNet implementation.

  Modernizes the classic ResNet design with techniques borrowed from
  transformers: depthwise-separable convolutions, inverted bottleneck
  blocks, GELU activation, LayerNorm, and fewer activation functions.

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
  +-----v--------------------+
  | Stem (4x4 strided conv)   |  [batch, H/4, W/4, dims[0]]
  +---------------------------+
        |
  +-----v--------------------+
  | Stage 1                   |  depths[0] ConvNeXt blocks at dims[0]
  |   DW Conv 7x7 -> LN      |
  |   -> PW Conv (expand 4x) |
  |   -> GELU                |
  |   -> PW Conv (project)   |
  |   -> LayerScale          |
  |   -> Residual            |
  +---------------------------+
        |
  +-----v--------------------+
  | Downsample                |  LN + 2x2 strided conv to dims[1]
  +---------------------------+
        |
  +-----v--------------------+
  | Stage 2                   |  depths[1] ConvNeXt blocks at dims[1]
  +---------------------------+
        |
        ... (repeat for each stage)
        |
  +-----v--------------------+
  | Global Average Pool       |  [batch, dims[-1]]
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

  ## ConvNeXt Block (faithful to paper)

  ```
  Input [batch, H, W, C]
    |
    +----------- Residual ----------+
    |                               |
  Depthwise Conv 7x7               |
    |                               |
  LayerNorm                         |
    |                               |
  Pointwise Conv (C -> 4C)         |
    |                               |
  GELU                              |
    |                               |
  Pointwise Conv (4C -> C)         |
    |                               |
  LayerScale (learnable gamma)     |
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

  ## References

  - "A ConvNet for the 2020s" (Liu et al., CVPR 2022)
  """

  use Edifice.Vision.Backbone

  @default_image_size 224
  @default_patch_size 4
  @default_in_channels 3
  @default_depths [3, 3, 9, 3]
  @default_dims [96, 192, 384, 768]
  @default_dropout 0.0
  @default_layer_scale_init 1.0e-6

  # ============================================================================
  # Model Builder
  # ============================================================================

  @doc """
  Build a ConvNeXt model.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Stem patchify stride (default: 4)
    - `:in_channels` - Number of input channels (default: 3)
    - `:depths` - Number of blocks per stage (default: [3, 3, 9, 3])
    - `:dims` - Channel dimensions per stage (default: [96, 192, 384, 768])
    - `:dropout` - Dropout rate (default: 0.0)
    - `:layer_scale_init` - Initial value for layer scale (default: 1e-6)
    - `:num_classes` - Number of classes for classification head (optional)

  ## Returns

    An Axon model. Without `:num_classes`, outputs `[batch, dims[-1]]`.
    With `:num_classes`, outputs `[batch, num_classes]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:depths, [pos_integer()]}
          | {:dims, [pos_integer()]}
          | {:dropout, float()}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:layer_scale_init, float()}
          | {:num_classes, pos_integer() | nil}
          | {:patch_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    depths = Keyword.get(opts, :depths, @default_depths)
    dims = Keyword.get(opts, :dims, @default_dims)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    layer_scale_init = Keyword.get(opts, :layer_scale_init, @default_layer_scale_init)
    num_classes = Keyword.get(opts, :num_classes, nil)

    num_stages = length(depths)
    first_dim = List.first(dims)

    # Input: [batch, channels, height, width] (NCHW)
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Transpose to NHWC for Axon.conv (channels_last default)
    x =
      Axon.nx(
        input,
        fn t ->
          Nx.transpose(t, axes: [0, 2, 3, 1])
        end,
        name: "nchw_to_nhwc"
      )

    # Stem: patchify with patch_size x patch_size strided convolution
    # This replaces PatchEmbed — a single strided conv is the ConvNeXt stem
    x =
      Axon.conv(x, first_dim,
        kernel_size: {patch_size, patch_size},
        strides: [patch_size, patch_size],
        name: "stem_conv"
      )

    x = Axon.layer_norm(x, name: "stem_norm")

    # Process through stages
    x =
      Enum.reduce(
        Enum.zip([depths, dims, 0..(num_stages - 1)]),
        x,
        fn {stage_depth, dim, stage_idx}, acc ->
          # ConvNeXt blocks for this stage
          acc =
            Enum.reduce(0..(stage_depth - 1), acc, fn block_idx, block_acc ->
              convnext_block(block_acc, dim, dropout, layer_scale_init,
                name: "stage#{stage_idx}_block#{block_idx}"
              )
            end)

          # Downsampling between stages (except last)
          if stage_idx < num_stages - 1 do
            next_dim = Enum.at(dims, stage_idx + 1)
            downsample(acc, next_dim, name: "downsample_#{stage_idx}")
          else
            acc
          end
        end
      )

    # Global average pool: [batch, H, W, C] -> [batch, C]
    x =
      Axon.nx(
        x,
        fn tensor ->
          Nx.mean(tensor, axes: [1, 2])
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
  # ConvNeXt Block (faithful to paper)
  # ============================================================================

  defp convnext_block(input, dim, dropout, layer_scale_init, opts) do
    name = Keyword.get(opts, :name, "cnx_block")
    expansion = 4

    # Depthwise conv 7x7: each channel gets its own filter
    x =
      Axon.conv(input, dim,
        kernel_size: {7, 7},
        padding: :same,
        feature_group_size: dim,
        name: "#{name}_dw_conv"
      )

    # LayerNorm (applied to last axis in NHWC)
    x = Axon.layer_norm(x, name: "#{name}_norm")

    # Pointwise expand: 1x1 conv equivalent, C -> 4C
    x =
      Axon.conv(x, dim * expansion,
        kernel_size: {1, 1},
        name: "#{name}_pw_expand"
      )

    x = Axon.activation(x, :gelu, name: "#{name}_gelu")

    # Pointwise project: 1x1 conv, 4C -> C
    x =
      Axon.conv(x, dim,
        kernel_size: {1, 1},
        name: "#{name}_pw_project"
      )

    # Layer scale: learnable per-channel scaling (initialized to small value per Liu et al. 2022)
    gamma_node =
      Axon.param("#{name}_gamma", {1, 1, 1, dim},
        initializer: fn shape, _type, _key ->
          Nx.broadcast(layer_scale_init, shape) |> Nx.as_type(:f32)
        end
      )

    x =
      Axon.layer(
        &layer_scale_impl/3,
        [x, gamma_node],
        name: "#{name}_layer_scale",
        op_name: :layer_scale
      )

    x =
      if dropout > 0.0 do
        Axon.dropout(x, rate: dropout, name: "#{name}_drop")
      else
        x
      end

    # Residual connection
    Axon.add(input, x, name: "#{name}_residual")
  end

  # Layer scale: multiply by learnable gamma parameter
  # gamma is initialized to a small value (1e-6) per the paper
  defp layer_scale_impl(x, gamma, _opts) do
    Nx.multiply(x, gamma)
  end

  # ============================================================================
  # Downsampling
  # ============================================================================

  defp downsample(input, next_dim, opts) do
    name = Keyword.get(opts, :name, "downsample")

    # LayerNorm -> 2x2 strided conv (spatial halving + channel projection)
    input
    |> Axon.layer_norm(name: "#{name}_norm")
    |> Axon.conv(next_dim,
      kernel_size: {2, 2},
      strides: [2, 2],
      name: "#{name}_conv"
    )
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

  # ============================================================================
  # Backbone Behaviour
  # ============================================================================

  @impl Edifice.Vision.Backbone
  def build_backbone(opts \\ []) do
    opts |> Keyword.delete(:num_classes) |> build()
  end

  @impl Edifice.Vision.Backbone
  def feature_size(opts \\ []) do
    dims = Keyword.get(opts, :dims, @default_dims)
    List.last(dims)
  end
end
