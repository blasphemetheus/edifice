defmodule Edifice.Vision.PoolFormer do
  @moduledoc """
  PoolFormer: MetaFormer with average pooling as token mixer (Yu et al., 2022).

  Demonstrates that the general MetaFormer architecture (norm -> token_mixer ->
  residual -> norm -> FFN -> residual) is more important than the specific
  attention mechanism. PoolFormer replaces self-attention with simple average
  pooling, achieving competitive performance with much lower computational cost.

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
  +-----v--------------------+
  | Patch Embedding           |  Split into P x P patches, linear project
  +---------------------------+
        |
        v
  [batch, num_patches, hidden_size]
        |
  +-----v--------------------+
  | PoolFormer Block x N      |
  |                           |
  | Token Mixing:             |
  |   LN -> AvgPool - x      |
  |   + Residual              |
  |                           |
  | Channel Mixing:           |
  |   LN -> Dense(4*h)       |
  |   -> GELU                |
  |   -> Dense(h)            |
  |   + Residual              |
  +---------------------------+
        |
        v
  +---------------------------+
  | LayerNorm -> Mean Pool    |
  +---------------------------+
        |
        v
  [batch, hidden_size]
  ```

  ## Key Insight

  The pooling token mixer subtracts the input from its average-pooled version,
  which creates a simple form of local context aggregation. This is much simpler
  and faster than attention while maintaining competitive accuracy.

  ## Usage

      model = PoolFormer.build(
        image_size: 224,
        patch_size: 16,
        hidden_size: 256,
        num_layers: 4,
        num_classes: 1000
      )

  ## References

  - Yu et al., "MetaFormer is Actually What You Need for Vision" (CVPR 2022)
  - https://arxiv.org/abs/2111.11418
  """

  use Edifice.Vision.Backbone

  alias Edifice.Blocks.{FFN, PatchEmbed}

  @default_image_size 224
  @default_patch_size 16
  @default_in_channels 3
  @default_hidden_size 256
  @default_num_layers 4
  @default_pool_size 3

  @doc """
  Build a PoolFormer model.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Patch size, square (default: 16)
    - `:in_channels` - Number of input channels (default: 3)
    - `:hidden_size` - Hidden dimension per patch (default: 256)
    - `:num_layers` - Number of PoolFormer blocks (default: 4)
    - `:pool_size` - Pooling kernel size for token mixer (default: 3)
    - `:num_classes` - Number of output classes (optional)

  ## Returns

    An Axon model. Without `:num_classes`, outputs `[batch, hidden_size]`.
    With `:num_classes`, outputs `[batch, num_classes]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:hidden_size, pos_integer()}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:num_layers, pos_integer()}
          | {:patch_size, pos_integer()}
          | {:pool_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    pool_size = Keyword.get(opts, :pool_size, @default_pool_size)
    num_classes = Keyword.get(opts, :num_classes, nil)

    # Input: [batch, channels, height, width]
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Patch embedding: [batch, num_patches, hidden_size]
    x =
      PatchEmbed.layer(input,
        image_size: image_size,
        patch_size: patch_size,
        in_channels: in_channels,
        embed_dim: hidden_size,
        name: "patch_embed"
      )

    # Stack of PoolFormer blocks
    x =
      Enum.reduce(0..(num_layers - 1), x, fn idx, acc ->
        poolformer_block(acc, hidden_size, pool_size, name: "poolformer_#{idx}")
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Global average pool: [batch, num_patches, hidden_size] -> [batch, hidden_size]
    x =
      Axon.nx(x, fn tensor -> Nx.mean(tensor, axes: [1]) end, name: "global_pool")

    # Optional classification head
    if num_classes do
      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  @doc """
  Get the output size of a PoolFormer model.

  Returns `:num_classes` if set, otherwise `:hidden_size`.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    case Keyword.get(opts, :num_classes) do
      nil -> Keyword.get(opts, :hidden_size, @default_hidden_size)
      num_classes -> num_classes
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
    Keyword.get(opts, :hidden_size, @default_hidden_size)
  end

  # PoolFormer block: pool token mixer + FFN with residual connections
  defp poolformer_block(input, hidden_size, pool_size, opts) do
    name = Keyword.get(opts, :name, "poolformer")

    # Token mixer: LayerNorm -> (AvgPool - identity) -> Residual
    normed = Axon.layer_norm(input, name: "#{name}_token_norm")

    pooled =
      Axon.nx(
        normed,
        fn tensor -> pool_subtract_compute(tensor, pool_size) end,
        name: "#{name}_pool_mix"
      )

    x = Axon.add(input, pooled, name: "#{name}_token_residual")

    # Channel mixer: LayerNorm -> FFN -> Residual
    channel_normed = Axon.layer_norm(x, name: "#{name}_channel_norm")

    ffn_out = FFN.layer(channel_normed, hidden_size: hidden_size, name: "#{name}_ffn")

    Axon.add(x, ffn_out, name: "#{name}_channel_residual")
  end

  # Average pooling minus identity along the patch dimension
  # Input: [batch, num_patches, hidden_size]
  # Pools along axis 1 (patch dimension) with the given kernel size
  defp pool_subtract_compute(input, pool_size) do
    {batch, seq_len, dim} = Nx.shape(input)
    pad_total = pool_size - 1
    pad_before = div(pad_total, 2)
    pad_after = pad_total - pad_before

    # Pad the sequence dimension
    padded =
      Nx.pad(input, 0.0, [{0, 0, 0}, {pad_before, pad_after, 0}, {0, 0, 0}])

    # Compute sliding average using slicing and accumulation
    pooled =
      Enum.reduce(0..(pool_size - 1), Nx.broadcast(0.0, {batch, seq_len, dim}), fn offset, acc ->
        slice = Nx.slice_along_axis(padded, offset, seq_len, axis: 1)
        Nx.add(acc, slice)
      end)

    pooled = Nx.divide(pooled, pool_size)

    # Subtract identity: pooled - input
    Nx.subtract(pooled, input)
  end
end
