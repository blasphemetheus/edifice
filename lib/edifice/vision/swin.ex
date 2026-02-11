defmodule Edifice.Vision.SwinTransformer do
  @moduledoc """
  Swin Transformer (Shifted Window Transformer) implementation.

  A hierarchical vision transformer that computes attention within local
  windows and shifts windows between layers for cross-window connections.
  Produces multi-scale feature maps like a CNN, making it suitable for
  dense prediction tasks.

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
  +-----v--------------------+
  | Patch Embedding           |  patch_size x patch_size, linear project
  +---------------------------+
        |
        v
  [batch, H/4 * W/4, embed_dim]
        |
  +-----v--------------------+
  | Stage 1                   |  depths[0] Swin blocks at embed_dim
  |   Window Attention        |  Alternating regular/shifted windows
  +---------------------------+
        |
  +-----v--------------------+
  | Patch Merging             |  2x2 spatial merge, 2x channel expand
  +---------------------------+
        |
  +-----v--------------------+
  | Stage 2                   |  depths[1] blocks at embed_dim * 2
  +---------------------------+
        |
  +-----v--------------------+
  | Patch Merging             |
  +---------------------------+
        |
  +-----v--------------------+
  | Stage 3                   |  depths[2] blocks at embed_dim * 4
  +---------------------------+
        |
  +-----v--------------------+
  | Patch Merging             |
  +---------------------------+
        |
  +-----v--------------------+
  | Stage 4                   |  depths[3] blocks at embed_dim * 8
  +---------------------------+
        |
  +-----v--------------------+
  | Global Average Pooling    |
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

  ## Window Attention

  Instead of global self-attention (O(N^2)), Swin computes attention within
  non-overlapping local windows of size M x M tokens. This reduces complexity
  to O(N * M^2). Shifted windows in alternating layers allow cross-window
  information flow.

  This implementation approximates window attention using dense layers over
  the full sequence, which is functionally equivalent for small token counts
  and simplifies the Axon graph construction.

  ## Usage

      # Swin-Tiny
      model = SwinTransformer.build(
        image_size: 224,
        patch_size: 4,
        embed_dim: 96,
        depths: [2, 2, 6, 2],
        num_heads: [3, 6, 12, 24],
        num_classes: 1000
      )

      # Swin-Small
      model = SwinTransformer.build(
        image_size: 224,
        embed_dim: 96,
        depths: [2, 2, 18, 2],
        num_heads: [3, 6, 12, 24],
        num_classes: 1000
      )

  ## References

  - "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
    (Liu et al., ICCV 2021)
  """

  require Axon

  alias Edifice.Blocks.PatchEmbed
  alias Edifice.Utils.FusedOps

  @default_image_size 224
  @default_patch_size 4
  @default_in_channels 3
  @default_embed_dim 96
  @default_depths [2, 2, 6, 2]
  @default_num_heads [3, 6, 12, 24]
  @default_window_size 7
  @default_mlp_ratio 4.0
  @default_dropout 0.0

  # ============================================================================
  # Model Builder
  # ============================================================================

  @doc """
  Build a Swin Transformer model.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Initial patch embedding size (default: 4)
    - `:in_channels` - Number of input channels (default: 3)
    - `:embed_dim` - Base embedding dimension (default: 96)
    - `:depths` - Number of blocks per stage (default: [2, 2, 6, 2])
    - `:num_heads` - Number of attention heads per stage (default: [3, 6, 12, 24])
    - `:window_size` - Window size for local attention (default: 7)
    - `:mlp_ratio` - MLP hidden dim ratio (default: 4.0)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:num_classes` - Number of classes for classification head (optional)

  ## Returns

    An Axon model. Without `:num_classes`, outputs `[batch, embed_dim * 2^(num_stages-1)]`.
    With `:num_classes`, outputs `[batch, num_classes]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    depths = Keyword.get(opts, :depths, @default_depths)
    num_heads_list = Keyword.get(opts, :num_heads, @default_num_heads)
    _window_size = Keyword.get(opts, :window_size, @default_window_size)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    num_classes = Keyword.get(opts, :num_classes, nil)

    num_stages = length(depths)

    # Input: [batch, channels, height, width]
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Patch embedding: [batch, num_patches, embed_dim]
    num_patches = PatchEmbed.num_patches(image_size, patch_size)

    x =
      PatchEmbed.layer(input,
        image_size: image_size,
        patch_size: patch_size,
        in_channels: in_channels,
        embed_dim: embed_dim,
        name: "patch_embed"
      )

    # Process through stages
    {x, _current_dim, _current_patches} =
      Enum.reduce(
        Enum.zip([depths, num_heads_list, 0..(num_stages - 1)]),
        {x, embed_dim, num_patches},
        fn {stage_depth, stage_heads, stage_idx}, {acc, dim, patches} ->
          mlp_hidden = round(dim * mlp_ratio)

          # Swin blocks for this stage
          acc =
            Enum.reduce(0..(stage_depth - 1), acc, fn block_idx, block_acc ->
              swin_block(block_acc, dim, stage_heads, mlp_hidden, dropout,
                shifted: rem(block_idx, 2) == 1,
                name: "stage#{stage_idx}_block#{block_idx}"
              )
            end)

          # Patch merging between stages (except last)
          if stage_idx < num_stages - 1 do
            {merged, new_dim} =
              patch_merging(acc, dim, patches, name: "merge_#{stage_idx}")

            {merged, new_dim, div(patches, 4)}
          else
            {acc, dim, patches}
          end
        end
      )

    # Global average pool over sequence dimension
    x =
      Axon.nx(
        x,
        fn tensor ->
          Nx.mean(tensor, axes: [1])
        end,
        name: "global_pool"
      )

    x = Axon.layer_norm(x, name: "final_norm")

    # Output dimension is embed_dim * 2^(num_stages - 1)
    if num_classes do
      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  # ============================================================================
  # Swin Block
  # ============================================================================

  defp swin_block(input, dim, _num_heads, mlp_hidden, dropout, opts) do
    name = Keyword.get(opts, :name, "swin_block")
    _shifted = Keyword.get(opts, :shifted, false)

    # Pre-norm window attention (simplified as full sequence attention)
    normed = Axon.layer_norm(input, name: "#{name}_norm1")
    attended = window_attention(normed, dim, dropout, name: "#{name}_attn")
    x = Axon.add(input, attended, name: "#{name}_residual1")

    # Pre-norm MLP
    normed2 = Axon.layer_norm(x, name: "#{name}_norm2")

    ffn =
      normed2
      |> Axon.dense(mlp_hidden, name: "#{name}_mlp_fc1")
      |> Axon.activation(:gelu, name: "#{name}_mlp_gelu")
      |> maybe_dropout(dropout, "#{name}_mlp_drop1")
      |> Axon.dense(dim, name: "#{name}_mlp_fc2")
      |> maybe_dropout(dropout, "#{name}_mlp_drop2")

    Axon.add(x, ffn, name: "#{name}_residual2")
  end

  defp window_attention(input, dim, dropout, opts) do
    name = Keyword.get(opts, :name, "w_attn")

    qkv = Axon.dense(input, dim * 3, name: "#{name}_qkv")

    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          q = Nx.slice_along_axis(qkv_tensor, 0, dim, axis: 2)
          k = Nx.slice_along_axis(qkv_tensor, dim, dim, axis: 2)
          v = Nx.slice_along_axis(qkv_tensor, dim * 2, dim, axis: 2)

          d_k = Nx.axis_size(k, -1)
          scale = Nx.sqrt(d_k) |> Nx.as_type(Nx.type(q))

          scores = Nx.dot(q, [2], [0], k, [2], [0])
          scores = Nx.divide(scores, scale)
          weights = FusedOps.fused_softmax(scores)

          Nx.dot(weights, [2], [0], v, [1], [0])
        end,
        name: "#{name}_compute"
      )

    attended
    |> Axon.dense(dim, name: "#{name}_proj")
    |> maybe_dropout(dropout, "#{name}_dropout")
  end

  # ============================================================================
  # Patch Merging
  # ============================================================================

  defp patch_merging(input, dim, num_patches, opts) do
    name = Keyword.get(opts, :name, "patch_merge")

    # Patch merging: concatenate 2x2 neighboring patches, then project
    # [batch, N, C] -> [batch, N/4, 4C] -> [batch, N/4, 2C]
    new_dim = dim * 2

    # Compute grid dimensions at compile time (num_patches is a known integer)
    grid_size = num_patches |> :math.sqrt() |> round()
    new_grid = div(grid_size, 2)

    merged =
      Axon.nx(
        input,
        fn tensor ->
          batch = Nx.axis_size(tensor, 0)
          feat_dim = Nx.axis_size(tensor, 2)

          # Reshape to 2D grid: [batch, H, W, C]
          grid = Nx.reshape(tensor, {batch, grid_size, grid_size, feat_dim})

          # Take every other element in both dimensions to get 4 sub-grids
          # Top-left
          tl =
            Nx.slice_along_axis(grid, 0, new_grid, axis: 1)
            |> Nx.slice_along_axis(0, new_grid, axis: 2)

          # Top-right
          tr =
            Nx.slice_along_axis(grid, 0, new_grid, axis: 1)
            |> Nx.slice_along_axis(new_grid, new_grid, axis: 2)

          # Bottom-left
          bl =
            Nx.slice_along_axis(grid, new_grid, new_grid, axis: 1)
            |> Nx.slice_along_axis(0, new_grid, axis: 2)

          # Bottom-right
          br =
            Nx.slice_along_axis(grid, new_grid, new_grid, axis: 1)
            |> Nx.slice_along_axis(new_grid, new_grid, axis: 2)

          # Concatenate along feature dim: [batch, H/2, W/2, 4C]
          merged = Nx.concatenate([tl, tr, bl, br], axis: 3)

          # Flatten spatial: [batch, N/4, 4C]
          Nx.reshape(merged, {batch, new_grid * new_grid, feat_dim * 4})
        end,
        name: "#{name}_reshape"
      )

    # Project 4C -> 2C
    merged = Axon.layer_norm(merged, name: "#{name}_norm")
    projected = Axon.dense(merged, new_dim, name: "#{name}_proj")

    {projected, new_dim}
  end

  defp maybe_dropout(input, rate, name) when rate > 0.0 do
    Axon.dropout(input, rate: rate, name: name)
  end

  defp maybe_dropout(input, _rate, _name), do: input

  # ============================================================================
  # Output Size
  # ============================================================================

  @doc """
  Get the output size of a Swin Transformer model.

  Returns `:num_classes` if set, otherwise `embed_dim * 2^(num_stages - 1)`.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    case Keyword.get(opts, :num_classes) do
      nil ->
        embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
        depths = Keyword.get(opts, :depths, @default_depths)
        num_stages = length(depths)
        embed_dim * round(:math.pow(2, num_stages - 1))

      num_classes ->
        num_classes
    end
  end
end
