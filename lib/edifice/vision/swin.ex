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

  Attention is computed within non-overlapping local windows of M x M tokens,
  reducing complexity from O(N^2) to O(N * M^2). Shifted windows in alternating
  layers enable cross-window information flow via cyclic shift and masked attention.

  Features:
  - Real window partitioning with M x M local attention
  - Multi-head scaled dot-product attention within each window
  - Cyclic shift for shifted window attention with boundary masking
  - Learnable relative position bias per attention head

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

  Spatial dimensions at each stage must be divisible by the effective window size.

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
    window_size = Keyword.get(opts, :window_size, @default_window_size)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    num_classes = Keyword.get(opts, :num_classes, nil)

    num_stages = length(depths)
    grid_size = div(image_size, patch_size)

    # Input: [batch, channels, height, width]
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Patch embedding: [batch, num_patches, embed_dim]
    x =
      PatchEmbed.layer(input,
        image_size: image_size,
        patch_size: patch_size,
        in_channels: in_channels,
        embed_dim: embed_dim,
        name: "patch_embed"
      )

    # Process through stages
    {x, _dim, _h, _w} =
      Enum.reduce(
        Enum.zip([depths, num_heads_list, 0..(num_stages - 1)]),
        {x, embed_dim, grid_size, grid_size},
        fn {stage_depth, stage_heads, stage_idx}, {acc, dim, h, w} ->
          mlp_hidden = round(dim * mlp_ratio)
          # Cap window size at spatial dimensions
          eff_ws = min(window_size, min(h, w))

          # Swin blocks for this stage
          acc =
            Enum.reduce(0..(stage_depth - 1), acc, fn block_idx, block_acc ->
              swin_block(block_acc, dim, stage_heads, mlp_hidden, dropout,
                shifted: rem(block_idx, 2) == 1,
                window_size: eff_ws,
                h: h,
                w: w,
                name: "stage#{stage_idx}_block#{block_idx}"
              )
            end)

          # Patch merging between stages (except last)
          if stage_idx < num_stages - 1 do
            {merged, new_dim} = patch_merging(acc, dim, h, w, name: "merge_#{stage_idx}")
            {merged, new_dim, div(h, 2), div(w, 2)}
          else
            {acc, dim, h, w}
          end
        end
      )

    # Global average pool over sequence dimension
    x =
      Axon.nx(x, fn tensor -> Nx.mean(tensor, axes: [1]) end, name: "global_pool")

    x = Axon.layer_norm(x, name: "final_norm")

    if num_classes do
      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  # ============================================================================
  # Swin Block
  # ============================================================================

  defp swin_block(input, dim, num_heads, mlp_hidden, dropout, opts) do
    name = Keyword.get(opts, :name, "swin_block")
    shifted = Keyword.get(opts, :shifted, false)
    window_size = Keyword.get(opts, :window_size, 7)
    h = Keyword.fetch!(opts, :h)
    w = Keyword.fetch!(opts, :w)

    # Pre-norm window attention
    normed = Axon.layer_norm(input, name: "#{name}_norm1")

    attended =
      build_window_attention(normed, dim, num_heads, dropout,
        window_size: window_size,
        shifted: shifted,
        h: h,
        w: w,
        name: "#{name}_attn"
      )

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

  # ============================================================================
  # Window Attention
  # ============================================================================

  defp build_window_attention(input, dim, num_heads, dropout, opts) do
    name = Keyword.get(opts, :name, "w_attn")
    window_size = Keyword.get(opts, :window_size, 7)
    shifted = Keyword.get(opts, :shifted, false)
    h = Keyword.fetch!(opts, :h)
    w = Keyword.fetch!(opts, :w)

    head_dim = div(dim, num_heads)

    # Only shift when there are multiple windows in both dimensions
    shift_size =
      if shifted and window_size < h and window_size < w do
        div(window_size, 2)
      else
        0
      end

    # Precompute relative position bias as a fixed constant
    # Uses distance-based decay per head (similar to ALiBi but for 2D windows)
    rel_pos_bias = compute_relative_position_bias(window_size, num_heads)
    rel_pos_node = Axon.constant(rel_pos_bias, name: "#{name}_rel_pos_bias")

    shift_mask_node =
      if shift_size > 0 do
        mask = compute_shift_mask(h, w, window_size, shift_size)
        Axon.constant(mask, name: "#{name}_shift_mask")
      else
        # Dummy constant (won't be used when shift_size=0)
        Axon.constant(Nx.tensor(0.0), name: "#{name}_no_mask")
      end

    # Q, K, V projections
    q = Axon.dense(input, dim, name: "#{name}_q")
    k = Axon.dense(input, dim, name: "#{name}_k")
    v = Axon.dense(input, dim, name: "#{name}_v")

    # Window attention with partitioning, shifting, and relative position bias
    has_shift = shift_size > 0

    attn_out =
      Axon.layer(
        fn q_t, k_t, v_t, bias_t, mask_t, layer_opts ->
          window_attention_impl(
            q_t,
            k_t,
            v_t,
            bias_t,
            if(has_shift, do: mask_t, else: nil),
            layer_opts
          )
        end,
        [q, k, v, rel_pos_node, shift_mask_node],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        window_size: window_size,
        shift_size: shift_size,
        h: h,
        w: w,
        op_name: :swin_window_attention
      )

    attn_out
    |> Axon.dense(dim, name: "#{name}_proj")
    |> maybe_dropout(dropout, "#{name}_dropout")
  end

  # Window attention: shift -> partition -> multi-head attention -> reverse -> unshift
  defp window_attention_impl(q, k, v, rel_pos_bias, shift_mask, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]
    ws = opts[:window_size]
    shift_size = opts[:shift_size]
    h = opts[:h]
    w = opts[:w]

    batch = Nx.axis_size(q, 0)
    dim = num_heads * head_dim
    ws_sq = ws * ws
    num_windows = div(h, ws) * div(w, ws)
    total_windows = batch * num_windows

    # Reshape to spatial: [B, H*W, C] -> [B, H, W, C]
    q_spatial = Nx.reshape(q, {batch, h, w, dim})
    k_spatial = Nx.reshape(k, {batch, h, w, dim})
    v_spatial = Nx.reshape(v, {batch, h, w, dim})

    # Cyclic shift for shifted window attention
    {q_spatial, k_spatial, v_spatial} =
      if shift_size > 0 do
        {cyclic_shift(q_spatial, shift_size, h, w), cyclic_shift(k_spatial, shift_size, h, w),
         cyclic_shift(v_spatial, shift_size, h, w)}
      else
        {q_spatial, k_spatial, v_spatial}
      end

    # Window partition: [B, H, W, C] -> [B*nW, ws*ws, C]
    q_win = window_partition(q_spatial, ws, h, w)
    k_win = window_partition(k_spatial, ws, h, w)
    v_win = window_partition(v_spatial, ws, h, w)

    # Multi-head: [B*nW, ws*ws, C] -> [B*nW, num_heads, ws*ws, head_dim]
    q_mh =
      q_win
      |> Nx.reshape({total_windows, ws_sq, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    k_mh =
      k_win
      |> Nx.reshape({total_windows, ws_sq, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    v_mh =
      v_win
      |> Nx.reshape({total_windows, ws_sq, num_heads, head_dim})
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Scaled dot-product attention
    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))

    # Scores: [B*nW, num_heads, ws*ws, ws*ws]
    scores =
      Nx.dot(q_mh, [3], [0, 1], k_mh, [3], [0, 1])
      |> Nx.divide(scale)

    # Add relative position bias: [1, num_heads, ws*ws, ws*ws]
    scores = Nx.add(scores, rel_pos_bias)

    # Apply shift mask for shifted windows
    scores =
      if shift_mask != nil do
        scores_5d = Nx.reshape(scores, {batch, num_windows, num_heads, ws_sq, ws_sq})
        mask = shift_mask |> Nx.new_axis(0) |> Nx.new_axis(2)
        scores_5d = Nx.add(scores_5d, mask)
        Nx.reshape(scores_5d, {total_windows, num_heads, ws_sq, ws_sq})
      else
        scores
      end

    # Softmax and weighted sum
    weights = FusedOps.fused_softmax(scores)

    # [B*nW, num_heads, ws*ws, head_dim]
    output = Nx.dot(weights, [3], [0, 1], v_mh, [2], [0, 1])

    # Reshape: [B*nW, ws*ws, C]
    output =
      output
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({total_windows, ws_sq, dim})

    # Window reverse: [B*nW, ws*ws, C] -> [B, H, W, C]
    output_spatial = window_reverse(output, ws, h, w, batch)

    # Reverse cyclic shift
    output_spatial =
      if shift_size > 0 do
        reverse_cyclic_shift(output_spatial, shift_size, h, w)
      else
        output_spatial
      end

    # Flatten: [B, H, W, C] -> [B, H*W, C]
    Nx.reshape(output_spatial, {batch, h * w, dim})
  end

  # ============================================================================
  # Window Operations
  # ============================================================================

  @doc """
  Partition a spatial tensor into non-overlapping windows.

  Input: [B, H, W, C] -> Output: [B*nW, ws*ws, C]
  """
  def window_partition(input, ws, h, w) do
    batch = Nx.axis_size(input, 0)
    c = Nx.axis_size(input, 3)
    num_h = div(h, ws)
    num_w = div(w, ws)

    input
    |> Nx.reshape({batch, num_h, ws, num_w, ws, c})
    |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
    |> Nx.reshape({batch * num_h * num_w, ws * ws, c})
  end

  @doc """
  Reverse window partition back to spatial layout.

  Input: [B*nW, ws*ws, C] -> Output: [B, H, W, C]
  """
  def window_reverse(input, ws, h, w, batch) do
    c = Nx.axis_size(input, 2)
    num_h = div(h, ws)
    num_w = div(w, ws)

    input
    |> Nx.reshape({batch, num_h, num_w, ws, ws, c})
    |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
    |> Nx.reshape({batch, h, w, c})
  end

  @doc """
  Cyclic shift: roll tensor by -shift_size along both H and W axes.
  """
  def cyclic_shift(input, shift_size, h, w) do
    # Shift along H axis
    top = Nx.slice_along_axis(input, shift_size, h - shift_size, axis: 1)
    bottom = Nx.slice_along_axis(input, 0, shift_size, axis: 1)
    shifted = Nx.concatenate([top, bottom], axis: 1)

    # Shift along W axis
    left = Nx.slice_along_axis(shifted, shift_size, w - shift_size, axis: 2)
    right = Nx.slice_along_axis(shifted, 0, shift_size, axis: 2)
    Nx.concatenate([left, right], axis: 2)
  end

  @doc """
  Reverse cyclic shift: roll tensor by +shift_size along both H and W axes.
  """
  def reverse_cyclic_shift(input, shift_size, h, w) do
    top = Nx.slice_along_axis(input, h - shift_size, shift_size, axis: 1)
    bottom = Nx.slice_along_axis(input, 0, h - shift_size, axis: 1)
    shifted = Nx.concatenate([top, bottom], axis: 1)

    left = Nx.slice_along_axis(shifted, w - shift_size, shift_size, axis: 2)
    right = Nx.slice_along_axis(shifted, 0, w - shift_size, axis: 2)
    Nx.concatenate([left, right], axis: 2)
  end

  # ============================================================================
  # Precomputed Constants
  # ============================================================================

  @doc """
  Compute relative position bias for window attention.

  Uses distance-based decay with per-head geometric slopes, similar to ALiBi
  but for 2D windows. Each head gets a different slope, providing diverse
  position sensitivity across heads.

  Returns a [1, num_heads, ws*ws, ws*ws] bias tensor.
  """
  def compute_relative_position_bias(window_size, num_heads) do
    ws_sq = window_size * window_size

    # Compute pairwise Manhattan distances between all positions in the window
    coords = for i <- 0..(window_size - 1), j <- 0..(window_size - 1), do: {i, j}

    distances =
      for {h1, w1} <- coords, {h2, w2} <- coords do
        abs(h1 - h2) + abs(w1 - w2)
      end

    dist_matrix =
      Nx.tensor(distances, type: :f32)
      |> Nx.reshape({ws_sq, ws_sq})

    # Geometric slopes per head (like ALiBi): 2^(-8/num_heads * (i+1))
    slopes =
      for i <- 0..(num_heads - 1) do
        :math.pow(2.0, -8.0 / num_heads * (i + 1))
      end

    slopes_tensor = Nx.tensor(slopes, type: :f32) |> Nx.reshape({num_heads, 1, 1})

    # bias[h, i, j] = -slope_h * distance(i, j)
    bias =
      dist_matrix
      |> Nx.new_axis(0)
      |> Nx.multiply(slopes_tensor)
      |> Nx.negate()

    # Add batch dimension: [1, num_heads, ws*ws, ws*ws]
    Nx.new_axis(bias, 0)
  end

  @doc """
  Compute attention mask for shifted windows.

  Assigns region IDs based on position relative to shift boundaries,
  then creates a pairwise mask that blocks attention between tokens
  from different regions within each window.

  Returns a [num_windows, ws*ws, ws*ws] mask tensor with 0.0 for
  allowed attention and -100.0 for blocked attention.
  """
  def compute_shift_mask(h, w, window_size, shift_size) do
    mask_data =
      for i <- 0..(h - 1), j <- 0..(w - 1) do
        h_region =
          cond do
            i < h - window_size -> 0
            i < h - shift_size -> 1
            true -> 2
          end

        w_region =
          cond do
            j < w - window_size -> 0
            j < w - shift_size -> 1
            true -> 2
          end

        h_region * 3 + w_region
      end

    mask = Nx.tensor(mask_data) |> Nx.reshape({h, w})

    num_h = div(h, window_size)
    num_w = div(w, window_size)
    ws_sq = window_size * window_size

    # Partition mask into windows: [nW, ws*ws]
    mask_windows =
      mask
      |> Nx.reshape({num_h, window_size, num_w, window_size})
      |> Nx.transpose(axes: [0, 2, 1, 3])
      |> Nx.reshape({num_h * num_w, ws_sq})

    # Pairwise mask: [nW, ws*ws, ws*ws]
    mask_a = Nx.new_axis(mask_windows, 2)
    mask_b = Nx.new_axis(mask_windows, 1)

    Nx.not_equal(mask_a, mask_b)
    |> Nx.select(Nx.tensor(-100.0), Nx.tensor(0.0))
  end

  # ============================================================================
  # Patch Merging
  # ============================================================================

  defp patch_merging(input, dim, h, w, opts) do
    name = Keyword.get(opts, :name, "patch_merge")
    new_dim = dim * 2
    new_h = div(h, 2)
    new_w = div(w, 2)

    merged =
      Axon.nx(
        input,
        fn tensor ->
          batch = Nx.axis_size(tensor, 0)
          c = Nx.axis_size(tensor, 2)

          # Reshape to spatial: [B, H, W, C]
          grid = Nx.reshape(tensor, {batch, h, w, c})

          # Group 2x2 adjacent patches: [B, H/2, 2, W/2, 2, C]
          reshaped = Nx.reshape(grid, {batch, new_h, 2, new_w, 2, c})

          # Reorder to [B, H/2, W/2, 2, 2, C]
          grouped = Nx.transpose(reshaped, axes: [0, 1, 3, 2, 4, 5])

          # Flatten 2x2 block features: [B, H/2, W/2, 4C]
          merged = Nx.reshape(grouped, {batch, new_h, new_w, 4 * c})

          # Flatten spatial: [B, H/2*W/2, 4C]
          Nx.reshape(merged, {batch, new_h * new_w, 4 * c})
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
  # Utilities
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
