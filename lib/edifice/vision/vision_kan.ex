defmodule Edifice.Vision.VisionKAN do
  @moduledoc """
  Vision KAN: Attention-free hierarchical vision backbone using
  Kolmogorov-Arnold Networks with Radial Basis Function activations.

  Each stage consists of ViK blocks combining:
  1. Patch-wise RBFKAN for nonlinear per-patch modeling
  2. Axis-wise separable depthwise convolution for local spatial mixing
  3. Low-rank global path for cross-patch interaction

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
  Patch Embed (4x4 strided conv)
        |
  +-----------+
  | Stage 1   |  N_1 ViK blocks, C_1 channels
  +-----------+
        |
  Downsample (2x2 strided conv)
        |
  +-----------+
  | Stage 2   |  N_2 ViK blocks, C_2 channels
  +-----------+
        |
  Downsample
        |
  +-----------+
  | Stage 3   |  N_3 ViK blocks, C_3 channels
  +-----------+
        |
  Downsample
        |
  +-----------+
  | Stage 4   |  N_4 ViK blocks, C_4 channels
  +-----------+
        |
  Global Average Pool -> [batch, C_4]
  ```

  Each ViK Block:
  ```
  Input [batch, H, W, C]
    |
  LayerNorm -> MultiPatch-RBFKAN -> + residual
    |
  LayerNorm -> FFN (1x1 conv) -> + residual
  ```

  ## Usage

      model = VisionKAN.build(
        image_size: 32,
        in_channels: 3,
        channels: [32, 64, 128, 256],
        depths: [2, 2, 4, 2]
      )

  ## Reference

  - "Vision KAN: Towards an Attention-Free Backbone for Vision with
    Kolmogorov-Arnold Networks" (arXiv:2601.21541, January 2026)
  """

  @behaviour Edifice.Vision.Backbone

  @default_image_size 32
  @default_in_channels 3
  @default_patch_size 4
  @default_channels [32, 64, 128, 256]
  @default_depths [2, 2, 4, 2]
  @default_num_rbf_centers 5
  @default_kan_patch_size 7
  @default_dw_kernel_size 7
  @default_global_reduction 4
  @default_dropout 0.1
  @default_ffn_expansion 4

  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:patch_size, pos_integer()}
          | {:channels, [pos_integer()]}
          | {:depths, [pos_integer()]}
          | {:num_rbf_centers, pos_integer()}
          | {:kan_patch_size, pos_integer()}
          | {:dw_kernel_size, pos_integer()}
          | {:global_reduction, pos_integer()}
          | {:dropout, float()}
          | {:ffn_expansion, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:input_shape, tuple()}

  @doc """
  Build a Vision KAN model.

  ## Options

    - `:image_size` - Input image spatial dimension (default: 32)
    - `:in_channels` - Input channels (default: 3)
    - `:patch_size` - Initial patch embedding stride (default: 4)
    - `:channels` - Per-stage channel dimensions (default: [32, 64, 128, 256])
    - `:depths` - ViK blocks per stage (default: [2, 2, 4, 2])
    - `:num_rbf_centers` - RBF grid points per activation (default: 5)
    - `:kan_patch_size` - Patch size for MultiPatch-RBFKAN grouping (default: 7)
    - `:dw_kernel_size` - Depthwise conv kernel size (default: 7)
    - `:global_reduction` - Reduction ratio for low-rank global path (default: 4)
    - `:dropout` - Dropout rate (default: 0.1)
    - `:ffn_expansion` - FFN expansion ratio (default: 4)
    - `:num_classes` - Classification head output (default: nil, no head)

  ## Returns

    An Axon model. If `:num_classes` is set, outputs `[batch, num_classes]`.
    Otherwise outputs `[batch, final_channel_dim]`.

  ## Examples

      iex> model = Edifice.Vision.VisionKAN.build(image_size: 16, in_channels: 1, channels: [8, 16], depths: [1, 1], patch_size: 4)
      iex> %Axon{} = model
  """
  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    channels = Keyword.get(opts, :channels, @default_channels)
    depths = Keyword.get(opts, :depths, @default_depths)
    num_rbf_centers = Keyword.get(opts, :num_rbf_centers, @default_num_rbf_centers)
    kan_patch_size = Keyword.get(opts, :kan_patch_size, @default_kan_patch_size)
    dw_kernel_size = Keyword.get(opts, :dw_kernel_size, @default_dw_kernel_size)
    global_reduction = Keyword.get(opts, :global_reduction, @default_global_reduction)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    ffn_expansion = Keyword.get(opts, :ffn_expansion, @default_ffn_expansion)
    num_classes = Keyword.get(opts, :num_classes, nil)

    input_shape =
      Keyword.get(opts, :input_shape, {nil, in_channels, image_size, image_size})

    num_stages = length(depths)
    first_dim = List.first(channels)

    # Input: NCHW
    input = Axon.input("image", shape: input_shape)

    # Transpose to NHWC for Axon conv
    x =
      Axon.nx(input, fn t -> Nx.transpose(t, axes: [0, 2, 3, 1]) end,
        name: "nchw_to_nhwc"
      )

    # Stem: patch embedding via strided conv
    x =
      Axon.conv(x, first_dim,
        kernel_size: {patch_size, patch_size},
        strides: [patch_size, patch_size],
        name: "stem_conv"
      )

    x = Axon.layer_norm(x, name: "stem_norm")

    # Spatial size after stem
    spatial = div(image_size, patch_size)

    # Process through stages
    {x, _spatial} =
      Enum.zip([depths, channels, 0..(num_stages - 1)])
      |> Enum.reduce({x, spatial}, fn {stage_depth, dim, stage_idx}, {acc, sp} ->
        # ViK blocks for this stage
        acc =
          Enum.reduce(0..(stage_depth - 1), acc, fn block_idx, block_acc ->
            vik_block(block_acc,
              dim: dim,
              spatial: sp,
              num_rbf_centers: num_rbf_centers,
              kan_patch_size: kan_patch_size,
              dw_kernel_size: dw_kernel_size,
              global_reduction: global_reduction,
              dropout: dropout,
              ffn_expansion: ffn_expansion,
              name: "stage#{stage_idx}_vik#{block_idx}"
            )
          end)

        # Downsample between stages (except last)
        if stage_idx < num_stages - 1 do
          next_dim = Enum.at(channels, stage_idx + 1)
          acc = downsample(acc, next_dim, name: "downsample_#{stage_idx}")
          {acc, div(sp, 2)}
        else
          {acc, sp}
        end
      end)

    # Global average pool: [batch, H, W, C] -> [batch, C]
    x =
      Axon.nx(x, fn tensor -> Nx.mean(tensor, axes: [1, 2]) end,
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
  # ViK Block
  # ============================================================================

  defp vik_block(input, opts) do
    dim = Keyword.fetch!(opts, :dim)
    spatial = Keyword.fetch!(opts, :spatial)
    num_rbf_centers = Keyword.fetch!(opts, :num_rbf_centers)
    kan_patch_size = Keyword.fetch!(opts, :kan_patch_size)
    dw_kernel_size = Keyword.fetch!(opts, :dw_kernel_size)
    global_reduction = Keyword.fetch!(opts, :global_reduction)
    dropout = Keyword.fetch!(opts, :dropout)
    ffn_expansion = Keyword.fetch!(opts, :ffn_expansion)
    name = Keyword.get(opts, :name, "vik")

    # --- Token mixer: MultiPatch-RBFKAN ---
    normed = Axon.layer_norm(input, name: "#{name}_norm1")

    mixed =
      multipatch_rbfkan(normed,
        dim: dim,
        spatial: spatial,
        num_rbf_centers: num_rbf_centers,
        kan_patch_size: kan_patch_size,
        dw_kernel_size: dw_kernel_size,
        global_reduction: global_reduction,
        name: "#{name}_mp_rbfkan"
      )

    mixed =
      if dropout > 0.0 do
        Axon.dropout(mixed, rate: dropout, name: "#{name}_drop1")
      else
        mixed
      end

    x = Axon.add(input, mixed, name: "#{name}_res1")

    # --- Channel mixer: FFN (1x1 conv) ---
    normed2 = Axon.layer_norm(x, name: "#{name}_norm2")

    ffn =
      normed2
      |> Axon.conv(dim * ffn_expansion, kernel_size: {1, 1}, name: "#{name}_ffn_expand")
      |> Axon.activation(:gelu, name: "#{name}_ffn_gelu")
      |> Axon.conv(dim, kernel_size: {1, 1}, name: "#{name}_ffn_project")

    ffn =
      if dropout > 0.0 do
        Axon.dropout(ffn, rate: dropout, name: "#{name}_drop2")
      else
        ffn
      end

    Axon.add(x, ffn, name: "#{name}_res2")
  end

  # ============================================================================
  # MultiPatch-RBFKAN
  # ============================================================================

  defp multipatch_rbfkan(input, opts) do
    dim = Keyword.fetch!(opts, :dim)
    spatial = Keyword.fetch!(opts, :spatial)
    num_rbf_centers = Keyword.fetch!(opts, :num_rbf_centers)
    kan_patch_size = Keyword.fetch!(opts, :kan_patch_size)
    dw_kernel_size = Keyword.fetch!(opts, :dw_kernel_size)
    global_reduction = Keyword.fetch!(opts, :global_reduction)
    name = Keyword.get(opts, :name, "mp_rbfkan")

    # Effective patch size: min of kan_patch_size and spatial dim
    eff_patch = min(kan_patch_size, spatial)

    # 1. RBFKAN path: per-patch KAN processing
    #    Reshape [B, H, W, C] to patches, apply RBFKAN, reshape back
    rbf_out =
      rbfkan_path(input,
        dim: dim,
        spatial: spatial,
        eff_patch: eff_patch,
        num_rbf_centers: num_rbf_centers,
        name: "#{name}_rbf"
      )

    # 2. Axis-wise depthwise conv: horizontal + vertical mixing
    local_out =
      axis_wise_conv(input,
        dim: dim,
        kernel_size: dw_kernel_size,
        name: "#{name}_axis"
      )

    # 3. Low-rank global path: GAP -> bottleneck MLP -> broadcast
    global_out =
      global_path(input,
        dim: dim,
        spatial: spatial,
        reduction: global_reduction,
        name: "#{name}_global"
      )

    # Combine all three paths
    Axon.layer(
      fn rbf, local, global_feat, _opts ->
        Nx.add(Nx.add(rbf, local), global_feat)
      end,
      [rbf_out, local_out, global_out],
      name: "#{name}_combine",
      op_name: :vik_combine
    )
  end

  # ============================================================================
  # RBFKAN Path
  # ============================================================================

  defp rbfkan_path(input, opts) do
    dim = Keyword.fetch!(opts, :dim)
    spatial = Keyword.fetch!(opts, :spatial)
    eff_patch = Keyword.fetch!(opts, :eff_patch)
    num_rbf_centers = Keyword.fetch!(opts, :num_rbf_centers)
    name = Keyword.get(opts, :name, "rbfkan")

    # Pad spatial dims to be divisible by eff_patch if needed
    num_patches_h = div(spatial + eff_patch - 1, eff_patch)
    padded_spatial = num_patches_h * eff_patch

    # RBF parameters: centers, widths, weights
    # For channel-wise RBFKAN, apply RBF per channel independently
    # Shape: [num_rbf_centers] for centers and sigmas, [dim, num_rbf_centers] for weights
    rbf_centers =
      Axon.param("#{name}_centers", {num_rbf_centers},
        initializer: fn shape, _type, _key ->
          # Evenly spaced centers from -1 to 1
          g = elem(shape, 0)
          Nx.linspace(-1.0, 1.0, n: g, type: :f32)
        end
      )

    rbf_sigmas =
      Axon.param("#{name}_sigmas", {num_rbf_centers},
        initializer: fn shape, _type, _key ->
          g = elem(shape, 0)
          Nx.broadcast(2.0 / g, shape) |> Nx.as_type(:f32)
        end
      )

    rbf_weights =
      Axon.param("#{name}_rbf_w", {dim, num_rbf_centers},
        initializer: :lecun_uniform
      )

    base_weight =
      Axon.param("#{name}_base_w", {dim},
        initializer: fn shape, _type, _key ->
          Nx.broadcast(0.5, shape) |> Nx.as_type(:f32)
        end
      )

    spline_weight =
      Axon.param("#{name}_spline_w", {dim},
        initializer: fn shape, _type, _key ->
          Nx.broadcast(0.5, shape) |> Nx.as_type(:f32)
        end
      )

    Axon.layer(
      &rbfkan_impl/7,
      [input, rbf_centers, rbf_sigmas, rbf_weights, base_weight, spline_weight],
      name: "#{name}_compute",
      op_name: :rbfkan,
      spatial: spatial,
      eff_patch: eff_patch,
      padded_spatial: padded_spatial,
      dim: dim,
      num_rbf_centers: num_rbf_centers
    )
  end

  defp rbfkan_impl(input, centers, sigmas, rbf_w, base_w, spline_w, opts) do
    spatial = opts[:spatial]
    eff_patch = opts[:eff_patch]
    padded_spatial = opts[:padded_spatial]
    dim = opts[:dim]
    _num_rbf_centers = opts[:num_rbf_centers]

    {batch, _h, _w, _c} = Nx.shape(input)

    # Pad if needed
    x =
      if padded_spatial > spatial do
        pad_h = padded_spatial - spatial
        pad_w = padded_spatial - spatial
        Nx.pad(input, 0.0, [{0, 0, 0}, {0, pad_h, 0}, {0, pad_w, 0}, {0, 0, 0}])
      else
        input
      end

    num_patches = div(padded_spatial, eff_patch)

    # Reshape to patches: [B, nH, P, nW, P, C] -> [B*nH*nW, P*P, C]
    x =
      x
      |> Nx.reshape({batch, num_patches, eff_patch, num_patches, eff_patch, dim})
      |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
      |> Nx.reshape({batch * num_patches * num_patches, eff_patch * eff_patch, dim})

    # Apply RBFKAN per channel:
    # phi(x) = base_w * silu(x) + spline_w * sum(rbf_w * rbf(x, centers, sigmas))

    # Base activation: silu(x)
    base = Nx.multiply(x, Nx.sigmoid(x))

    # RBF activation: for each element, compute RBF basis values
    # x: [BPP, PP, C], centers: [G], sigmas: [G]
    # Expand for broadcasting: x -> [BPP, PP, C, 1], centers -> [1, 1, 1, G]
    x_exp = Nx.new_axis(x, 3)
    c_exp = Nx.reshape(centers, {1, 1, 1, Nx.axis_size(centers, 0)})
    s_exp = Nx.reshape(sigmas, {1, 1, 1, Nx.axis_size(sigmas, 0)})

    # RBF: exp(-||x - c||^2 / (2 * sigma^2))
    diff = Nx.subtract(x_exp, c_exp)
    rbf_vals = Nx.exp(Nx.negate(Nx.divide(Nx.multiply(diff, diff), Nx.multiply(2.0, Nx.multiply(s_exp, s_exp)))))
    # rbf_vals: [BPP, PP, C, G]

    # Weighted sum: sum over G with per-channel weights
    # rbf_w: [C, G] -> [1, 1, C, G]
    w_exp = Nx.reshape(rbf_w, {1, 1, dim, Nx.axis_size(rbf_w, 1)})
    spline = Nx.sum(Nx.multiply(rbf_vals, w_exp), axes: [3])
    # spline: [BPP, PP, C]

    # Combine: base_w * base + spline_w * spline
    # base_w, spline_w: [C] -> [1, 1, C]
    bw = Nx.reshape(base_w, {1, 1, dim})
    sw = Nx.reshape(spline_w, {1, 1, dim})
    out = Nx.add(Nx.multiply(bw, base), Nx.multiply(sw, spline))

    # Reshape back to spatial: [B, nH, nW, P, P, C] -> [B, H, W, C]
    out =
      out
      |> Nx.reshape({batch, num_patches, num_patches, eff_patch, eff_patch, dim})
      |> Nx.transpose(axes: [0, 1, 3, 2, 4, 5])
      |> Nx.reshape({batch, padded_spatial, padded_spatial, dim})

    # Crop back to original spatial size
    if padded_spatial > spatial do
      Nx.slice_along_axis(out, 0, spatial, axis: 1)
      |> Nx.slice_along_axis(0, spatial, axis: 2)
    else
      out
    end
  end

  # ============================================================================
  # Axis-wise Depthwise Convolution
  # ============================================================================

  defp axis_wise_conv(input, opts) do
    dim = Keyword.fetch!(opts, :dim)
    kernel_size = Keyword.fetch!(opts, :kernel_size)
    name = Keyword.get(opts, :name, "axis_conv")

    # Horizontal: [1, K] depthwise conv
    h_conv =
      Axon.conv(input, dim,
        kernel_size: {1, kernel_size},
        padding: [{0, 0}, {div(kernel_size - 1, 2), div(kernel_size, 2)}],
        feature_group_size: dim,
        name: "#{name}_horizontal"
      )

    # Vertical: [K, 1] depthwise conv
    v_conv =
      Axon.conv(input, dim,
        kernel_size: {kernel_size, 1},
        padding: [{div(kernel_size - 1, 2), div(kernel_size, 2)}, {0, 0}],
        feature_group_size: dim,
        name: "#{name}_vertical"
      )

    Axon.add(h_conv, v_conv, name: "#{name}_sum")
  end

  # ============================================================================
  # Low-rank Global Path
  # ============================================================================

  defp global_path(input, opts) do
    dim = Keyword.fetch!(opts, :dim)
    spatial = Keyword.fetch!(opts, :spatial)
    reduction = Keyword.fetch!(opts, :reduction)
    name = Keyword.get(opts, :name, "global")

    bottleneck = max(div(dim, reduction), 1)

    # Global average pool: [B, H, W, C] -> [B, C]
    pooled =
      Axon.nx(input, fn t -> Nx.mean(t, axes: [1, 2]) end,
        name: "#{name}_gap"
      )

    # Bottleneck MLP: C -> C/r -> C
    x =
      pooled
      |> Axon.dense(bottleneck, name: "#{name}_down")
      |> Axon.activation(:gelu, name: "#{name}_gelu")
      |> Axon.dense(dim, name: "#{name}_up")

    # Broadcast back to spatial: [B, C] -> [B, H, W, C]
    Axon.nx(x, fn t -> Nx.reshape(t, {Nx.axis_size(t, 0), 1, 1, Nx.axis_size(t, 1)}) |> Nx.broadcast({Nx.axis_size(t, 0), spatial, spatial, Nx.axis_size(t, 1)}) end,
      name: "#{name}_broadcast"
    )
  end

  # ============================================================================
  # Downsampling
  # ============================================================================

  defp downsample(input, next_dim, opts) do
    name = Keyword.get(opts, :name, "downsample")

    input
    |> Axon.layer_norm(name: "#{name}_norm")
    |> Axon.conv(next_dim,
      kernel_size: {2, 2},
      strides: [2, 2],
      name: "#{name}_conv"
    )
  end

  # ============================================================================
  # Backbone Behaviour + Utilities
  # ============================================================================

  @impl Edifice.Vision.Backbone
  def build_backbone(opts) do
    opts |> Keyword.delete(:num_classes) |> build()
  end

  @impl Edifice.Vision.Backbone
  def feature_size(opts) do
    channels = Keyword.get(opts, :channels, @default_channels)
    List.last(channels)
  end

  @impl Edifice.Vision.Backbone
  def input_shape(opts) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    {nil, in_channels, image_size, image_size}
  end

  @doc """
  Get the output size of a VisionKAN model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    channels = Keyword.get(opts, :channels, @default_channels)
    List.last(channels)
  end

  @doc """
  Recommended default configuration.
  """
  @spec recommended_defaults() :: keyword()
  def recommended_defaults do
    [
      image_size: 32,
      in_channels: 3,
      patch_size: 4,
      channels: [32, 64, 128, 256],
      depths: [2, 2, 4, 2],
      num_rbf_centers: 5,
      kan_patch_size: 7,
      dw_kernel_size: 7,
      global_reduction: 4,
      dropout: 0.1,
      ffn_expansion: 4
    ]
  end
end
