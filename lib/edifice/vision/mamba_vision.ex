defmodule Edifice.Vision.MambaVision do
  @moduledoc """
  MambaVision: A Hybrid Mamba-Transformer Vision Backbone.

  Implements the MambaVision architecture from "MambaVision: A Hybrid
  Mamba-Transformer Vision Backbone" (Hatamizadeh & Kautz, NVIDIA, 2024).
  A hierarchical 4-stage vision backbone that uses CNN blocks in early
  stages and hybrid Mamba SSM + windowed self-attention in later stages.

  ## Key Innovation

  Instead of applying Mamba uniformly (like Vim/VMamba), MambaVision uses a
  stage-appropriate mix:
  - **Stages 1-2**: Pure CNN blocks (fast at high resolution)
  - **Stages 3-4**: First half Mamba SSM, second half windowed attention

  The MambaVisionMixer modifies standard Mamba with:
  1. **Non-causal convolution** (no directional bias for 2D data)
  2. **Dual-branch**: SSM on half channels, symmetric Conv+SiLU on other half
  3. **Concatenation** instead of multiplicative gating

  ## Architecture

  ```
  Input: (B, 3, 224, 224)
    -> PatchEmbed (2x Conv3x3 stride 2 = 4x downsample)
    -> Stage 1 (ConvBlocks)           -> Downsample (Conv stride 2)
    -> Stage 2 (ConvBlocks)           -> Downsample
    -> Stage 3 (Mamba + Attention)    -> Downsample
    -> Stage 4 (Mamba + Attention)
    -> LayerNorm -> Global Avg Pool -> Linear -> Output
  ```

  Channel progression: dim -> 2*dim -> 4*dim -> 8*dim

  ## Model Variants

  | Variant | dim | depths     | Params |
  |---------|-----|------------|--------|
  | Tiny    | 80  | [1,3,8,4]  | ~32M   |
  | Small   | 96  | [3,3,7,5]  | ~50M   |
  | Base    | 128 | [3,3,10,5] | ~98M   |

  ## Usage

      model = MambaVision.build(
        image_size: 224,
        dim: 80,
        depths: [1, 3, 8, 4],
        num_heads: [2, 4, 8, 16],
        num_classes: 10
      )

  ## References
  - Paper: https://arxiv.org/abs/2407.08083
  - Code: https://github.com/NVlabs/MambaVision
  """

  use Edifice.Vision.Backbone

  alias Edifice.Utils.FusedOps

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @default_image_size 224
  @default_in_channels 3
  @default_dim 80
  @default_depths [1, 3, 8, 4]
  @default_num_heads [2, 4, 8, 16]
  @default_mlp_ratio 4
  @default_dropout 0.0
  @default_d_state 8
  @default_d_conv 3

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a MambaVision model.

  ## Options
    - `:image_size` - Input image size, square (default: 224)
    - `:in_channels` - Number of input channels (default: 3)
    - `:dim` - Base channel dimension, doubles each stage (default: 80)
    - `:depths` - Number of blocks per stage (default: [1, 3, 8, 4])
    - `:num_heads` - Attention heads per stage (default: [2, 4, 8, 16])
    - `:mlp_ratio` - MLP expansion ratio in hybrid stages (default: 4)
    - `:dropout` - Dropout/drop path rate (default: 0.0)
    - `:d_state` - SSM state dimension (default: 8)
    - `:d_conv` - SSM convolution kernel size (default: 3)
    - `:num_classes` - Classification head size (optional)

  ## Returns
    Without `:num_classes`: `[batch, 8*dim]` feature vector.
    With `:num_classes`: `[batch, num_classes]` logits.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:d_conv, pos_integer()}
          | {:d_state, pos_integer()}
          | {:depths, [pos_integer()]}
          | {:dim, pos_integer()}
          | {:dropout, float()}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:mlp_ratio, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:num_heads, [pos_integer()]}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    dim = Keyword.get(opts, :dim, @default_dim)
    depths = Keyword.get(opts, :depths, @default_depths)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    d_state = Keyword.get(opts, :d_state, @default_d_state)
    d_conv = Keyword.get(opts, :d_conv, @default_d_conv)
    num_classes = Keyword.get(opts, :num_classes, nil)

    # Channel dims per stage: dim, 2*dim, 4*dim, 8*dim
    dims = [dim, dim * 2, dim * 4, dim * 8]

    # Input: [batch, channels, height, width]
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Patch embedding stem: two Conv3x3 stride 2 -> 4x downsample
    # Output: [batch, dim, H/4, W/4]
    x = build_stem(input, in_channels, dim)

    # Build 4 stages
    x =
      Enum.reduce(0..3, x, fn stage_idx, acc ->
        stage_dim = Enum.at(dims, stage_idx)
        stage_depth = Enum.at(depths, stage_idx)
        stage_heads = Enum.at(num_heads, stage_idx)
        is_hybrid = stage_idx >= 2

        # Build stage blocks
        stage_out =
          if is_hybrid do
            build_hybrid_stage(acc, stage_dim, stage_depth, stage_heads,
              mlp_ratio: mlp_ratio,
              dropout: dropout,
              d_state: d_state,
              d_conv: d_conv,
              name: "stage_#{stage_idx}"
            )
          else
            build_conv_stage(acc, stage_dim, stage_depth,
              dropout: dropout,
              name: "stage_#{stage_idx}"
            )
          end

        # Downsample between stages (not after the last stage)
        if stage_idx < 3 do
          next_dim = Enum.at(dims, stage_idx + 1)
          build_downsample(stage_out, next_dim, "downsample_#{stage_idx}")
        else
          stage_out
        end
      end)

    # Final: LayerNorm -> Global Average Pooling -> Flatten
    x = Axon.layer_norm(x, name: "final_norm")

    # Global average pool: [batch, C, H, W] -> [batch, C]
    x =
      Axon.nx(
        x,
        fn tensor ->
          Nx.mean(tensor, axes: [2, 3])
        end,
        name: "global_avg_pool"
      )

    # Classification head
    if num_classes do
      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  # ============================================================================
  # Stem (Patch Embedding)
  # ============================================================================

  defp build_stem(input, _in_channels, dim) do
    # Two Conv2d(3x3, stride=2) + BatchNorm + ReLU
    # We use Axon.conv for 2D convolution
    # Input: [batch, in_channels, H, W] -> [batch, dim, H/4, W/4]

    # Intermediate dimension (32 for small models)
    in_dim = min(dim, 64)

    input
    |> Axon.conv(in_dim,
      kernel_size: 3,
      strides: 2,
      padding: [{1, 1}, {1, 1}],
      channels: :first,
      name: "stem_conv1"
    )
    |> Axon.batch_norm(name: "stem_bn1", channel_index: 1)
    |> Axon.activation(:relu, name: "stem_relu1")
    |> Axon.conv(dim,
      kernel_size: 3,
      strides: 2,
      padding: [{1, 1}, {1, 1}],
      channels: :first,
      name: "stem_conv2"
    )
    |> Axon.batch_norm(name: "stem_bn2", channel_index: 1)
    |> Axon.activation(:relu, name: "stem_relu2")
  end

  # ============================================================================
  # Convolutional Stage (Stages 1-2)
  # ============================================================================

  defp build_conv_stage(input, dim, depth, opts) do
    name = Keyword.get(opts, :name, "conv_stage")

    Enum.reduce(0..(depth - 1), input, fn block_idx, acc ->
      build_conv_block(acc, dim, "#{name}_block_#{block_idx}")
    end)
  end

  # ConvBlock: Conv3x3 -> BN -> GELU -> Conv3x3 -> BN + residual
  defp build_conv_block(input, dim, name) do
    x =
      input
      |> Axon.conv(dim,
        kernel_size: 3,
        padding: [{1, 1}, {1, 1}],
        channels: :first,
        name: "#{name}_conv1"
      )
      |> Axon.batch_norm(name: "#{name}_bn1", channel_index: 1)
      |> Axon.activation(:gelu, name: "#{name}_gelu")
      |> Axon.conv(dim,
        kernel_size: 3,
        padding: [{1, 1}, {1, 1}],
        channels: :first,
        name: "#{name}_conv2"
      )
      |> Axon.batch_norm(name: "#{name}_bn2", channel_index: 1)

    Axon.add(input, x, name: "#{name}_residual")
  end

  # ============================================================================
  # Hybrid Stage (Stages 3-4): Mamba + Attention
  # ============================================================================

  defp build_hybrid_stage(input, dim, depth, num_heads, opts) do
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    d_state = Keyword.get(opts, :d_state, @default_d_state)
    d_conv = Keyword.get(opts, :d_conv, @default_d_conv)
    name = Keyword.get(opts, :name, "hybrid_stage")

    # First half: Mamba blocks, second half: attention blocks
    mamba_count = div(depth, 2) + rem(depth, 2)

    # Reshape 2D -> 1D sequence for hybrid blocks: [B, C, H, W] -> [B, H*W, C]
    x =
      Axon.nx(
        input,
        fn tensor ->
          batch = Nx.axis_size(tensor, 0)
          c = Nx.axis_size(tensor, 1)
          h = Nx.axis_size(tensor, 2)
          w = Nx.axis_size(tensor, 3)
          tensor |> Nx.transpose(axes: [0, 2, 3, 1]) |> Nx.reshape({batch, h * w, c})
        end,
        name: "#{name}_to_seq"
      )

    # Build all blocks
    x =
      Enum.reduce(0..(depth - 1), x, fn block_idx, acc ->
        if block_idx < mamba_count do
          build_mamba_vision_block(acc, dim,
            mlp_ratio: mlp_ratio,
            d_state: d_state,
            d_conv: d_conv,
            name: "#{name}_mamba_#{block_idx}"
          )
        else
          build_attention_block(acc, dim, num_heads,
            mlp_ratio: mlp_ratio,
            name: "#{name}_attn_#{block_idx}"
          )
        end
      end)

    # Reshape back: [B, H*W, C] -> [B, C, H, W]
    Axon.layer(
      &reshape_seq_to_2d/3,
      [x, input],
      name: "#{name}_to_2d",
      op_name: :reshape_seq_to_2d
    )
  end

  defp reshape_seq_to_2d(seq, original, _opts) do
    batch = Nx.axis_size(original, 0)
    c = Nx.axis_size(original, 1)
    h = Nx.axis_size(original, 2)
    w = Nx.axis_size(original, 3)
    seq |> Nx.reshape({batch, h, w, c}) |> Nx.transpose(axes: [0, 3, 1, 2])
  end

  # ============================================================================
  # MambaVision Mixer Block
  # ============================================================================

  # Block: LayerNorm -> MambaVisionMixer + residual -> LayerNorm -> MLP + residual
  defp build_mamba_vision_block(input, dim, opts) do
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    d_state = Keyword.get(opts, :d_state, @default_d_state)
    d_conv = Keyword.get(opts, :d_conv, @default_d_conv)
    name = Keyword.get(opts, :name, "mamba_block")

    # Mixer sub-block
    normed = Axon.layer_norm(input, name: "#{name}_norm1")

    mixed =
      build_mamba_vision_mixer(normed, dim,
        d_state: d_state,
        d_conv: d_conv,
        name: "#{name}_mixer"
      )

    x = Axon.add(input, mixed, name: "#{name}_residual1")

    # MLP sub-block
    normed2 = Axon.layer_norm(x, name: "#{name}_norm2")
    mlp_dim = dim * mlp_ratio

    ffn =
      normed2
      |> Axon.dense(mlp_dim, name: "#{name}_mlp_up")
      |> Axon.activation(:gelu, name: "#{name}_mlp_gelu")
      |> Axon.dense(dim, name: "#{name}_mlp_down")

    Axon.add(x, ffn, name: "#{name}_residual2")
  end

  # MambaVisionMixer: dual-branch (SSM on half channels + symmetric on other half)
  defp build_mamba_vision_mixer(input, dim, opts) do
    d_state = Keyword.get(opts, :d_state, @default_d_state)
    d_conv = Keyword.get(opts, :d_conv, @default_d_conv)
    name = Keyword.get(opts, :name, "mixer")

    d_inner = dim
    half_dim = div(d_inner, 2)

    # Input projection
    projected = Axon.dense(input, d_inner, name: "#{name}_in_proj")

    # Split into SSM branch (x) and symmetric branch (z)
    x_branch =
      Axon.nx(projected, fn t -> Nx.slice_along_axis(t, 0, half_dim, axis: 2) end,
        name: "#{name}_split_x"
      )

    z_branch =
      Axon.nx(projected, fn t -> Nx.slice_along_axis(t, half_dim, half_dim, axis: 2) end,
        name: "#{name}_split_z"
      )

    # SSM branch: Conv1d + SiLU + SSM scan
    ssm_out =
      x_branch
      |> Axon.conv(half_dim,
        kernel_size: d_conv,
        padding: :same,
        feature_group_size: half_dim,
        name: "#{name}_ssm_conv"
      )
      |> Axon.activation(:silu, name: "#{name}_ssm_silu")
      |> build_simple_ssm(half_dim, d_state, "#{name}_ssm")

    # Symmetric branch: Conv1d + SiLU (no SSM)
    sym_out =
      z_branch
      |> Axon.conv(half_dim,
        kernel_size: d_conv,
        padding: :same,
        feature_group_size: half_dim,
        name: "#{name}_sym_conv"
      )
      |> Axon.activation(:silu, name: "#{name}_sym_silu")

    # Concatenate branches
    combined =
      Axon.layer(
        fn ssm, sym, _opts ->
          Nx.concatenate([ssm, sym], axis: -1)
        end,
        [ssm_out, sym_out],
        name: "#{name}_concat",
        op_name: :concat
      )

    # Output projection
    Axon.dense(combined, dim, name: "#{name}_out_proj")
  end

  # Simple SSM: project to B, C, dt -> discretize -> sequential scan -> output
  defp build_simple_ssm(input, d_inner, d_state, name) do
    dt_rank = max(div(d_inner, 16), 1)

    # Project to dt, B, C
    params = Axon.dense(input, dt_rank + 2 * d_state, name: "#{name}_x_proj")

    dt_proj =
      Axon.dense(
        Axon.nx(params, fn t -> Nx.slice_along_axis(t, 0, dt_rank, axis: 2) end,
          name: "#{name}_dt_slice"
        ),
        d_inner,
        name: "#{name}_dt_proj"
      )

    # Run the SSM
    Axon.layer(
      &ssm_scan_impl/4,
      [input, params, dt_proj],
      name: "#{name}_scan",
      d_inner: d_inner,
      d_state: d_state,
      dt_rank: dt_rank,
      op_name: :ssm_scan
    )
  end

  defp ssm_scan_impl(x, params, dt, opts) do
    d_inner = opts[:d_inner]
    d_state = opts[:d_state]
    dt_rank = opts[:dt_rank]
    batch = Nx.axis_size(x, 0)
    seq_len = Nx.axis_size(x, 1)

    # Extract B and C from params
    b = Nx.slice_along_axis(params, dt_rank, d_state, axis: 2)
    c = Nx.slice_along_axis(params, dt_rank + d_state, d_state, axis: 2)

    # dt with softplus activation
    delta = Nx.log1p(Nx.exp(dt))

    # A parameter: initialized as negative log-spaced values
    a_log =
      Nx.iota({d_inner, d_state}, type: :f32, axis: 1) |> Nx.add(1) |> Nx.log() |> Nx.negate()

    # Discretize: A_bar = exp(delta * A), B_bar = delta * B
    # delta: [batch, seq, d_inner], a_log: [d_inner, d_state]
    delta_expanded = Nx.new_axis(delta, 3)
    a_bar = Nx.exp(Nx.multiply(delta_expanded, a_log))
    b_bar = Nx.multiply(Nx.new_axis(delta, 3), Nx.new_axis(b, 2))

    # Sequential scan
    h_init = Nx.broadcast(Nx.tensor(0.0, type: :f32), {batch, d_inner, d_state})

    {_, outputs} =
      Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
        a_t = Nx.slice_along_axis(a_bar, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        bx_t = Nx.slice_along_axis(b_bar, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        x_t = Nx.slice_along_axis(x, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(c, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        # h = A_bar * h + B_bar * x
        bx = Nx.multiply(bx_t, Nx.new_axis(x_t, 2))
        h = Nx.add(Nx.multiply(a_t, h_prev), bx)

        # y = C * h (sum over state dim)
        y = Nx.sum(Nx.multiply(Nx.new_axis(c_t, 1), h), axes: [2])

        {h, [y | acc]}
      end)

    outputs |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # Attention Block (windowed self-attention)
  # ============================================================================

  # Block: LayerNorm -> Attention + residual -> LayerNorm -> MLP + residual
  defp build_attention_block(input, dim, num_heads, opts) do
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    name = Keyword.get(opts, :name, "attn_block")

    # Attention sub-block
    normed = Axon.layer_norm(input, name: "#{name}_norm1")

    attended = build_self_attention(normed, dim, num_heads, "#{name}_attn")

    x = Axon.add(input, attended, name: "#{name}_residual1")

    # MLP sub-block
    normed2 = Axon.layer_norm(x, name: "#{name}_norm2")
    mlp_dim = dim * mlp_ratio

    ffn =
      normed2
      |> Axon.dense(mlp_dim, name: "#{name}_mlp_up")
      |> Axon.activation(:gelu, name: "#{name}_mlp_gelu")
      |> Axon.dense(dim, name: "#{name}_mlp_down")

    Axon.add(x, ffn, name: "#{name}_residual2")
  end

  # Standard multi-head self-attention (no position bias, no window shifting)
  defp build_self_attention(input, dim, num_heads, name) do
    qkv = Axon.dense(input, dim * 3, name: "#{name}_qkv")

    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          q = Nx.slice_along_axis(qkv_tensor, 0, dim, axis: 2)
          k = Nx.slice_along_axis(qkv_tensor, dim, dim, axis: 2)
          v = Nx.slice_along_axis(qkv_tensor, dim * 2, dim, axis: 2)

          head_dim = div(dim, num_heads)
          scale = Nx.sqrt(Nx.tensor(head_dim, type: :f32))

          scores = Nx.dot(q, [2], [0], k, [2], [0])
          scores = Nx.divide(scores, scale)
          weights = FusedOps.fused_softmax(scores)

          Nx.dot(weights, [2], [0], v, [1], [0])
        end,
        name: "#{name}_compute"
      )

    Axon.dense(attended, dim, name: "#{name}_proj")
  end

  # ============================================================================
  # Downsample
  # ============================================================================

  # Conv2d stride 2 to halve spatial and double channels
  defp build_downsample(input, out_dim, name) do
    Axon.conv(input, out_dim,
      kernel_size: 3,
      strides: 2,
      padding: [{1, 1}, {1, 1}],
      channels: :first,
      name: name
    )
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a MambaVision model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    dim = Keyword.get(opts, :dim, @default_dim)

    case Keyword.get(opts, :num_classes) do
      nil -> dim * 8
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
    dim = Keyword.get(opts, :dim, @default_dim)
    dim * 8
  end

  @doc """
  Get the Tiny variant configuration.
  """
  @spec tiny_config() :: keyword()
  def tiny_config do
    [dim: 80, depths: [1, 3, 8, 4], num_heads: [2, 4, 8, 16]]
  end

  @doc """
  Get the Small variant configuration.
  """
  @spec small_config() :: keyword()
  def small_config do
    [dim: 96, depths: [3, 3, 7, 5], num_heads: [2, 4, 8, 16]]
  end

  @doc """
  Get the Base variant configuration.
  """
  @spec base_config() :: keyword()
  def base_config do
    [dim: 128, depths: [3, 3, 10, 5], num_heads: [2, 4, 8, 16]]
  end
end
