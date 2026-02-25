defmodule Edifice.Vision.MetaFormer do
  @moduledoc """
  MetaFormer: The general architecture behind ViT's success.

  Implements the MetaFormer framework and CAFormer from "MetaFormer Baselines
  for Vision" (Yu et al., 2022/2023). The key insight: ViT's power comes from
  the overall architecture (norm → token mixer → residual → norm → FFN → residual),
  not from the specific choice of self-attention as the token mixer.

  ## Key Insight

  Even replacing attention with **average pooling** (PoolFormer) achieves
  competitive results. This proves the MetaFormer architecture itself is the
  main contributor to performance, not the specific token mixer.

  ## MetaFormer Block

  ```
  Input
    |
    v
  +---------------------+
  | LayerNorm           |
  | Token Mixer (any)   |  ← pooling, conv, attention, etc.
  | + Residual          |
  +---------------------+
    |
    v
  +---------------------+
  | LayerNorm           |
  | FFN (MLP)           |
  | + Residual          |
  +---------------------+
    |
    v
  Output
  ```

  ## CAFormer (Conv-Attention Former)

  Best-performing MetaFormer variant using the optimal mixer for each stage:
  - Stages 1-2: Depthwise separable convolution (good for local patterns)
  - Stages 3-4: Self-attention (good for global patterns)

  ```
  Image → PatchEmbed → [Conv×3] → [Conv×3] → [Attn×9] → [Attn×3] → Pool → Head
           Stage 0       Stage 1    Stage 2     Stage 3    Stage 4
           dim=64        dim=128    dim=320     dim=512
  ```

  ## Token Mixers

  - `:pooling` — Average pooling (PoolFormer)
  - `:conv` — Depthwise separable convolution
  - `:attention` — Standard self-attention
  - Custom function — Any `(Axon.t(), keyword()) -> Axon.t()`

  ## Usage

      # Generic MetaFormer with any mixer
      model = MetaFormer.build_metaformer(
        image_size: 224,
        patch_size: 4,
        depths: [3, 3, 9, 3],
        dims: [64, 128, 320, 512],
        token_mixer: :attention
      )

      # CAFormer: conv stages then attention stages
      model = MetaFormer.build_caformer(
        image_size: 224,
        patch_size: 4,
        depths: [3, 3, 9, 3],
        dims: [64, 128, 320, 512]
      )

  ## References

  - "MetaFormer is Actually What You Need for Vision" (Yu et al., CVPR 2022)
  - "MetaFormer Baselines for Vision" (Yu et al., TPAMI 2023)
  - https://arxiv.org/abs/2210.13452
  """

  alias Edifice.Blocks.{FFN, PatchEmbed}

  @default_image_size 224
  @default_patch_size 4
  @default_in_channels 3
  @default_depths [3, 3, 9, 3]
  @default_dims [64, 128, 320, 512]
  @default_pool_size 3

  # ============================================================================
  # MetaFormer (Generic)
  # ============================================================================

  @doc """
  Build a MetaFormer model with a configurable token mixer.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Initial patch size (default: 4)
    - `:in_channels` - Number of input channels (default: 3)
    - `:depths` - Number of blocks per stage (default: [3, 3, 9, 3])
    - `:dims` - Hidden dimension per stage (default: [64, 128, 320, 512])
    - `:token_mixer` - Token mixer type: `:pooling`, `:conv`, `:attention` (default: `:pooling`)
    - `:pool_size` - Pooling kernel size when mixer is `:pooling` (default: 3)
    - `:num_classes` - Number of output classes (optional)

  ## Returns

    An Axon model. Without `:num_classes`, outputs `[batch, last_dim]`.
    With `:num_classes`, outputs `[batch, num_classes]`.
  """
  @typedoc "Options for `build_metaformer/1`."
  @type metaformer_opt ::
          {:depths, [pos_integer()]}
          | {:dims, [pos_integer()]}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:patch_size, pos_integer()}
          | {:pool_size, pos_integer()}
          | {:token_mixer, atom()}

  @spec build_metaformer([metaformer_opt()]) :: Axon.t()
  def build_metaformer(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    depths = Keyword.get(opts, :depths, @default_depths)
    dims = Keyword.get(opts, :dims, @default_dims)
    token_mixer = Keyword.get(opts, :token_mixer, :pooling)
    num_classes = Keyword.get(opts, :num_classes, nil)

    # Input: [batch, channels, height, width]
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Initial patch embedding into first stage dimension
    x =
      PatchEmbed.layer(input,
        image_size: image_size,
        patch_size: patch_size,
        in_channels: in_channels,
        embed_dim: hd(dims),
        name: "patch_embed"
      )

    # Build stages with same mixer for all stages
    x = build_stages(x, depths, dims, make_mixers(token_mixer, length(depths)), opts)

    # Final norm + global pool
    x = Axon.layer_norm(x, name: "final_norm")
    x = Axon.nx(x, fn tensor -> Nx.mean(tensor, axes: [1]) end, name: "global_pool")

    if num_classes do
      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  @doc """
  Build via `Edifice.build/2`. Dispatches to `build_metaformer/1` or
  `build_caformer/1` based on `:variant` option.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    case Keyword.get(opts, :variant) do
      :caformer -> build_caformer(opts)
      _ -> build_metaformer(opts)
    end
  end

  # ============================================================================
  # CAFormer
  # ============================================================================

  @doc """
  Build a CAFormer model (Conv stages + Attention stages).

  CAFormer uses depthwise convolution for the first two stages (local patterns)
  and self-attention for the last two stages (global patterns).

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Initial patch size (default: 4)
    - `:in_channels` - Number of input channels (default: 3)
    - `:depths` - Number of blocks per stage (default: [3, 3, 9, 3])
    - `:dims` - Hidden dimension per stage (default: [64, 128, 320, 512])
    - `:num_classes` - Number of output classes (optional)

  ## Returns

    An Axon model. Without `:num_classes`, outputs `[batch, last_dim]`.
    With `:num_classes`, outputs `[batch, num_classes]`.
  """
  @typedoc "Options for `build_caformer/1`."
  @type caformer_opt ::
          {:depths, [pos_integer()]}
          | {:dims, [pos_integer()]}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:patch_size, pos_integer()}

  @spec build_caformer([caformer_opt()]) :: Axon.t()
  def build_caformer(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    depths = Keyword.get(opts, :depths, @default_depths)
    dims = Keyword.get(opts, :dims, @default_dims)
    num_classes = Keyword.get(opts, :num_classes, nil)

    num_stages = length(depths)

    # CAFormer: first half conv, second half attention
    split = div(num_stages, 2)

    mixers =
      Enum.map(0..(num_stages - 1), fn idx ->
        if idx < split, do: :conv, else: :attention
      end)

    # Input
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Initial patch embedding
    x =
      PatchEmbed.layer(input,
        image_size: image_size,
        patch_size: patch_size,
        in_channels: in_channels,
        embed_dim: hd(dims),
        name: "patch_embed"
      )

    # Build stages with mixed token mixers
    x = build_stages(x, depths, dims, mixers, opts)

    # Final norm + global pool
    x = Axon.layer_norm(x, name: "final_norm")
    x = Axon.nx(x, fn tensor -> Nx.mean(tensor, axes: [1]) end, name: "global_pool")

    if num_classes do
      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  # ============================================================================
  # Stage Builder
  # ============================================================================

  defp build_stages(x, depths, dims, mixers, opts) do
    pool_size = Keyword.get(opts, :pool_size, @default_pool_size)

    stages = Enum.zip([depths, dims, mixers]) |> Enum.with_index()

    Enum.reduce(stages, x, fn {{depth, dim, mixer}, stage_idx}, acc ->
      # Downsample between stages (project to new dim)
      acc =
        if stage_idx > 0 do
          Axon.dense(acc, dim, name: "downsample_#{stage_idx}")
        else
          acc
        end

      # Stack of MetaFormer blocks
      Enum.reduce(0..(depth - 1), acc, fn block_idx, block_acc ->
        metaformer_block(block_acc, dim, mixer,
          pool_size: pool_size,
          name: "stage_#{stage_idx}_block_#{block_idx}"
        )
      end)
    end)
  end

  # ============================================================================
  # MetaFormer Block
  # ============================================================================

  defp metaformer_block(input, hidden_size, mixer, opts) do
    name = Keyword.get(opts, :name, "metaformer")
    pool_size = Keyword.get(opts, :pool_size, @default_pool_size)

    # Token mixing: LayerNorm -> Mixer -> Residual
    normed = Axon.layer_norm(input, name: "#{name}_token_norm")

    mixed =
      case mixer do
        :pooling -> pool_mixer(normed, pool_size, name: "#{name}_pool")
        :conv -> conv_mixer(normed, hidden_size, name: "#{name}_conv")
        :attention -> attention_mixer(normed, hidden_size, name: "#{name}_attn")
      end

    x = Axon.add(input, mixed, name: "#{name}_token_residual")

    # Channel mixing: LayerNorm -> FFN -> Residual
    channel_normed = Axon.layer_norm(x, name: "#{name}_channel_norm")
    ffn_out = FFN.layer(channel_normed, hidden_size: hidden_size, name: "#{name}_ffn")

    Axon.add(x, ffn_out, name: "#{name}_channel_residual")
  end

  # ============================================================================
  # Token Mixers
  # ============================================================================

  # Average pooling mixer (PoolFormer style): pooled - input
  defp pool_mixer(input, pool_size, opts) do
    name = Keyword.get(opts, :name, "pool")

    Axon.nx(
      input,
      fn tensor -> pool_subtract_compute(tensor, pool_size) end,
      name: "#{name}_mix"
    )
  end

  defp pool_subtract_compute(input, pool_size) do
    {batch, seq_len, dim} = Nx.shape(input)
    pad_total = pool_size - 1
    pad_before = div(pad_total, 2)
    pad_after = pad_total - pad_before

    padded =
      Nx.pad(input, 0.0, [{0, 0, 0}, {pad_before, pad_after, 0}, {0, 0, 0}])

    pooled =
      Enum.reduce(0..(pool_size - 1), Nx.broadcast(0.0, {batch, seq_len, dim}), fn offset, acc ->
        slice = Nx.slice_along_axis(padded, offset, seq_len, axis: 1)
        Nx.add(acc, slice)
      end)

    pooled = Nx.divide(pooled, pool_size)
    Nx.subtract(pooled, input)
  end

  # Depthwise separable convolution mixer (simulated on sequence)
  defp conv_mixer(input, hidden_size, opts) do
    name = Keyword.get(opts, :name, "conv")

    # Depthwise conv approximated as: dense -> local mixing -> dense
    x = Axon.dense(input, hidden_size, name: "#{name}_dw_in")

    # Local mixing via sliding window average (like depthwise conv with kernel 3)
    x =
      Axon.nx(
        x,
        fn tensor -> local_conv_compute(tensor) end,
        name: "#{name}_dw_mix"
      )

    x = Axon.activation(x, :gelu, name: "#{name}_gelu")
    Axon.dense(x, hidden_size, name: "#{name}_dw_out")
  end

  # 1D depthwise conv simulation with kernel size 3
  defp local_conv_compute(input) do
    {_batch, seq_len, _dim} = Nx.shape(input)

    padded = Nx.pad(input, 0.0, [{0, 0, 0}, {1, 1, 0}, {0, 0, 0}])

    left = Nx.slice_along_axis(padded, 0, seq_len, axis: 1)
    center = Nx.slice_along_axis(padded, 1, seq_len, axis: 1)
    right = Nx.slice_along_axis(padded, 2, seq_len, axis: 1)

    # Weighted sum: 0.25 * left + 0.5 * center + 0.25 * right
    Nx.add(
      Nx.multiply(0.25, Nx.add(left, right)),
      Nx.multiply(0.5, center)
    )
  end

  # Self-attention mixer
  defp attention_mixer(input, hidden_size, opts) do
    name = Keyword.get(opts, :name, "attn")
    num_heads = max(div(hidden_size, 64), 1)

    qkv = Axon.dense(input, hidden_size * 3, name: "#{name}_qkv")

    attended =
      Axon.layer(
        &mha_impl/2,
        [qkv],
        name: "#{name}_compute",
        embed_dim: hidden_size,
        num_heads: num_heads,
        head_dim: div(hidden_size, num_heads),
        op_name: :multi_head_attention
      )

    Axon.dense(attended, hidden_size, name: "#{name}_proj")
  end

  defp mha_impl(qkv, opts) do
    embed_dim = opts[:embed_dim]
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(qkv, 0)
    seq_len = Nx.axis_size(qkv, 1)

    q = Nx.slice_along_axis(qkv, 0, embed_dim, axis: 2)
    k = Nx.slice_along_axis(qkv, embed_dim, embed_dim, axis: 2)
    v = Nx.slice_along_axis(qkv, embed_dim * 2, embed_dim, axis: 2)

    q = q |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({batch, seq_len, num_heads, head_dim}) |> Nx.transpose(axes: [0, 2, 1, 3])

    scale = Nx.sqrt(Nx.tensor(head_dim, type: Nx.type(q)))
    scores = Nx.divide(Nx.dot(q, [3], [0, 1], k, [3], [0, 1]), scale)

    max_scores = Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp_scores = Nx.exp(Nx.subtract(scores, max_scores))
    attn_weights = Nx.divide(exp_scores, Nx.add(Nx.sum(exp_scores, axes: [-1], keep_axes: true), 1.0e-8))

    output = Nx.dot(attn_weights, [3], [0, 1], v, [2], [0, 1])

    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  defp make_mixers(mixer, num_stages) when is_atom(mixer) do
    List.duplicate(mixer, num_stages)
  end

  @doc """
  Get the output size of a MetaFormer model.
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
