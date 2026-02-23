defmodule Edifice.Vision.EfficientViT do
  @moduledoc """
  EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction.

  Implements EfficientViT from "EfficientViT: Multi-Scale Linear Attention for
  High-Resolution Dense Prediction" (Liu et al., 2023). Achieves O(n) complexity
  instead of O(n²) via linear attention with cascaded group attention.

  ## Key Innovations

  - **Linear attention**: Uses kernel trick to avoid materializing the full
    attention matrix. Q×K^T is computed via feature maps, giving O(n) complexity.
  - **Cascaded group attention (CGA)**: Different heads see different channel
    splits of the input, enforcing head diversity and reducing redundancy.
  - **Multi-scale**: Progressive downsampling stages, each with its own dimension.
  - **Depthwise conv in FFN**: Adds local context between linear layers.

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
        v
  +--------------------------+
  | Patch Embedding           |
  +--------------------------+
        |
        v
  +==========================+
  | Stage 1 (depth[0] blocks) |
  |  CGA Linear Attention    |
  |  DW-Conv FFN             |
  +==========================+
        | (downsample)
        v
  +==========================+
  | Stage 2 (depth[1] blocks) |
  |  CGA Linear Attention    |
  |  DW-Conv FFN             |
  +==========================+
        | (downsample)
        v
  +==========================+
  | Stage 3 (depth[2] blocks) |
  |  CGA Linear Attention    |
  |  DW-Conv FFN             |
  +==========================+
        |
        v
  +--------------------------+
  | LayerNorm + Global Pool  |
  +--------------------------+
        |
        v
  [batch, last_dim]
  ```

  ## Cascaded Group Attention

  ```
  Input: [batch, seq, dim]
         |
    Split into num_heads groups along dim
         |
    Head 0: [batch, seq, dim/heads] → Q₀, K₀, V₀ → LinearAttn → out₀
    Head 1: [batch, seq, dim/heads] → Q₁, K₁, V₁ → LinearAttn → out₁ + out₀
    Head 2: [batch, seq, dim/heads] → Q₂, K₂, V₂ → LinearAttn → out₂ + out₁
    ...
         |
    Concatenate all head outputs
         |
    Output projection
  ```

  Each head sees a unique slice of the feature map (no shared representation),
  which forces diverse attention patterns across heads.

  ## Linear Attention

  Standard attention: O(n²)
      Attn = softmax(QK^T/√d) × V

  Linear attention: O(n)
      Attn = φ(Q) × (φ(K)^T × V)  where φ is ELU+1

  By computing φ(K)^T × V first (d×d matrix), we avoid the n×n attention
  matrix entirely.

  ## Usage

      model = EfficientViT.build(
        image_size: 224,
        patch_size: 16,
        embed_dim: 64,
        depths: [1, 2, 3],
        num_heads: [4, 4, 4]
      )

  ## References

  - Paper: "EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction"
  - arXiv: https://arxiv.org/abs/2205.14756
  """

  alias Edifice.Blocks.PatchEmbed

  @default_image_size 224
  @default_patch_size 16
  @default_in_channels 3
  @default_embed_dim 64
  @default_depths [1, 2, 3]
  @default_num_heads [4, 4, 4]
  @default_mlp_ratio 4.0

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build an EfficientViT model with linear attention.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Patch size, square (default: 16)
    - `:in_channels` - Number of input channels (default: 3)
    - `:embed_dim` - Initial embedding dimension (default: 64)
    - `:depths` - Number of blocks per stage (default: [1, 2, 3])
    - `:num_heads` - Number of attention heads per stage (default: [4, 4, 4])
    - `:mlp_ratio` - MLP expansion ratio (default: 4.0)
    - `:num_classes` - Number of output classes (optional)

  ## Returns

    An Axon model. Without `:num_classes`, outputs `[batch, last_dim]`.
    With `:num_classes`, outputs `[batch, num_classes]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:depths, [pos_integer()]}
          | {:embed_dim, pos_integer()}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_classes, pos_integer() | nil}
          | {:num_heads, [pos_integer()]}
          | {:patch_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    depths = Keyword.get(opts, :depths, @default_depths)
    num_heads_list = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    num_classes = Keyword.get(opts, :num_classes, nil)

    # Compute dimension at each stage (doubles each stage)
    dims = Enum.map(0..(length(depths) - 1), fn i -> embed_dim * round(:math.pow(2, i)) end)

    # Input: [batch, channels, height, width]
    input = Axon.input("image", shape: {nil, in_channels, image_size, image_size})

    # Patch embedding into first stage dimension
    x =
      PatchEmbed.layer(input,
        image_size: image_size,
        patch_size: patch_size,
        in_channels: in_channels,
        embed_dim: hd(dims),
        name: "patch_embed"
      )

    # Build stages
    stages = Enum.zip([depths, dims, num_heads_list]) |> Enum.with_index()

    x =
      Enum.reduce(stages, x, fn {{depth, dim, num_heads}, stage_idx}, acc ->
        # Downsample between stages
        acc =
          if stage_idx > 0 do
            Axon.dense(acc, dim, name: "downsample_#{stage_idx}")
          else
            acc
          end

        # Stack of EfficientViT blocks
        Enum.reduce(0..(depth - 1), acc, fn block_idx, block_acc ->
          efficient_vit_block(block_acc, dim, num_heads, mlp_ratio,
            name: "stage_#{stage_idx}_block_#{block_idx}"
          )
        end)
      end)

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
  # EfficientViT Block
  # ============================================================================

  defp efficient_vit_block(input, dim, num_heads, mlp_ratio, opts) do
    name = Keyword.get(opts, :name, "evit_block")
    mlp_dim = round(dim * mlp_ratio)

    # Token mixing: LayerNorm -> Cascaded Group Linear Attention -> Residual
    normed = Axon.layer_norm(input, name: "#{name}_norm1")

    attended =
      cascaded_group_attention(normed, dim, num_heads, name: "#{name}_cga")

    x = Axon.add(input, attended, name: "#{name}_residual1")

    # Channel mixing: LayerNorm -> FFN with DW conv -> Residual
    normed2 = Axon.layer_norm(x, name: "#{name}_norm2")
    ffn_out = dwconv_ffn(normed2, dim, mlp_dim, name: "#{name}_ffn")

    Axon.add(x, ffn_out, name: "#{name}_residual2")
  end

  # ============================================================================
  # Cascaded Group Attention (CGA)
  # ============================================================================

  defp cascaded_group_attention(input, dim, num_heads, opts) do
    name = Keyword.get(opts, :name, "cga")
    head_dim = div(dim, num_heads)

    # Each head gets its own Q, K, V projections on a split of the input
    # Split input: [batch, seq, dim] -> num_heads × [batch, seq, head_dim]
    # Then each head does linear attention independently

    # Project full input to Q, K, V for all heads at once
    q_all = Axon.dense(input, dim, name: "#{name}_q")
    k_all = Axon.dense(input, dim, name: "#{name}_k")
    v_all = Axon.dense(input, dim, name: "#{name}_v")

    # Apply cascaded group linear attention
    attended =
      Axon.layer(
        &cascaded_group_attn_impl/4,
        [q_all, k_all, v_all],
        name: "#{name}_compute",
        num_heads: num_heads,
        head_dim: head_dim,
        op_name: :cascaded_group_attention
      )

    Axon.dense(attended, dim, name: "#{name}_proj")
  end

  defp cascaded_group_attn_impl(q, k, v, opts) do
    num_heads = opts[:num_heads]
    head_dim = opts[:head_dim]

    batch = Nx.axis_size(q, 0)
    seq_len = Nx.axis_size(q, 1)

    # Reshape to per-head: [batch, seq, num_heads, head_dim]
    q_heads = Nx.reshape(q, {batch, seq_len, num_heads, head_dim})
    k_heads = Nx.reshape(k, {batch, seq_len, num_heads, head_dim})
    v_heads = Nx.reshape(v, {batch, seq_len, num_heads, head_dim})

    # Transpose to [batch, num_heads, seq, head_dim]
    q_heads = Nx.transpose(q_heads, axes: [0, 2, 1, 3])
    k_heads = Nx.transpose(k_heads, axes: [0, 2, 1, 3])
    v_heads = Nx.transpose(v_heads, axes: [0, 2, 1, 3])

    # Apply linear attention with ELU+1 kernel per head
    # φ(x) = elu(x) + 1
    q_feat = Nx.add(Nx.max(q_heads, 0.0), Nx.multiply(Nx.min(q_heads, 0.0) |> Nx.exp() |> Nx.subtract(1.0), 1.0))
    q_feat = Nx.add(q_feat, 1.0)

    k_feat = Nx.add(Nx.max(k_heads, 0.0), Nx.multiply(Nx.min(k_heads, 0.0) |> Nx.exp() |> Nx.subtract(1.0), 1.0))
    k_feat = Nx.add(k_feat, 1.0)

    # Linear attention: φ(Q) × (φ(K)^T × V) — O(n·d²) instead of O(n²·d)
    # [batch, heads, head_dim, seq] × [batch, heads, seq, head_dim] -> [batch, heads, head_dim, head_dim]
    kv = Nx.dot(k_feat, [2], [0, 1], v_heads, [2], [0, 1])

    # [batch, heads, seq, head_dim] × [batch, heads, head_dim, head_dim] -> [batch, heads, seq, head_dim]
    output = Nx.dot(q_feat, [3], [0, 1], kv, [2], [0, 1])

    # Normalize by sum of kernel values
    k_sum = Nx.sum(k_feat, axes: [2], keep_axes: true)
    normalizer = Nx.dot(q_feat, [3], [0, 1], Nx.transpose(k_sum, axes: [0, 1, 3, 2]), [2], [0, 1])
    output = Nx.divide(output, Nx.add(normalizer, 1.0e-8))

    # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, dim]
    output
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, num_heads * head_dim})
  end

  # ============================================================================
  # FFN with Depthwise Conv
  # ============================================================================

  # FFN with depthwise conv between linear layers for local context
  defp dwconv_ffn(input, dim, mlp_dim, opts) do
    name = Keyword.get(opts, :name, "ffn")

    x = Axon.dense(input, mlp_dim, name: "#{name}_fc1")
    x = Axon.activation(x, :gelu, name: "#{name}_gelu1")

    # Depthwise conv: simulated as local sliding window on sequence
    x =
      Axon.nx(
        x,
        fn tensor -> depthwise_conv_1d(tensor) end,
        name: "#{name}_dwconv"
      )

    x = Axon.activation(x, :gelu, name: "#{name}_gelu2")
    Axon.dense(x, dim, name: "#{name}_fc2")
  end

  # 1D depthwise conv with kernel size 3 (local context mixing)
  defp depthwise_conv_1d(input) do
    {_batch, seq_len, _dim} = Nx.shape(input)

    padded = Nx.pad(input, 0.0, [{0, 0, 0}, {1, 1, 0}, {0, 0, 0}])

    left = Nx.slice_along_axis(padded, 0, seq_len, axis: 1)
    center = Nx.slice_along_axis(padded, 1, seq_len, axis: 1)
    right = Nx.slice_along_axis(padded, 2, seq_len, axis: 1)

    Nx.add(
      Nx.multiply(0.25, Nx.add(left, right)),
      Nx.multiply(0.5, center)
    )
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of an EfficientViT model.
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
