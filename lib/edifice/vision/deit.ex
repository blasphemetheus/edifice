defmodule Edifice.Vision.DeiT do
  @moduledoc """
  Data-efficient Image Transformer (DeiT) implementation.

  Extends ViT with a distillation token that learns from a teacher model.
  During training, the CLS token produces the classification output and the
  distillation token produces the teacher-aligned output. At inference, both
  token outputs can be averaged for improved accuracy.

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
  +-----v--------------------+
  | Patch Embedding           |  Split into P x P patches, linear project
  +---------------------------+
        |
        v
  [batch, num_patches, embed_dim]
        |
  +-----v--------------------+
  | Prepend CLS + Dist Tokens |  Two learnable [1, 1, embed_dim] tokens
  +---------------------------+
        |
        v
  [batch, num_patches + 2, embed_dim]
        |
  +-----v--------------------+
  | Add Position Embedding    |  Learnable [1, num_patches + 2, embed_dim]
  +---------------------------+
        |
        v
  +-----v--------------------+
  | Transformer Block x N     |
  |   LayerNorm -> Attention  |
  |   Residual -> LayerNorm   |
  |   MLP -> Residual         |
  +---------------------------+
        |
        v
  +-----v--------------------+
  | Extract CLS (idx 0)       |  -> Classification head
  | Extract Dist (idx 1)      |  -> Distillation head (teacher loss)
  +---------------------------+
  ```

  ## Usage

      # DeiT-Base for ImageNet with distillation
      model = DeiT.build(
        image_size: 224,
        patch_size: 16,
        embed_dim: 768,
        depth: 12,
        num_heads: 12,
        num_classes: 1000,
        teacher_num_classes: 1000
      )

      # DeiT-Small without distillation head
      model = DeiT.build(
        image_size: 224,
        patch_size: 16,
        embed_dim: 384,
        depth: 12,
        num_heads: 6,
        num_classes: 1000
      )

  ## References

  - "Training data-efficient image transformers & distillation through attention"
    (Touvron et al., ICML 2021)
  """

  require Axon

  alias Edifice.Blocks.PatchEmbed
  alias Edifice.Utils.FusedOps

  @default_image_size 224
  @default_patch_size 16
  @default_in_channels 3
  @default_embed_dim 768
  @default_depth 12
  @default_num_heads 12
  @default_mlp_ratio 4.0
  @default_dropout 0.0

  # ============================================================================
  # Model Builder
  # ============================================================================

  @doc """
  Build a DeiT model.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Patch size, square (default: 16)
    - `:in_channels` - Number of input channels (default: 3)
    - `:embed_dim` - Embedding dimension (default: 768)
    - `:depth` - Number of transformer blocks (default: 12)
    - `:num_heads` - Number of attention heads (default: 12)
    - `:mlp_ratio` - MLP hidden dim ratio relative to embed_dim (default: 4.0)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:num_classes` - Number of classes for classification head (optional)
    - `:teacher_num_classes` - Number of classes for distillation head (optional).
      If set, model returns `{cls_output, dist_output}` via a container output.

  ## Returns

    An Axon model. When both `:num_classes` and `:teacher_num_classes` are set,
    outputs a container `%{cls: [batch, num_classes], dist: [batch, teacher_num_classes]}`.
    When only `:num_classes` is set, outputs `[batch, num_classes]`.
    Otherwise, outputs `[batch, embed_dim]` from the CLS token.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:depth, pos_integer()}
          | {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:mlp_ratio, float()}
          | {:num_classes, pos_integer() | nil}
          | {:num_heads, pos_integer()}
          | {:patch_size, pos_integer()}
          | {:teacher_num_classes, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    embed_dim = Keyword.get(opts, :embed_dim, @default_embed_dim)
    depth = Keyword.get(opts, :depth, @default_depth)
    num_heads = Keyword.get(opts, :num_heads, @default_num_heads)
    mlp_ratio = Keyword.get(opts, :mlp_ratio, @default_mlp_ratio)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    num_classes = Keyword.get(opts, :num_classes, nil)
    teacher_num_classes = Keyword.get(opts, :teacher_num_classes, nil)

    num_patches = PatchEmbed.num_patches(image_size, patch_size)
    mlp_hidden = round(embed_dim * mlp_ratio)

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

    # Prepend CLS and distillation tokens: [batch, num_patches + 2, embed_dim]
    x = prepend_cls_and_dist_tokens(x, embed_dim, name: "tokens")

    # Add learnable position embeddings
    x = add_position_embedding(x, num_patches + 2, embed_dim, name: "pos_embed")

    x =
      if dropout > 0.0 do
        Axon.dropout(x, rate: dropout, name: "embed_dropout")
      else
        x
      end

    # Transformer encoder blocks
    x =
      Enum.reduce(0..(depth - 1), x, fn block_idx, acc ->
        transformer_block(acc, embed_dim, num_heads, mlp_hidden, dropout,
          name: "block_#{block_idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Extract CLS token (index 0)
    cls_output =
      Axon.nx(
        x,
        fn tensor ->
          Nx.slice_along_axis(tensor, 0, 1, axis: 1)
          |> Nx.squeeze(axes: [1])
        end,
        name: "extract_cls"
      )

    # Extract distillation token (index 1)
    dist_output =
      Axon.nx(
        x,
        fn tensor ->
          Nx.slice_along_axis(tensor, 1, 1, axis: 1)
          |> Nx.squeeze(axes: [1])
        end,
        name: "extract_dist"
      )

    # Build output based on options
    cond do
      num_classes != nil and teacher_num_classes != nil ->
        # Both heads: return container with cls and dist outputs
        cls_head = Axon.dense(cls_output, num_classes, name: "cls_head")
        dist_head = Axon.dense(dist_output, teacher_num_classes, name: "dist_head")
        Axon.container(%{cls: cls_head, dist: dist_head})

      num_classes != nil ->
        # Classification only: average CLS and dist tokens, then classify
        averaged =
          Axon.layer(
            &average_tokens_impl/3,
            [cls_output, dist_output],
            name: "avg_tokens",
            op_name: :average_tokens
          )

        Axon.dense(averaged, num_classes, name: "classifier")

      true ->
        # No classification head: return CLS token features
        cls_output
    end
  end

  # ============================================================================
  # Transformer Block
  # ============================================================================

  defp transformer_block(input, embed_dim, num_heads, mlp_hidden, dropout, opts) do
    name = Keyword.get(opts, :name, "block")

    # Pre-norm self-attention
    normed = Axon.layer_norm(input, name: "#{name}_norm1")

    attended = self_attention(normed, embed_dim, num_heads, dropout, name: "#{name}_attn")

    x = Axon.add(input, attended, name: "#{name}_residual1")

    # Pre-norm MLP
    normed2 = Axon.layer_norm(x, name: "#{name}_norm2")

    ffn =
      normed2
      |> Axon.dense(mlp_hidden, name: "#{name}_mlp_fc1")
      |> Axon.activation(:gelu, name: "#{name}_mlp_gelu")
      |> maybe_dropout(dropout, "#{name}_mlp_drop1")
      |> Axon.dense(embed_dim, name: "#{name}_mlp_fc2")
      |> maybe_dropout(dropout, "#{name}_mlp_drop2")

    Axon.add(x, ffn, name: "#{name}_residual2")
  end

  defp self_attention(input, embed_dim, _num_heads, dropout, opts) do
    name = Keyword.get(opts, :name, "attn")

    qkv = Axon.dense(input, embed_dim * 3, name: "#{name}_qkv")

    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          q = Nx.slice_along_axis(qkv_tensor, 0, embed_dim, axis: 2)
          k = Nx.slice_along_axis(qkv_tensor, embed_dim, embed_dim, axis: 2)
          v = Nx.slice_along_axis(qkv_tensor, embed_dim * 2, embed_dim, axis: 2)

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
    |> Axon.dense(embed_dim, name: "#{name}_proj")
    |> maybe_dropout(dropout, "#{name}_dropout")
  end

  # ============================================================================
  # Special Tokens and Embeddings
  # ============================================================================

  defp prepend_cls_and_dist_tokens(input, embed_dim, opts) do
    name = Keyword.get(opts, :name, "tokens")

    # CLS token
    cls_src = Axon.nx(input, fn _t -> Nx.broadcast(1.0, {1, 1}) end, name: "#{name}_cls_src")
    cls_proj = Axon.dense(cls_src, embed_dim, name: "#{name}_cls_proj")

    # Distillation token
    dist_src = Axon.nx(input, fn _t -> Nx.broadcast(0.5, {1, 1}) end, name: "#{name}_dist_src")
    dist_proj = Axon.dense(dist_src, embed_dim, name: "#{name}_dist_proj")

    Axon.layer(
      &prepend_two_tokens_impl/4,
      [input, cls_proj, dist_proj],
      name: "#{name}_prepend",
      op_name: :prepend_tokens
    )
  end

  defp prepend_two_tokens_impl(patches, cls_token, dist_token, _opts) do
    batch_size = Nx.axis_size(patches, 0)
    embed_dim = Nx.axis_size(cls_token, 1)

    cls = Nx.reshape(cls_token, {1, 1, embed_dim}) |> Nx.broadcast({batch_size, 1, embed_dim})
    dist = Nx.reshape(dist_token, {1, 1, embed_dim}) |> Nx.broadcast({batch_size, 1, embed_dim})

    Nx.concatenate([cls, dist, patches], axis: 1)
  end

  defp add_position_embedding(input, seq_len, embed_dim, opts) do
    name = Keyword.get(opts, :name, "pos_embed")

    pos_source =
      Axon.nx(input, fn _t -> Nx.iota({1, seq_len}, axis: 1) |> Nx.divide(seq_len) end,
        name: "#{name}_src"
      )

    pos_proj = Axon.dense(pos_source, embed_dim, name: "#{name}_proj")

    Axon.layer(
      &add_embedding_impl/3,
      [input, pos_proj],
      name: "#{name}_add",
      op_name: :add_pos_embed
    )
  end

  defp add_embedding_impl(input, pos_embed, _opts) do
    Nx.add(input, pos_embed)
  end

  defp average_tokens_impl(cls, dist, _opts) do
    Nx.add(cls, dist) |> Nx.divide(2.0)
  end

  defp maybe_dropout(input, rate, name) when rate > 0.0 do
    Axon.dropout(input, rate: rate, name: name)
  end

  defp maybe_dropout(input, _rate, _name), do: input

  # ============================================================================
  # Output Size
  # ============================================================================

  @doc """
  Get the output size of a DeiT model.

  Returns `:num_classes` if set, otherwise `:embed_dim`.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    case Keyword.get(opts, :num_classes) do
      nil -> Keyword.get(opts, :embed_dim, @default_embed_dim)
      num_classes -> num_classes
    end
  end
end
