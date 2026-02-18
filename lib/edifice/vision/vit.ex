defmodule Edifice.Vision.ViT do
  @moduledoc """
  Vision Transformer (ViT) implementation.

  Treats an image as a sequence of fixed-size patches, linearly embeds each
  patch, prepends a learnable CLS token, adds position embeddings, and
  processes the resulting sequence through standard transformer encoder blocks.

  ## Architecture

  ```
  Image [batch, channels, height, width]
        |
  +-----v-----------------+
  | Patch Embedding        |  Split into P x P patches, linear project
  +------------------------+
        |
        v
  [batch, num_patches, embed_dim]
        |
  +-----v-----------------+
  | Prepend CLS Token      |  Learnable [1, 1, embed_dim] token
  +------------------------+
        |
        v
  [batch, num_patches + 1, embed_dim]
        |
  +-----v-----------------+
  | Add Position Embedding |  Learnable [1, num_patches + 1, embed_dim]
  +------------------------+
        |
        v
  +-----v-----------------+
  | Transformer Block x N  |
  |   LayerNorm            |
  |   Self-Attention       |
  |   Residual             |
  |   LayerNorm            |
  |   MLP (expand + GELU)  |
  |   Residual             |
  +------------------------+
        |
        v
  +-----v-----------------+
  | Extract CLS Token      |  [batch, embed_dim]
  +------------------------+
        |
        v
  +-----v-----------------+
  | LayerNorm              |
  +------------------------+
        |
        v
  +-----v-----------------+
  | Optional Classifier    |  Dense -> num_classes
  +------------------------+
  ```

  ## Usage

      # ViT-Base for ImageNet
      model = ViT.build(
        image_size: 224,
        patch_size: 16,
        embed_dim: 768,
        depth: 12,
        num_heads: 12
      )

      # Small ViT for CIFAR-10
      model = ViT.build(
        image_size: 32,
        patch_size: 4,
        embed_dim: 192,
        depth: 6,
        num_heads: 3,
        num_classes: 10
      )

  ## References

  - "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    (Dosovitskiy et al., ICLR 2021)
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
  Build a Vision Transformer model.

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

  ## Returns

    An Axon model. Without `:num_classes`, outputs `[batch, embed_dim]`.
    With `:num_classes`, outputs `[batch, num_classes]`.
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

    # Prepend CLS token: [batch, num_patches + 1, embed_dim]
    x = prepend_cls_token(x, embed_dim, name: "cls_token")

    # Add learnable position embeddings
    x = add_position_embedding(x, num_patches + 1, embed_dim, name: "pos_embed")

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

    # Extract CLS token: [batch, embed_dim]
    x =
      Axon.nx(
        x,
        fn tensor ->
          Nx.slice_along_axis(tensor, 0, 1, axis: 1)
          |> Nx.squeeze(axes: [1])
        end,
        name: "extract_cls"
      )

    # Optional classification head
    if num_classes do
      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  # ============================================================================
  # Transformer Block
  # ============================================================================

  @doc false
  @spec transformer_block(
          Axon.t(),
          pos_integer(),
          pos_integer(),
          pos_integer(),
          float(),
          keyword()
        ) ::
          Axon.t()
  def transformer_block(input, embed_dim, num_heads, mlp_hidden, dropout, opts \\ []) do
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

  # ============================================================================
  # Self-Attention
  # ============================================================================

  defp self_attention(input, embed_dim, _num_heads, dropout, opts) do
    name = Keyword.get(opts, :name, "attn")

    # Project to Q, K, V
    qkv = Axon.dense(input, embed_dim * 3, name: "#{name}_qkv")

    attended =
      Axon.nx(
        qkv,
        fn qkv_tensor ->
          # Split into Q, K, V
          q = Nx.slice_along_axis(qkv_tensor, 0, embed_dim, axis: 2)
          k = Nx.slice_along_axis(qkv_tensor, embed_dim, embed_dim, axis: 2)
          v = Nx.slice_along_axis(qkv_tensor, embed_dim * 2, embed_dim, axis: 2)

          # Scaled dot-product attention (no causal mask for ViT)
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

  defp prepend_cls_token(input, embed_dim, opts) do
    name = Keyword.get(opts, :name, "cls_token")

    # Create a learnable CLS token via a dense projection of a constant
    cls_source = Axon.nx(input, fn _tensor -> Nx.broadcast(1.0, {1, 1}) end, name: "#{name}_src")
    cls_proj = Axon.dense(cls_source, embed_dim, name: "#{name}_proj")

    # Expand CLS to [batch, 1, embed_dim] and concatenate
    Axon.layer(
      &prepend_token_impl/3,
      [input, cls_proj],
      name: "#{name}_prepend",
      op_name: :prepend_token
    )
  end

  defp prepend_token_impl(patches, cls_token, _opts) do
    # patches: [batch, num_patches, embed_dim]
    # cls_token: [1, embed_dim]
    batch_size = Nx.axis_size(patches, 0)
    embed_dim = Nx.axis_size(cls_token, 1)

    # Expand CLS token to [batch, 1, embed_dim]
    cls = Nx.reshape(cls_token, {1, 1, embed_dim})
    cls = Nx.broadcast(cls, {batch_size, 1, embed_dim})

    Nx.concatenate([cls, patches], axis: 1)
  end

  defp add_position_embedding(input, seq_len, embed_dim, opts) do
    name = Keyword.get(opts, :name, "pos_embed")

    # Create learnable position embedding via dense projection
    pos_source =
      Axon.nx(input, fn _tensor -> Nx.iota({1, seq_len}, axis: 1) |> Nx.divide(seq_len) end,
        name: "#{name}_src"
      )

    pos_proj = Axon.dense(pos_source, embed_dim, name: "#{name}_proj")

    # Add position embeddings to input (broadcasts over batch)
    Axon.layer(
      &add_embedding_impl/3,
      [input, pos_proj],
      name: "#{name}_add",
      op_name: :add_pos_embed
    )
  end

  defp add_embedding_impl(input, pos_embed, _opts) do
    # input: [batch, seq_len, embed_dim]
    # pos_embed: [1, seq_len, embed_dim]
    Nx.add(input, pos_embed)
  end

  defp maybe_dropout(input, rate, name) when rate > 0.0 do
    Axon.dropout(input, rate: rate, name: name)
  end

  defp maybe_dropout(input, _rate, _name), do: input

  # ============================================================================
  # Output Size
  # ============================================================================

  @doc """
  Get the output size of a ViT model.

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
