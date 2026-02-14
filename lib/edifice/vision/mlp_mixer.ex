defmodule Edifice.Vision.MLPMixer do
  @moduledoc """
  MLP-Mixer - All-MLP architecture for vision.

  Replaces attention and convolutions entirely with MLPs. Uses two types of
  MLP layers applied alternately: token-mixing MLPs that operate across
  spatial locations (patches), and channel-mixing MLPs that operate within
  each location independently.

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
  | Mixer Layer x N           |
  |                           |
  | Token Mixing:             |
  |   LN -> Transpose         |
  |   -> Dense(token_mlp_dim) |
  |   -> GELU                 |
  |   -> Dense(num_patches)   |
  |   -> Transpose            |
  |   + Residual              |
  |                           |
  | Channel Mixing:           |
  |   LN -> Dense(ch_mlp_dim) |
  |   -> GELU                 |
  |   -> Dense(hidden_size)   |
  |   + Residual              |
  +---------------------------+
        |
        v
  +-----v--------------------+
  | LayerNorm                 |
  +---------------------------+
        |
  +-----v--------------------+
  | Global Average Pool       |  Mean over patches
  +---------------------------+
        |
        v
  [batch, hidden_size]
        |
  +-----v--------------------+
  | Optional Classifier       |
  +---------------------------+
  ```

  ## Key Insight

  Token-mixing MLPs allow communication between different spatial locations,
  while channel-mixing MLPs process features within each location. This
  separation is analogous to depthwise separable convolutions but uses
  fully-connected layers, achieving competitive results without attention.

  ## Usage

      # MLP-Mixer-B/16
      model = MLPMixer.build(
        image_size: 224,
        patch_size: 16,
        hidden_size: 768,
        num_layers: 12,
        token_mlp_dim: 384,
        channel_mlp_dim: 3072,
        num_classes: 1000
      )

      # Small Mixer for CIFAR-10
      model = MLPMixer.build(
        image_size: 32,
        patch_size: 4,
        hidden_size: 256,
        num_layers: 8,
        token_mlp_dim: 128,
        channel_mlp_dim: 1024,
        num_classes: 10
      )

  ## References

  - "MLP-Mixer: An all-MLP Architecture for Vision"
    (Tolstikhin et al., NeurIPS 2021)
  """

  require Axon

  alias Edifice.Blocks.PatchEmbed

  @default_image_size 224
  @default_patch_size 16
  @default_in_channels 3
  @default_hidden_size 512
  @default_num_layers 8
  @default_token_mlp_dim 256
  @default_channel_mlp_dim 2048
  @default_dropout 0.0

  # ============================================================================
  # Model Builder
  # ============================================================================

  @doc """
  Build an MLP-Mixer model.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Patch size, square (default: 16)
    - `:in_channels` - Number of input channels (default: 3)
    - `:hidden_size` - Hidden dimension per patch (default: 512)
    - `:num_layers` - Number of mixer layers (default: 8)
    - `:token_mlp_dim` - Token-mixing MLP hidden dimension (default: 256)
    - `:channel_mlp_dim` - Channel-mixing MLP hidden dimension (default: 2048)
    - `:dropout` - Dropout rate (default: 0.0)
    - `:num_classes` - Number of classes for classification head (optional)

  ## Returns

    An Axon model. Without `:num_classes`, outputs `[batch, hidden_size]`.
    With `:num_classes`, outputs `[batch, num_classes]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    token_mlp_dim = Keyword.get(opts, :token_mlp_dim, @default_token_mlp_dim)
    channel_mlp_dim = Keyword.get(opts, :channel_mlp_dim, @default_channel_mlp_dim)
    dropout = Keyword.get(opts, :dropout, @default_dropout)
    num_classes = Keyword.get(opts, :num_classes, nil)

    num_patches = PatchEmbed.num_patches(image_size, patch_size)

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

    # Stack of mixer layers
    x =
      Enum.reduce(0..(num_layers - 1), x, fn layer_idx, acc ->
        mixer_layer(acc, num_patches, hidden_size, token_mlp_dim, channel_mlp_dim, dropout,
          name: "mixer_#{layer_idx}"
        )
      end)

    # Final layer norm
    x = Axon.layer_norm(x, name: "final_norm")

    # Global average pool: [batch, num_patches, hidden_size] -> [batch, hidden_size]
    x =
      Axon.nx(
        x,
        fn tensor ->
          Nx.mean(tensor, axes: [1])
        end,
        name: "global_pool"
      )

    # Optional classification head
    if num_classes do
      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  # ============================================================================
  # Mixer Layer
  # ============================================================================

  defp mixer_layer(input, num_patches, hidden_size, token_mlp_dim, channel_mlp_dim, dropout, opts) do
    name = Keyword.get(opts, :name, "mixer")

    # Token-mixing MLP: operates across patches (spatial mixing)
    # LN -> transpose -> MLP -> transpose -> residual
    token_normed = Axon.layer_norm(input, name: "#{name}_token_norm")

    # Transpose: [batch, num_patches, hidden_size] -> [batch, hidden_size, num_patches]
    token_transposed =
      Axon.nx(
        token_normed,
        fn tensor ->
          Nx.transpose(tensor, axes: [0, 2, 1])
        end,
        name: "#{name}_transpose1"
      )

    token_mixed =
      token_transposed
      |> Axon.dense(token_mlp_dim, name: "#{name}_token_fc1")
      |> Axon.activation(:gelu, name: "#{name}_token_gelu")
      |> maybe_dropout(dropout, "#{name}_token_drop1")
      |> Axon.dense(num_patches, name: "#{name}_token_fc2")
      |> maybe_dropout(dropout, "#{name}_token_drop2")

    # Transpose back: [batch, hidden_size, num_patches] -> [batch, num_patches, hidden_size]
    token_mixed =
      Axon.nx(
        token_mixed,
        fn tensor ->
          Nx.transpose(tensor, axes: [0, 2, 1])
        end,
        name: "#{name}_transpose2"
      )

    # Residual
    x = Axon.add(input, token_mixed, name: "#{name}_token_residual")

    # Channel-mixing MLP: operates within each patch (feature mixing)
    # LN -> MLP -> residual
    channel_normed = Axon.layer_norm(x, name: "#{name}_channel_norm")

    channel_mixed =
      channel_normed
      |> Axon.dense(channel_mlp_dim, name: "#{name}_channel_fc1")
      |> Axon.activation(:gelu, name: "#{name}_channel_gelu")
      |> maybe_dropout(dropout, "#{name}_channel_drop1")
      |> Axon.dense(hidden_size, name: "#{name}_channel_fc2")
      |> maybe_dropout(dropout, "#{name}_channel_drop2")

    # Residual
    Axon.add(x, channel_mixed, name: "#{name}_channel_residual")
  end

  defp maybe_dropout(input, rate, name) when rate > 0.0 do
    Axon.dropout(input, rate: rate, name: name)
  end

  defp maybe_dropout(input, _rate, _name), do: input

  # ============================================================================
  # Output Size
  # ============================================================================

  @doc """
  Get the output size of an MLP-Mixer model.

  Returns `:num_classes` if set, otherwise `:hidden_size`.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    case Keyword.get(opts, :num_classes) do
      nil -> Keyword.get(opts, :hidden_size, @default_hidden_size)
      num_classes -> num_classes
    end
  end
end
