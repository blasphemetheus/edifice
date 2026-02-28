defmodule Edifice.Vision.FocalNet do
  @moduledoc """
  FocalNet: Focal Modulation Networks for vision (Yang et al., 2022).

  Replaces self-attention with focal modulation, which aggregates context at
  multiple granularity levels using hierarchical depthwise convolutions and
  gated aggregation. This provides a simple yet effective alternative to
  attention that captures both local and global context.

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
  | FocalNet Block x N        |
  |                           |
  | Focal Modulation:         |
  |   q = Dense(x)            |
  |   For each level l:       |
  |     ctx += gelu(conv_l)   |
  |   gate = sigmoid(Dense(x))|
  |   out = q * gate * ctx    |
  |   + Residual              |
  |                           |
  | FFN:                      |
  |   Dense(4*h) -> GELU      |
  |   -> Dense(h)             |
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

  ## Usage

      model = FocalNet.build(
        image_size: 224,
        patch_size: 16,
        hidden_size: 256,
        num_layers: 4,
        focal_levels: 3,
        num_classes: 1000
      )

  ## References

  - Yang et al., "Focal Modulation Networks" (NeurIPS 2022)
  - https://arxiv.org/abs/2203.11926
  """

  use Edifice.Vision.Backbone

  alias Edifice.Blocks.{FFN, PatchEmbed}

  @default_image_size 224
  @default_patch_size 16
  @default_in_channels 3
  @default_hidden_size 256
  @default_num_layers 4
  @default_focal_levels 3
  @default_focal_kernel 3

  @doc """
  Build a FocalNet model.

  ## Options

    - `:image_size` - Input image size, square (default: 224)
    - `:patch_size` - Patch size, square (default: 16)
    - `:in_channels` - Number of input channels (default: 3)
    - `:hidden_size` - Hidden dimension per patch (default: 256)
    - `:num_layers` - Number of FocalNet blocks (default: 4)
    - `:focal_levels` - Number of focal context levels (default: 3)
    - `:focal_kernel` - Base kernel size for focal convolutions (default: 3)
    - `:num_classes` - Number of output classes (optional)

  ## Returns

    An Axon model. Without `:num_classes`, outputs `[batch, hidden_size]`.
    With `:num_classes`, outputs `[batch, num_classes]`.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:focal_kernel, pos_integer()}
          | {:focal_levels, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:image_size, pos_integer()}
          | {:in_channels, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:num_layers, pos_integer()}
          | {:patch_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    image_size = Keyword.get(opts, :image_size, @default_image_size)
    patch_size = Keyword.get(opts, :patch_size, @default_patch_size)
    in_channels = Keyword.get(opts, :in_channels, @default_in_channels)
    hidden_size = Keyword.get(opts, :hidden_size, @default_hidden_size)
    num_layers = Keyword.get(opts, :num_layers, @default_num_layers)
    focal_levels = Keyword.get(opts, :focal_levels, @default_focal_levels)
    focal_kernel = Keyword.get(opts, :focal_kernel, @default_focal_kernel)
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

    # Stack of FocalNet blocks
    x =
      Enum.reduce(0..(num_layers - 1), x, fn idx, acc ->
        focalnet_block(acc, hidden_size, focal_levels, focal_kernel, name: "focal_#{idx}")
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
  Get the output size of a FocalNet model.

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

  # FocalNet block: focal modulation + FFN with residual connections
  defp focalnet_block(input, hidden_size, focal_levels, focal_kernel, opts) do
    name = Keyword.get(opts, :name, "focal")

    # Focal modulation sublayer
    normed = Axon.layer_norm(input, name: "#{name}_mod_norm")

    # Query projection
    query = Axon.dense(normed, hidden_size, name: "#{name}_query")

    # Build focal context: hierarchical dense layers simulating increasing receptive fields
    # Each level applies a dense projection (simulating depthwise conv at increasing kernel)
    # and accumulates with GELU activation
    ctx =
      Enum.reduce(0..(focal_levels - 1), nil, fn level, acc ->
        # Each level projects to hidden_size, simulating conv at kernel_size * (level + 1)
        # We use dense layers since patches are 1D sequences
        level_proj =
          normed
          |> Axon.dense(hidden_size, name: "#{name}_focal_level_#{level}")
          |> Axon.activation(:gelu, name: "#{name}_focal_gelu_#{level}")

        # For level > 0, apply additional dense to simulate wider receptive field
        level_proj =
          if level > 0 do
            kernel_size = focal_kernel * (level + 1)

            Axon.nx(
              level_proj,
              fn tensor -> focal_context_compute(tensor, kernel_size) end,
              name: "#{name}_focal_ctx_#{level}"
            )
          else
            level_proj
          end

        if acc == nil do
          level_proj
        else
          Axon.add(acc, level_proj, name: "#{name}_focal_accum_#{level}")
        end
      end)

    # Gate: sigmoid(Dense(x))
    gate =
      normed
      |> Axon.dense(hidden_size, name: "#{name}_gate")
      |> Axon.sigmoid(name: "#{name}_gate_sigmoid")

    # Output: q * gate * ctx
    modulated =
      Axon.layer(
        fn q, g, c, _opts ->
          Nx.multiply(Nx.multiply(q, g), c)
        end,
        [query, gate, ctx],
        name: "#{name}_modulate",
        op_name: :focal_modulate
      )

    # Output projection
    mod_out = Axon.dense(modulated, hidden_size, name: "#{name}_mod_proj")

    # Residual
    x = Axon.add(input, mod_out, name: "#{name}_mod_residual")

    # FFN sublayer
    ffn_normed = Axon.layer_norm(x, name: "#{name}_ffn_norm")
    ffn_out = FFN.layer(ffn_normed, hidden_size: hidden_size, name: "#{name}_ffn")
    Axon.add(x, ffn_out, name: "#{name}_ffn_residual")
  end

  # Simulate focal context with wider receptive field via local averaging
  # Input: [batch, seq_len, dim]
  defp focal_context_compute(input, kernel_size) do
    {batch, seq_len, dim} = Nx.shape(input)
    pad_total = kernel_size - 1
    pad_before = div(pad_total, 2)
    pad_after = pad_total - pad_before

    padded =
      Nx.pad(input, 0.0, [{0, 0, 0}, {pad_before, pad_after, 0}, {0, 0, 0}])

    # Sliding average over the sequence dimension
    pooled =
      Enum.reduce(0..(kernel_size - 1), Nx.broadcast(0.0, {batch, seq_len, dim}), fn offset,
                                                                                     acc ->
        slice = Nx.slice_along_axis(padded, offset, seq_len, axis: 1)
        Nx.add(acc, slice)
      end)

    Nx.divide(pooled, kernel_size)
  end
end
