defmodule Edifice.Convolutional.EfficientNet do
  @moduledoc """
  EfficientNet - Compound Scaling of Neural Networks.

  EfficientNet uses a compound scaling method that uniformly scales depth,
  width, and resolution with a set of fixed scaling coefficients. The core
  building block is the MBConv (Mobile Inverted Bottleneck Convolution) with
  squeeze-and-excitation attention.

  Since this library targets 1D feature vectors (not images), we implement
  a simplified dense-layer version of EfficientNet's architecture:
  - MBConv blocks use inverted residual structure with dense layers
  - Squeeze-Excitation attention operates on feature channels
  - Compound scaling adjusts depth and width

  ## Architecture

  ```
  Input [batch, input_dim]
        |
        v
  +--------------------------------------+
  | Stem Dense                           |
  +--------------------------------------+
        |
        v
  +--------------------------------------+
  | MBConv Block 1:                      |
  |   Expand -> Transform -> SE -> Proj  |
  |   + Residual                         |
  +--------------------------------------+
        |  (repeat, scaled by depth_mult)
        v
  +--------------------------------------+
  | Head Dense + Classifier              |
  +--------------------------------------+
        |
        v
  Output [batch, num_classes or last_dim]
  ```

  ## Usage

      model = EfficientNet.build(
        input_dim: 256,
        base_dim: 32,
        depth_multiplier: 1.0,
        width_multiplier: 1.0,
        num_classes: 10
      )

  ## References

  - Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional
    Neural Networks" (ICML 2019)
  - https://arxiv.org/abs/1905.11946
  """

  require Axon

  @default_base_dim 32
  @default_depth_multiplier 1.0
  @default_width_multiplier 1.0
  @default_expand_ratio 6
  @default_se_ratio 0.25

  # Base block configuration: {expand_ratio, output_channels, num_repeats}
  @base_blocks [
    {1, 16, 1},
    {6, 24, 2},
    {6, 40, 2},
    {6, 80, 3},
    {6, 112, 3},
    {6, 192, 4},
    {6, 320, 1}
  ]

  @doc """
  Build an EfficientNet-style model.

  ## Options

  - `:input_dim` - Input feature dimension (required)
  - `:base_dim` - Base stem dimension (default: 32)
  - `:depth_multiplier` - Depth scaling factor (default: 1.0)
  - `:width_multiplier` - Width scaling factor (default: 1.0)
  - `:num_classes` - If provided, adds classifier head (default: nil)
  - `:dropout` - Dropout rate before classifier (default: 0.0)

  ## Returns

  An Axon model: `[batch, input_dim]` -> `[batch, last_dim or num_classes]`
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:base_dim, pos_integer()}
          | {:depth_multiplier, pos_integer()}
          | {:dropout, float()}
          | {:input_dim, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:width_multiplier, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    base_dim = Keyword.get(opts, :base_dim, @default_base_dim)
    depth_mult = Keyword.get(opts, :depth_multiplier, @default_depth_multiplier)
    width_mult = Keyword.get(opts, :width_multiplier, @default_width_multiplier)
    num_classes = Keyword.get(opts, :num_classes, nil)
    dropout = Keyword.get(opts, :dropout, 0.0)

    input = Axon.input("input", shape: {nil, input_dim})

    # Stem
    stem_dim = scale_width(base_dim, width_mult)

    x =
      input
      |> Axon.dense(stem_dim, name: "stem")
      |> Axon.layer_norm(name: "stem_bn")
      |> Axon.activation(:silu, name: "stem_act")

    # MBConv blocks with compound scaling
    {x, _block_idx} =
      @base_blocks
      |> Enum.reduce({x, 0}, fn {expand_ratio, channels, repeats}, {acc, block_idx} ->
        scaled_channels = scale_width(channels, width_mult)
        scaled_repeats = scale_depth(repeats, depth_mult)

        acc =
          Enum.reduce(0..(scaled_repeats - 1), acc, fn rep, inner_acc ->
            mbconv_block(inner_acc, scaled_channels,
              expand_ratio: expand_ratio,
              name: "mbconv_#{block_idx}_#{rep}"
            )
          end)

        {acc, block_idx + 1}
      end)

    # Head
    head_dim = scale_width(1280, width_mult)

    x =
      x
      |> Axon.dense(head_dim, name: "head")
      |> Axon.layer_norm(name: "head_bn")
      |> Axon.activation(:silu, name: "head_act")

    x =
      if dropout > 0.0 do
        Axon.dropout(x, rate: dropout, name: "head_drop")
      else
        x
      end

    if num_classes do
      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  @doc """
  MBConv (Mobile Inverted Bottleneck) block with Squeeze-Excitation.

  ## Options

  - `:expand_ratio` - Expansion ratio for inverted bottleneck (default: 6)
  - `:se_ratio` - Squeeze-Excitation reduction ratio (default: 0.25)
  - `:name` - Layer name prefix
  """
  @spec mbconv_block(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def mbconv_block(input, output_dim, opts \\ []) do
    expand_ratio = Keyword.get(opts, :expand_ratio, @default_expand_ratio)
    se_ratio = Keyword.get(opts, :se_ratio, @default_se_ratio)
    name = Keyword.get(opts, :name, "mbconv")

    # Expansion phase - always apply to ensure consistent dimensions for SE
    expanded_dim = output_dim * expand_ratio

    x =
      input
      |> Axon.dense(expanded_dim, name: "#{name}_expand")
      |> Axon.layer_norm(name: "#{name}_expand_bn")
      |> Axon.activation(:silu, name: "#{name}_expand_act")

    # Squeeze-Excitation
    se_dim = max(1, round(expanded_dim * se_ratio))
    x = squeeze_excitation(x, se_dim, expand_dim: expanded_dim, name: "#{name}_se")

    # Projection phase (linear, no activation)
    projected =
      x
      |> Axon.dense(output_dim, name: "#{name}_project")
      |> Axon.layer_norm(name: "#{name}_project_bn")

    # Residual connection (only when input and output dims match)
    Axon.layer(
      fn proj, inp, _opts ->
        if Nx.axis_size(proj, 1) == Nx.axis_size(inp, 1) do
          Nx.add(proj, inp)
        else
          proj
        end
      end,
      [projected, input],
      name: "#{name}_residual",
      op_name: :mbconv_residual
    )
  end

  @doc """
  Squeeze-Excitation attention block.

  Computes channel attention weights by squeezing spatial information and
  exciting (re-weighting) channels based on their importance.
  """
  @spec squeeze_excitation(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def squeeze_excitation(input, se_dim, opts \\ []) do
    name = Keyword.get(opts, :name, "se")

    # Squeeze: global information via the features themselves
    se =
      input
      |> Axon.dense(se_dim, name: "#{name}_reduce")
      |> Axon.activation(:silu, name: "#{name}_act")

    # Excite: produce channel weights (restore to input dimension)
    expand_dim = Keyword.fetch!(opts, :expand_dim)
    weights = Axon.dense(se, expand_dim, name: "#{name}_expand")

    # Apply sigmoid gating
    gate = Axon.sigmoid(weights)
    Axon.multiply(input, gate, name: "#{name}_gate")
  end

  @doc """
  Get the output size of an EfficientNet model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    width_mult = Keyword.get(opts, :width_multiplier, @default_width_multiplier)
    num_classes = Keyword.get(opts, :num_classes, nil)
    if num_classes, do: num_classes, else: scale_width(1280, width_mult)
  end

  # Scale channel width by multiplier, rounded to nearest 8 for tensor cores
  defp scale_width(channels, multiplier) do
    scaled = channels * multiplier
    rounded = max(8, round(scaled / 8) * 8)

    # Ensure rounding doesn't reduce by more than 10%
    if rounded < scaled * 0.9, do: rounded + 8, else: rounded
  end

  # Scale depth (repeats) by multiplier, ceiling
  defp scale_depth(repeats, multiplier) do
    max(1, ceil(repeats * multiplier))
  end
end
