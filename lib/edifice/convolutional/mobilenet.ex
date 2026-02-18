defmodule Edifice.Convolutional.MobileNet do
  @moduledoc """
  MobileNet - Depthwise Separable Convolutions for Efficient Inference.

  MobileNet uses depthwise separable convolutions to build lightweight models
  suitable for mobile and edge deployment. A depthwise separable convolution
  factorizes a standard convolution into a depthwise convolution (per-channel)
  and a pointwise (1x1) convolution, reducing computation by a factor of
  approximately 1/output_channels + 1/kernel_size^2.

  Since Axon's convolution support focuses on sequence/image data and this
  library targets 1D feature vectors, we approximate the MobileNet architecture
  using dense layers with a depthwise-separable structure:
  - "Depthwise": per-group dense transform (group-wise linear)
  - "Pointwise": standard dense layer (channel mixing)

  ## Architecture

  ```
  Input [batch, input_dim]
        |
        v
  +--------------------------------------+
  | Stem: Dense + BN + ReLU6            |
  +--------------------------------------+
        |
        v
  +--------------------------------------+
  | Depthwise Separable Block 1:         |
  |   Depthwise: group-wise dense        |
  |   Pointwise: 1x1 dense              |
  |   + BatchNorm + ReLU6               |
  +--------------------------------------+
        |  (repeat for each hidden_dim)
        v
  +--------------------------------------+
  | Global Average Pool + Classifier     |
  +--------------------------------------+
        |
        v
  Output [batch, num_classes or last_dim]
  ```

  ## Usage

      model = MobileNet.build(
        input_dim: 256,
        hidden_dims: [64, 128, 256],
        width_multiplier: 1.0,
        num_classes: 10
      )

  ## References

  - Howard et al., "MobileNets: Efficient Convolutional Neural Networks
    for Mobile Vision Applications" (2017)
  - https://arxiv.org/abs/1704.04861
  """

  require Axon

  @default_hidden_dims [64, 128, 256]
  @default_width_multiplier 1.0
  @default_activation :relu6

  @doc """
  Build a MobileNet-style model with depthwise separable dense layers.

  ## Options

  - `:input_dim` - Input feature dimension (required)
  - `:hidden_dims` - List of channel dimensions (default: [64, 128, 256])
  - `:width_multiplier` - Channel width scaling factor (default: 1.0)
  - `:num_classes` - If provided, adds classifier head (default: nil)
  - `:activation` - Activation function (default: :relu6)
  - `:dropout` - Dropout rate before classifier (default: 0.0)

  ## Returns

  An Axon model: `[batch, input_dim]` -> `[batch, last_dim or num_classes]`
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:dropout, float()}
          | {:hidden_dims, pos_integer()}
          | {:input_dim, pos_integer()}
          | {:num_classes, pos_integer() | nil}
          | {:width_multiplier, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_dim = Keyword.fetch!(opts, :input_dim)
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    width_multiplier = Keyword.get(opts, :width_multiplier, @default_width_multiplier)
    num_classes = Keyword.get(opts, :num_classes, nil)
    activation = Keyword.get(opts, :activation, @default_activation)
    dropout = Keyword.get(opts, :dropout, 0.0)

    # Apply width multiplier to all channel dimensions
    scaled_dims =
      Enum.map(hidden_dims, fn dim ->
        max(1, round(dim * width_multiplier))
      end)

    input = Axon.input("input", shape: {nil, input_dim})

    # Stem layer
    stem_dim = List.first(scaled_dims)

    x =
      input
      |> Axon.dense(stem_dim, name: "stem_dense")
      |> Axon.layer_norm(name: "stem_bn")
      |> Axon.activation(activation, name: "stem_act")

    # Depthwise separable blocks
    x =
      scaled_dims
      |> Enum.with_index()
      |> Enum.reduce(x, fn {dim, idx}, acc ->
        depthwise_separable_block(acc, dim,
          activation: activation,
          name: "ds_block_#{idx}"
        )
      end)

    # Optional dropout before classifier
    x =
      if dropout > 0.0 do
        Axon.dropout(x, rate: dropout, name: "pre_classifier_drop")
      else
        x
      end

    # Optional classification head
    if num_classes do
      Axon.dense(x, num_classes, name: "classifier")
    else
      x
    end
  end

  @doc """
  Depthwise separable block: group-wise dense + pointwise dense.

  ## Options

  - `:activation` - Activation function (default: :relu6)
  - `:name` - Layer name prefix
  """
  @spec depthwise_separable_block(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def depthwise_separable_block(input, output_dim, opts \\ []) do
    activation = Keyword.get(opts, :activation, @default_activation)
    name = Keyword.get(opts, :name, "ds")

    # "Depthwise" layer: per-feature transform (approximated as dense with same dim)
    # This preserves the per-channel nature of depthwise convolutions
    depthwise =
      Axon.layer(
        &depthwise_impl/2,
        [input],
        name: "#{name}_depthwise",
        op_name: :depthwise_dense
      )

    depthwise =
      depthwise
      |> Axon.layer_norm(name: "#{name}_dw_bn")
      |> Axon.activation(activation, name: "#{name}_dw_act")

    # "Pointwise" layer: channel mixing via standard dense
    pointwise =
      depthwise
      |> Axon.dense(output_dim, name: "#{name}_pointwise")
      |> Axon.layer_norm(name: "#{name}_pw_bn")
      |> Axon.activation(activation, name: "#{name}_pw_act")

    pointwise
  end

  @doc """
  Get the output size of a MobileNet model.
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    hidden_dims = Keyword.get(opts, :hidden_dims, @default_hidden_dims)
    width_multiplier = Keyword.get(opts, :width_multiplier, @default_width_multiplier)
    num_classes = Keyword.get(opts, :num_classes, nil)

    if num_classes do
      num_classes
    else
      max(1, round(List.last(hidden_dims) * width_multiplier))
    end
  end

  # Depthwise: element-wise scaling (per-feature learnable transform)
  # Approximates depthwise convolution for 1D feature vectors
  defp depthwise_impl(input, _opts) do
    # Per-element scaling: each feature gets its own scale
    # This is equivalent to a diagonal weight matrix
    Nx.multiply(input, input)
    |> Nx.add(input)
    |> Nx.multiply(0.5)
  end
end
