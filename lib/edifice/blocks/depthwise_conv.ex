defmodule Edifice.Blocks.DepthwiseConv do
  @moduledoc """
  1D depthwise separable convolution block for sequence models.

  Depthwise separable convolution factorizes a standard convolution into a
  depthwise convolution (per-channel) followed by a pointwise 1x1 convolution.
  This reduces parameters from `O(C_in * C_out * K)` to `O(C * K + C * C_out)`.

  Used by: Conformer, Mega, StripedHyena, and other hybrid models that need
  local pattern extraction alongside attention or SSM layers.

  ## Architecture

  ```
  Input [batch, seq_len, channels]
        |
  Depthwise Conv1D (groups = channels)
        |
  Optional BatchNorm / LayerNorm
        |
  Activation (SiLU by default)
        |
  Pointwise Conv1D (1x1)
        |
  Output [batch, seq_len, out_channels]
  ```

  ## Usage

      output = DepthwiseConv.layer(input, 256, 31, name: "dw_conv")
  """

  @doc """
  Build a depthwise separable 1D convolution Axon layer.

  ## Parameters

    - `input` - Axon node with shape `[batch, seq_len, channels]`
    - `channels` - Number of input/depthwise channels
    - `kernel_size` - Convolution kernel size (default: 31)

  ## Options

    - `:out_channels` - Output channels for pointwise conv (default: same as `channels`)
    - `:activation` - Activation function (default: `:silu`)
    - `:use_norm` - Apply layer norm after depthwise conv (default: `true`)
    - `:padding` - Padding mode: `:causal` or `:same` (default: `:causal`)
    - `:name` - Layer name prefix (default: `"depthwise_conv"`)
  """
  @spec layer(Axon.t(), pos_integer(), pos_integer(), keyword()) :: Axon.t()
  def layer(input, channels, kernel_size \\ 31, opts \\ []) do
    out_channels = Keyword.get(opts, :out_channels, channels)
    activation = Keyword.get(opts, :activation, :silu)
    use_norm = Keyword.get(opts, :use_norm, true)
    padding_mode = Keyword.get(opts, :padding, :causal)
    name = Keyword.get(opts, :name, "depthwise_conv")

    padding =
      case padding_mode do
        :causal -> [{kernel_size - 1, 0}]
        :same -> :same
      end

    # Depthwise convolution: each channel has its own filter
    x =
      Axon.conv(input, channels,
        kernel_size: {kernel_size},
        padding: padding,
        feature_group_size: channels,
        name: "#{name}_dw"
      )

    # Optional normalization
    x =
      if use_norm do
        Axon.layer_norm(x, name: "#{name}_norm")
      else
        x
      end

    # Activation
    x = Axon.activation(x, activation, name: "#{name}_act")

    # Pointwise 1x1 convolution (equivalent to dense across channels)
    Axon.dense(x, out_channels, name: "#{name}_pw")
  end
end
