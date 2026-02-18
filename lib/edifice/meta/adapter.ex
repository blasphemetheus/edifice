defmodule Edifice.Meta.Adapter do
  @moduledoc """
  Bottleneck Adapter modules for parameter-efficient finetuning.

  Adapter layers are small bottleneck modules inserted between frozen
  pretrained layers. Each adapter consists of a down-projection, nonlinearity,
  and up-projection with a residual connection, adding only a small number
  of trainable parameters.

  ## Architecture

  ```
  Input x [batch, hidden_size]
        |
        +---> Down-project to bottleneck [batch, bottleneck_size]
        |                    |
        |                    v
        |              Activation (ReLU)
        |                    |
        |                    v
        |         Up-project [batch, hidden_size]
        |                    |
        v                    v
        x    +    adapter_output
        |
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      # Standalone adapter
      adapter = Adapter.build(hidden_size: 768, bottleneck_size: 64)

      # Wrap an existing layer with an adapter
      original_output = Axon.dense(input, 768, name: "pretrained_layer")
      adapted = Adapter.wrap(original_output, hidden_size: 768, bottleneck_size: 64)

  ## References

  - Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (ICML 2019)
  - https://arxiv.org/abs/1902.00751
  """

  @default_bottleneck_size 64
  @default_activation :relu

  @doc """
  Build a standalone bottleneck adapter.

  ## Options

  - `:hidden_size` - Input/output dimension (required)
  - `:bottleneck_size` - Bottleneck dimension (default: 64)
  - `:activation` - Activation function (default: :relu)
  - `:name` - Layer name prefix (default: "adapter")

  ## Returns

  An Axon model: `[batch, hidden_size]` -> `[batch, hidden_size]`
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:bottleneck_size, pos_integer()}
          | {:hidden_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    bottleneck_size = Keyword.get(opts, :bottleneck_size, @default_bottleneck_size)
    activation = Keyword.get(opts, :activation, @default_activation)
    name = Keyword.get(opts, :name, "adapter")

    input = Axon.input("input", shape: {nil, hidden_size})

    adapter_output =
      adapter_block(input, hidden_size,
        bottleneck_size: bottleneck_size,
        activation: activation,
        name: name
      )

    adapter_output
  end

  @doc """
  Wrap an existing layer output with an adapter (residual bottleneck).

  Inserts the adapter after the given layer with a residual connection:

      output = layer_output + adapter(layer_output)

  ## Parameters

  - `layer_output` - Axon node from the existing (frozen) layer

  ## Options

  - `:hidden_size` - Hidden dimension matching the layer output (required)
  - `:bottleneck_size` - Bottleneck dimension (default: 64)
  - `:activation` - Activation function (default: :relu)
  - `:name` - Layer name prefix (default: "adapter")

  ## Returns

  An Axon node with the adapted output.
  """
  @spec wrap(Axon.t(), keyword()) :: Axon.t()
  def wrap(layer_output, opts \\ []) do
    hidden_size = Keyword.fetch!(opts, :hidden_size)
    bottleneck_size = Keyword.get(opts, :bottleneck_size, @default_bottleneck_size)
    activation = Keyword.get(opts, :activation, @default_activation)
    name = Keyword.get(opts, :name, "adapter")

    adapter_block(layer_output, hidden_size,
      bottleneck_size: bottleneck_size,
      activation: activation,
      name: name
    )
  end

  @doc """
  Build the adapter bottleneck: down-project -> activate -> up-project -> residual add.

  ## Parameters

  - `input` - Axon input node
  - `hidden_size` - Input/output dimension

  ## Options

  - `:bottleneck_size` - Bottleneck dimension (default: 64)
  - `:activation` - Activation function (default: :relu)
  - `:name` - Layer name prefix
  """
  @spec adapter_block(Axon.t(), pos_integer(), keyword()) :: Axon.t()
  def adapter_block(input, hidden_size, opts \\ []) do
    bottleneck_size = Keyword.get(opts, :bottleneck_size, @default_bottleneck_size)
    activation = Keyword.get(opts, :activation, @default_activation)
    name = Keyword.get(opts, :name, "adapter")

    # Down-project to bottleneck
    down = Axon.dense(input, bottleneck_size, name: "#{name}_down")

    # Activation
    activated = Axon.activation(down, activation, name: "#{name}_act")

    # Up-project back to hidden size
    up =
      Axon.dense(activated, hidden_size,
        name: "#{name}_up",
        kernel_initializer: Axon.Initializers.zeros()
      )

    # Residual connection
    Axon.add(input, up, name: "#{name}_residual")
  end

  @doc """
  Get the output size of an adapter (same as input).
  """
  @spec output_size(keyword()) :: pos_integer()
  def output_size(opts \\ []) do
    Keyword.fetch!(opts, :hidden_size)
  end
end
