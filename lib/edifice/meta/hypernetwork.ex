defmodule Edifice.Meta.Hypernetwork do
  @moduledoc """
  Hypernetworks that generate weights for a target network.

  A hypernetwork is a neural network that produces the weights for another
  neural network (the target network). This enables:

  1. **Conditional computation**: Different inputs produce different target weights
  2. **Weight sharing**: One hypernetwork can generate weights for many target layers
  3. **Task adaptation**: Condition on task embeddings to generate task-specific networks
  4. **Compression**: The hypernetwork can be smaller than the target weight space

  ## Architecture

  ```
  Conditioning Input [batch, conditioning_size]
        |
        v
  +----------------------------+
  |      Hypernetwork          |
  |  (generates weight chunks) |
  +----------------------------+
        |
        v
  Weight Matrices for Target Network
  [W1: in1 x out1, W2: in2 x out2, ...]
        |
        v
  +----------------------------+
  |     Target Network         |
  |  (uses generated weights)  |
  +----------------------------+
        |
        v
  Output [batch, final_output_size]
  ```

  ## Usage

      # Build hypernetwork
      model = Hypernetwork.build(
        conditioning_size: 64,
        target_layer_sizes: [{128, 64}, {64, 32}],
        hidden_sizes: [256, 256]
      )

  ## References
  - Ha et al., "HyperNetworks" (2016)
  - https://arxiv.org/abs/1609.09106
  """

  require Axon

  @default_hidden_sizes [256, 256]

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a hypernetwork that generates weights for a target network.

  The hypernetwork takes a conditioning input and produces weight matrices
  for each layer of the target network. The target network then processes
  data input using these generated weights.

  ## Options
    - `:conditioning_size` - Dimension of the conditioning input (required)
    - `:target_layer_sizes` - List of `{input_dim, output_dim}` tuples for
      each target layer (required)
    - `:hidden_sizes` - Hidden layer sizes for the weight generator (default: [256, 256])
    - `:input_size` - Size of the data input to the target network (required)
    - `:activation` - Activation for target network layers (default: :relu)

  ## Returns
    An Axon model taking conditioning `[batch, conditioning_size]` and
    data `[batch, input_size]`, producing `[batch, last_output_dim]`.
  """
  @spec build(keyword()) :: Axon.t()
  def build(opts \\ []) do
    conditioning_size = Keyword.fetch!(opts, :conditioning_size)
    target_layer_sizes = Keyword.fetch!(opts, :target_layer_sizes)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    input_size = Keyword.fetch!(opts, :input_size)
    activation = Keyword.get(opts, :activation, :relu)

    # Inputs
    conditioning = Axon.input("conditioning", shape: {nil, conditioning_size})
    data_input = Axon.input("data_input", shape: {nil, input_size})

    # Build weight generator: conditioning -> weight chunks
    weight_generators =
      target_layer_sizes
      |> Enum.with_index()
      |> Enum.map(fn {{in_dim, out_dim}, idx} ->
        build_weight_generator(conditioning, in_dim, out_dim,
          hidden_sizes: hidden_sizes,
          name: "weight_gen_#{idx}"
        )
      end)

    # Apply generated weights to data input sequentially
    apply_generated_weights(data_input, weight_generators, target_layer_sizes,
      activation: activation)
  end

  @doc """
  Build a weight generator network.

  Takes a conditioning input and outputs a flattened weight matrix
  and bias vector for one target layer.

  ## Parameters
    - `conditioning` - Axon node with conditioning input `[batch, conditioning_size]`
    - `target_in` - Input dimension of the target layer
    - `target_out` - Output dimension of the target layer

  ## Options
    - `:hidden_sizes` - Hidden layer sizes (default: [256, 256])
    - `:name` - Layer name prefix

  ## Returns
    An Axon node producing concatenated weight + bias: `[batch, target_in * target_out + target_out]`
  """
  @spec build_weight_generator(Axon.t(), pos_integer(), pos_integer(), keyword()) :: Axon.t()
  def build_weight_generator(conditioning, target_in, target_out, opts \\ []) do
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    name = Keyword.get(opts, :name, "weight_gen")

    # Total parameters to generate: weight matrix + bias
    total_params = target_in * target_out + target_out

    # Build the generator MLP
    x =
      hidden_sizes
      |> Enum.with_index()
      |> Enum.reduce(conditioning, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "#{name}_hidden_#{idx}")
        |> Axon.relu()
      end)

    # Output: flat vector of generated parameters
    # Scale initial output to be small (prevents target network explosion)
    Axon.dense(x, total_params,
      name: "#{name}_params",
      kernel_initializer: Axon.Initializers.uniform(scale: 0.01))
  end

  @doc """
  Apply hypernetwork-generated weights to compute the target network output.

  Takes data input and generated weight vectors, reshapes them into proper
  weight matrices and biases, and applies them sequentially.

  ## Parameters
    - `data_input` - Axon node with data `[batch, input_dim]`
    - `weight_generators` - List of Axon nodes, each producing weight params
    - `target_layer_sizes` - List of `{in_dim, out_dim}` tuples

  ## Options
    - `:activation` - Activation function between layers (default: :relu)

  ## Returns
    An Axon node with shape `[batch, last_out_dim]`
  """
  @spec apply_generated_weights(Axon.t(), [Axon.t()], [{pos_integer(), pos_integer()}], keyword()) ::
          Axon.t()
  def apply_generated_weights(data_input, weight_generators, target_layer_sizes, opts \\ []) do
    activation = Keyword.get(opts, :activation, :relu)

    # Apply each generated layer sequentially
    weight_generators
    |> Enum.zip(target_layer_sizes)
    |> Enum.with_index()
    |> Enum.reduce(data_input, fn {{weight_gen, {in_dim, out_dim}}, idx}, acc ->
      is_last = idx == length(weight_generators) - 1

      # Apply generated weights to data
      result =
        Axon.layer(
          &apply_weight_impl/3,
          [acc, weight_gen],
          name: "target_layer_#{idx}",
          in_dim: in_dim,
          out_dim: out_dim,
          op_name: :hyper_apply
        )

      # Apply activation (except on last layer)
      if is_last do
        result
      else
        Axon.activation(result, activation, name: "target_act_#{idx}")
      end
    end)
  end

  # ============================================================================
  # Private Implementation
  # ============================================================================

  # Apply generated weights: reshape flat params into weight matrix + bias,
  # then compute output = input * W + b
  defp apply_weight_impl(data, weight_params, opts) do
    in_dim = opts[:in_dim]
    out_dim = opts[:out_dim]

    batch_size = Nx.axis_size(data, 0)

    # Split generated params into weight and bias
    # weight_params: [batch, in_dim * out_dim + out_dim]
    weight_flat = Nx.slice_along_axis(weight_params, 0, in_dim * out_dim, axis: 1)
    bias = Nx.slice_along_axis(weight_params, in_dim * out_dim, out_dim, axis: 1)

    # Reshape weight: [batch, in_dim, out_dim]
    weight = Nx.reshape(weight_flat, {batch_size, in_dim, out_dim})

    # Batch matrix multiply: [batch, 1, in_dim] @ [batch, in_dim, out_dim]
    data_expanded = Nx.new_axis(data, 1)
    output = Nx.dot(data_expanded, [2], [0], weight, [1], [0])
    output = Nx.squeeze(output, axes: [1])

    # Add bias
    Nx.add(output, bias)
  end
end
