defmodule Edifice.Meta.Capsule do
  @moduledoc """
  Capsule Networks with dynamic routing (Sabour et al., 2017).

  Capsule Networks replace scalar neuron activations with vector "capsules"
  that encode both the probability of an entity's existence (vector length)
  and its instantiation parameters (vector direction). This preserves
  spatial hierarchies that CNNs lose through max-pooling.

  ## Key Concepts

  - **Capsule**: A group of neurons whose activity vector represents an entity.
    Vector length = probability of entity, direction = entity properties.
  - **Squash**: Non-linear activation that preserves direction but squashes
    length to [0, 1]: `v = (||s||^2 / (1 + ||s||^2)) * (s / ||s||)`
  - **Dynamic Routing**: Agreement-based routing where lower capsules send
    output to higher capsules that "agree" with their predictions.

  ## Architecture

  ```
  Input [batch, height, width, channels]
        |
        v
  +----------------------------+
  |    Conv Layer              |
  +----------------------------+
        |
        v
  +----------------------------+
  | Primary Capsule Layer      |
  | (Conv -> reshape to caps)  |
  +----------------------------+
        |
        v
  +----------------------------+
  | Dynamic Routing            |
  | (routing by agreement)     |
  +----------------------------+
        |
        v
  +----------------------------+
  | Digit/Output Capsules      |
  +----------------------------+
        |
        v
  Output: capsule vectors [batch, num_digit_caps, digit_cap_dim]
  Length of each capsule = class probability
  ```

  ## Usage

      model = Capsule.build(
        input_shape: {nil, 28, 28, 1},
        num_primary_caps: 32,
        primary_cap_dim: 8,
        num_digit_caps: 10,
        digit_cap_dim: 16,
        routing_iterations: 3
      )

  ## References
  - Sabour et al., "Dynamic Routing Between Capsules" (2017)
  - https://arxiv.org/abs/1710.09829
  """

  require Axon
  import Nx.Defn

  @default_num_primary_caps 32
  @default_primary_cap_dim 8
  @default_num_digit_caps 10
  @default_digit_cap_dim 16
  @default_routing_iterations 3

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a Capsule Network (CapsNet).

  ## Options
    - `:input_shape` - Input shape as `{nil, height, width, channels}` (required)
    - `:num_primary_caps` - Number of primary capsule types (default: 32)
    - `:primary_cap_dim` - Dimension of each primary capsule (default: 8)
    - `:num_digit_caps` - Number of output capsules (default: 10)
    - `:digit_cap_dim` - Dimension of each output capsule (default: 16)
    - `:routing_iterations` - Number of dynamic routing iterations (default: 3)
    - `:conv_channels` - Initial convolution channels (default: 256)
    - `:conv_kernel` - Initial convolution kernel size (default: 9)

  ## Returns
    An Axon model producing capsule norms `[batch, num_digit_caps]`
    representing class probabilities.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:conv_channels, pos_integer()}
          | {:conv_kernel, pos_integer()}
          | {:digit_cap_dim, pos_integer()}
          | {:input_shape, tuple()}
          | {:num_digit_caps, pos_integer()}
          | {:num_primary_caps, pos_integer()}
          | {:primary_cap_dim, pos_integer()}
          | {:routing_iterations, float()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_shape = Keyword.fetch!(opts, :input_shape)
    num_primary_caps = Keyword.get(opts, :num_primary_caps, @default_num_primary_caps)
    primary_cap_dim = Keyword.get(opts, :primary_cap_dim, @default_primary_cap_dim)
    num_digit_caps = Keyword.get(opts, :num_digit_caps, @default_num_digit_caps)
    digit_cap_dim = Keyword.get(opts, :digit_cap_dim, @default_digit_cap_dim)
    routing_iterations = Keyword.get(opts, :routing_iterations, @default_routing_iterations)
    conv_channels = Keyword.get(opts, :conv_channels, 256)
    conv_kernel = Keyword.get(opts, :conv_kernel, 9)

    input = Axon.input("input", shape: input_shape)

    # Initial convolution
    conv =
      Axon.conv(input, conv_channels,
        kernel_size: {conv_kernel, conv_kernel},
        padding: :valid,
        name: "initial_conv"
      )

    conv = Axon.relu(conv)

    # Primary capsule layer: outputs capsule vectors
    primary_caps =
      primary_capsule_layer(conv, num_primary_caps, primary_cap_dim, name: "primary_caps")

    # Dynamic routing to digit/output capsules
    digit_caps =
      dynamic_routing(primary_caps, num_digit_caps, digit_cap_dim,
        routing_iterations: routing_iterations,
        name: "digit_caps"
      )

    # Output: length of each digit capsule = class probability
    Axon.nx(
      digit_caps,
      fn caps ->
        # caps: [batch, num_digit_caps, digit_cap_dim]
        Nx.sqrt(Nx.add(Nx.sum(Nx.pow(caps, 2), axes: [2]), 1.0e-8))
      end,
      name: "capsule_norms"
    )
  end

  @doc """
  Build a primary capsule layer.

  Converts a standard convolutional feature map into capsule vectors.
  Uses convolution to produce `num_caps * cap_dim` channels, then
  reshapes into capsule vectors and applies the squash activation.

  ## Parameters
    - `input` - Axon node with conv features `[batch, height, width, channels]`
    - `num_caps` - Number of capsule types
    - `cap_dim` - Dimension of each capsule vector

  ## Options
    - `:kernel_size` - Convolution kernel size (default: 9)
    - `:strides` - Convolution strides (default: 2)
    - `:name` - Layer name prefix

  ## Returns
    An Axon node with shape `[batch, total_num_capsules, cap_dim]`
    where total_num_capsules = num_caps * spatial_positions
  """
  @spec primary_capsule_layer(Axon.t(), pos_integer(), pos_integer(), keyword()) :: Axon.t()
  def primary_capsule_layer(input, num_caps, cap_dim, opts \\ []) do
    kernel_size = Keyword.get(opts, :kernel_size, 9)
    strides = Keyword.get(opts, :strides, 2)
    name = Keyword.get(opts, :name, "primary_caps")

    # Convolution: output channels = num_caps * cap_dim
    total_channels = num_caps * cap_dim

    conv_out =
      Axon.conv(input, total_channels,
        kernel_size: {kernel_size, kernel_size},
        strides: [strides, strides],
        padding: :valid,
        name: "#{name}_conv"
      )

    # Reshape to capsule vectors and apply squash
    Axon.nx(
      conv_out,
      fn tensor ->
        # tensor: [batch, h, w, num_caps * cap_dim]
        batch = Nx.axis_size(tensor, 0)
        h = Nx.axis_size(tensor, 1)
        w = Nx.axis_size(tensor, 2)

        # Reshape: [batch, h * w * num_caps, cap_dim]
        total_caps = h * w * num_caps
        reshaped = Nx.reshape(tensor, {batch, total_caps, cap_dim})

        # Apply squash activation
        squash_impl(reshaped)
      end,
      name: "#{name}_squash"
    )
  end

  @doc """
  Squash activation function for capsule vectors.

  Non-linear "squashing" that preserves the direction of the vector
  but scales its magnitude to be between 0 and 1.

      v = (||s||^2 / (1 + ||s||^2)) * (s / ||s||)

  Short vectors get shrunk to near zero length, long vectors get
  shrunk to just below 1. Direction is preserved.

  ## Parameters
    - `tensor` - Input tensor `[..., cap_dim]`

  ## Returns
    Squashed tensor with same shape, magnitudes in [0, 1)
  """
  @spec squash(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn squash(tensor) do
    squash_impl(tensor)
  end

  defnp squash_impl(tensor) do
    # Squared norm along last axis
    squared_norm = Nx.sum(Nx.pow(tensor, 2), axes: [-1], keep_axes: true)
    norm = Nx.sqrt(squared_norm + 1.0e-8)

    # Scale factor: ||s||^2 / (1 + ||s||^2)
    scale = squared_norm / (1.0 + squared_norm)

    # Unit vector: s / ||s||
    unit = tensor / norm

    # Squashed output
    scale * unit
  end

  @doc """
  Dynamic routing by agreement between capsule layers.

  Lower-level capsules predict the output of higher-level capsules via
  learned transformation matrices. Routing coefficients are iteratively
  updated based on agreement between predictions and actual outputs.

  ## Algorithm
  1. Initialize routing logits b_ij = 0
  2. For each iteration:
     a. Compute routing coefficients: c_ij = softmax(b_ij)
     b. Compute weighted prediction sum: s_j = sum(c_ij * u_hat_ij)
     c. Apply squash: v_j = squash(s_j)
     d. Update logits: b_ij += u_hat_ij . v_j (agreement)

  ## Parameters
    - `input_caps` - Axon node with input capsules `[batch, num_input_caps, input_cap_dim]`
    - `num_output_caps` - Number of output capsules
    - `output_cap_dim` - Dimension of each output capsule

  ## Options
    - `:routing_iterations` - Number of routing iterations (default: 3)
    - `:name` - Layer name prefix

  ## Returns
    An Axon node with shape `[batch, num_output_caps, output_cap_dim]`
  """
  @spec dynamic_routing(Axon.t(), pos_integer(), pos_integer(), keyword()) :: Axon.t()
  def dynamic_routing(input_caps, num_output_caps, output_cap_dim, opts \\ []) do
    routing_iterations = Keyword.get(opts, :routing_iterations, @default_routing_iterations)
    name = Keyword.get(opts, :name, "routing")

    # Per-pair transformation matrices W_ij: each (input_cap, output_cap) pair
    # has its own transformation. Shape determined dynamically from input.
    w_param =
      Axon.param(
        "#{name}_W",
        fn input_shape ->
          # input_shape: {batch, num_input_caps, input_cap_dim}
          num_input_caps = elem(input_shape, 1)
          input_cap_dim = elem(input_shape, 2)
          {num_input_caps, input_cap_dim, num_output_caps * output_cap_dim}
        end,
        initializer: :glorot_uniform
      )

    # Apply per-capsule transformation + routing
    Axon.layer(
      &routing_impl/3,
      [input_caps, w_param],
      name: name,
      num_output_caps: num_output_caps,
      output_cap_dim: output_cap_dim,
      routing_iterations: routing_iterations,
      op_name: :dynamic_routing
    )
  end

  # ============================================================================
  # Private Implementation
  # ============================================================================

  # Dynamic routing with per-pair transformation matrices
  defp routing_impl(input_caps, w_param, opts) do
    num_output_caps = opts[:num_output_caps]
    output_cap_dim = opts[:output_cap_dim]
    routing_iterations = opts[:routing_iterations]

    batch = Nx.axis_size(input_caps, 0)
    num_input_caps = Nx.axis_size(input_caps, 1)

    # Per-capsule transform: u_hat_j|i = W_ij @ u_i
    # input_caps: [batch, num_input, cap_dim]
    # w_param: [num_input, cap_dim, num_output * out_dim]
    # Result: [batch, num_input, num_output * out_dim]
    #
    # For each input capsule i, multiply by its own W_i matrix.
    # This is an element-wise matmul along the num_input dimension.
    # input[:, i, :] @ w[i, :, :] for each i
    u_hat_flat =
      Nx.sum(
        Nx.multiply(
          # [batch, num_input, cap_dim, 1]
          Nx.new_axis(input_caps, 3),
          # [1, num_input, cap_dim, num_out*out_dim]
          Nx.new_axis(w_param, 0)
        ),
        axes: [2]
      )

    # Reshape: [batch, num_input_caps, num_output_caps, output_cap_dim]
    u_hat =
      Nx.reshape(
        u_hat_flat,
        {batch, num_input_caps, num_output_caps, output_cap_dim}
      )

    # Initialize routing logits to zero
    # b: [batch, num_input_caps, num_output_caps]
    b = Nx.broadcast(Nx.tensor(0.0), {batch, num_input_caps, num_output_caps})

    # Iterative routing
    {_b_final, v_final} =
      Enum.reduce(1..routing_iterations, {b, nil}, fn _iter, {b_curr, _v} ->
        # Routing coefficients: softmax over output capsules
        # c: [batch, num_input_caps, num_output_caps]
        max_b = Nx.reduce_max(b_curr, axes: [2], keep_axes: true)
        exp_b = Nx.exp(Nx.subtract(b_curr, max_b))
        c = Nx.divide(exp_b, Nx.sum(exp_b, axes: [2], keep_axes: true))

        # Weighted sum: s_j = sum_i(c_ij * u_hat_ij)
        # c: [batch, num_input_caps, num_output_caps, 1]
        # u_hat: [batch, num_input_caps, num_output_caps, output_cap_dim]
        c_expanded = Nx.new_axis(c, 3)
        weighted = Nx.multiply(c_expanded, u_hat)

        # Sum over input capsules: [batch, num_output_caps, output_cap_dim]
        s = Nx.sum(weighted, axes: [1])

        # Squash: v_j = squash(s_j)
        v = squash_impl(s)

        # Update routing logits: b_ij += u_hat_ij . v_j
        # v: [batch, 1, num_output_caps, output_cap_dim]
        v_expanded = Nx.new_axis(v, 1)

        # Agreement: dot product between prediction and actual output
        # [batch, num_input_caps, num_output_caps]
        agreement = Nx.sum(Nx.multiply(u_hat, v_expanded), axes: [3])

        b_new = Nx.add(b_curr, agreement)

        {b_new, v}
      end)

    v_final
  end
end
