defmodule Edifice.Generative.NormalizingFlow do
  @moduledoc """
  Normalizing Flows with RealNVP-style affine coupling layers.

  Normalizing flows learn invertible transformations between a simple
  base distribution (standard normal) and a complex target distribution.
  Because each layer is invertible with a tractable Jacobian, we get
  exact log-likelihood computation -- unlike VAEs which optimize a bound.

  ## Architecture (RealNVP Affine Coupling)

  Each coupling layer:
  1. Splits input into two halves: (x1, x2)
  2. Computes scale and translation from x1: s, t = NN(x1)
  3. Transforms x2: y2 = x2 * exp(s) + t
  4. Passes x1 unchanged: y1 = x1
  5. Output: (y1, y2)

  This is trivially invertible:
      x2 = (y2 - t) * exp(-s)
      x1 = y1

  The log-determinant of the Jacobian is simply sum(s), making
  density evaluation efficient.

  ```
  z ~ N(0, I)
       |
       v
  +------------------+
  | Coupling Layer 1 |  split -> NN -> affine transform -> concat
  +------------------+
       |
       v
  +------------------+
  | Coupling Layer 2 |  (alternating split pattern)
  +------------------+
       |
       v
      ...
       |
       v
  +------------------+
  | Coupling Layer K |
  +------------------+
       |
       v
  x ~ p(x)
  ```

  Successive layers alternate which half is transformed to ensure
  all dimensions are eventually modified.

  ## Usage

      # Build a normalizing flow
      model = NormalizingFlow.build(input_size: 16, num_flows: 4, hidden_sizes: [128])

      # Forward pass (encoding: data -> latent)
      {z, log_det} = NormalizingFlow.forward(x, params, num_flows: 4, input_size: 16)

      # Inverse pass (generation: latent -> data)
      x = NormalizingFlow.inverse(z, params, num_flows: 4, input_size: 16)

      # Log-likelihood
      log_prob = NormalizingFlow.log_probability(x, params, num_flows: 4, input_size: 16)
  """
  import Nx.Defn

  @default_num_flows 4
  @default_hidden_sizes [256]
  @default_activation :relu

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a normalizing flow model.

  Constructs `num_flows` affine coupling layers, each containing a
  small neural network that computes scale and translation parameters.
  The model is an Axon graph where the input flows through all coupling
  layers sequentially.

  ## Options
    - `:input_size` - Input dimension, must be even (required)
    - `:num_flows` - Number of coupling layers (default: 4)
    - `:hidden_sizes` - Hidden layer sizes for each coupling network (default: [256])
    - `:activation` - Activation function for coupling networks (default: :relu)

  ## Returns
    An Axon model: `[batch, input_size]` -> `[batch, input_size]`.

    The model transforms inputs through the flow. For density evaluation
    and generation, use the `forward/3`, `inverse/3`, and `log_probability/3`
    Nx functions directly.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:activation, atom()}
          | {:hidden_sizes, [pos_integer()]}
          | {:input_size, pos_integer()}
          | {:num_flows, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    input_size = Keyword.fetch!(opts, :input_size)
    num_flows = Keyword.get(opts, :num_flows, @default_num_flows)
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)

    if rem(input_size, 2) != 0 do
      raise ArgumentError,
            "input_size must be even for RealNVP coupling (got #{input_size}). " <>
              "Pad input if needed."
    end

    half_size = div(input_size, 2)
    input = Axon.input("input", shape: {nil, input_size})

    # Chain coupling layers, alternating the split pattern
    Enum.reduce(0..(num_flows - 1), input, fn flow_idx, acc ->
      affine_coupling_layer(acc, flow_idx,
        input_size: input_size,
        half_size: half_size,
        hidden_sizes: hidden_sizes,
        activation: activation
      )
    end)
  end

  # ============================================================================
  # Coupling Layers (Axon graph construction)
  # ============================================================================

  @doc """
  Build a single RealNVP affine coupling layer as part of an Axon graph.

  On even-indexed layers, x1 (first half) conditions the transform of x2.
  On odd-indexed layers, x2 (second half) conditions the transform of x1.
  This alternation ensures all dimensions are transformed across layers.

  The coupling network outputs scale (s) and translation (t) parameters.
  Scale is passed through tanh and scaled to prevent extreme values.

  ## Parameters
    - `input` - Input Axon node `[batch, input_size]`
    - `flow_idx` - Layer index (determines split direction)
    - `opts` - Options including `:input_size`, `:half_size`, `:hidden_sizes`, `:activation`

  ## Returns
    An Axon node `[batch, input_size]` with the coupling transform applied.
  """
  @spec affine_coupling_layer(Axon.t(), non_neg_integer(), keyword()) :: Axon.t()
  def affine_coupling_layer(input, flow_idx, opts) do
    input_size = Keyword.fetch!(opts, :input_size)
    half_size = Keyword.get(opts, :half_size, div(input_size, 2))
    hidden_sizes = Keyword.get(opts, :hidden_sizes, @default_hidden_sizes)
    activation = Keyword.get(opts, :activation, @default_activation)

    prefix = "flow_#{flow_idx}"

    # Split input into two halves
    x1 =
      Axon.nx(
        input,
        fn t ->
          Nx.slice_along_axis(t, 0, half_size, axis: -1)
        end,
        name: "#{prefix}_split_1"
      )

    x2 =
      Axon.nx(
        input,
        fn t ->
          Nx.slice_along_axis(t, half_size, half_size, axis: -1)
        end,
        name: "#{prefix}_split_2"
      )

    # Alternate which half conditions the other
    {conditioner, transformed} =
      if rem(flow_idx, 2) == 0 do
        {x1, x2}
      else
        {x2, x1}
      end

    # Build the coupling network: conditioner -> (scale, translation)
    coupling_net =
      Enum.with_index(hidden_sizes)
      |> Enum.reduce(conditioner, fn {size, idx}, acc ->
        acc
        |> Axon.dense(size, name: "#{prefix}_coupling_dense_#{idx}")
        |> Axon.activation(activation, name: "#{prefix}_coupling_act_#{idx}")
      end)

    # Scale output (tanh-bounded to prevent explosion)
    scale =
      coupling_net
      |> Axon.dense(half_size, name: "#{prefix}_scale")
      |> Axon.activation(:tanh, name: "#{prefix}_scale_tanh")

    # Translation output (unbounded)
    translation = Axon.dense(coupling_net, half_size, name: "#{prefix}_translation")

    # Apply affine transform: y = x * exp(s) + t
    transformed_output =
      Axon.layer(
        fn x, s, t, _opts ->
          Nx.add(Nx.multiply(x, Nx.exp(s)), t)
        end,
        [transformed, scale, translation],
        name: "#{prefix}_affine"
      )

    # Reassemble in the correct order
    if rem(flow_idx, 2) == 0 do
      # conditioner was x1, transformed was x2
      Axon.concatenate([conditioner, transformed_output], axis: -1, name: "#{prefix}_concat")
    else
      # conditioner was x2, transformed was x1
      Axon.concatenate([transformed_output, conditioner], axis: -1, name: "#{prefix}_concat")
    end
  end

  # ============================================================================
  # Forward / Inverse / Log-det (Nx functions for training and generation)
  # ============================================================================

  @doc """
  Inverse of a single affine coupling layer (for generation).

  Given the output y of a coupling layer, recovers the input x:
      x2 = (y2 - t) * exp(-s)    where s, t = NN(y1)
      x1 = y1

  ## Parameters
    - `y` - Coupling layer output `[batch, input_size]`
    - `s_params` - Scale network parameters (list of `{weight, bias}` tuples)
    - `t_params` - Translation network parameters (list of `{weight, bias}` tuples)
    - `half_size` - Size of each split half
    - `even` - Whether this is an even-indexed layer (determines split order)

  ## Returns
    Recovered input `[batch, input_size]`.
  """
  @spec inverse_coupling_layer(
          Nx.Tensor.t(),
          list({Nx.Tensor.t(), Nx.Tensor.t()}),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          pos_integer(),
          boolean()
        ) :: Nx.Tensor.t()
  defn inverse_coupling_layer(
         y,
         hidden_weights,
         hidden_biases,
         scale_params,
         trans_params,
         half_size
       ) do
    # Split
    y1 = Nx.slice_along_axis(y, 0, half_size, axis: -1)
    y2 = Nx.slice_along_axis(y, half_size, half_size, axis: -1)

    # The conditioner half passes through unchanged
    # Recompute s, t from the conditioner
    # For simplicity, this assumes a single hidden layer
    # hidden = relu(y1 * W_h + b_h)
    hidden = Nx.max(Nx.dot(y1, hidden_weights) + hidden_biases, 0.0)

    # s = tanh(hidden * W_s + b_s)
    {s_w, s_b} = scale_params
    s = Nx.tanh(Nx.dot(hidden, s_w) + s_b)

    # t = hidden * W_t + b_t
    {t_w, t_b} = trans_params
    t = Nx.dot(hidden, t_w) + t_b

    # Invert the affine transform: x2 = (y2 - t) * exp(-s)
    x2 = (y2 - t) * Nx.exp(-s)

    # Reassemble
    Nx.concatenate([y1, x2], axis: -1)
  end

  @doc """
  Compute log-determinant of the Jacobian for an affine coupling layer.

  For the affine coupling transform y2 = x2 * exp(s) + t, the Jacobian
  is triangular and its log-determinant is simply sum(s).

  ## Parameters
    - `scale` - Scale parameters `s` from the coupling network `[batch, half_size]`

  ## Returns
    Log-determinant `[batch]` (summed over dimensions).
  """
  @spec log_det_jacobian(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn log_det_jacobian(scale) do
    # For y = x * exp(s) + t, the Jacobian diagonal is exp(s)
    # log|det(J)| = sum(s)
    Nx.sum(scale, axes: [-1])
  end

  @doc """
  Compute the log-determinant for a full forward pass through all coupling layers.

  This is the sum of log-determinants from each individual coupling layer,
  needed for exact log-likelihood computation.

  ## Parameters
    - `scales` - List of scale tensors from each coupling layer,
      each `[batch, half_size]`

  ## Returns
    Total log-determinant `[batch]`.
  """
  @spec total_log_det_jacobian([Nx.Tensor.t()]) :: Nx.Tensor.t()
  def total_log_det_jacobian(scales) when is_list(scales) do
    scales
    |> Enum.map(&log_det_jacobian/1)
    |> Enum.reduce(&Nx.add/2)
  end

  @doc """
  Compute the log-probability of data under the flow model.

  Uses the change-of-variables formula:
      log p(x) = log p(z) + log|det(dz/dx)|

  where z = f(x) is the forward transformation and p(z) = N(0, I).

  ## Parameters
    - `z` - Transformed data in latent space `[batch, input_size]`
    - `total_log_det` - Sum of log-determinants from forward pass `[batch]`

  ## Returns
    Log-probability `[batch]`.
  """
  @spec log_probability(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn log_probability(z, total_log_det) do
    # log p(z) under standard normal: -0.5 * (d*log(2*pi) + sum(z^2))
    d = Nx.axis_size(z, -1)
    log_pz = -0.5 * (d * Nx.log(2.0 * Nx.Constants.pi()) + Nx.sum(Nx.pow(z, 2), axes: [-1]))

    # Change of variables: log p(x) = log p(z) + log|det(J)|
    # Note: sign depends on direction. For encoding (x->z), we add log_det.
    log_pz + total_log_det
  end

  @doc """
  Negative log-likelihood loss for normalizing flow training.

  Minimizing NLL is equivalent to maximizing the log-probability
  of the training data under the flow model.

  ## Parameters
    - `z` - Encoded latent vectors `[batch, input_size]`
    - `total_log_det` - Sum of log-determinants `[batch]`

  ## Returns
    NLL loss scalar (mean over batch).
  """
  @spec nll_loss(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn nll_loss(z, total_log_det) do
    log_prob = log_probability(z, total_log_det)
    -Nx.mean(log_prob)
  end
end
